"""DistributedOffloadAdam: dim-0 sharded optimizer state (ZeRO-1 style).

Design axes — every param has a role along two orthogonal axes:

- ``state_location`` ∈ {device, host}: small params (below
  ``inplace_param_threshold``) keep optimizer state resident on GPU;
  the rest live in pinned host memory.
- ``shard_strategy`` ∈ {sharded, replicated}: offload params with
  ``p.shape[0] % world_size == 0`` are dim-0 sharded (state and post-reduce
  grad are slice-shaped); everything else is replicated per rank. Inplace
  state is always replicated in v1.

Grad reduction always happens on the last microbatch inside the grad hook,
so norm / step / d2h downstream always see the correctly-reduced grad:

- sharded:    ``reduce_scatter`` full → slice (slice stays on device until
              the step consumes it; host grad buffer is never written post-
              reduce because of the shape mismatch).
- replicated: ``all_reduce`` in place (full-shape grad → averaged full grad,
              then d2h for offload / stays on GPU for inplace).

After the kernel runs on the (slice / full) grad, sharded params
``all_gather`` their updated dim-0 slice back to every rank.

QAT: ``p.data`` (packed) is replicated per rank; the fp32 master is
sharded when ``p.shape[0] % world_size == 0``. After step_fn on the slice,
the master slices are all-gathered and ``p.data.copy_(full_master)``
re-encodes the packed weight on every rank.
"""

import torch
import torch.distributed as dist

from .offload_adam import OffloadAdam
from .pinned_alloc import zeros_pinned


class DistributedOffloadAdam(OffloadAdam):
    """**Experimental.** Sharded-state variant of :class:`OffloadAdam`
    (ZeRO-1 style). See module docstring for the design. Tested on a
    narrow set of topologies; expect rough edges outside the validated
    configurations."""

    def __init__(self, model, *, process_group=None, **kwargs):
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "DistributedOffloadAdam requires torch.distributed to be "
                "initialized; call dist.init_process_group first."
            )
        self._pg = process_group if process_group is not None else dist.group.WORLD
        self._world_size = dist.get_world_size(self._pg)
        self._rank = dist.get_rank(self._pg)
        self._sharded = set()
        super().__init__(model, **kwargs)

    def _shard_slice(self, p):
        """Local dim-0 slice view into ``p.data``."""
        ds = p.shape[0] // self._world_size
        return p.data[self._rank * ds : (self._rank + 1) * ds]

    def _all_gather_param(self, p):
        local = self._shard_slice(p).contiguous().view(-1)
        dist.all_gather_into_tensor(p.data.view(-1), local, group=self._pg)

    def _reduce_grad_to_slice(self, p):
        """Reduce-scatter ``device_states[p]["grad"]`` (full) to its dim-0
        slice; replace the entry in place so downstream kernels see the slice.
        """
        ds = p.shape[0] // self._world_size
        dev_grad = self.device_states[p]["grad"]
        slice_grad = torch.empty(
            (ds, *p.shape[1:]),
            dtype=dev_grad.dtype,
            device=dev_grad.device,
        )
        dist.reduce_scatter_tensor(
            slice_grad.view(-1),
            dev_grad.contiguous().view(-1),
            op=dist.ReduceOp.AVG,
            group=self._pg,
        )
        self.device_states[p]["grad"] = slice_grad

    def _reduce_grad(self, p):
        """Single entry point for last-microbatch grad reduction. Dispatches
        to reduce_scatter (sharded) or all_reduce (replicated), and reads
        from the right buffer (``state[p]`` for inplace, ``device_states[p]``
        for offload)."""
        if self._world_size <= 1:
            return
        if p in self._sharded:
            self._reduce_grad_to_slice(p)
        elif p in self._inplace_params:
            dist.all_reduce(
                self.state[p]["grad"],
                op=dist.ReduceOp.AVG,
                group=self._pg,
            )
        else:
            dist.all_reduce(
                self.device_states[p]["grad"],
                op=dist.ReduceOp.AVG,
                group=self._pg,
            )

    def _is_sharded_shape(self, p):
        """Whether ``p`` will be dim-0 sharded across ranks."""
        return self._world_size > 1 and p.shape[0] % self._world_size == 0

    def _estimate_pinned_bytes(self, params):
        """Grad is full-shape per rank; non-grad state is 1/world_size
        when sharded, else full."""
        need = 0
        for p in params:
            sharded = self._is_sharded_shape(p)
            for name, dtype in self.offload_config.items():
                elem_bytes = p.numel() * torch.empty(0, dtype=dtype).element_size()
                if sharded and name != "grad":
                    elem_bytes //= self._world_size
                need += elem_bytes
        return need

    def _alloc_pinned_states(self, params, numa_node):
        """Grad stays full-shape per rank; non-grad state is dim-0 sharded
        when ``p.shape[0] % world_size == 0``, else replicated."""
        total_bytes = 0
        for p in params:
            state = self.state[p]
            is_quant = p in self._quant_params
            sharded = self._is_sharded_shape(p)
            ds = p.shape[0] // self._world_size if sharded else None
            shard_shape = (ds, *p.shape[1:]) if sharded else p.shape
            if sharded:
                self._sharded.add(p)
            for name, dtype in self.offload_config.items():
                shape = p.shape if name == "grad" else shard_shape
                t = zeros_pinned(shape, dtype, numa_node=numa_node)
                if name == "master_params":
                    # QWeight has no flat-slice view: dequant to fp32 first.
                    if is_quant:
                        tmp = torch.empty(
                            p.shape,
                            dtype=torch.float32,
                            device="cpu",
                        )
                        tmp.copy_(p.data)
                        src = tmp
                    else:
                        src = p.data
                    if sharded:
                        t.copy_(src[self._rank * ds : (self._rank + 1) * ds])
                    else:
                        t.copy_(src)
                state[name] = t
                total_bytes += t.numel() * t.element_size()
            if is_quant and "master_params" in self.offload_config:
                state["master_filled"] = False
        if self.verbose > 0 and params:
            print(
                f"[rank {self._rank}] Pinned host memory: "
                f"{total_bytes / (1024**3):.2f} GB"
            )

    @torch.no_grad()
    def _inplace_grad_hook(self, p):
        """Inplace (replicated, on-device) grad hook: accumulate, reduce on
        the last microbatch, then record norm / step."""
        if p.grad is None:
            return
        state = self.state[p]
        state["grad"].add_(p.grad)
        p.grad = None
        if not self.ready_for_optimizer_step:
            return
        self._reduce_grad(p)
        if self.max_grad_norm is not None:
            state["grad_norm"] = torch.norm(state["grad"], 2.0)
        if self._step_in_backward:
            state["step"] += 1
            self._step_inplace(p, self._param2group[p])

    @torch.no_grad()
    def _grad_hook(self, p):
        """Offload grad hook. Reduction happens here on the last microbatch
        (so downstream norm / step see the correctly-reduced grad), not
        inside ``_step_offload``. Between microbatches we d2h the running
        full-shape accumulator; after the last-mb reduce, sharded params
        hold their slice-grad on device until ``.step()`` (or the step-in-
        backward branch) consumes it — the slice shape doesn't fit the
        full-shape host buffer."""
        if p.grad is None:
            return
        self._accumulate_grad_on_device(p)
        state = self.state[p]
        group = self._param2group[p]
        if not self.ready_for_optimizer_step:
            self._issue_d2h(p, ["grad"])
            return
        self._reduce_grad(p)
        if self.max_grad_norm is not None:
            state["grad_norm"] = torch.norm(self.device_states[p]["grad"], 2.0)
        if self._step_in_backward:
            state["step"] += 1
            self._step_offload(p, group)
            state["host_grad_valid"] = False
        elif p not in self._sharded:
            # Replicated offload: full-shape grad fits the host buffer; d2h
            # now so ``.step()`` can rehydrate it later.
            self._issue_d2h(p, ["grad"])
        # Sharded: slice grad stays on device, consumed by ``.step()`` below.

    def _call_step_fn(self, p, group, device_states):
        """For sharded params feed the kernel the dim-0 slice (or the grad
        sink for quant); non-sharded defers to super."""
        if p not in self._sharded:
            return super()._call_step_fn(p, group, device_states)
        state = self.state[p]
        beta1, beta2 = group["betas"]
        is_quant = p in self._quant_params
        if is_quant:
            self._maybe_warn_quant_master(state)
            # bf16 sink — re-encode done by _step_offload via all-gather.
            param_view = device_states["grad"]
        else:
            param_view = self._shard_slice(p)
        self.step_fn(
            param_view,
            device_states,
            group["lr"],
            group["weight_decay"],
            beta1,
            beta2,
            group["eps"],
            state["step"],
            decoupled_weight_decay=self.decoupled_weight_decay,
        )

    def _step_offload(self, p, group, clip_coef=None):
        """Sharded path. Precondition: ``device_states[p]["grad"]`` already
        holds the slice-shape reduced grad (written by ``_reduce_grad`` in
        the grad hook on the last microbatch). Kernel runs on the slice,
        then param / master slices are all-gathered back to every rank."""
        if p not in self._sharded:
            return super()._step_offload(p, group, clip_coef=clip_coef)
        device_states = self.device_states[p]
        self._ensure_on_device(p, self._non_grad_keys)
        if clip_coef is not None:
            device_states["grad"].mul_(clip_coef)
        self._call_step_fn(p, group, device_states)
        if p in self._quant_params:
            # Gather full fp32 master; QWeightBase.copy_ re-encodes p.data.
            full_master = torch.empty(
                p.shape,
                dtype=torch.float32,
                device=p.device,
            )
            dist.all_gather_into_tensor(
                full_master.view(-1),
                device_states["master_params"].contiguous().view(-1),
                group=self._pg,
            )
            p.data.copy_(full_master)
        else:
            self._all_gather_param(p)
        self._issue_d2h(p, self._non_grad_keys)
        device_states["grad"] = None

    @torch.no_grad()
    def step(self, closure=None):
        """Clipping path only. Grad hooks have already reduced each param's
        grad on the last microbatch, so ``state[p]["grad_norm"]`` is the
        norm of the post-reduce grad. Here we just assemble the global L2
        norm across ranks and per-param."""
        if closure is not None:
            raise NotImplementedError(
                "DistributedOffloadAdam does not support closure-based step."
            )
        if self._step_in_backward or self.max_grad_norm is None:
            return super().step()
        # Sharded params: each rank has ||slice||² of the reduced grad;
        #   Σ_ranks ||slice_r||² = ||full_reduced_grad||² (disjoint partition).
        # Replicated params: every rank holds the same ||full_reduced_grad||²;
        #   count it once (rank 0 only) so the cross-rank SUM below doesn't
        #   inflate by world_size.
        local_sq = torch.zeros((), device="cuda")
        for group in self.param_groups:
            for p in group["params"]:
                n = self.state[p]["grad_norm"]
                if p in self._sharded or self._rank == 0:
                    local_sq = local_sq + n * n
        if self._world_size > 1:
            dist.all_reduce(local_sq, op=dist.ReduceOp.SUM, group=self._pg)
        total_norm = torch.sqrt(local_sq)
        clip_coef = torch.clamp(
            self.max_grad_norm / (total_norm + 1e-6),
            max=1.0,
        )
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] += 1
                if p in self._inplace_params:
                    self._step_inplace(p, group, clip_coef=clip_coef)
                else:
                    self._step_offload(p, group, clip_coef=clip_coef)
                    state["host_grad_valid"] = False

    def _copy_master(self, p, tensor):
        """Shard-aware: slice the source tensor to this rank's portion for
        sharded params, otherwise full-shape copy."""
        if p in self._sharded:
            ds = p.shape[0] // self._world_size
            tensor = tensor[self._rank * ds : (self._rank + 1) * ds]
        self.state[p]["master_params"].copy_(tensor)
