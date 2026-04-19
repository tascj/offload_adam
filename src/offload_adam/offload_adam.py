"""OffloadAdam: Adam with gradients and optimizer states offloaded to host memory.

Two execution paths share the same transfer primitives and are selected by
whether `max_grad_norm` is set:

- No clipping → step runs inside each param's post-accumulate-grad hook on
  the last micro-batch, overlapping load/compute/writeback with backward.
- Clipping → backward only accumulates grad and per-param norms; the full
  step loop (global clip, load/compute/writeback) runs in `.step()`.
"""

import warnings

import torch
from torch.optim.optimizer import Optimizer

from ._pretrained import stream_master_from_pretrained
from .kernels import (
    adam_step_fp31_master,
    adam_step_fp32_master,
    adam_step_fp32_master_custom_rounding,
    adam_step_stochastic_rounding,
)
from .pinned_alloc import zeros_pinned
from .qweight.base import QWeightBase


def get_leaf_modules_with_params(module):
    """Recursively collect leaf modules with parameters from a PyTorch model."""
    leaf_modules = []
    for child in module.children():
        if list(child.children()):
            leaf_modules.extend(get_leaf_modules_with_params(child))
        elif any(p.requires_grad for p in child.parameters(recurse=False)):
            leaf_modules.append(child)
    return leaf_modules


class OffloadAdam(Optimizer):
    """Adam optimizer that offloads gradients and optimizer states to host memory.

    Args:
        model (nn.Module): Model containing parameters to optimize.
        lr (float): Learning rate. Default: 1e-3.
        betas (Tuple[float, float]): Running-average coefficients.
            Default: (0.9, 0.999).
        eps (float): Denominator term for numerical stability. Default: 1e-8.
        weight_decay (float): Weight decay coefficient. Default: 0.01.
        mode (str): One of `'stochastic_rounding'`, `'fp32_master'`,
            `'fp31_master'`, `'fp32_master_custom_rounding'`. Default:
            `'stochastic_rounding'`.
        max_grad_norm (float | None): If set, apply L2 global-norm clipping
            with this max norm before the step. `None` (default) disables
            clipping and routes the step into the backward hook for maximum
            backward/step overlap.
        numa_node (str | int | None): NUMA policy for pinned allocations —
            `'auto'`, an int node id, or `None`.
        decoupled_weight_decay (bool): Decouple weight decay from gradient
            (AdamW style). Default: False.
        inplace_param_threshold (int | None): Params with fewer elements than
            this stay fully on GPU (states + grad), bypassing PCIe entirely.
            The default covers typical LLM normalization weights and biases
            while leaving attention/MLP projections and embeddings on the
            offload path. Set `0` to force every param onto the offload path.
            Default: 1M elements.
        prefetch_policy (str): `'eager'` (default) prefetches at the first
            `pre_backward_hook` fire that sees a param. For params shared
            across multiple leaf modules (tied embeddings) this pins the
            non-grad states on GPU for the duration of backward. `'lazy'`
            defers prefetch to the leaf module that owns each param
            earliest in forward order — for shared params this collapses
            the residency at a potential cost in PCIe/compute overlap.
        verbose (int): Non-zero prints total pinned bytes allocated.
    """

    supported_modes = {
        "stochastic_rounding": {
            "step": adam_step_stochastic_rounding,
            "offload": {
                "grad": torch.bfloat16,
                "exp_avg": torch.bfloat16,
                "exp_avg_sq": torch.bfloat16,
            },
        },
        "fp32_master": {
            "step": adam_step_fp32_master,
            "offload": {
                "grad": torch.bfloat16,
                "exp_avg": torch.bfloat16,
                "exp_avg_sq": torch.bfloat16,
                "master_params": torch.float32,
            },
        },
        "fp31_master": {
            "step": adam_step_fp31_master,
            "offload": {
                "grad": torch.bfloat16,
                "exp_avg": torch.bfloat16,
                "exp_avg_sq": torch.bfloat16,
                "rounding_error": torch.int16,
            },
        },
        "fp32_master_custom_rounding": {
            "step": adam_step_fp32_master_custom_rounding,
            "offload": {
                "grad": torch.bfloat16,
                "exp_avg": torch.bfloat16,
                "exp_avg_sq": torch.bfloat16,
                "rounding_error": torch.int16,
            },
        },
    }

    _DEFAULT_INPLACE_PARAM_THRESHOLD = 1024 * 1024

    def __init__(
        self,
        model,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        mode="stochastic_rounding",
        max_grad_norm=None,
        numa_node="auto",
        decoupled_weight_decay=False,
        inplace_param_threshold=None,
        prefetch_policy="eager",
        verbose=0,
    ):
        if prefetch_policy not in ("eager", "lazy"):
            raise ValueError(
                f"prefetch_policy must be 'eager' or 'lazy' (got "
                f"{prefetch_policy!r})"
            )
        self._prefetch_policy = prefetch_policy
        assert mode in self.supported_modes, (
            f"Invalid mode: {mode}, available modes: {list(self.supported_modes)}"
        )
        self.mode = mode
        self.step_fn = self.supported_modes[mode]["step"]
        self.offload_config = self.supported_modes[mode]["offload"]
        self.offload_state_keys = list(self.offload_config.keys())
        self._non_grad_keys = [k for k in self.offload_state_keys if k != "grad"]

        self.max_grad_norm = max_grad_norm
        # Without clipping we fuse load/compute/writeback into backward
        # hooks. Clipping needs a global norm, which forces the step back
        # into `.step()`.
        self._step_in_backward = max_grad_norm is None

        self.decoupled_weight_decay = decoupled_weight_decay
        self.verbose = verbose
        self.ready_for_optimizer_step = False

        modules = get_leaf_modules_with_params(model)
        # Dedupe by identity so tied embeddings don't enter the optimizer
        # twice (`.step()` would otherwise apply Adam twice).
        seen = set()
        params = []
        module_count = {}
        for m in modules:
            for p in m.parameters():
                if not p.requires_grad:
                    continue
                module_count[id(p)] = module_count.get(id(p), 0) + 1
                if id(p) not in seen:
                    seen.add(id(p))
                    params.append(p)
        shared_params = [p for p in params if module_count[id(p)] > 1]
        if shared_params and self._prefetch_policy == "eager":
            total_elems = sum(p.numel() for p in shared_params)
            warnings.warn(
                f"OffloadAdam: {len(shared_params)} parameter(s) shared "
                f"across multiple leaf modules ({total_elems:,} elements; "
                f"e.g. tied embeddings). With prefetch_policy='eager' their "
                f"non-grad states sit on GPU through the whole backward. "
                f"Pass prefetch_policy='lazy' to defer prefetch to the "
                f"input-side module.",
                stacklevel=2,
            )
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # Quant params: fp32 master + re-quant via dispatch.
        self._quant_params = {
            p for p in params if isinstance(p.data, QWeightBase)
        }
        if self._quant_params and mode != "fp32_master":
            raise ValueError(
                f"QAT params require mode='fp32_master'; got {mode!r} "
                f"({len(self._quant_params)} quant params detected)"
            )
        # One-shot warning flag (not persisted). The per-state
        # `master_filled` bool is what survives state_dict round-trips.
        self._quant_master_warned = False

        # Partition params: small ones stay fully on GPU; big ones get offloaded.
        # Quant params are always offloaded — their master sits on host pinned.
        if inplace_param_threshold is None:
            inplace_param_threshold = self._DEFAULT_INPLACE_PARAM_THRESHOLD
        self.inplace_param_threshold = inplace_param_threshold
        self._inplace_params = {
            p for p in params
            if p.numel() < inplace_param_threshold
            and p not in self._quant_params
        }
        self._offload_params = [
            p for p in params if p not in self._inplace_params
        ]

        self.device_states = {}
        self.h2d_events = {}
        self.d2h_events = {}
        self.h2d_stream = torch.cuda.Stream()
        self.d2h_stream = torch.cuda.Stream()

        self._param2group = {
            p: group for group in self.param_groups for p in group["params"]
        }
        # First forward-ordered module that owns each offload param;
        # 'lazy' policy gates prefetch on this so a shared param's
        # h2d lands at its last backward fire, not its first.
        self._prefetch_module = {}
        for m in modules:
            for p in m.parameters():
                if p.requires_grad and p not in self._inplace_params:
                    self._prefetch_module.setdefault(p, m)
        self._register_hooks(modules, params)
        self._alloc_pinned_states(self._offload_params, numa_node)
        self._alloc_inplace_states(self._inplace_params)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _register_hooks(self, modules, params):
        # pre_backward per-module (need module entry timing); grad hook
        # per unique param so tied embeds don't get double-registered.
        for module in modules:
            if any(
                p.requires_grad and p not in self._inplace_params
                for p in module.parameters()
            ):
                module.register_full_backward_pre_hook(self._pre_backward_hook)
        for p in params:
            state = self.state[p]
            state["step"] = 0
            if self.max_grad_norm is not None:
                state["grad_norm"] = torch.tensor(0.0, device="cuda")
            if p in self._inplace_params:
                p.register_post_accumulate_grad_hook(self._inplace_grad_hook)
            else:
                p.register_post_accumulate_grad_hook(self._grad_hook)
                self.device_states[p] = {}
                state["host_grad_valid"] = False

    def _alloc_pinned_states(self, params, numa_node):
        total_bytes = 0
        for p in params:
            state = self.state[p]
            is_quant = p in self._quant_params
            for name, dtype in self.offload_config.items():
                t = zeros_pinned(p.shape, dtype, numa_node=numa_node)
                if name == "master_params":
                    # Quant: `copy_` dequantizes — lossy seed. Plain: lossless widen.
                    # `master_filled=False` below flags lossy for load_master_from_pretrained.
                    t.copy_(p.data)
                state[name] = t
                total_bytes += t.numel() * t.element_size()
            if is_quant and "master_params" in self.offload_config:
                state["master_filled"] = False
        if self.verbose > 0 and params:
            print(
                f"Pinned host memory allocated: {total_bytes / (1024 ** 3):.2f} GB"
            )

    def _alloc_inplace_states(self, params):
        """Allocate optimizer states on GPU for params kept in-place.

        Grad starts at zero and is zeroed after each step, so the grad-hook
        can unconditionally `add_(p.grad)` without an adopt/accumulate flag.
        """
        total_bytes = 0
        for p in params:
            state = self.state[p]
            for name, dtype in self.offload_config.items():
                t = torch.zeros_like(p, dtype=dtype)
                if name == "master_params":
                    t.copy_(p.data)
                state[name] = t
                total_bytes += t.numel() * t.element_size()
        if self.verbose > 0 and params:
            print(
                f"In-place GPU states for {len(params)} small params: "
                f"{total_bytes / (1024 ** 2):.2f} MB"
            )

    # ------------------------------------------------------------------
    # Layer 1 — transfer primitives
    # ------------------------------------------------------------------

    def _host_view(self, p, name):
        return self.state[p][name]

    def _issue_h2d(self, p, keys):
        main_stream = torch.cuda.current_stream()
        device_states = self.device_states[p]
        if p in self.d2h_events:
            self.h2d_stream.wait_event(self.d2h_events[p])
        self.h2d_stream.wait_stream(main_stream)
        for key in keys:
            with torch.cuda.stream(self.h2d_stream):
                device_states[key] = self._host_view(p, key).to(
                    p.device, non_blocking=True
                )
        self.h2d_events[p] = self.h2d_stream.record_event()

    def _issue_d2h(self, p, keys):
        # No explicit wait on h2d_events[p]: every caller already routes the
        # device tensor through main_stream (accumulate / step_fn), and
        # wait_stream(main_stream) below propagates that h2d dependency.
        main_stream = torch.cuda.current_stream()
        device_states = self.device_states[p]
        self.d2h_stream.wait_stream(main_stream)
        for key in keys:
            with torch.cuda.stream(self.d2h_stream):
                self._host_view(p, key).copy_(
                    device_states[key], non_blocking=True
                )
            device_states[key].record_stream(self.d2h_stream)
            device_states[key] = None
        self.d2h_events[p] = self.d2h_stream.record_event()

    def _ensure_on_device(self, p, keys):
        """Issue h2d for any `keys` not yet on device, then make
        main_stream wait on the h2d event."""
        device_states = self.device_states[p]
        missing = [k for k in keys if device_states.get(k) is None]
        if missing:
            self._issue_h2d(p, missing)
        if p in self.h2d_events:
            torch.cuda.current_stream().wait_event(self.h2d_events[p])

    # ------------------------------------------------------------------
    # Layer 2 — compound per-param ops
    # ------------------------------------------------------------------

    def _accumulate_grad_on_device(self, p):
        """Fold p.grad into the device-resident grad buffer.

        First micro-batch of a window adopts p.grad as the device grad; later
        micro-batches wait for the prefetched host grad and add p.grad into it.
        """
        device_states = self.device_states[p]
        state = self.state[p]
        if state["host_grad_valid"]:
            self._ensure_on_device(p, ["grad"])
            device_states["grad"].add_(p.grad)
        else:
            device_states["grad"] = p.grad
            state["host_grad_valid"] = True
        p.grad = None

    def _call_step_fn(self, p, group, device_states):
        state = self.state[p]
        beta1, beta2 = group["betas"]
        is_quant = p in self._quant_params
        if (
            is_quant
            and not self._quant_master_warned
            and state.get("master_filled") is False
        ):
            warnings.warn(
                "OffloadAdam: quantized params running with a lossy "
                "dequant-seeded fp32 master. For higher precision, call "
                "`optim.load_master_from_pretrained("
                "pretrained_name_or_path, model)` with the same "
                "bf16/fp16 checkpoint `from_pretrained` used before "
                "`quantize_linears` ran.",
                stacklevel=2,
            )
            self._quant_master_warned = True  # warn once
        if is_quant:
            # bf16 master write-back would smash packed storage; redirect
            # to grad (dummy sink) and re-quantize via `copy_` below.
            param_view = device_states["grad"]
        else:
            param_view = p.data
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
        if is_quant:
            p.data.copy_(device_states["master_params"])

    def _step_offload(self, p, group, clip_coef=None):
        """PCIe-mediated step: ensure state on device → kernel → d2h."""
        device_states = self.device_states[p]
        self._ensure_on_device(p, self.offload_state_keys)
        if clip_coef is not None:
            device_states["grad"].mul_(clip_coef)
        self._call_step_fn(p, group, device_states)
        self._issue_d2h(p, self._non_grad_keys)
        device_states["grad"] = None

    def _step_inplace(self, p, group, clip_coef=None):
        """Run the step for an in-place (small) param entirely on GPU.

        State buffers are persistent on device, grad has been accumulated
        into `state["grad"]` by the in-place grad hook. After step, grad is
        zeroed so the next training step starts from a clean slate.
        """
        state = self.state[p]
        if clip_coef is not None:
            state["grad"].mul_(clip_coef)
        self._call_step_fn(
            p, group, {k: state[k] for k in self.offload_state_keys}
        )
        state["grad"].zero_()

    # ------------------------------------------------------------------
    # Layer 3 — hooks (the two strategies diverge here)
    # ------------------------------------------------------------------

    def _pre_backward_hook(self, module, grad_output):
        # `dev.get(...) is None` makes the hook idempotent across multiple
        # fires per backward (tied embed: two modules; shared module: many
        # invocations). Lazy adds a topology check to defer tied-embed
        # prefetch to the input-side module.
        lazy = self._prefetch_policy == "lazy"
        for p in module.parameters():
            if p in self._inplace_params:
                continue
            if lazy and self._prefetch_module.get(p) is not module:
                continue
            dev = self.device_states[p]
            keys = []
            if self.state[p]["host_grad_valid"] and dev.get("grad") is None:
                keys.append("grad")
            if self._step_in_backward and self.ready_for_optimizer_step:
                keys.extend(
                    k for k in self._non_grad_keys if dev.get(k) is None
                )
            if keys:
                self._issue_h2d(p, keys)

    @torch.no_grad()
    def _grad_hook(self, p):
        if p.grad is None:
            return

        self._accumulate_grad_on_device(p)
        state = self.state[p]
        group = self._param2group[p]

        # Clipping path: record per-param L2 norm; .step() reduces it into
        # the global norm and applies the clip coefficient at step time.
        if self.max_grad_norm is not None and self.ready_for_optimizer_step:
            state["grad_norm"] = torch.norm(
                self.device_states[p]["grad"], 2.0
            )

        # Step-in-backward path: run the whole step here on the last mb;
        # every other case just writes the accumulated grad back to host.
        if self._step_in_backward and self.ready_for_optimizer_step:
            state["step"] += 1
            self._step_offload(p, group)
            state["host_grad_valid"] = False
        else:
            self._issue_d2h(p, ["grad"])

    @torch.no_grad()
    def _inplace_grad_hook(self, p):
        """Grad hook for small in-place params: no h2d/d2h, all on GPU.

        `state["grad"]` is pre-zeroed; every micro-batch unconditionally
        accumulates into it. On the last mb, step-in-backward path executes
        the full step here; clipping path defers to `.step()`.
        """
        if p.grad is None:
            return
        state = self.state[p]
        group = self._param2group[p]
        state["grad"].add_(p.grad)
        p.grad = None

        if self.max_grad_norm is not None and self.ready_for_optimizer_step:
            state["grad_norm"] = torch.norm(state["grad"], 2.0)

        if self._step_in_backward and self.ready_for_optimizer_step:
            state["step"] += 1
            self._step_inplace(p, group)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise NotImplementedError(
                "OffloadAdam does not support closure-based step: gradients "
                "are consumed by backward hooks, so re-running backward "
                "inside a closure would corrupt optimizer state."
            )
        if self._step_in_backward:
            return

        clip_coef = None
        if self.max_grad_norm is not None:
            norms = [
                self.state[p]["grad_norm"]
                for group in self.param_groups
                for p in group["params"]
            ]
            total_norm = torch.linalg.vector_norm(torch.stack(norms), 2.0)
            clip_coef = torch.clamp(
                self.max_grad_norm / (total_norm + 1e-6), max=1.0
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

    def zero_grad(self, set_to_none=False):
        """No-op; grads are managed by the hooks."""
        return

    @torch.no_grad()
    def load_master_state_dict(self, state_dict, model, strict=False):
        """Overwrite `master_params` buffers from an in-memory
        ``{name: tensor}`` mapping.

        Advanced / test-time entry point — the canonical way to upgrade
        the lossy dequant-seeded master to a lossless source is
        ``load_master_from_pretrained``, which streams from disk.

        Source tensors may be any float dtype (bf16 / fp16 / fp32);
        they are copied into the optimizer's fp32 `master_params` —
        narrower dtypes widen losslessly.

        Args:
            state_dict: ``{param_name: tensor}`` mapping.
            model: Used to resolve param names.
            strict: If True, raise ``ValueError`` on any name in
                ``state_dict`` that doesn't match a trainable parameter
                tracked by this optimizer.

        Returns:
            ``(unexpected_keys,)`` tuple — list of names in
            ``state_dict`` that no param matched.
        """
        if "master_params" not in self.offload_config:
            raise RuntimeError(
                "load_master_state_dict requires a mode that maintains "
                f"master_params (e.g. 'fp32_master'); current mode is {self.mode!r}."
            )
        name2param = {
            n: p for n, p in model.named_parameters() if p.requires_grad
        }
        unexpected = []
        for name, tensor in state_dict.items():
            p = name2param.get(name)
            if p is None or p not in self.state:
                unexpected.append(name)
                continue
            state = self.state[p]
            state["master_params"].copy_(tensor)
            # Loaded → mark filled so the step-time warning skips this
            # param. Persisted via `Optimizer.state_dict()`.
            if "master_filled" in state:
                state["master_filled"] = True
        if strict and unexpected:
            raise ValueError(
                "load_master_state_dict(strict=True): no param matches "
                f"{len(unexpected)} key(s): {unexpected}"
            )
        return (unexpected,)

    @torch.no_grad()
    def load_master_from_pretrained(
        self, pretrained_name_or_path, model, *, strict=False,
    ):
        """Refill the fp32 master from a ``from_pretrained``-style
        checkpoint (single file, sharded directory, or HF Hub repo ID).

        Reads each tensor once from the on-disk bf16/fp16 checkpoint,
        widens to fp32 as it copies into ``state[p]["master_params"]``
        (lossless), and releases the read buffer before moving on —
        per-layer peak host memory stays bounded. Only acts on params
        whose ``p.data`` is a ``QWeightBase`` subclass; non-quant
        params already got the same widen at optimizer init.

        Args:
            pretrained_name_or_path: Same shape
                ``AutoModelForCausalLM.from_pretrained`` accepts — a
                local file, a local directory (single or sharded), or a
                HuggingFace Hub repo ID (resolved via
                ``huggingface_hub``).
            model: Used to resolve key → ``Parameter``.
            strict: If True, raise ``ValueError`` when one or more quant
                params have no matching key across the provided files.

        Returns:
            Sorted list of quant param names that had no match across
            the provided files. Empty list means every quant master was
            refilled losslessly.
        """
        if "master_params" not in self.offload_config:
            raise RuntimeError(
                "load_master_from_pretrained requires a mode that "
                "maintains master_params (e.g. 'fp32_master'); current "
                f"mode is {self.mode!r}."
            )
        return stream_master_from_pretrained(
            self, pretrained_name_or_path, model, strict=strict,
        )

    def load_state_dict(self, state_dict):
        raise NotImplementedError(
            "OffloadAdam.load_state_dict is not supported: the default "
            "Optimizer.load_state_dict would move pinned-host state to "
            "GPU and cast fp32 master to bf16, defeating the offload "
            "contract. File an issue if resume is needed."
        )
