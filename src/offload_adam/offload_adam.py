"""OffloadAdam: Adam with gradients and optimizer states offloaded to host memory.

Two execution paths share the same transfer primitives and are selected by
whether `gradient_clipping` is passed:

- No clipping → step runs inside each param's post-accumulate-grad hook on
  the last micro-batch, overlapping load/compute/writeback with backward.
- Clipping → backward only accumulates grad and per-param norms; the full
  step loop (global clip, chunked load/compute/writeback) runs in `.step()`.
"""

import warnings

import torch
from torch.optim.optimizer import Optimizer

from .kernels import (
    adam_step_fp31_master,
    adam_step_fp32_master,
    adam_step_fp32_master_custom_rounding,
    adam_step_stochastic_rounding,
)
from .pinned_alloc import zeros_pinned


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
        gradient_clipping (dict | None): `{'max_norm': float, 'norm_type': float}`
            for global-norm clipping. `None` disables clipping and routes the
            step into the backward hook for maximum backward/step overlap.
        numa_node (str | int | None): NUMA policy for pinned allocations —
            `'auto'`, an int node id, or `None`.
        decoupled_weight_decay (bool): Decouple weight decay from gradient
            (AdamW style). Default: False.
        step_chunk_size (int | None): Chunk size used by the `.step()` path
            (only active when `gradient_clipping` is set). Default: 100M.
        inplace_param_threshold (int | None): Params with fewer elements than
            this stay fully on GPU (states + grad), bypassing PCIe entirely.
            The default covers typical LLM normalization weights and biases
            while leaving attention/MLP projections and embeddings on the
            offload path. Set `0` to force every param onto the offload path.
            Default: 1M elements.
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

    _DEFAULT_STEP_CHUNK_SIZE = 1024 ** 2 * 100
    _DEFAULT_INPLACE_PARAM_THRESHOLD = 1024 * 1024
    # Max in-flight chunks before the chunked-step loop CPU-waits on the
    # oldest d2h. K=2 gives full 3-stream (h2d/compute/d2h) overlap with
    # peak device memory bounded to ~3× chunk_size.
    _CHUNK_PIPELINE_DEPTH = 2

    def __init__(
        self,
        model,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        mode="stochastic_rounding",
        gradient_clipping=None,
        numa_node="auto",
        decoupled_weight_decay=False,
        step_chunk_size=None,
        inplace_param_threshold=None,
        verbose=0,
    ):
        assert mode in self.supported_modes, (
            f"Invalid mode: {mode}, available modes: {list(self.supported_modes)}"
        )
        self.mode = mode
        self.step_fn = self.supported_modes[mode]["step"]
        self.offload_config = self.supported_modes[mode]["offload"]
        self.offload_state_keys = list(self.offload_config.keys())
        self._non_grad_keys = [k for k in self.offload_state_keys if k != "grad"]

        self.gradient_clipping = gradient_clipping
        # Without a clip spec we can fuse load/compute/writeback into backward
        # hooks. Clipping needs a global norm, which forces the step back into
        # `.step()` with a chunked loop.
        self._step_in_backward = gradient_clipping is None

        if step_chunk_size is None:
            step_chunk_size = self._DEFAULT_STEP_CHUNK_SIZE
        elif self._step_in_backward:
            warnings.warn(
                "step_chunk_size is ignored when gradient_clipping is None; "
                "the step runs per-parameter inside backward hooks.",
                stacklevel=2,
            )
        self.step_chunk_size = step_chunk_size

        self.decoupled_weight_decay = decoupled_weight_decay
        self.verbose = verbose
        self.ready_for_optimizer_step = False

        modules = get_leaf_modules_with_params(model)
        params = [
            p for m in modules for p in m.parameters() if p.requires_grad
        ]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # Partition params: small ones stay fully on GPU; big ones get offloaded.
        if inplace_param_threshold is None:
            inplace_param_threshold = self._DEFAULT_INPLACE_PARAM_THRESHOLD
        self.inplace_param_threshold = inplace_param_threshold
        self._inplace_params = {
            p for p in params if p.numel() < inplace_param_threshold
        }
        self._offload_params = [
            p for p in params if p not in self._inplace_params
        ]

        self.device_states = {}
        self.h2d_events = {}
        self.d2h_events = {}
        self.h2d_stream = torch.cuda.Stream()
        self.d2h_stream = torch.cuda.Stream()

        param2group = {
            p: group for group in self.param_groups for p in group["params"]
        }
        self._register_hooks(modules, param2group)
        self._alloc_pinned_states(self._offload_params, numa_node)
        self._alloc_inplace_states(self._inplace_params)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _register_hooks(self, modules, param2group):
        for module in modules:
            module.register_full_backward_pre_hook(self._pre_backward_hook)
            for p in module.parameters():
                if not p.requires_grad:
                    continue
                state = self.state[p]
                state["step"] = 0
                if self.gradient_clipping is not None:
                    state["grad_norm"] = torch.tensor(0.0, device="cuda")
                if p in self._inplace_params:
                    p.register_post_accumulate_grad_hook(
                        self._make_inplace_grad_hook(p, param2group[p])
                    )
                else:
                    p.register_post_accumulate_grad_hook(
                        self._make_grad_hook(p, param2group[p])
                    )
                    self.device_states[p] = {}
                    state["host_grad_valid"] = False

    def _alloc_pinned_states(self, params, numa_node):
        total_bytes = 0
        for p in params:
            state = self.state[p]
            for name, dtype in self.offload_config.items():
                t = zeros_pinned(p.shape, dtype, numa_node=numa_node)
                if name == "master_params":
                    t.copy_(p.data)
                state[name] = t
                total_bytes += t.numel() * t.element_size()
        if self.verbose > 0:
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

    def _host_view(self, p, name, chunk=None):
        t = self.state[p][name]
        if chunk is not None:
            t = t.view(-1)[chunk]
        return t

    def _issue_h2d(self, p, keys, chunk=None):
        main_stream = torch.cuda.current_stream()
        device_states = self.device_states[p]
        if p in self.d2h_events:
            self.h2d_stream.wait_event(self.d2h_events[p])
        self.h2d_stream.wait_stream(main_stream)
        for key in keys:
            with torch.cuda.stream(self.h2d_stream):
                device_states[key] = self._host_view(p, key, chunk).to(
                    p.device, non_blocking=True
                )
        self.h2d_events[p] = self.h2d_stream.record_event()

    def _issue_d2h(self, p, keys, chunk=None):
        # No explicit wait on h2d_events[p]: every caller already routes the
        # device tensor through main_stream (accumulate / step_fn), and
        # wait_stream(main_stream) below propagates that h2d dependency.
        main_stream = torch.cuda.current_stream()
        device_states = self.device_states[p]
        self.d2h_stream.wait_stream(main_stream)
        for key in keys:
            with torch.cuda.stream(self.d2h_stream):
                self._host_view(p, key, chunk).copy_(
                    device_states[key], non_blocking=True
                )
            device_states[key].record_stream(self.d2h_stream)
            device_states[key] = None
        self.d2h_events[p] = self.d2h_stream.record_event()

    def _ensure_on_device(self, p, keys, chunk=None):
        """Make the main stream wait for the pending h2d; synchronously fetch
        any key that wasn't prefetched."""
        main_stream = torch.cuda.current_stream()
        device_states = self.device_states[p]
        if p in self.h2d_events:
            main_stream.wait_event(self.h2d_events[p])
        for key in keys:
            if device_states.get(key, None) is None:
                device_states[key] = self._host_view(p, key, chunk).to(
                    p.device, non_blocking=True
                )

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

    def _call_step_fn(self, p, group, device_states, chunk=None):
        state = self.state[p]
        param_view = p.data if chunk is None else p.data.view(-1)[chunk]
        beta1, beta2 = group["betas"]
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

    def _step_overlapped(self, p, group):
        """Run the step for `p` from inside its grad hook.

        Non-grad states were prefetched in `_pre_backward_hook`; grad is
        already on device from `_accumulate_grad_on_device`.
        """
        self._ensure_on_device(p, self._non_grad_keys)
        self._call_step_fn(p, group, self.device_states[p])
        self._issue_d2h(p, self._non_grad_keys)
        self.device_states[p]["grad"] = None

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

    def _step_chunked(self, p, group, clip_coef=None):
        """Run the step for `p` in fixed-size chunks from `.step()`.

        Each chunk pulls grad + non-grad states from host, optionally applies
        the global clip coefficient, runs the step, and writes non-grad states
        back. Chunking keeps peak device memory bounded for huge params.

        Without the windowed CPU sync at the bottom, the per-chunk transfer
        primitives are fully async: the caching allocator would issue a fresh
        cudaMalloc per chunk and device memory would grow linearly with chunk
        count. The sync caps in-flight chunks at `_CHUNK_PIPELINE_DEPTH`,
        which is enough for full h2d/compute/d2h overlap.
        """
        size = p.numel()
        device_states = self.device_states[p]
        pending = []
        for start in range(0, size, self.step_chunk_size):
            end = min(start + self.step_chunk_size, size)
            chunk = slice(start, end)
            self._issue_h2d(p, self.offload_state_keys, chunk)
            self._ensure_on_device(p, self.offload_state_keys, chunk)
            if clip_coef is not None:
                device_states["grad"].mul_(clip_coef)
            self._call_step_fn(p, group, device_states, chunk)
            self._issue_d2h(p, self._non_grad_keys, chunk)
            device_states["grad"] = None
            pending.append(self.d2h_events[p])
            if len(pending) > self._CHUNK_PIPELINE_DEPTH:
                pending.pop(0).synchronize()

    # ------------------------------------------------------------------
    # Layer 3 — hooks (the two strategies diverge here)
    # ------------------------------------------------------------------

    def _pre_backward_hook(self, module, grad_output):
        for p in module.parameters():
            if p in self._inplace_params:
                continue
            keys = []
            if self.state[p]["host_grad_valid"]:
                keys.append("grad")
            if self._step_in_backward and self.ready_for_optimizer_step:
                keys.extend(self._non_grad_keys)
            if keys:
                self._issue_h2d(p, keys)

    def _make_grad_hook(self, p, group):
        @torch.no_grad()
        def hook(*_unused):
            if p.grad is None:
                return

            self._accumulate_grad_on_device(p)
            state = self.state[p]

            # Clipping path: record per-param norm; .step() reduces it into
            # the global norm and applies the clip coefficient chunk by chunk.
            if self.gradient_clipping is not None and self.ready_for_optimizer_step:
                state["grad_norm"] = torch.norm(
                    self.device_states[p]["grad"],
                    self.gradient_clipping["norm_type"],
                )

            # Step-in-backward path: run the whole step here on the last mb;
            # every other case just writes the accumulated grad back to host.
            if self._step_in_backward and self.ready_for_optimizer_step:
                state["step"] += 1
                self._step_overlapped(p, group)
                state["host_grad_valid"] = False
            else:
                self._issue_d2h(p, ["grad"])

        return hook

    def _make_inplace_grad_hook(self, p, group):
        """Grad hook for small in-place params: no h2d/d2h, all on GPU.

        `state["grad"]` is pre-zeroed; every micro-batch unconditionally
        accumulates into it. On the last mb, step-in-backward path executes
        the full step here; chunked path defers to `.step()`.
        """
        @torch.no_grad()
        def hook(*_unused):
            if p.grad is None:
                return
            state = self.state[p]
            state["grad"].add_(p.grad)
            p.grad = None

            if self.gradient_clipping is not None and self.ready_for_optimizer_step:
                state["grad_norm"] = torch.norm(
                    state["grad"], self.gradient_clipping["norm_type"]
                )

            if self._step_in_backward and self.ready_for_optimizer_step:
                state["step"] += 1
                self._step_inplace(p, group)

        return hook

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        if self._step_in_backward:
            return

        clip_coef = None
        if self.gradient_clipping is not None:
            norms = [
                self.state[p]["grad_norm"]
                for group in self.param_groups
                for p in group["params"]
            ]
            total_norm = torch.linalg.vector_norm(
                torch.stack(norms), self.gradient_clipping["norm_type"]
            )
            clip_coef = torch.clamp(
                self.gradient_clipping["max_norm"] / (total_norm + 1e-6),
                max=1.0,
            )

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] += 1
                if p in self._inplace_params:
                    self._step_inplace(p, group, clip_coef=clip_coef)
                else:
                    self._step_chunked(p, group, clip_coef=clip_coef)
                    state["host_grad_valid"] = False

    def zero_grad(self, set_to_none=False):
        """No-op; grads are managed by the hooks."""
        return
