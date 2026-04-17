import torch
from torch.optim.optimizer import Optimizer

from .pinned_alloc import zeros_pinned
from .kernels import (
    adam_step_stochastic_rounding,
    adam_step_fp32_master,
    adam_step_fp31_master,
    adam_step_fp32_master_custom_rounding,
)


def get_leaf_modules_with_params(module):
    """Recursively collect leaf modules with parameters from a PyTorch model."""
    leaf_modules = []
    for child in module.children():
        if list(child.children()):  # If the module has children, recurse
            leaf_modules.extend(get_leaf_modules_with_params(child))
        elif any(
            p.requires_grad for p in child.parameters(recurse=False)
        ):  # Check if it has parameters
            leaf_modules.append(child)
    return leaf_modules


class OffloadAdam(Optimizer):
    """Adam optimizer that offloads gradients and optimizer states to host memory.

    Args:
        model (nn.Module): Model containing parameters to optimize
        lr (float, optional): Learning rate. Default: 1e-3
        betas (Tuple[float, float], optional): Coefficients for computing running averages of
            gradient and its square. Default: (0.9, 0.999)
        eps (float, optional): Term added to denominator to improve numerical stability. Default: 1e-8
        weight_decay (float, optional): Weight decay (L2 penalty). Default: 0.01
        mode (str, optional): Optimization step mode - one of 'stochastic_rounding', 'fp32_master',
            'fp31_master', or 'fp32_master_custom_rounding'. Default: 'stochastic_rounding'
        gradient_clipping (dict, optional): Dict with 'max_norm' and 'norm_type' for gradient clipping.
            Default: None
        numa_node (str | int | None, optional): NUMA policy for pinned host allocations.
            'auto' (default) binds each allocation to the current CUDA device's NUMA
            node when libnuma and the sysfs topology are available; an int specifies
            a node explicitly; None disables NUMA binding.
        decoupled_weight_decay (bool, optional): Whether to decouple weight decay. Default: False
        step_chunk_size (int, optional): Update in chunks smaller than parameter size. Useful for single
            large parameter (e.g. token embedding). Default: 10M
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

    def __init__(
        self,
        model,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        mode="stochastic_rounding",
        gradient_clipping=None,  # dict(max_norm=1.0, norm_type=2.0)
        numa_node="auto",
        decoupled_weight_decay=False,
        step_chunk_size=1024**2 * 10,
        verbose=0,
    ):
        assert mode in self.supported_modes, (
            f"Invalid mode: {mode}, available modes: {self.supported_modes.keys()}"
        )
        self.mode = mode
        self.step_fn = self.supported_modes[self.mode]["step"]

        modules = get_leaf_modules_with_params(model)
        params = []
        for module in modules:
            for p in module.parameters():
                if p.requires_grad:
                    params.append(p)

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(OffloadAdam, self).__init__(params, defaults)

        self.gradient_clipping = gradient_clipping
        self.decoupled_weight_decay = decoupled_weight_decay
        self.verbose = verbose
        self.ready_for_optimizer_step = False
        self.step_chunk_size = step_chunk_size

        self.device_states = {}
        self.d2h_events = {}
        self.h2d_events = {}
        self.d2h_stream = torch.cuda.Stream()
        self.h2d_stream = torch.cuda.Stream()

        param2group = {}
        for group in self.param_groups:
            for param in group["params"]:
                param2group[param] = group

        # hooks for h2d transfer
        for module in modules:
            module.register_full_backward_pre_hook(self.pre_backward_hook)
            for param in module.parameters():
                if not param.requires_grad:
                    continue

                group = param2group[param]
                param.register_post_accumulate_grad_hook(
                    self._create_post_accumulate_grad_hook(param, group)
                )
                self.device_states[param] = {}
                state = self.state[param]
                state["step"] = 0
                state["host_grad_valid"] = False
                if self.gradient_clipping is not None:
                    state["grad_norm"] = torch.tensor(0.0, device="cuda")

        offload_config = self.supported_modes[self.mode]["offload"]
        self.offload_config = offload_config
        self.offload_state_keys = list(offload_config.keys())

        total_bytes = 0
        for param in params:
            state = self.state[param]
            for name, dtype in offload_config.items():
                t = zeros_pinned(param.shape, dtype, numa_node=numa_node)
                if name == "master_params":
                    t.copy_(param.data)
                state[name] = t
                total_bytes += t.numel() * t.element_size()

        if self.verbose > 0:
            print(
                f"Pinned host memory allocated: {total_bytes / (1024**3):.2f} GB"
            )

    def _host_view(self, param, name, param_chunk=None):
        t = self.state[param][name]
        if param_chunk is not None:
            t = t.view(-1)[param_chunk]
        return t

    def ensure_on_device(
        self, device_states, param, offload_state_keys, param_chunk=None
    ):
        """Ensure the given states are on the device.
        In case prefetch is not set, copy the states from host to device here."""
        if param in self.h2d_events:
            self.h2d_events[param].synchronize()
        for offload_state_key in offload_state_keys:
            if device_states.get(offload_state_key, None) is None:
                device_states[offload_state_key] = self._host_view(
                    param, offload_state_key, param_chunk
                ).to(param.device, non_blocking=True)

    def issue_h2d_transfer(self, param, offload_state_keys, param_chunk=None):
        """Issue host to device transfers for the given states"""
        main_stream = torch.cuda.current_stream()
        device_states = self.device_states[param]
        if param in self.d2h_events:
            self.d2h_events[param].synchronize()
        self.h2d_stream.wait_stream(main_stream)
        for offload_state_key in offload_state_keys:
            with torch.cuda.stream(self.h2d_stream):
                device_states[offload_state_key] = self._host_view(
                    param, offload_state_key, param_chunk
                ).to(param.device, non_blocking=True)
        self.h2d_events[param] = self.h2d_stream.record_event()

    def issue_d2h_transfer(self, param, offload_state_keys, param_chunk=None):
        """Issue device to host transfers for the given states"""
        main_stream = torch.cuda.current_stream()
        device_states = self.device_states[param]
        if param in self.h2d_events:
            self.h2d_events[param].synchronize()
        self.d2h_stream.wait_stream(main_stream)
        for offload_state_key in offload_state_keys:
            with torch.cuda.stream(self.d2h_stream):
                self._host_view(param, offload_state_key, param_chunk).copy_(
                    device_states[offload_state_key], non_blocking=True
                )
            # release device memory
            device_states[offload_state_key].record_stream(self.d2h_stream)
            device_states[offload_state_key] = None
        self.d2h_events[param] = self.d2h_stream.record_event()

    def pre_backward_hook(self, module, grad_output):
        # Prefetch gradients from Host to Device
        # Overlap with backward computation
        for param in module.parameters():
            if self.state[param]["host_grad_valid"]:
                self.issue_h2d_transfer(param, ["grad"])

    def _create_post_accumulate_grad_hook(self, param, group):
        @torch.no_grad()
        def grad_accumulate_hook(*_unused):
            if param.grad is None:
                return

            # states prefetched in pre_backward_hook
            device_states = self.device_states[param]
            state = self.state[param]

            if state["host_grad_valid"]:
                # accumulate grad on host
                self.ensure_on_device(device_states, param, ["grad"])
                device_states["grad"].add_(param.grad)
            else:
                # first accumulate, copy grad to host
                device_states["grad"] = param.grad
                state["host_grad_valid"] = True
            if self.ready_for_optimizer_step and self.gradient_clipping is not None:
                self.state[param]["grad_norm"] = torch.norm(
                    device_states["grad"], self.gradient_clipping["norm_type"]
                )
            param.grad = None
            self.issue_d2h_transfer(param, ["grad"])
            return

        return grad_accumulate_hook

    @torch.no_grad()
    def step(self, closure=None):
        # clip_grad_norm_
        if self.gradient_clipping is not None:
            norms = []
            for group in self.param_groups:
                for p in group["params"]:
                    norms.append(self.state[p]["grad_norm"])
            total_norm = torch.linalg.vector_norm(
                torch.stack(norms), self.gradient_clipping["norm_type"]
            )
            clip_coef = self.gradient_clipping["max_norm"] / (total_norm + 1e-6)
            clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] += 1

                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                step = state["step"]

                device_states = self.device_states[p]

                param_size = p.numel()
                for start in range(0, param_size, self.step_chunk_size):
                    end = min(start + self.step_chunk_size, param_size)
                    param_chunk = slice(start, end)
                    self.issue_h2d_transfer(p, self.offload_state_keys, param_chunk)
                    self.ensure_on_device(
                        device_states, p, self.offload_state_keys, param_chunk
                    )
                    if self.gradient_clipping is not None:
                        device_states["grad"].mul_(clip_coef_clamped)

                    self.step_fn(
                        p.data.view(-1)[param_chunk],
                        device_states,
                        lr,
                        weight_decay,
                        beta1,
                        beta2,
                        eps,
                        step,
                        decoupled_weight_decay=self.decoupled_weight_decay,
                    )

                    self.issue_d2h_transfer(
                        p,
                        [name for name in self.offload_state_keys if name != "grad"],
                        param_chunk,
                    )
                    device_states["grad"] = None
                    # mark grad on host as invalid
                state["host_grad_valid"] = False

        return

    def zero_grad(self, set_to_none=False):
        """No-op."""
        return
