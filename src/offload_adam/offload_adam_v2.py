import torch
from .offload_adam import OffloadAdam


class OffloadAdamV2(OffloadAdam):
    """A faster version that does not support gradient clipping."""

    def __init__(
        self,
        model,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        mode="stochastic_rounding",
        bucket_size=4 * (1024**3),
        decoupled_weight_decay=False,
        verbose=0,
    ):
        super().__init__(
            model=model,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            mode=mode,
            bucket_size=bucket_size,
            decoupled_weight_decay=decoupled_weight_decay,
            verbose=verbose,
        )

    def pre_backward_hook(self, module, grad_output):
        for param in module.parameters():
            state = self.state[param]
            offload_state_keys = []
            if state["host_grad_valid"]:
                offload_state_keys.append("grad")
            if self.ready_for_optimizer_step:
                offload_state_keys.extend(
                    [_ for _ in self.offload_state_keys if _ != "grad"]
                )
            if offload_state_keys:
                self.issue_h2d_transfer(param, offload_state_keys)

    def _create_post_accumulate_grad_hook(self, param, group):

        @torch.no_grad()
        def optimizer_step_hook(*unused):
            if param.grad is None:
                return

            # states prefetched in pre_backward_hook
            device_states = self.device_states[param]
            state = self.state[param]

            # handle gradients
            if not state["host_grad_valid"]:
                device_states["grad"] = param.grad
                state["host_grad_valid"] = True
            else:
                self.ensure_on_device(device_states, param, ["grad"])
                device_states["grad"].add_(param.grad)
            param.grad = None
            if not self.ready_for_optimizer_step:
                self.issue_d2h_transfer(param, ["grad"])
                return

            # optimizer step
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            state = self.state[param]
            state["step"] += 1

            self.ensure_on_device(
                device_states, param, [_ for _ in self.offload_state_keys if _ != "grad"]
            )
            self.step_fn(
                param.data,
                device_states,
                lr,
                weight_decay,
                beta1,
                beta2,
                eps,
                state["step"],
                decoupled_weight_decay=self.decoupled_weight_decay,
            )
            # Copy states back to Host
            self.issue_d2h_transfer(
                param, [name for name in self.offload_state_keys if name != "grad"]
            )
            device_states["grad"] = None
            state["host_grad_valid"] = False

        return optimizer_step_hook

    def step(self, closure=None):
        """No-op."""
        return

    def zero_grad(self, set_to_none: bool = False):
        """No-op."""
        return
