import torch
from torch.optim.optimizer import Optimizer

from .kernels import (
    adam_step_stochastic_rounding,
    adam_step_fp32_master,
    adam_step_fp31_master,
    adam_step_fp32_master_custom_rounding,
)


class Adam(Optimizer):
    configs = {
        "stochastic_rounding": {
            "step": adam_step_stochastic_rounding,
            "states": {
                "exp_avg": torch.bfloat16,
                "exp_avg_sq": torch.bfloat16,
            },
        },
        "fp32_master": {
            "step": adam_step_fp32_master,
            "states": {
                "exp_avg": torch.bfloat16,
                "exp_avg_sq": torch.bfloat16,
                "master_params": torch.float32,
            },
        },
        "fp31_master": {
            "step": adam_step_fp31_master,
            "states": {
                "exp_avg": torch.bfloat16,
                "exp_avg_sq": torch.bfloat16,
                "rounding_error": torch.int16,
            },
        },
        "fp32_master_custom_rounding": {
            "step": adam_step_fp32_master_custom_rounding,
            "states": {
                "exp_avg": torch.bfloat16,
                "exp_avg_sq": torch.bfloat16,
                "rounding_error": torch.int16,
            },
        },
    }

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        mode="stochastic_rounding",
        decoupled_weight_decay=False,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Adam, self).__init__(params, defaults)

        assert mode in self.configs, f"Invalid mode: {mode}"
        self.mode = mode
        self.config = self.configs[self.mode]
        self.step_fn = self.config["step"]
        self.decoupled_weight_decay = decoupled_weight_decay

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Sparse gradient is not supported")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    for state_name, dtype in self.config["states"].items():
                        state[state_name] = torch.zeros_like(p, dtype=dtype)
                    if "master_params" in state:
                        state["master_params"].copy_(p.data)
                state["step"] += 1

                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                step = state["step"]

                device_states = {name: state[name] for name in self.config["states"]}
                device_states["grad"] = grad
                self.step_fn(
                    p.data,
                    device_states,
                    lr,
                    weight_decay,
                    beta1,
                    beta2,
                    eps,
                    step,
                    decoupled_weight_decay=self.decoupled_weight_decay,
                )

        return loss
