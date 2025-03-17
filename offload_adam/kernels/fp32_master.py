"""
+----------------+-------------+
| State          | Size (bytes)|
+----------------+-------------+
| grad           | 2           |
| exp_avg        | 2           |
| exp_avg_sq     | 2           |
| master_params  | 4           |
+----------------+-------------+
| Total          | 10          |
+----------------+-------------+
"""

import torch
import triton
import triton.language as tl


@triton.jit
def adam_step_fp32_master_kernel(
    params_ptr,
    grad_ptr,
    exp_avg_ptr,
    exp_avg_sq_ptr,
    master_params_ptr,
    lr,
    weight_decay,
    beta1_hat,
    beta2_hat,
    eps,
    n_elements,
    DECOUPLED_WEIGHT_DECAY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load
    master_params = tl.load(master_params_ptr + offsets, mask=mask)  # float32
    grad = tl.load(grad_ptr + offsets, mask=mask).to(tl.float32)
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask).to(tl.float32)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask).to(tl.float32)

    if DECOUPLED_WEIGHT_DECAY:
        master_params = master_params - lr *weight_decay * master_params
    else:
        grad = grad + weight_decay * master_params
    exp_avg = exp_avg + (1 - beta1_hat) * (grad - exp_avg)
    exp_avg_sq = exp_avg_sq + (1 - beta2_hat) * (grad * grad - exp_avg_sq)
    denom = tl.sqrt_rn(exp_avg_sq) + eps
    master_params = master_params - tl.div_rn(lr * exp_avg, denom)

    # Store
    tl.store(params_ptr + offsets, master_params.to(tl.bfloat16), mask=mask)
    tl.store(exp_avg_ptr + offsets, exp_avg.to(tl.bfloat16), mask=mask)
    tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq.to(tl.bfloat16), mask=mask)
    tl.store(master_params_ptr + offsets, master_params, mask=mask)


def adam_step_fp32_master(
    params,
    states,
    lr,
    weight_decay,
    beta1,
    beta2,
    eps,
    step,
    decoupled_weight_decay=False,
    BLOCK_SIZE=1024,
):
    beta1_hat = (beta1**step - beta1) / (beta1**step - 1)
    beta2_hat = (beta2**step - beta2) / (beta2**step - 1)

    # launch kernel
    grid = (triton.cdiv(params.numel(), BLOCK_SIZE),)
    adam_step_fp32_master_kernel[grid](
        params,
        states["grad"],
        states["exp_avg"],
        states["exp_avg_sq"],
        states["master_params"],
        lr,
        weight_decay,
        beta1_hat,
        beta2_hat,
        eps,
        params.numel(),
        decoupled_weight_decay,
        BLOCK_SIZE,
    )
    torch.cuda.synchronize()

