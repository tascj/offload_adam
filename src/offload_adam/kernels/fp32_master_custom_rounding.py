"""
Experimental implementation to reduce memory consumption of fp32 master params.

float32 -> bfloat16 round-to-nearest-even makes it difficult to store the rounding error in 16 bits.
For example, all 65537 values in [0x00018000, 0x00028000] rounds to 0x0002
We need at least 17 bits to store the rounding error.

In this implementation, 0x____8000 is always rounded up when converting float32 to bfloat16,
so that the rounding error is in the range [-32768, 32767].
We could use 16 bits to store the rounding error.


Host memory consumption per parameter:
+----------------+-------------+
| State          | Size (bytes)|
+----------------+-------------+
| grad           | 2           |
| exp_avg        | 2           |
| exp_avg_sq     | 2           |
| rounding_error | 2           |
+----------------+-------------+
| Total          | 8           |
+----------------+-------------+
"""

import torch
import triton
import triton.language as tl

@triton.jit
def decompose(x_fp32):
    # float32 -> bfloat16
    x_up = x_fp32.to(tl.int32, bitcast=True) + 0x8000
    x_bf16 = (x_up >> 16).to(tl.int16).to(tl.bfloat16, bitcast=True)
    # calculate and store the rounding error
    x_32bit = x_fp32.to(tl.int32, bitcast=True)
    xq_32bit = x_bf16.to(tl.float32).to(tl.int32, bitcast=True)
    error_32bit = x_32bit - xq_32bit
    error_q = error_32bit.to(tl.int16)  # [-32768, 32767]
    return x_bf16, error_q


@triton.jit
def reconstruct(x_bf16, error_q):
    error_is_valid = error_q != 0x8000
    xq_32bit = x_bf16.to(tl.float32).to(tl.int32, bitcast=True)
    error_q_32bit = error_q.to(tl.int32)
    x_32bit = xq_32bit + error_q_32bit
    x_32bit = tl.where(error_is_valid, x_32bit, xq_32bit)
    x = x_32bit.to(tl.float32, bitcast=True)
    return x


@triton.jit
def adam_step_fp32_master_custom_rounding_kernel(
    params_ptr,
    grad_ptr,
    exp_avg_ptr,
    exp_avg_sq_ptr,
    rounding_error_ptr,
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
    params = tl.load(params_ptr + offsets, mask=mask).to(tl.float32)
    grad = tl.load(grad_ptr + offsets, mask=mask).to(tl.float32)
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask).to(tl.float32)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask).to(tl.float32)
    rounding_error = tl.load(rounding_error_ptr + offsets, mask=mask).to(tl.int16)
    master_params = reconstruct(params, rounding_error)

    if DECOUPLED_WEIGHT_DECAY:
        master_params = master_params - lr * weight_decay * master_params
    else:
        grad = grad + weight_decay * master_params
    exp_avg = exp_avg + (1 - beta1_hat) * (grad - exp_avg)
    exp_avg_sq = exp_avg_sq + (1 - beta2_hat) * (grad * grad - exp_avg_sq)
    denom = tl.sqrt_rn(exp_avg_sq) + eps
    master_params = master_params - tl.div_rn(lr * exp_avg, denom)

    params, rounding_error = decompose(master_params)
    # Store
    tl.store(exp_avg_ptr + offsets, exp_avg.to(tl.bfloat16), mask=mask)
    tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq.to(tl.bfloat16), mask=mask)
    tl.store(params_ptr + offsets, params, mask=mask)
    tl.store(rounding_error_ptr + offsets, rounding_error, mask=mask)


def adam_step_fp32_master_custom_rounding(
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
    assert params.dtype == torch.bfloat16, "params must be bfloat16"

    beta1_hat = (beta1**step - beta1) / (beta1**step - 1)
    beta2_hat = (beta2**step - beta2) / (beta2**step - 1)

    # launch kernel
    grid = (triton.cdiv(params.numel(), BLOCK_SIZE),)
    adam_step_fp32_master_custom_rounding_kernel[grid](
        params,
        states["grad"],
        states["exp_avg"],
        states["exp_avg_sq"],
        states["rounding_error"],
        lr,
        weight_decay,
        beta1_hat,
        beta2_hat,
        eps,
        params.numel(),
        decoupled_weight_decay,
        BLOCK_SIZE,
    )
