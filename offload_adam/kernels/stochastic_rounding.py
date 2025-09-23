"""
AdamW step:
1. convert all states from bfloat16 to float32
2. perform the AdamW step
3. convert the params back to bfloat16, with stochastic rounding

Host memory consumption per parameter:
+----------------+-------------+
| State          | Size (bytes)|
+----------------+-------------+
| grad           | 2           |
| exp_avg        | 2           |
| exp_avg_sq     | 2           |
+----------------+-------------+
| Total          | 6           |
+----------------+-------------+
"""

import random
import triton
import triton.language as tl


@triton.jit
def _fp32_to_bf16_sr(x_f32, rand_16bit):
    # Adapted from torchao
    x_f32_bits = tl.cast(x_f32, tl.int32, bitcast=True)
    x_fraction = x_f32_bits & 0xFFFF
    x_bf16_towards_zero = x_f32_bits & (-0x10000)
    x_f32_bits = tl.where(
        rand_16bit < x_fraction, x_bf16_towards_zero + 0x10000, x_bf16_towards_zero
    )
    return tl.cast(x_f32_bits, tl.float32, bitcast=True).to(tl.bfloat16)


@triton.jit
def adam_step_stochastic_rounding_kernel(
    params_ptr,
    grad_ptr,
    exp_avg_ptr,
    exp_avg_sq_ptr,
    lr,
    weight_decay,
    beta1_hat,
    beta2_hat,
    eps,
    n_elements,
    seed,
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

    if DECOUPLED_WEIGHT_DECAY:
        params = params - lr * weight_decay * params
    else:
        grad = grad + weight_decay * params
    exp_avg = exp_avg + (1 - beta1_hat) * (grad - exp_avg)
    exp_avg_sq = exp_avg_sq + (1 - beta2_hat) * (grad * grad - exp_avg_sq)
    denom = tl.sqrt_rn(exp_avg_sq) + eps
    params = params - tl.div_rn(lr * exp_avg, denom)

    # Store
    tl.store(exp_avg_ptr + offsets, exp_avg.to(tl.bfloat16), mask=mask)
    tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq.to(tl.bfloat16), mask=mask)
    # stochastic rounding
    rand_16bit = tl.randint(seed, offsets) & 0xFFFF
    params = _fp32_to_bf16_sr(params, rand_16bit)
    tl.store(params_ptr + offsets, params, mask=mask)


def adam_step_stochastic_rounding(
    params,
    states,
    lr,
    weight_decay,
    beta1,
    beta2,
    eps,
    step,
    seed=None,
    decoupled_weight_decay=False,
    BLOCK_SIZE=1024,
):
    beta1_hat = (beta1**step - beta1) / (beta1**step - 1)
    beta2_hat = (beta2**step - beta2) / (beta2**step - 1)

    if seed is None:
        seed = random.getrandbits(64)
    # launch kernel
    grid = (triton.cdiv(params.numel(), BLOCK_SIZE),)
    adam_step_stochastic_rounding_kernel[grid](
        params,
        states["grad"],
        states["exp_avg"],
        states["exp_avg_sq"],
        lr,
        weight_decay,
        beta1_hat,
        beta2_hat,
        eps,
        params.numel(),
        seed,
        decoupled_weight_decay,
        BLOCK_SIZE,
    )
