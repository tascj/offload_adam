"""Int8 symmetric per-channel QAT weight (compressed-tensors W8A16).

Storage:
  weight  : (out_f, in_f) int8 — plain row-major.
  scales  : (out_f,) scales.dtype — amax / 127, symmetric, no zero point.

Unpacked in VRAM (int8 is natively addressable); the on-disk W8A16 layout
(``(out_f, in_f // 4) int32``) is a ``state_dict``-time re-pack.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.utils._python_dispatch import return_and_correct_aliasing

from .base import QWeightBase

aten = torch.ops.aten


# ----------------------------------------------------------------------
# Python reference: quantize / dequantize
# ----------------------------------------------------------------------

@torch.no_grad()
def quantize_int8_per_channel(
    tensor: Tensor, eps: float = 1e-12,
) -> Tuple[Tensor, Tensor]:
    """Reference python quantize, per-output-channel symmetric.

    Args:
        tensor: (out_f, in_f) float tensor.
    Returns:
        weight: (out_f, in_f) int8 in [-128, 127].
        scales: (out_f,) same dtype as input — per-row amax / 127.
    """
    assert tensor.ndim == 2
    orig_dtype = tensor.dtype
    t = tensor.float()
    absmax = t.abs().amax(dim=1)                       # (out_f,) fp32
    scale = absmax / 127.0
    inv_scale = 1.0 / scale.clamp(min=eps)
    q = (t * inv_scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
    return q, scale.to(orig_dtype)


def dequantize_int8_per_channel(
    weight: Tensor, scales: Tensor,
) -> Tensor:
    """Inverse of `quantize_int8_per_channel`. Computes at scales.dtype."""
    return weight.to(scales.dtype) * scales.unsqueeze(1)


# ----------------------------------------------------------------------
# Triton kernels
# ----------------------------------------------------------------------

@triton.jit
def _dequant_int8_kernel(
    weight_ptr,      # (out_f, in_f) int8
    scales_ptr,      # (out_f,) scales.dtype
    out_ptr,         # (out_f, in_f) scales.dtype
    weight_stride_row, weight_stride_col,
    out_stride_row, out_stride_col,
    n_out_rows, n_in_cols,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused int8→float cast + per-row scale multiply, one (BLOCK_M, BLOCK_N) tile per program."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < n_out_rows
    col_mask = cols < n_in_cols
    mask2d = row_mask[:, None] & col_mask[None, :]

    i8 = tl.load(
        weight_ptr
        + rows[:, None] * weight_stride_row
        + cols[None, :] * weight_stride_col,
        mask=mask2d,
        other=0,
    )
    scale = tl.load(scales_ptr + rows, mask=row_mask)

    out = i8.to(scale.dtype) * scale[:, None]
    tl.store(
        out_ptr + rows[:, None] * out_stride_row + cols[None, :] * out_stride_col,
        out,
        mask=mask2d,
    )


def _dequant_int8_triton(weight: Tensor, scales: Tensor) -> Tensor:
    out_f, in_f = weight.shape
    out = torch.empty((out_f, in_f), device=weight.device, dtype=scales.dtype)
    BLOCK_M = 32
    BLOCK_N = 128
    grid = (triton.cdiv(out_f, BLOCK_M), triton.cdiv(in_f, BLOCK_N))
    _dequant_int8_kernel[grid](
        weight, scales, out,
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        out_f, in_f,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return out


@triton.jit
def _quantize_int8_kernel(
    input_ptr,       # (out_f, in_f) bf16 / fp16 / fp32
    weight_ptr,      # (out_f, in_f) int8
    scales_ptr,      # (out_f,) scales.dtype
    input_stride_row, input_stride_col,
    weight_stride_row, weight_stride_col,
    n_out_rows, n_in_cols,
    EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Per-row absmax reduce → quantize → store int8 + scale.

    Two passes over in_f (absmax then quantize). Single-pass atomic amax
    would need cross-program sync for no bandwidth win — rows big enough
    that pass-2 hits L2.
    """
    pid_m = tl.program_id(0)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < n_out_rows

    # Pass 1: per-row absmax across all of in_f.
    absmax = tl.zeros([BLOCK_M], dtype=tl.float32)
    for k in range(0, tl.cdiv(n_in_cols, BLOCK_N)):
        cols = k * BLOCK_N + tl.arange(0, BLOCK_N)
        col_mask = cols < n_in_cols
        tile = tl.load(
            input_ptr
            + rows[:, None] * input_stride_row
            + cols[None, :] * input_stride_col,
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        local_max = tl.max(tl.abs(tile), axis=1)
        absmax = tl.maximum(absmax, local_max)

    scale = absmax / 127.0
    # rcp.rn matches torch's 1/x at exactly-halfway values.
    inv_scale = tl.extra.libdevice.rcp_rn(tl.maximum(scale, EPS))

    tl.store(scales_ptr + rows, scale, mask=row_mask)

    # Pass 2: quantize + store int8.
    for k in range(0, tl.cdiv(n_in_cols, BLOCK_N)):
        cols = k * BLOCK_N + tl.arange(0, BLOCK_N)
        col_mask = cols < n_in_cols
        mask2d = row_mask[:, None] & col_mask[None, :]
        tile = tl.load(
            input_ptr
            + rows[:, None] * input_stride_row
            + cols[None, :] * input_stride_col,
            mask=mask2d,
            other=0.0,
        ).to(tl.float32)
        scaled = tile * inv_scale[:, None]
        q_f = tl.extra.libdevice.rint(scaled)
        q_f = tl.maximum(q_f, -128.0)
        q_f = tl.minimum(q_f, 127.0)
        q = q_f.to(tl.int8)
        tl.store(
            weight_ptr
            + rows[:, None] * weight_stride_row
            + cols[None, :] * weight_stride_col,
            q,
            mask=mask2d,
        )


def _quantize_int8_triton(
    tensor: Tensor, scale_dtype: torch.dtype,
) -> Tuple[Tensor, Tensor]:
    """Fused triton absmax + int8 quantize. Returns (weight, scales)."""
    assert tensor.ndim == 2
    out_f, in_f = tensor.shape
    weight = torch.empty(
        (out_f, in_f), dtype=torch.int8, device=tensor.device,
    )
    scales = torch.empty((out_f,), dtype=scale_dtype, device=tensor.device)
    BLOCK_M = 32
    BLOCK_N = 128
    grid = (triton.cdiv(out_f, BLOCK_M),)
    _quantize_int8_kernel[grid](
        tensor, weight, scales,
        tensor.stride(0), tensor.stride(1),
        weight.stride(0), weight.stride(1),
        out_f, in_f,
        EPS=1e-12,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return weight, scales


# ----------------------------------------------------------------------
# Tensor subclass
# ----------------------------------------------------------------------

class Int8QWeight(QWeightBase):
    """Int8 per-channel symmetric QAT weight.

    Outer (out_f, in_f) at scales.dtype; re-quantize happens in ``aten.copy_``.
    """

    @staticmethod
    @torch._dynamo.disable
    def __new__(cls, weight: Tensor, scales: Tensor):
        out_f, in_f = weight.shape
        return Tensor._make_wrapper_subclass(
            cls, (out_f, in_f), dtype=scales.dtype, device=weight.device,
        )

    @torch._dynamo.disable
    def __init__(self, weight: Tensor, scales: Tensor):
        assert weight.dtype is torch.int8 and weight.ndim == 2
        assert scales.ndim == 1 and scales.shape[0] == weight.shape[0]
        self.weight = weight
        self.scales = scales

    def __tensor_flatten__(self):
        return ["weight", "scales"], []

    @classmethod
    def __tensor_unflatten__(cls, tensors, attrs, outer_size=None, outer_stride=None):
        return cls(tensors["weight"], tensors["scales"])

    @classmethod
    def from_float(
        cls, tensor: Tensor,
        scale_dtype: torch.dtype = torch.bfloat16,
        **_,
    ):
        """Quantize ``tensor`` into an ``Int8QWeight``.

        ``scale_dtype`` pins both the stored scale dtype and the subclass's
        outer ``dtype``. Default ``bfloat16``; pass ``float16`` for fp16 models.
        """
        weight, scales = quantize_int8_per_channel(tensor.detach())
        if scales.dtype != scale_dtype:
            scales = scales.to(scale_dtype)
        out = cls(weight, scales)
        out.requires_grad_(tensor.requires_grad)
        return out

    @classmethod
    def can_quantize(cls, tensor: Tensor, **_) -> bool:
        # in_f % 4 == 0 for the int32 pack at serialize time.
        return tensor.dim() == 2 and tensor.shape[1] % 4 == 0

    def dequantize(self, dtype=None) -> Tensor:
        if dtype is None or dtype == self.scales.dtype:
            return _dequant_int8_triton(self.weight, self.scales)
        # fp32 upcast or cross-dtype: python fallback.
        compute_dtype = torch.float32 if dtype == torch.float32 else self.scales.dtype
        return (
            self.weight.to(compute_dtype)
            * self.scales.to(compute_dtype).unsqueeze(1)
        ).to(dtype)

    @classmethod
    def canonical_key_suffixes(cls) -> Tuple[str, ...]:
        return ("weight_packed", "weight_scale", "weight_shape")

    def to_plain_state_dict(self) -> dict:
        out_f, in_f = self.weight.shape
        assert in_f % 4 == 0, (
            f"in_features={in_f} must be a multiple of 4 to pack int8 into int32"
        )
        # Shift int8 [-128, 127] → uint8 [0, 255] (zero-point 128), then pack 4
        # per int32 LSB-first. Matches compressed-tensors' `uint8b128` W8A16.
        u = ((self.weight.to(torch.int32) + 128) & 0xFF).reshape(out_f, in_f // 4, 4)
        shifts = torch.tensor(
            [0, 8, 16, 24], dtype=torch.int32, device=u.device,
        )
        weight_packed = (u << shifts).sum(dim=-1, dtype=torch.int32)
        return {
            "weight_packed": weight_packed,
            "weight_scale": self.scales.unsqueeze(1),     # (out_f, 1)
            "weight_shape": torch.tensor(
                [out_f, in_f], dtype=torch.int64, device=u.device,
            ),
        }

    @classmethod
    def from_plain_state_dict(cls, plain: dict, reference=None) -> "Int8QWeight":
        packed = plain["weight_packed"]
        scale = plain["weight_scale"].squeeze(-1)         # (out_f, 1) → (out_f,)
        out_f, half = packed.shape
        in_f = half * 4
        shifts = torch.tensor(
            [0, 8, 16, 24], dtype=torch.int32, device=packed.device,
        )
        u = (packed.unsqueeze(-1) >> shifts) & 0xFF       # uint8 [0, 255]
        weight = (u.reshape(out_f, in_f) - 128).to(torch.int8)  # undo zero-point 128
        return cls(weight, scale)

    def __repr__(self):
        return (
            f"Int8QWeight(shape={tuple(self.shape)}, dtype={self.dtype}, "
            f"device={self.device}, requires_grad={self.requires_grad})"
        )


# ----------------------------------------------------------------------
# Autograd: F.linear through the dequantize path
# ----------------------------------------------------------------------

class _Int8WeightOnlyLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Int8QWeight, bias: Optional[Tensor] = None):
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        w = weight.dequantize(dtype=x.dtype)
        out = x @ w.T
        if bias is not None:
            out = out + bias
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        w = weight.dequantize(dtype=grad_output.dtype)
        grad_x = grad_output @ w
        grad_flat = grad_output.reshape(-1, weight.shape[0])
        grad_w = grad_flat.T @ x.reshape(-1, weight.shape[1])
        grad_b = grad_flat.sum(0) if ctx.has_bias else None
        return grad_x, grad_w, grad_b


# ----------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------

@Int8QWeight.implements_torch_function(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    return _Int8WeightOnlyLinear.apply(*args, **kwargs)


@Int8QWeight.implements([aten.detach.default, aten.clone.default])
def _(func, types, args, kwargs):
    out = Int8QWeight(
        func(args[0].weight, *args[1:], **kwargs),
        func(args[0].scales, *args[1:], **kwargs),
    )
    return return_and_correct_aliasing(func, args, kwargs, out)


@Int8QWeight.implements(aten._to_copy.default)
def _(func, types, args, kwargs):
    device = kwargs.get("device", None)
    dtype = kwargs.get("dtype", None)
    out = Int8QWeight(
        args[0].weight.to(device=device),
        args[0].scales.to(device=device, dtype=dtype),
    )
    return return_and_correct_aliasing(func, args, kwargs, out)


@Int8QWeight.implements(aten.copy_.default)
def _(func, types, args, kwargs):
    dst, src = args[0], args[1]
    if isinstance(dst, Int8QWeight) and isinstance(src, Int8QWeight):
        dst.weight.copy_(src.weight)
        dst.scales.copy_(src.scales)
    elif isinstance(dst, Int8QWeight):
        weight, scales = _quantize_int8_triton(src, dst.scales.dtype)
        dst.weight.copy_(weight)
        dst.scales.copy_(scales)
    else:
        dst.copy_(src.dequantize())
    return dst


@Int8QWeight.implements(aten.zeros_like.default)
def _(func, types, args, kwargs):
    dtype = kwargs.get("dtype", args[0].dtype)
    device = kwargs.get("device", args[0].device)
    return torch.zeros(args[0].shape, dtype=dtype, device=device)
