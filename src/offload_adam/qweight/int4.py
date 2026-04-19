"""Int4 symmetric groupwise QAT weight, GPTQ storage.

Storage:
  qweight : (in_f // 8, out_f) int32 — 8 int4 values packed LSB-first
            per int32 along in_f; signed v stored as uint4 (v + 8).
  scales  : (n_groups, out_f) — per (group, out_col) absmax / 8.

Symmetric (zero point = 8), no activation reordering. Outer appearance
is (out_f, in_f) at scales.dtype; re-quantize happens in ``aten.copy_``.
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
# Python reference: pack / unpack / quantize
# ----------------------------------------------------------------------

def pack_int4_gptq(int_data: Tensor) -> Tensor:
    """Pack signed int4 into the GPTQ int32 layout.

    Args:
        int_data: (out_f, in_f) int8 in [-8, 7]. `in_f` must be a multiple of 8.
    Returns:
        (in_f // 8, out_f) int32 — each int32 packs 8 values from the same
        out column across 8 consecutive in positions LSB-first:
            packed[k, n] = Σᵢ ((int_data[n, 8k+i] + 8) << 4i), i=0..7.
    """
    assert int_data.shape[1] % 8 == 0, (
        f"in_features={int_data.shape[1]} must be a multiple of 8"
    )
    out_f, in_f = int_data.shape
    u = (int_data + 8).to(torch.int32)             # uint4 range [0, 15] in int32
    u = u.reshape(out_f, in_f // 8, 8)
    shifts = (torch.arange(8, device=u.device) * 4).to(torch.int32)
    packed = (u << shifts).sum(dim=-1, dtype=torch.int32)   # (out_f, in_f//8)
    return packed.T.contiguous()                   # (in_f//8, out_f)


def unpack_int4_gptq(packed: Tensor) -> Tensor:
    """Inverse of `pack_int4_gptq`. Returns (out_f, in_f) int8 in [-8, 7]."""
    in_f_div_8, out_f = packed.shape
    in_f = in_f_div_8 * 8
    p = packed.T.contiguous()                      # (out_f, in_f//8) int32
    shifts = (torch.arange(8, device=p.device) * 4).to(torch.int32)
    u = (p.unsqueeze(-1) >> shifts) & 0xF          # (out_f, in_f//8, 8) int32
    return (u - 8).to(torch.int8).reshape(out_f, in_f)


@torch.no_grad()
def quantize_int4_groupwise(
    tensor: Tensor, group_size: int = 128, eps: float = 1e-12,
) -> Tuple[Tensor, Tensor]:
    """Reference python quantize → GPTQ layout.

    Args:
        tensor: (out_f, in_f) float tensor. `in_f % group_size == 0` and
            `in_f % 8 == 0` are both required.
    Returns:
        qweight: (in_f // 8, out_f) int32 — GPTQ-packed.
        scales:  (n_groups, out_f) same dtype as input — per (group, col).
    """
    out_f, in_f = tensor.shape
    assert in_f % group_size == 0, (
        f"in_features={in_f} not divisible by group_size={group_size}"
    )
    assert in_f % 8 == 0, f"in_features={in_f} must be a multiple of 8"
    n_groups = in_f // group_size
    orig_dtype = tensor.dtype

    t = tensor.float().reshape(out_f, n_groups, group_size)
    scale = t.abs().amax(dim=-1) / 8                   # (out_f, n_groups) fp32
    inv_scale = 1.0 / scale.clamp(min=eps)
    q = (t * inv_scale.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int8)
    q = q.reshape(out_f, in_f)                         # (out_f, in_f) int8

    qweight = pack_int4_gptq(q)                        # (in_f//8, out_f) int32
    scales = scale.T.contiguous().to(orig_dtype)       # (n_groups, out_f)
    return qweight, scales


# ----------------------------------------------------------------------
# Triton kernels
# ----------------------------------------------------------------------

@triton.jit
def _dequant_int4_kernel(
    qweight_ptr,     # (in_f // 8, out_f) int32
    scales_ptr,      # (n_groups, out_f) scales.dtype
    out_ptr,         # (out_f, in_f) out.dtype
    qweight_stride_k, qweight_stride_n,
    scales_stride_g, scales_stride_n,
    out_stride_row, out_stride_col,
    n_out_rows,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Fused unpack + scale + dtype cast, one (BLOCK_M rows, 1 group) tile per program."""
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < n_out_rows

    k_packed_offs = tl.arange(0, GROUP_SIZE // 8)
    k_packed_start = pid_g * (GROUP_SIZE // 8)

    q_tile = tl.load(
        qweight_ptr
        + (k_packed_start + k_packed_offs)[:, None] * qweight_stride_k
        + rows[None, :] * qweight_stride_n,
        mask=row_mask[None, :],
        other=0,
    )                                                     # (K_PACKED, BLOCK_M) int32

    scale = tl.load(
        scales_ptr + pid_g * scales_stride_g + rows * scales_stride_n,
        mask=row_mask,
    )                                                     # (BLOCK_M,)

    shifts = tl.arange(0, 8) * 4                          # (8,) int32
    unpacked = (q_tile[:, :, None] >> shifts[None, None, :]) & 0xF
    signed = unpacked.to(tl.int8) - 8                     # (K_PACKED, BLOCK_M, 8)
    scaled = signed.to(scale.dtype) * scale[None, :, None]

    # (K_PACKED, BLOCK_M, 8) → (BLOCK_M, K_PACKED, 8) → (BLOCK_M, GROUP_SIZE)
    out_tile = tl.trans(scaled, 1, 0, 2).reshape(BLOCK_M, GROUP_SIZE)

    cols = pid_g * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    tl.store(
        out_ptr + rows[:, None] * out_stride_row + cols[None, :] * out_stride_col,
        out_tile,
        mask=row_mask[:, None],
    )


def _dequant_int4_triton(
    qweight: Tensor, scales: Tensor, group_size: int,
) -> Tensor:
    """Dequantize (in_f//8, out_f) int32 qweight + (n_groups, out_f) scales
    into (out_f, in_f) at scales.dtype."""
    in_f = qweight.shape[0] * 8
    out_f = qweight.shape[1]
    n_groups = scales.shape[0]
    assert n_groups * group_size == in_f
    out = torch.empty((out_f, in_f), device=qweight.device, dtype=scales.dtype)
    BLOCK_M = 32
    grid = (triton.cdiv(out_f, BLOCK_M), n_groups)
    _dequant_int4_kernel[grid](
        qweight, scales, out,
        qweight.stride(0), qweight.stride(1),
        scales.stride(0), scales.stride(1),
        out.stride(0), out.stride(1),
        out_f,
        GROUP_SIZE=group_size,
        BLOCK_M=BLOCK_M,
    )
    return out


@triton.jit
def _quantize_pack_int4_kernel(
    input_ptr,       # (out_f, in_f) bf16 / fp16 / fp32
    qweight_ptr,     # (in_f // 8, out_f) int32
    scales_ptr,      # (n_groups, out_f) scales.dtype
    input_stride_row, input_stride_col,
    qweight_stride_k, qweight_stride_n,
    scales_stride_g, scales_stride_n,
    n_out_rows,
    EPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Fused per-group absmax / int4 round / 8-into-int32 pack, one (BLOCK_M, 1 group) tile per program."""
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < n_out_rows

    cols = pid_g * GROUP_SIZE + tl.arange(0, GROUP_SIZE)

    tile = tl.load(
        input_ptr
        + rows[:, None] * input_stride_row
        + cols[None, :] * input_stride_col,
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    absmax = tl.max(tl.abs(tile), axis=1)
    scale = absmax / 8.0
    # rcp.rn matches torch's 1/x at exactly-halfway values.
    inv_scale = tl.extra.libdevice.rcp_rn(tl.maximum(scale, EPS))

    scaled = tile * inv_scale[:, None]
    q_f = tl.extra.libdevice.rint(scaled)               # round-half-to-even
    q_f = tl.maximum(q_f, -8.0)
    q_f = tl.minimum(q_f, 7.0)
    q_int = q_f.to(tl.int32)                            # keep in int32 for shifts

    u = q_int + 8                                       # uint4 [0, 15] in int32
    u = u.reshape(BLOCK_M, GROUP_SIZE // 8, 8)

    shifts = tl.arange(0, 8) * 4                        # (8,) int32
    packed_per_row = tl.sum(
        u * (1 << shifts)[None, None, :], axis=2,
    )                                                   # (BLOCK_M, K_PACKED) int32

    # (BLOCK_M, K_PACKED) → (K_PACKED, BLOCK_M) for m-inner store.
    packed_kn = tl.trans(packed_per_row, 1, 0)

    k_packed_offs = tl.arange(0, GROUP_SIZE // 8)
    k_packed_start = pid_g * (GROUP_SIZE // 8)
    tl.store(
        qweight_ptr
        + (k_packed_start + k_packed_offs)[:, None] * qweight_stride_k
        + rows[None, :] * qweight_stride_n,
        packed_kn,
        mask=row_mask[None, :],
    )

    tl.store(
        scales_ptr + pid_g * scales_stride_g + rows * scales_stride_n,
        scale,
        mask=row_mask,
    )


def _quantize_pack_int4_triton(
    tensor: Tensor, group_size: int, scale_dtype: torch.dtype,
) -> Tuple[Tensor, Tensor]:
    """Fused triton quantize + GPTQ pack. Returns (qweight, scales)."""
    assert tensor.ndim == 2
    out_f, in_f = tensor.shape
    assert in_f % group_size == 0
    assert in_f % 8 == 0
    n_groups = in_f // group_size
    qweight = torch.empty(
        (in_f // 8, out_f), dtype=torch.int32, device=tensor.device,
    )
    scales = torch.empty(
        (n_groups, out_f), dtype=scale_dtype, device=tensor.device,
    )
    BLOCK_M = 32
    grid = (triton.cdiv(out_f, BLOCK_M), n_groups)
    _quantize_pack_int4_kernel[grid](
        tensor, qweight, scales,
        tensor.stride(0), tensor.stride(1),
        qweight.stride(0), qweight.stride(1),
        scales.stride(0), scales.stride(1),
        out_f,
        EPS=1e-12,
        GROUP_SIZE=group_size,
        BLOCK_M=BLOCK_M,
    )
    return qweight, scales


# ----------------------------------------------------------------------
# Tensor subclass
# ----------------------------------------------------------------------

class Int4QWeight(QWeightBase):
    """Int4 groupwise symmetric QAT weight, raw GPTQ storage."""

    @staticmethod
    @torch._dynamo.disable
    def __new__(cls, qweight: Tensor, scales: Tensor, group_size: int):
        in_f = qweight.shape[0] * 8
        out_f = qweight.shape[1]
        return Tensor._make_wrapper_subclass(
            cls, (out_f, in_f), dtype=scales.dtype, device=qweight.device,
        )

    @torch._dynamo.disable
    def __init__(self, qweight: Tensor, scales: Tensor, group_size: int):
        assert qweight.dtype is torch.int32 and qweight.ndim == 2
        assert scales.ndim == 2
        self.qweight = qweight
        self.scales = scales
        self.group_size = group_size

    def __tensor_flatten__(self):
        return ["qweight", "scales"], [self.group_size]

    @classmethod
    def __tensor_unflatten__(cls, tensors, attrs, outer_size=None, outer_stride=None):
        return cls(tensors["qweight"], tensors["scales"], *attrs)

    @classmethod
    def from_float(
        cls, tensor: Tensor, group_size: int = 128,
        scale_dtype: torch.dtype = torch.bfloat16,
    ):
        """Quantize ``tensor`` into an ``Int4QWeight``.

        ``scale_dtype`` pins both the stored scale dtype and the subclass's
        outer ``dtype``. Default ``bfloat16``; pass ``float16`` for fp16 models.
        """
        qweight, scales = quantize_int4_groupwise(tensor.detach(), group_size)
        if scales.dtype != scale_dtype:
            scales = scales.to(scale_dtype)
        out = cls(qweight, scales, group_size)
        out.requires_grad_(tensor.requires_grad)
        return out

    @classmethod
    def can_quantize(cls, tensor: Tensor, group_size: int = 128, **_) -> bool:
        return (
            tensor.dim() == 2
            and tensor.shape[1] % group_size == 0
            and tensor.shape[1] % 8 == 0
        )

    def dequantize(self, dtype=None) -> Tensor:
        if dtype is None or dtype == self.scales.dtype:
            return _dequant_int4_triton(
                self.qweight, self.scales, self.group_size,
            )
        # fp32 upcast or cross-dtype request: python fallback.
        n_groups = self.scales.shape[0]
        compute_dtype = torch.float32 if dtype == torch.float32 else self.scales.dtype
        return (
            unpack_int4_gptq(self.qweight)
            .reshape(self.shape[0], n_groups, self.group_size)
            .to(compute_dtype)
            * self.scales.T.to(compute_dtype).unsqueeze(-1)
        ).reshape(self.shape).to(dtype)

    @classmethod
    def canonical_key_suffixes(cls) -> Tuple[str, ...]:
        # GPTQ set; `g_idx` omitted (desc_act=False, vllm treats it as optional).
        return ("qweight", "scales", "qzeros")

    def to_plain_state_dict(self) -> dict:
        n_groups, out_f = self.scales.shape
        assert out_f % 8 == 0, (
            f"out_features={out_f} must be a multiple of 8 to serialize "
            f"symmetric qzeros"
        )
        # Symmetric zero-point 8 at every slot → 0x88888888 per int32 (signed: -0x77777778).
        qzeros = torch.full(
            (n_groups, out_f // 8),
            fill_value=-0x77777778,
            dtype=torch.int32,
            device=self.qweight.device,
        )
        return {
            "qweight": self.qweight,
            "scales": self.scales,
            "qzeros": qzeros,
        }

    @classmethod
    def from_plain_state_dict(cls, plain: dict, reference=None) -> "Int4QWeight":
        # qzeros dropped on load; symmetric assumed at construction.
        qweight = plain["qweight"]
        scales = plain["scales"]
        in_f = qweight.shape[0] * 8
        group_size = in_f // scales.shape[0]
        return cls(qweight, scales, group_size)

    def __repr__(self):
        return (
            f"Int4QWeight(shape={tuple(self.shape)}, dtype={self.dtype}, "
            f"device={self.device}, group_size={self.group_size}, "
            f"requires_grad={self.requires_grad})"
        )


# ----------------------------------------------------------------------
# Autograd: F.linear through the dequantize path
# ----------------------------------------------------------------------

class _Int4WeightOnlyLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Int4QWeight, bias: Optional[Tensor] = None):
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

@Int4QWeight.implements_torch_function(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    return _Int4WeightOnlyLinear.apply(*args, **kwargs)


@Int4QWeight.implements([aten.detach.default, aten.clone.default])
def _(func, types, args, kwargs):
    out = Int4QWeight(
        func(args[0].qweight, *args[1:], **kwargs),
        func(args[0].scales, *args[1:], **kwargs),
        args[0].group_size,
    )
    return return_and_correct_aliasing(func, args, kwargs, out)


@Int4QWeight.implements(aten._to_copy.default)
def _(func, types, args, kwargs):
    device = kwargs.get("device", None)
    dtype = kwargs.get("dtype", None)
    out = Int4QWeight(
        args[0].qweight.to(device=device),
        args[0].scales.to(device=device, dtype=dtype),
        args[0].group_size,
    )
    return return_and_correct_aliasing(func, args, kwargs, out)


@Int4QWeight.implements(aten.copy_.default)
def _(func, types, args, kwargs):
    dst, src = args[0], args[1]
    if isinstance(dst, Int4QWeight) and isinstance(src, Int4QWeight):
        dst.qweight.copy_(src.qweight)
        dst.scales.copy_(src.scales)
    elif isinstance(dst, Int4QWeight):
        qweight, scales = _quantize_pack_int4_triton(
            src, dst.group_size, dst.scales.dtype,
        )
        dst.qweight.copy_(qweight)
        dst.scales.copy_(scales)
    else:
        dst.copy_(src.dequantize())
    return dst


@Int4QWeight.implements(aten.zeros_like.default)
def _(func, types, args, kwargs):
    dtype = kwargs.get("dtype", args[0].dtype)
    device = kwargs.get("device", args[0].device)
    return torch.zeros(args[0].shape, dtype=dtype, device=device)
