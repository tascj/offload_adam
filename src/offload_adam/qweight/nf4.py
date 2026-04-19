"""NF4 blockwise QAT weight (bitsandbytes storage).

Storage:
  weight_packed : (out_f, in_f // 2) uint8 — two 4-bit NF4 indices per byte.
                  Even col → high nibble, odd → low (bnb `kQuantizeBlockwise`).
  absmax        : (out_f * n_groups,) fp32 — per-block abs-max, flat
                  row-major across (out_f, n_groups).

Blockwise NF4: split each row into `blocksize` groups (default 64),
round to nearest in the 16-entry NF4 codebook (quantiles of N(0, 1))
after dividing by the block's abs-max. Double-quant is skipped.
"""

import json
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.utils._python_dispatch import return_and_correct_aliasing

from .base import QWeightBase

_TORCH_DTYPE_TO_STR = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
}
_STR_TO_TORCH_DTYPE = {v: k for k, v in _TORCH_DTYPE_TO_STR.items()}

aten = torch.ops.aten

# The 16-entry NF4 lookup table (bitsandbytes canonical values).
NF4_LUT = (
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
)


_NF4_LUT_CACHE: dict = {}


def _nf4_lut(device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
    """Cached per-(device, dtype) copy of the NF4 LUT — avoids a host→device
    rebuild on every kernel launch. Immutable; safe to share."""
    key = (device, dtype)
    lut = _NF4_LUT_CACHE.get(key)
    if lut is None:
        lut = torch.tensor(NF4_LUT, dtype=dtype, device=device)
        _NF4_LUT_CACHE[key] = lut
    return lut


# ----------------------------------------------------------------------
# Python reference: pack / unpack / quantize
# ----------------------------------------------------------------------

def pack_nf4(indices: Tensor) -> Tensor:
    """(out_f, in_f) uint8 indices (0..15) → (out_f, in_f // 2) uint8.

    Even column → high nibble, odd → low nibble (bnb convention).
    """
    assert indices.dtype is torch.uint8
    assert indices.shape[1] % 2 == 0
    high = indices[:, 0::2] << 4
    low = indices[:, 1::2]
    return (high | low).to(torch.uint8)


def unpack_nf4(packed: Tensor) -> Tensor:
    """Inverse of `pack_nf4`. Returns (out_f, in_f) uint8."""
    assert packed.dtype is torch.uint8
    high = (packed >> 4) & 0xF
    low = packed & 0xF
    out_f, half_in_f = packed.shape
    result = torch.empty(
        (out_f, 2 * half_in_f), dtype=torch.uint8, device=packed.device,
    )
    result[:, 0::2] = high
    result[:, 1::2] = low
    return result


@torch.no_grad()
def quantize_nf4_blockwise(
    tensor: Tensor, blocksize: int = 64,
) -> Tuple[Tensor, Tensor]:
    """Reference python NF4 quantize.

    Args:
        tensor: (out_f, in_f) float tensor. `in_f % blocksize == 0`
            and `in_f % 2 == 0` are both required.
    Returns:
        weight_packed: (out_f, in_f // 2) uint8.
        absmax: (out_f * n_groups,) fp32 — flat row-major over groups.
    """
    assert tensor.ndim == 2
    out_f, in_f = tensor.shape
    assert in_f % blocksize == 0
    assert in_f % 2 == 0
    n_groups = in_f // blocksize

    t = tensor.float().reshape(out_f, n_groups, blocksize)
    absmax = t.abs().amax(dim=-1)                          # (out_f, n_groups) fp32
    # Use `t * (1/absmax)` instead of `t / absmax` to match the triton
    # kernel's recip-then-multiply path bit-for-bit.
    inv_absmax = 1.0 / absmax.clamp(min=1e-12)
    normalized = t * inv_absmax.unsqueeze(-1)
    lut = _nf4_lut(tensor.device)
    dists = (normalized.unsqueeze(-1) - lut).abs()         # (..., 16)
    # (out_f, n_groups, blocksize) uint8
    indices = dists.argmin(dim=-1).to(torch.uint8)
    indices = indices.reshape(out_f, in_f)

    packed = pack_nf4(indices)
    return packed, absmax.reshape(-1)


def dequantize_nf4_blockwise(
    packed: Tensor, absmax: Tensor, in_f: int, blocksize: int = 64,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Inverse of `quantize_nf4_blockwise`."""
    out_f = packed.shape[0]
    n_groups = in_f // blocksize
    indices = unpack_nf4(packed).to(torch.long)
    lut = _nf4_lut(packed.device, dtype=dtype)
    vals = lut[indices].reshape(out_f, n_groups, blocksize)
    absmax_2d = absmax.to(dtype).reshape(out_f, n_groups)
    return (vals * absmax_2d.unsqueeze(-1)).reshape(out_f, in_f)


# ----------------------------------------------------------------------
# Triton kernels
# ----------------------------------------------------------------------

@triton.jit
def _dequant_nf4_kernel(
    packed_ptr,      # (out_f, in_f // 2) uint8
    absmax_ptr,      # (out_f * n_groups,) fp32
    lut_ptr,         # (16,) fp32
    out_ptr,         # (out_f, in_f) out.dtype
    packed_stride_row, packed_stride_col,
    out_stride_row, out_stride_col,
    n_out_rows, n_groups,
    BLOCKSIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """Fused unpack + LUT lookup + scale, one (BLOCK_M, 1 block) tile per program."""
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < n_out_rows

    col_offs = tl.arange(0, BLOCKSIZE // 2)
    packed_cols = pid_g * (BLOCKSIZE // 2) + col_offs

    # (BLOCK_M, BLOCKSIZE//2) uint8
    p = tl.load(
        packed_ptr
        + rows[:, None] * packed_stride_row
        + packed_cols[None, :] * packed_stride_col,
        mask=row_mask[:, None],
        other=0,
    )

    high_idx = ((p >> 4) & 0xF).to(tl.int32)
    low_idx = (p & 0xF).to(tl.int32)

    # Direct gather from the 16-entry LUT → (BLOCK_M, BLOCKSIZE//2) fp32.
    high_val = tl.load(lut_ptr + high_idx)
    low_val = tl.load(lut_ptr + low_idx)

    # absmax for this block, per row.
    absmax = tl.load(
        absmax_ptr + rows * n_groups + pid_g, mask=row_mask,
    )                                                     # (BLOCK_M,) fp32

    high_val = high_val * absmax[:, None]
    low_val = low_val * absmax[:, None]

    # Interleave high/low pairs into a contiguous tile, then one store.
    joined = tl.join(high_val, low_val)                    # (BLOCK_M, BLOCKSIZE//2, 2)
    out_tile = joined.reshape(BLOCK_M, BLOCKSIZE)

    out_cols = pid_g * BLOCKSIZE + tl.arange(0, BLOCKSIZE)
    tl.store(
        out_ptr
        + rows[:, None] * out_stride_row
        + out_cols[None, :] * out_stride_col,
        out_tile.to(OUT_DTYPE),
        mask=row_mask[:, None],
    )


_TL_DTYPE_MAP = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


def _dequant_nf4_triton(
    packed: Tensor, absmax: Tensor, in_f: int, blocksize: int,
    out_dtype: torch.dtype,
) -> Tensor:
    out_f = packed.shape[0]
    assert packed.shape[1] * 2 == in_f
    n_groups = in_f // blocksize
    assert absmax.numel() == out_f * n_groups
    out = torch.empty((out_f, in_f), device=packed.device, dtype=out_dtype)
    lut = _nf4_lut(packed.device)
    BLOCK_M = 32
    grid = (triton.cdiv(out_f, BLOCK_M), n_groups)
    _dequant_nf4_kernel[grid](
        packed, absmax, lut, out,
        packed.stride(0), packed.stride(1),
        out.stride(0), out.stride(1),
        out_f, n_groups,
        BLOCKSIZE=blocksize,
        BLOCK_M=BLOCK_M,
        OUT_DTYPE=_TL_DTYPE_MAP[out_dtype],
    )
    return out


@triton.jit
def _quantize_nf4_kernel(
    input_ptr,       # (out_f, in_f) fp tensor
    packed_ptr,      # (out_f, in_f // 2) uint8
    absmax_ptr,      # (out_f * n_groups,) fp32
    lut_ptr,         # (16,) fp32
    input_stride_row, input_stride_col,
    packed_stride_row, packed_stride_col,
    n_out_rows, n_groups,
    EPS: tl.constexpr,
    BLOCKSIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Per-block absmax + nearest-LUT + 2-nibble pack, one (BLOCK_M, 1 block) tile per program."""
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < n_out_rows

    col_offs = tl.arange(0, BLOCKSIZE)
    cols = pid_g * BLOCKSIZE + col_offs

    tile = tl.load(
        input_ptr
        + rows[:, None] * input_stride_row
        + cols[None, :] * input_stride_col,
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)                                        # (BLOCK_M, BLOCKSIZE)

    absmax = tl.max(tl.abs(tile), axis=1)                   # (BLOCK_M,)
    inv_abs = tl.extra.libdevice.rcp_rn(tl.maximum(absmax, EPS))
    normalized = tile * inv_abs[:, None]                    # (BLOCK_M, BLOCKSIZE)

    # Sequential 16-way min — avoids a (BLOCK_M, BLOCKSIZE, 16) distance tensor.
    best_idx = tl.zeros([BLOCK_M, BLOCKSIZE], dtype=tl.int32)
    best_dist = tl.full([BLOCK_M, BLOCKSIZE], float("inf"), dtype=tl.float32)
    for i in tl.static_range(16):
        lut_val = tl.load(lut_ptr + i)                      # scalar fp32
        d = tl.abs(normalized - lut_val)
        closer = d < best_dist
        best_idx = tl.where(closer, i, best_idx)
        best_dist = tl.where(closer, d, best_dist)
    # Pack (even → high, odd → low) via weighted-sum reduce.
    idx_pairs = best_idx.reshape(BLOCK_M, BLOCKSIZE // 2, 2)
    pair_weights = (1 - tl.arange(0, 2)) * 15 + 1            # [16, 1]
    packed = tl.sum(
        idx_pairs * pair_weights[None, None, :], axis=2,
    ).to(tl.uint8)                                           # (BLOCK_M, BLOCKSIZE//2)

    packed_cols = pid_g * (BLOCKSIZE // 2) + tl.arange(0, BLOCKSIZE // 2)
    tl.store(
        packed_ptr
        + rows[:, None] * packed_stride_row
        + packed_cols[None, :] * packed_stride_col,
        packed,
        mask=row_mask[:, None],
    )

    # Write one absmax per (row, group).
    tl.store(
        absmax_ptr + rows * n_groups + pid_g, absmax, mask=row_mask,
    )


def _quantize_nf4_triton(
    tensor: Tensor, blocksize: int,
) -> Tuple[Tensor, Tensor]:
    out_f, in_f = tensor.shape
    assert in_f % blocksize == 0
    assert in_f % 2 == 0
    n_groups = in_f // blocksize

    packed = torch.empty(
        (out_f, in_f // 2), dtype=torch.uint8, device=tensor.device,
    )
    absmax = torch.empty(
        (out_f * n_groups,), dtype=torch.float32, device=tensor.device,
    )
    lut = _nf4_lut(tensor.device)

    BLOCK_M = 32
    grid = (triton.cdiv(out_f, BLOCK_M), n_groups)
    _quantize_nf4_kernel[grid](
        tensor, packed, absmax, lut,
        tensor.stride(0), tensor.stride(1),
        packed.stride(0), packed.stride(1),
        out_f, n_groups,
        EPS=1e-12,
        BLOCKSIZE=blocksize,
        BLOCK_M=BLOCK_M,
    )
    return packed, absmax


# ----------------------------------------------------------------------
# Tensor subclass
# ----------------------------------------------------------------------

class NF4QWeight(QWeightBase):
    """NF4 blockwise QAT weight, bnb storage format (no double-quant)."""

    @staticmethod
    @torch._dynamo.disable
    def __new__(
        cls, weight_packed: Tensor, absmax: Tensor,
        in_f: int, blocksize: int, outer_dtype: torch.dtype,
    ):
        out_f = weight_packed.shape[0]
        return Tensor._make_wrapper_subclass(
            cls, (out_f, in_f), dtype=outer_dtype, device=weight_packed.device,
        )

    @torch._dynamo.disable
    def __init__(
        self, weight_packed: Tensor, absmax: Tensor,
        in_f: int, blocksize: int, outer_dtype: torch.dtype,
    ):
        assert weight_packed.dtype is torch.uint8 and weight_packed.ndim == 2
        assert absmax.dtype is torch.float32 and absmax.ndim == 1
        self.weight_packed = weight_packed
        self.absmax = absmax
        self.in_f = in_f
        self.blocksize = blocksize
        self.outer_dtype = outer_dtype

    def __tensor_flatten__(self):
        return ["weight_packed", "absmax"], [
            self.in_f, self.blocksize, self.outer_dtype,
        ]

    @classmethod
    def __tensor_unflatten__(cls, tensors, attrs, outer_size=None, outer_stride=None):
        return cls(tensors["weight_packed"], tensors["absmax"], *attrs)

    @classmethod
    def from_float(cls, tensor: Tensor, blocksize: int = 64, **_):
        packed, absmax = quantize_nf4_blockwise(tensor.detach(), blocksize)
        out = cls(packed, absmax, tensor.shape[1], blocksize, tensor.dtype)
        out.requires_grad_(tensor.requires_grad)
        return out

    @classmethod
    def can_quantize(cls, tensor: Tensor, blocksize: int = 64, **_) -> bool:
        return (
            tensor.dim() == 2
            and tensor.shape[1] % blocksize == 0
            and tensor.shape[1] % 2 == 0
        )

    def dequantize(self, dtype=None) -> Tensor:
        if dtype is None:
            dtype = self.outer_dtype
        if dtype in _TL_DTYPE_MAP:
            return _dequant_nf4_triton(
                self.weight_packed, self.absmax,
                self.in_f, self.blocksize, dtype,
            )
        # Unsupported dtype: python fallback.
        return dequantize_nf4_blockwise(
            self.weight_packed, self.absmax, self.in_f, self.blocksize, dtype,
        )

    @classmethod
    def canonical_key_suffixes(cls) -> Tuple[str, ...]:
        # bitsandbytes canonical layout (`QuantState.from_dict` consumes these):
        #   weight: uint8 (numel/2, 1) flat-packed (bnb stores as column vector)
        #   weight.absmax: fp32 (num_blocks,)
        #   weight.quant_map: fp32 (16,) NF4 LUT
        #   weight.quant_state.bitsandbytes__nf4: uint8 JSON {type, blocksize, dtype, shape}
        return (
            "weight",
            "weight.absmax",
            "weight.quant_map",
            "weight.quant_state.bitsandbytes__nf4",
        )

    def to_plain_state_dict(self) -> dict:
        out_f = self.weight_packed.shape[0]
        in_f = self.weight_packed.shape[1] * 2
        device = self.weight_packed.device
        # bnb serializes as flat (numel/2, 1); row-major flatten matches.
        weight_flat = self.weight_packed.reshape(-1, 1)
        quant_map = torch.tensor(NF4_LUT, dtype=torch.float32, device=device)
        meta = {
            "quant_type": "nf4",
            "blocksize": self.blocksize,
            "dtype": _TORCH_DTYPE_TO_STR[self.outer_dtype],
            "shape": [out_f, in_f],
        }
        meta_bytes = json.dumps(meta).encode("utf-8")
        meta_tensor = torch.tensor(
            list(meta_bytes), dtype=torch.uint8, device=device,
        )
        return {
            "weight": weight_flat,
            "weight.absmax": self.absmax,
            "weight.quant_map": quant_map,
            "weight.quant_state.bitsandbytes__nf4": meta_tensor,
        }

    @classmethod
    def from_plain_state_dict(cls, plain: dict, reference=None) -> "NF4QWeight":
        meta_bytes = bytes(
            plain["weight.quant_state.bitsandbytes__nf4"].cpu().tolist()
        )
        meta = json.loads(meta_bytes.decode("utf-8"))
        out_f, in_f = meta["shape"]
        blocksize = meta["blocksize"]
        outer_dtype = _STR_TO_TORCH_DTYPE[meta["dtype"]]
        weight_packed = plain["weight"].reshape(out_f, in_f // 2)
        absmax = plain["weight.absmax"]
        return cls(weight_packed, absmax, in_f, blocksize, outer_dtype)

    @classmethod
    def build_hf_quantization_config(
        cls, skip_patterns=(), compute_dtype: str = "bfloat16", **_,
    ) -> dict:
        return {
            "quant_method": "bitsandbytes",
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": False,
            "bnb_4bit_compute_dtype": compute_dtype,
            "llm_int8_skip_modules": list(skip_patterns),
        }

    def __repr__(self):
        return (
            f"NF4QWeight(shape={tuple(self.shape)}, dtype={self.dtype}, "
            f"device={self.device}, blocksize={self.blocksize}, "
            f"requires_grad={self.requires_grad})"
        )


# ----------------------------------------------------------------------
# Autograd: F.linear through the dequantize path
# ----------------------------------------------------------------------

class _NF4WeightOnlyLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: NF4QWeight, bias: Optional[Tensor] = None):
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

@NF4QWeight.implements_torch_function(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    return _NF4WeightOnlyLinear.apply(*args, **kwargs)


@NF4QWeight.implements([aten.detach.default, aten.clone.default])
def _(func, types, args, kwargs):
    out = NF4QWeight(
        func(args[0].weight_packed, *args[1:], **kwargs),
        func(args[0].absmax, *args[1:], **kwargs),
        args[0].in_f, args[0].blocksize, args[0].outer_dtype,
    )
    return return_and_correct_aliasing(func, args, kwargs, out)


@NF4QWeight.implements(aten._to_copy.default)
def _(func, types, args, kwargs):
    device = kwargs.get("device", None)
    dtype = kwargs.get("dtype", None)
    outer = dtype if dtype is not None else args[0].outer_dtype
    out = NF4QWeight(
        args[0].weight_packed.to(device=device),
        args[0].absmax.to(device=device),
        args[0].in_f, args[0].blocksize, outer,
    )
    return return_and_correct_aliasing(func, args, kwargs, out)


@NF4QWeight.implements(aten.copy_.default)
def _(func, types, args, kwargs):
    dst, src = args[0], args[1]
    if isinstance(dst, NF4QWeight) and isinstance(src, NF4QWeight):
        dst.weight_packed.copy_(src.weight_packed)
        dst.absmax.copy_(src.absmax)
    elif isinstance(dst, NF4QWeight):
        packed, absmax = _quantize_nf4_triton(src, dst.blocksize)
        dst.weight_packed.copy_(packed)
        dst.absmax.copy_(absmax)
    else:
        dst.copy_(src.dequantize())
    return dst


@NF4QWeight.implements(aten.zeros_like.default)
def _(func, types, args, kwargs):
    dtype = kwargs.get("dtype", args[0].dtype)
    device = kwargs.get("device", args[0].device)
    return torch.zeros(args[0].shape, dtype=dtype, device=device)
