"""NVFP4 blockwise QAT weight (compressed-tensors W4A16 NVFP4).

Storage:
  weight_packed       : (out_f, in_f // 2) uint8 — two FP4 (E2M1) per byte.
                        Even col → low nibble, odd → high (cuBLAS/TRT FP4).
  weight_scale        : (out_f, n_groups) float8_e4m3fn — per 16-element block scale.
  weight_global_scale : () fp32 — per-tensor scalar; effective scale =
                        block_scale.to(fp32) * global_scale.

Two-level scaling keeps per-block scale in FP8's range (≤ 448) while the
global scale absorbs tensor-wide dynamic range. Blackwell 5th-gen tensor
cores ingest this natively; training-time fake-quant dequantizes to bf16.
Block size is fixed at 16 by the format spec.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.utils._python_dispatch import return_and_correct_aliasing

from .base import QWeightBase

aten = torch.ops.aten

NVFP4_BLOCKSIZE = 16
FP4_E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0

# FP4 E2M1 codebook indexed by 4-bit code:
#   bit 3 = sign, bits 2..1 = exponent (bias 1), bit 0 = mantissa.
FP4_E2M1_LUT = (
     0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


_FP4_LUT_CACHE: dict = {}


def _fp4_lut(device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
    """Cached per-(device, dtype) copy of the FP4 E2M1 LUT. Immutable, shareable."""
    key = (device, dtype)
    lut = _FP4_LUT_CACHE.get(key)
    if lut is None:
        lut = torch.tensor(FP4_E2M1_LUT, dtype=dtype, device=device)
        _FP4_LUT_CACHE[key] = lut
    return lut


# ----------------------------------------------------------------------
# Python reference: pack / unpack / quantize
# ----------------------------------------------------------------------

def pack_fp4(indices: Tensor) -> Tensor:
    """(out_f, in_f) uint8 indices (0..15) → (out_f, in_f // 2) uint8.

    Even column → low nibble, odd → high nibble (cuBLAS FP4 convention).
    """
    assert indices.dtype is torch.uint8
    assert indices.shape[1] % 2 == 0
    low = indices[:, 0::2]
    high = indices[:, 1::2] << 4
    return (high | low).to(torch.uint8)


def unpack_fp4(packed: Tensor) -> Tensor:
    """Inverse of `pack_fp4`. Returns (out_f, in_f) uint8."""
    assert packed.dtype is torch.uint8
    low = packed & 0xF
    high = (packed >> 4) & 0xF
    out_f, half_in_f = packed.shape
    result = torch.empty(
        (out_f, 2 * half_in_f), dtype=torch.uint8, device=packed.device,
    )
    result[:, 0::2] = low
    result[:, 1::2] = high
    return result


def _quantize_fp4_e2m1_rne(x: Tensor) -> Tensor:
    """fp32 → uint8 FP4 E2M1 code, bit-for-bit ``cvt.rn.satfinite.e2m1x2.f32``.

    Round-to-nearest-even at tied midpoints, IEEE ``signbit`` for ±0.
    Positive code layout {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0} has
    even-mantissa codes at {0, 2, 4, 6}, odd at {1, 3, 5, 7}. At ties we
    bias toward even-mantissa: ``<=`` at a boundary means snap down iff
    the lower neighbour is even, else ``<``.
    """
    ax = x.abs()
    code = torch.zeros_like(ax, dtype=torch.uint8)
    u8 = lambda v: torch.tensor(v, dtype=torch.uint8, device=x.device)  # noqa: E731
    # Boundaries (positive side):
    #   [0, 0.25]    → 0         tie at 0.25 → 0  (even)
    #   (0.25, 0.75) → 0.5       tie at 0.75 → 1  (even)
    #   [0.75, 1.25] → 1.0       tie at 1.25 → 1  (even)
    #   (1.25, 1.75) → 1.5       tie at 1.75 → 2  (even)
    #   [1.75, 2.5]  → 2.0       tie at 2.5  → 2  (even)
    #   (2.5, 3.5)   → 3.0       tie at 3.5  → 4  (even)
    #   [3.5, 5.0]   → 4.0       tie at 5.0  → 4  (even)
    #   (5.0, ∞)     → 6.0       satfinite clamps at 6
    code = torch.where(ax > 0.25, u8(1), code)
    code = torch.where(ax >= 0.75, u8(2), code)
    code = torch.where(ax > 1.25, u8(3), code)
    code = torch.where(ax >= 1.75, u8(4), code)
    code = torch.where(ax > 2.5, u8(5), code)
    code = torch.where(ax >= 3.5, u8(6), code)
    code = torch.where(ax > 5.0, u8(7), code)
    # Preserve sign bit (IEEE signbit so -0.0 maps to code 8).
    sign_bit = torch.signbit(x).to(torch.uint8) << 3
    return code | sign_bit


@torch.no_grad()
def quantize_nvfp4_blockwise(
    tensor: Tensor, eps: float = 1e-12,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Reference python NVFP4 quantize — bit-matches the native PTX path.

    Args:
        tensor: (out_f, in_f) float tensor. ``in_f % 16 == 0`` required.
    Returns:
        (weight_packed (out_f, in_f//2) uint8,
         weight_scale  (out_f, n_groups) float8_e4m3fn,
         weight_global_scale () fp32).
    """
    assert tensor.ndim == 2
    out_f, in_f = tensor.shape
    assert in_f % NVFP4_BLOCKSIZE == 0
    assert in_f % 2 == 0
    n_groups = in_f // NVFP4_BLOCKSIZE

    t = tensor.float()
    tensor_amax = t.abs().amax().clamp(min=eps)
    global_scale = tensor_amax / (FP8_E4M3_MAX * FP4_E2M1_MAX)

    inv_global = 1.0 / global_scale

    t_b = t.reshape(out_f, n_groups, NVFP4_BLOCKSIZE)
    block_amax = t_b.abs().amax(dim=-1).clamp(min=eps)     # (out_f, n_groups)
    block_scale_fp32 = block_amax / FP4_E2M1_MAX
    block_scale_fp8 = (block_scale_fp32 * inv_global).to(torch.float8_e4m3fn)

    # Effective fp32 scale per block after fp8 round-trip.
    effective = block_scale_fp8.to(torch.float32) * global_scale
    inv_eff = 1.0 / effective.clamp(min=eps)

    # (out_f, n_groups, BLOCKSIZE)
    normalized = t_b * inv_eff.unsqueeze(-1)

    indices = _quantize_fp4_e2m1_rne(normalized).reshape(out_f, in_f)
    packed = pack_fp4(indices)
    return packed, block_scale_fp8, global_scale


def dequantize_nvfp4_blockwise(
    packed: Tensor, block_scale_fp8: Tensor, global_scale: Tensor,
    in_f: int, dtype: torch.dtype = torch.float32,
) -> Tensor:
    out_f = packed.shape[0]
    n_groups = in_f // NVFP4_BLOCKSIZE
    indices = unpack_fp4(packed).to(torch.long)
    lut = _fp4_lut(packed.device, dtype=torch.float32)
    vals_fp32 = lut[indices].reshape(out_f, n_groups, NVFP4_BLOCKSIZE)
    effective = (
        block_scale_fp8.to(torch.float32) * global_scale
    ).reshape(out_f, n_groups)
    return (vals_fp32 * effective.unsqueeze(-1)).reshape(out_f, in_f).to(dtype)


# ----------------------------------------------------------------------
# Triton kernels
# ----------------------------------------------------------------------

_TL_DTYPE_MAP = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


def _supports_native_fp4() -> bool:
    """Blackwell (sm_100+) has ``cvt.{f16x2,e2m1x2}``; else LUT fallback."""
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
    except Exception:
        return False
    return major >= 10


@triton.jit
def _fp4_code_to_fp32(idx):
    """4-bit FP4 E2M1 code → fp32 via direct bit synthesis.

    Layout of idx[3:0]: s | e1 e0 | m. The magnitude codes map to:
        pos 0  → 0                (fp32 zero)
        pos 1  → 0.5              (E2M1 subnormal → fp32 normal,
                                    exp=126, mantissa=0 — NOT m<<22)
        pos ≥2 → (1 + 0.5m) · 2^(e-1)
                 fp32 exp = 126 + (pos>>1), mantissa = (pos&1)<<22

    Sign bit flows directly to fp32 bit 31 so -0 is preserved.
    """
    pos = idx & 0x7
    is_zero = pos == 0
    is_subnormal = pos == 1
    exp_field = (126 + (pos >> 1)) << 23
    mant_field = tl.where(is_subnormal, 0, (pos & 1) << 22)
    s = (idx >> 3) & 1
    bits = (s << 31) | tl.where(is_zero, 0, exp_field | mant_field)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _dequant_nvfp4_kernel_lut(
    packed_ptr,           # (out_f, in_f // 2) uint8
    block_scale_fp8_ptr,  # (out_f, n_groups) fp8_e4m3
    global_scale_ptr,     # () fp32 scalar
    out_ptr,              # (out_f, in_f) out.dtype
    packed_stride_row, packed_stride_col,
    scale_stride_row, scale_stride_col,
    out_stride_row, out_stride_col,
    n_out_rows, n_groups,
    BLOCKSIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """Software dequant: bit-synthesize fp32 from FP4 code, LUT-free. Pre-Blackwell fallback."""
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < n_out_rows

    col_offs = tl.arange(0, BLOCKSIZE // 2)
    packed_cols = pid_g * (BLOCKSIZE // 2) + col_offs

    p = tl.load(
        packed_ptr
        + rows[:, None] * packed_stride_row
        + packed_cols[None, :] * packed_stride_col,
        mask=row_mask[:, None],
        other=0,
    )                                                       # (BLOCK_M, BLOCKSIZE//2)

    # Even col → low nibble, odd → high.
    low_idx = (p & 0xF).to(tl.int32)
    high_idx = ((p >> 4) & 0xF).to(tl.int32)

    low_val = _fp4_code_to_fp32(low_idx)
    high_val = _fp4_code_to_fp32(high_idx)

    # Reconstruct effective fp32 scale for this block.
    block_scale_fp8 = tl.load(
        block_scale_fp8_ptr
        + rows * scale_stride_row
        + pid_g * scale_stride_col,
        mask=row_mask,
    )                                                       # (BLOCK_M,) fp8
    global_scale = tl.load(global_scale_ptr)                # scalar fp32
    effective = block_scale_fp8.to(tl.float32) * global_scale

    low_val = low_val * effective[:, None]
    high_val = high_val * effective[:, None]

    # Interleave: even cols ← low, odd cols ← high (matches packing).
    joined = tl.join(low_val, high_val)                     # (BLOCK_M, BLOCKSIZE//2, 2)
    out_tile = joined.reshape(BLOCK_M, BLOCKSIZE)

    out_cols = pid_g * BLOCKSIZE + tl.arange(0, BLOCKSIZE)
    tl.store(
        out_ptr
        + rows[:, None] * out_stride_row
        + out_cols[None, :] * out_stride_col,
        out_tile.to(OUT_DTYPE),
        mask=row_mask[:, None],
    )


@triton.jit
def _dequant_nvfp4_kernel_native(
    packed_ptr,           # (out_f, in_f // 2) uint8
    block_scale_fp8_ptr,  # (out_f, n_groups) fp8_e4m3
    global_scale_ptr,     # () fp32 scalar
    out_ptr,              # (out_f, in_f) out.dtype
    packed_stride_row, packed_stride_col,
    scale_stride_row, scale_stride_col,
    out_stride_row, out_stride_col,
    n_out_rows, n_groups,
    BLOCKSIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """Blackwell-native dequant: ``cvt.rn.f16x2.e2m1x2`` decodes 2 FP4 → 2 FP16 per byte."""
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < n_out_rows

    packed_cols = pid_g * (BLOCKSIZE // 2) + tl.arange(0, BLOCKSIZE // 2)

    # (BLOCK_M, BLOCKSIZE//2) uint8
    p = tl.load(
        packed_ptr
        + rows[:, None] * packed_stride_row
        + packed_cols[None, :] * packed_stride_col,
        mask=row_mask[:, None],
        other=0,
    )

    # PTX `cvt.rn.f16x2.e2m1x2 d.b32, s.b8`: src[3:0] → low fp16, src[7:4] → high.
    # Triton holds the uint8 in a 32-bit register; narrow to .b8 before cvt.
    # Emit the fp16 pair as int32 and bit-split into two fp16 streams
    # (Triton has no direct int32 → fp16x2 expand).
    pair_i32 = tl.inline_asm_elementwise(
        asm=(
            "{\n"
            "  .reg .b8 src8;\n"
            "  cvt.u8.u32 src8, $1;\n"
            "  cvt.rn.f16x2.e2m1x2 $0, src8;\n"
            "}"
        ),
        constraints="=r,r",
        args=[p],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )  # (BLOCK_M, BLOCKSIZE//2) int32
    low_bits = (pair_i32 & 0xFFFF).to(tl.int16)
    high_bits = (pair_i32 >> 16).to(tl.int16)
    even_fp16 = low_bits.to(tl.float16, bitcast=True)
    odd_fp16 = high_bits.to(tl.float16, bitcast=True)
    joined = tl.join(even_fp16, odd_fp16)                   # (BLOCK_M, BLOCKSIZE//2, 2)
    tile_fp16 = joined.reshape(BLOCK_M, BLOCKSIZE)

    # Reconstruct effective fp32 scale per block.
    block_scale_fp8 = tl.load(
        block_scale_fp8_ptr
        + rows * scale_stride_row
        + pid_g * scale_stride_col,
        mask=row_mask,
    )
    global_scale = tl.load(global_scale_ptr)
    effective = block_scale_fp8.to(tl.float32) * global_scale   # (BLOCK_M,)

    out_tile = tile_fp16.to(tl.float32) * effective[:, None]

    out_cols = pid_g * BLOCKSIZE + tl.arange(0, BLOCKSIZE)
    tl.store(
        out_ptr
        + rows[:, None] * out_stride_row
        + out_cols[None, :] * out_stride_col,
        out_tile.to(OUT_DTYPE),
        mask=row_mask[:, None],
    )


def _dequant_nvfp4_triton(
    packed: Tensor, block_scale_fp8: Tensor, global_scale: Tensor,
    in_f: int, out_dtype: torch.dtype,
) -> Tensor:
    out_f = packed.shape[0]
    n_groups = in_f // NVFP4_BLOCKSIZE
    assert block_scale_fp8.shape == (out_f, n_groups)
    out = torch.empty((out_f, in_f), device=packed.device, dtype=out_dtype)
    BLOCK_M = 32
    grid = (triton.cdiv(out_f, BLOCK_M), n_groups)
    if _supports_native_fp4():
        _dequant_nvfp4_kernel_native[grid](
            packed, block_scale_fp8, global_scale, out,
            packed.stride(0), packed.stride(1),
            block_scale_fp8.stride(0), block_scale_fp8.stride(1),
            out.stride(0), out.stride(1),
            out_f, n_groups,
            BLOCKSIZE=NVFP4_BLOCKSIZE,
            BLOCK_M=BLOCK_M,
            OUT_DTYPE=_TL_DTYPE_MAP[out_dtype],
        )
    else:
        _dequant_nvfp4_kernel_lut[grid](
            packed, block_scale_fp8, global_scale, out,
            packed.stride(0), packed.stride(1),
            block_scale_fp8.stride(0), block_scale_fp8.stride(1),
            out.stride(0), out.stride(1),
            out_f, n_groups,
            BLOCKSIZE=NVFP4_BLOCKSIZE,
            BLOCK_M=BLOCK_M,
            OUT_DTYPE=_TL_DTYPE_MAP[out_dtype],
        )
    return out


@triton.jit
def _quantize_nvfp4_prologue(
    input_ptr, block_scale_fp8_ptr, global_scale_ptr,
    input_stride_row, input_stride_col,
    scale_stride_row, scale_stride_col,
    n_out_rows,
    EPS: tl.constexpr,
    BLOCKSIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    pid_m, pid_g,
):
    """Shared prologue: load tile, store per-block fp8 scale, return normalized fp32 tile."""
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < n_out_rows
    cols = pid_g * BLOCKSIZE + tl.arange(0, BLOCKSIZE)

    tile = tl.load(
        input_ptr
        + rows[:, None] * input_stride_row
        + cols[None, :] * input_stride_col,
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    global_scale = tl.load(global_scale_ptr)
    inv_global = tl.extra.libdevice.rcp_rn(tl.maximum(global_scale, EPS))

    block_amax = tl.maximum(tl.max(tl.abs(tile), axis=1), EPS)
    block_scale_fp32 = block_amax / 6.0  # FP4 E2M1 max
    block_scale_fp8 = (block_scale_fp32 * inv_global).to(tl.float8e4nv)

    tl.store(
        block_scale_fp8_ptr
        + rows * scale_stride_row
        + pid_g * scale_stride_col,
        block_scale_fp8, mask=row_mask,
    )

    effective = block_scale_fp8.to(tl.float32) * global_scale
    inv_eff = tl.extra.libdevice.rcp_rn(tl.maximum(effective, EPS))
    return tile * inv_eff[:, None], row_mask


@triton.jit
def _quantize_nvfp4_kernel_native(
    input_ptr, packed_ptr, block_scale_fp8_ptr, global_scale_ptr,
    input_stride_row, input_stride_col,
    packed_stride_row, packed_stride_col,
    scale_stride_row, scale_stride_col,
    n_out_rows, n_groups,
    EPS: tl.constexpr,
    BLOCKSIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Blackwell-native quant: ``cvt.rn.satfinite.e2m1x2.f32`` packs 2 fp32 → 1 byte."""
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)
    normalized, row_mask = _quantize_nvfp4_prologue(
        input_ptr, block_scale_fp8_ptr, global_scale_ptr,
        input_stride_row, input_stride_col,
        scale_stride_row, scale_stride_col,
        n_out_rows, EPS, BLOCKSIZE, BLOCK_M, pid_m, pid_g,
    )

    # Even col → low nibble, odd → high (cuBLAS FP4 convention).
    paired = normalized.reshape(BLOCK_M, BLOCKSIZE // 2, 2)
    even, odd = tl.split(paired)

    packed = tl.inline_asm_elementwise(
        asm=(
            "{\n"
            "  .reg .b8 tmp;\n"
            "  cvt.rn.satfinite.e2m1x2.f32 tmp, $1, $2;\n"
            "  cvt.u32.u8 $0, tmp;\n"
            "}"
        ),
        constraints="=r,r,r",
        args=[odd, even],
        dtype=tl.uint8,
        is_pure=True,
        pack=1,
    )

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    packed_cols = pid_g * (BLOCKSIZE // 2) + tl.arange(0, BLOCKSIZE // 2)
    tl.store(
        packed_ptr
        + rows[:, None] * packed_stride_row
        + packed_cols[None, :] * packed_stride_col,
        packed,
        mask=row_mask[:, None],
    )


@triton.jit
def _quantize_nvfp4_kernel_lut(
    input_ptr, packed_ptr, block_scale_fp8_ptr, global_scale_ptr,
    input_stride_row, input_stride_col,
    packed_stride_row, packed_stride_col,
    scale_stride_row, scale_stride_col,
    n_out_rows, n_groups,
    EPS: tl.constexpr,
    BLOCKSIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Software quant: threshold RNE + IEEE signbit, bit-matches the native path."""
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)
    normalized, row_mask = _quantize_nvfp4_prologue(
        input_ptr, block_scale_fp8_ptr, global_scale_ptr,
        input_stride_row, input_stride_col,
        scale_stride_row, scale_stride_col,
        n_out_rows, EPS, BLOCKSIZE, BLOCK_M, pid_m, pid_g,
    )

    # Positive-code selection on abs(x) via RNE midpoint thresholds.
    # `<=` at a boundary means the lower neighbour is the even-mantissa
    # one and wins the tie; `<` means the upper neighbour is even.
    ax = tl.abs(normalized)
    pos = tl.zeros([BLOCK_M, BLOCKSIZE], dtype=tl.int32)
    pos = tl.where(ax > 0.25, 1, pos)
    pos = tl.where(ax >= 0.75, 2, pos)
    pos = tl.where(ax > 1.25, 3, pos)
    pos = tl.where(ax >= 1.75, 4, pos)
    pos = tl.where(ax > 2.5, 5, pos)
    pos = tl.where(ax >= 3.5, 6, pos)
    pos = tl.where(ax > 5.0, 7, pos)

    # IEEE sign bit: bit 31 of the fp32 bit pattern (preserves -0).
    fp32_bits = normalized.to(tl.int32, bitcast=True)
    sign = (fp32_bits >> 31) & 1
    best_idx = pos | (sign << 3)                             # (BLOCK_M, BLOCKSIZE)

    # Pack: even col → low nibble, odd → high.
    idx_pairs = best_idx.reshape(BLOCK_M, BLOCKSIZE // 2, 2)
    pair_weights = tl.arange(0, 2) * 15 + 1                 # [1, 16]
    packed = tl.sum(
        idx_pairs * pair_weights[None, None, :], axis=2,
    ).to(tl.uint8)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    packed_cols = pid_g * (BLOCKSIZE // 2) + tl.arange(0, BLOCKSIZE // 2)
    tl.store(
        packed_ptr
        + rows[:, None] * packed_stride_row
        + packed_cols[None, :] * packed_stride_col,
        packed,
        mask=row_mask[:, None],
    )


def _quantize_nvfp4_triton(
    tensor: Tensor, eps: float = 1e-12,
) -> Tuple[Tensor, Tensor, Tensor]:
    assert tensor.ndim == 2
    out_f, in_f = tensor.shape
    assert in_f % NVFP4_BLOCKSIZE == 0
    assert in_f % 2 == 0
    n_groups = in_f // NVFP4_BLOCKSIZE

    # Global scale computed on the host via torch (single reduction).
    tensor_amax = tensor.abs().amax().float().clamp(min=eps)
    global_scale = tensor_amax / (FP8_E4M3_MAX * FP4_E2M1_MAX)

    packed = torch.empty(
        (out_f, in_f // 2), dtype=torch.uint8, device=tensor.device,
    )
    block_scale_fp8 = torch.empty(
        (out_f, n_groups), dtype=torch.float8_e4m3fn, device=tensor.device,
    )

    BLOCK_M = 32
    grid = (triton.cdiv(out_f, BLOCK_M), n_groups)
    if _supports_native_fp4():
        _quantize_nvfp4_kernel_native[grid](
            tensor, packed, block_scale_fp8, global_scale,
            tensor.stride(0), tensor.stride(1),
            packed.stride(0), packed.stride(1),
            block_scale_fp8.stride(0), block_scale_fp8.stride(1),
            out_f, n_groups,
            EPS=eps,
            BLOCKSIZE=NVFP4_BLOCKSIZE,
            BLOCK_M=BLOCK_M,
        )
    else:
        _quantize_nvfp4_kernel_lut[grid](
            tensor, packed, block_scale_fp8, global_scale,
            tensor.stride(0), tensor.stride(1),
            packed.stride(0), packed.stride(1),
            block_scale_fp8.stride(0), block_scale_fp8.stride(1),
            out_f, n_groups,
            EPS=eps,
            BLOCKSIZE=NVFP4_BLOCKSIZE,
            BLOCK_M=BLOCK_M,
        )
    return packed, block_scale_fp8, global_scale


# ----------------------------------------------------------------------
# Tensor subclass
# ----------------------------------------------------------------------

class NVFP4QWeight(QWeightBase):
    """NVFP4 blockwise QAT weight, compressed-tensors W4A16 format."""

    @staticmethod
    @torch._dynamo.disable
    def __new__(
        cls, weight_packed: Tensor, block_scale_fp8: Tensor,
        global_scale: Tensor, in_f: int, outer_dtype: torch.dtype,
    ):
        out_f = weight_packed.shape[0]
        return Tensor._make_wrapper_subclass(
            cls, (out_f, in_f), dtype=outer_dtype, device=weight_packed.device,
        )

    @torch._dynamo.disable
    def __init__(
        self, weight_packed: Tensor, block_scale_fp8: Tensor,
        global_scale: Tensor, in_f: int, outer_dtype: torch.dtype,
    ):
        assert weight_packed.dtype is torch.uint8 and weight_packed.ndim == 2
        assert block_scale_fp8.dtype is torch.float8_e4m3fn
        assert block_scale_fp8.ndim == 2
        assert global_scale.dtype is torch.float32
        self.weight_packed = weight_packed
        self.block_scale_fp8 = block_scale_fp8
        self.global_scale = global_scale
        self.in_f = in_f
        self.outer_dtype = outer_dtype

    def __tensor_flatten__(self):
        return ["weight_packed", "block_scale_fp8", "global_scale"], [
            self.in_f, self.outer_dtype,
        ]

    @classmethod
    def __tensor_unflatten__(cls, tensors, attrs, outer_size=None, outer_stride=None):
        return cls(
            tensors["weight_packed"], tensors["block_scale_fp8"],
            tensors["global_scale"], *attrs,
        )

    @classmethod
    def from_float(cls, tensor: Tensor, **_):
        packed, scale, gs = quantize_nvfp4_blockwise(tensor.detach())
        out = cls(packed, scale, gs, tensor.shape[1], tensor.dtype)
        out.requires_grad_(tensor.requires_grad)
        return out

    @classmethod
    def can_quantize(cls, tensor: Tensor, **_) -> bool:
        return (
            tensor.dim() == 2
            and tensor.shape[1] % NVFP4_BLOCKSIZE == 0
            and tensor.shape[1] % 2 == 0
        )

    def dequantize(self, dtype=None) -> Tensor:
        if dtype is None:
            dtype = self.outer_dtype
        if dtype in _TL_DTYPE_MAP:
            return _dequant_nvfp4_triton(
                self.weight_packed, self.block_scale_fp8, self.global_scale,
                self.in_f, dtype,
            )
        return dequantize_nvfp4_blockwise(
            self.weight_packed, self.block_scale_fp8, self.global_scale,
            self.in_f, dtype,
        )

    @classmethod
    def canonical_key_suffixes(cls) -> Tuple[str, ...]:
        return ("weight_packed", "weight_scale", "weight_global_scale")

    def to_plain_state_dict(self) -> dict:
        # compressed-tensors stores `weight_global_scale` as the reciprocal
        # of our internal convention (tensor_amax / (FP8_MAX * FP4_MAX)).
        return {
            "weight_packed": self.weight_packed,
            "weight_scale": self.block_scale_fp8,
            "weight_global_scale": (1.0 / self.global_scale).reshape(1),
        }

    @classmethod
    def from_plain_state_dict(cls, plain: dict, reference=None) -> "NVFP4QWeight":
        # outer_dtype isn't carried in the canonical state_dict; must come
        # from a reference (the pre-hook in `quantize_linears` supplies one).
        if reference is None:
            raise ValueError(
                "NVFP4QWeight.from_plain_state_dict requires a reference "
                "instance to resolve outer_dtype; the canonical state_dict "
                "carries no outer-dtype field."
            )
        weight_packed = plain["weight_packed"]
        block_scale = plain["weight_scale"]
        # Undo the reciprocal applied on save.
        global_scale = (1.0 / plain["weight_global_scale"]).reshape(())
        in_f = weight_packed.shape[1] * 2
        return cls(weight_packed, block_scale, global_scale, in_f, reference.outer_dtype)

    @classmethod
    def build_hf_quantization_config(cls, skip_patterns=(), **_) -> dict:
        return {
            "quant_method": "compressed-tensors",
            "format": "nvfp4-pack-quantized",
            "config_groups": {
                "group_0": {
                    "weights": {
                        "num_bits": 4,
                        "type": "float",
                        "symmetric": True,
                        "strategy": "tensor_group",
                        "group_size": NVFP4_BLOCKSIZE,
                    },
                    "targets": ["Linear"],
                },
            },
            "ignore": list(skip_patterns),
        }

    def __repr__(self):
        return (
            f"NVFP4QWeight(shape={tuple(self.shape)}, dtype={self.dtype}, "
            f"device={self.device}, requires_grad={self.requires_grad})"
        )


# ----------------------------------------------------------------------
# Autograd: F.linear through the dequantize path
# ----------------------------------------------------------------------

class _NVFP4WeightOnlyLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: NVFP4QWeight, bias: Optional[Tensor] = None):
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

@NVFP4QWeight.implements_torch_function(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    return _NVFP4WeightOnlyLinear.apply(*args, **kwargs)


@NVFP4QWeight.implements([aten.detach.default, aten.clone.default])
def _(func, types, args, kwargs):
    out = NVFP4QWeight(
        func(args[0].weight_packed, *args[1:], **kwargs),
        func(args[0].block_scale_fp8, *args[1:], **kwargs),
        func(args[0].global_scale, *args[1:], **kwargs),
        args[0].in_f, args[0].outer_dtype,
    )
    return return_and_correct_aliasing(func, args, kwargs, out)


@NVFP4QWeight.implements(aten._to_copy.default)
def _(func, types, args, kwargs):
    device = kwargs.get("device", None)
    dtype = kwargs.get("dtype", None)
    outer = dtype if dtype is not None else args[0].outer_dtype
    out = NVFP4QWeight(
        args[0].weight_packed.to(device=device),
        args[0].block_scale_fp8.to(device=device),
        args[0].global_scale.to(device=device),
        args[0].in_f, outer,
    )
    return return_and_correct_aliasing(func, args, kwargs, out)


@NVFP4QWeight.implements(aten.copy_.default)
def _(func, types, args, kwargs):
    dst, src = args[0], args[1]
    if isinstance(dst, NVFP4QWeight) and isinstance(src, NVFP4QWeight):
        dst.weight_packed.copy_(src.weight_packed)
        dst.block_scale_fp8.copy_(src.block_scale_fp8)
        dst.global_scale.copy_(src.global_scale)
    elif isinstance(dst, NVFP4QWeight):
        packed, scale, gs = _quantize_nvfp4_triton(src)
        dst.weight_packed.copy_(packed)
        dst.block_scale_fp8.copy_(scale)
        dst.global_scale.copy_(gs)
    else:
        dst.copy_(src.dequantize())
    return dst


@NVFP4QWeight.implements(aten.zeros_like.default)
def _(func, types, args, kwargs):
    dtype = kwargs.get("dtype", args[0].dtype)
    device = kwargs.get("device", args[0].device)
    return torch.zeros(args[0].shape, dtype=dtype, device=device)
