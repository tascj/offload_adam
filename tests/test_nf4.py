"""Correctness tests for NF4 quantization, bitsandbytes storage format.

Covers:
- pack / unpack nibble convention (even → high, odd → low)
- layout anchor: explicit byte value for a hand-crafted index row
- quantize_nf4_blockwise python reference shape / range
- triton dequant + quantize parity vs python
- F.linear autograd through the dispatch
- aten.copy_(nf4, fp32) triggers re-quantize that matches fresh quantize
"""

import pytest
import torch
import torch.nn.functional as F

from offload_adam.qweight.nf4 import (
    NF4_LUT,
    NF4QWeight,
    _dequant_nf4_triton,
    _nf4_lut,
    _quantize_nf4_triton,
    pack_nf4,
    quantize_nf4_blockwise,
    unpack_nf4,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# -------- pack / unpack convention --------------------------------------

def test_pack_unpack_roundtrip_cpu():
    torch.manual_seed(0)
    x = torch.randint(0, 16, (6, 32), dtype=torch.uint8)
    packed = pack_nf4(x)
    assert packed.dtype == torch.uint8
    assert packed.shape == (6, 32 // 2)
    assert torch.equal(unpack_nf4(packed), x)


def test_nf4_layout_convention():
    """Packing: position 2k (even) → high nibble, 2k+1 (odd) → low nibble.

    Regressing this silently would break bitsandbytes checkpoint interop.
    """
    # Single row: indices 0..7 across 8 columns → 4 packed bytes.
    row = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.uint8)
    packed = pack_nf4(row)
    assert packed.shape == (1, 4)
    # byte[k] = (2k << 4) | (2k+1)
    assert packed[0, 0].item() == (0 << 4) | 1
    assert packed[0, 1].item() == (2 << 4) | 3
    assert packed[0, 2].item() == (4 << 4) | 5
    assert packed[0, 3].item() == (6 << 4) | 7
    assert torch.equal(unpack_nf4(packed), row)


def test_nf4_lut_has_sixteen_sorted_values():
    assert len(NF4_LUT) == 16
    vals = list(NF4_LUT)
    assert vals == sorted(vals)
    assert vals[0] == -1.0 and vals[-1] == 1.0
    assert 0.0 in vals                                     # zero is a code point


# -------- python quantize -----------------------------------------------

def test_quantize_nf4_shape_and_dtype():
    torch.manual_seed(0)
    w = torch.randn(64, 256, dtype=torch.float32) * 0.05
    packed, absmax = quantize_nf4_blockwise(w, blocksize=64)
    assert packed.shape == (64, 256 // 2)
    assert packed.dtype == torch.uint8
    assert absmax.shape == (64 * 4,)                        # out_f * n_groups
    assert absmax.dtype == torch.float32
    # Indices stored fit in 4 bits.
    idx = unpack_nf4(packed)
    assert idx.min().item() >= 0 and idx.max().item() < 16


# -------- dequant triton parity -----------------------------------------

@CUDA
@pytest.mark.parametrize("out_f,in_f,blocksize", [
    (128, 256, 64),
    (4096, 4096, 64),
    (12288, 4096, 64),
    (4096, 11008, 64),
    (35, 256, 64),       # BLOCK_M-misaligned rows
    (64, 128, 32),       # smaller blocksize
])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dequant_triton_parity(out_f, in_f, blocksize, dtype):
    torch.manual_seed(0)
    w = torch.randn(out_f, in_f, dtype=dtype, device="cuda") * 0.05
    qw = NF4QWeight.from_float(w, blocksize=blocksize)
    # fp32-compute reference: gather from LUT in fp32, multiply fp32
    # absmax, cast to output dtype at the end (matches the triton kernel).
    idx = unpack_nf4(qw.weight_packed).to(torch.long)
    lut_fp32 = _nf4_lut(qw.weight_packed.device, dtype=torch.float32)
    n_groups = in_f // blocksize
    vals_fp32 = lut_fp32[idx].reshape(out_f, n_groups, blocksize)
    am_fp32 = qw.absmax.reshape(out_f, n_groups)
    ref = (vals_fp32 * am_fp32.unsqueeze(-1)).reshape(out_f, in_f).to(dtype)
    got = _dequant_nf4_triton(
        qw.weight_packed, qw.absmax, in_f, blocksize, dtype,
    )
    assert got.shape == ref.shape
    assert got.dtype == ref.dtype
    assert torch.equal(got, ref), (
        f"shape=({out_f},{in_f}) bs={blocksize} dtype={dtype}: "
        f"max_abs={(got.float() - ref.float()).abs().max().item()}"
    )


# -------- F.linear dispatch ---------------------------------------------

@CUDA
def test_linear_autograd_matches_dequant_reference():
    torch.manual_seed(0)
    in_f, out_f, batch = 256, 64, 8
    w = torch.randn(out_f, in_f, dtype=torch.bfloat16, device="cuda") * 0.05
    b = torch.randn(out_f, dtype=torch.bfloat16, device="cuda") * 0.05
    qw = NF4QWeight.from_float(w, blocksize=64)
    qw.requires_grad_(True)
    b.requires_grad_(True)
    x = torch.randn(
        batch, in_f, dtype=torch.bfloat16, device="cuda", requires_grad=True,
    )

    y = F.linear(x, qw, b)
    y.sum().backward()

    x_ref = x.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    w_ref = qw.dequantize().detach().clone().requires_grad_(True)
    y_ref = F.linear(x_ref, w_ref, b_ref)
    y_ref.sum().backward()

    assert torch.allclose(y, y_ref, atol=1e-2, rtol=1e-2)
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-2, rtol=1e-2)
    assert torch.allclose(b.grad, b_ref.grad, atol=1e-2, rtol=1e-2)
    assert torch.allclose(qw.grad, w_ref.grad, atol=1e-2, rtol=1e-2)
    assert type(qw.grad) is torch.Tensor


# -------- triton quantize parity ----------------------------------------

@CUDA
@pytest.mark.parametrize("out_f,in_f,blocksize", [
    (128, 256, 64),
    (4096, 4096, 64),
    (12288, 4096, 64),
    (4096, 11008, 64),
    (35, 256, 64),
    (64, 128, 32),
])
@pytest.mark.parametrize("src_dtype", [torch.bfloat16, torch.float32])
def test_quantize_triton_parity(out_f, in_f, blocksize, src_dtype):
    torch.manual_seed(0)
    w = torch.randn(out_f, in_f, dtype=src_dtype, device="cuda") * 0.05
    ref_packed, ref_absmax = quantize_nf4_blockwise(w, blocksize=blocksize)
    got_packed, got_absmax = _quantize_nf4_triton(w, blocksize)
    assert got_packed.shape == ref_packed.shape
    assert got_packed.dtype == torch.uint8
    assert got_absmax.shape == ref_absmax.shape
    assert got_absmax.dtype == torch.float32
    assert torch.equal(got_absmax, ref_absmax), (
        f"absmax mismatch: max_abs="
        f"{(got_absmax - ref_absmax).abs().max()}"
    )
    # Rare tied-distance cases: a value exactly at the midpoint between
    # two LUT codes can resolve to either side depending on fp precision.
    # Allow a tiny mismatch budget (<1 ulp of the full tensor).
    diff = (got_packed.to(torch.int16) - ref_packed.to(torch.int16)) != 0
    diff_ratio = diff.sum().item() / diff.numel()
    assert diff_ratio < 1e-5, (
        f"packed mismatch: shape=({out_f},{in_f}) bs={blocksize} dtype={src_dtype} "
        f"diff_bytes={diff.sum().item()}/{diff.numel()} ({diff_ratio:.2e})"
    )


@CUDA
def test_copy_fp32_to_nf4_matches_fresh_quantize():
    """Emulates the optimizer's `p.data.copy_(master)` re-quantize path."""
    torch.manual_seed(0)
    w = torch.randn(64, 256, dtype=torch.bfloat16, device="cuda") * 0.05
    qw = NF4QWeight.from_float(w, blocksize=64)

    new_w = torch.randn(64, 256, dtype=torch.float32, device="cuda")
    qw.copy_(new_w)

    ref_packed, ref_absmax = _quantize_nf4_triton(new_w, 64)
    assert torch.equal(qw.weight_packed, ref_packed)
    assert torch.equal(qw.absmax, ref_absmax)


def test_can_quantize():
    assert NF4QWeight.can_quantize(torch.empty(4, 64))
    assert NF4QWeight.can_quantize(torch.empty(4, 128))
    assert not NF4QWeight.can_quantize(torch.empty(64))
    assert not NF4QWeight.can_quantize(torch.empty(4, 4, 4))
    assert not NF4QWeight.can_quantize(torch.empty(4, 63))  # not multiple of blocksize
