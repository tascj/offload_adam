"""Correctness tests for int4 quantization, raw GPTQ layout.

Covers:
- pack / unpack bit-packing convention (8 int4 LSB-first per int32)
- layout anchor: explicit byte/int32 values for a hand-crafted input
- quantize_int4_groupwise python reference shape / range
- triton dequant + quantize parity vs python, including bf16 rounding
- F.linear autograd through the dispatch
- aten.copy_(int4, fp32) triggers re-quantize that matches fresh quantize
"""

import pytest
import torch
import torch.nn.functional as F

from offload_adam.qweight.int4 import (
    Int4QWeight,
    _dequant_int4_triton,
    _quantize_pack_int4_triton,
    pack_int4_gptq,
    quantize_int4_groupwise,
    unpack_int4_gptq,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _python_dequant(qweight, scales, group_size):
    """Python reference matching the triton fast path (compute at scales.dtype)."""
    in_f = qweight.shape[0] * 8
    out_f = qweight.shape[1]
    n_groups = scales.shape[0]
    return (
        unpack_int4_gptq(qweight)
        .reshape(out_f, n_groups, group_size)
        .to(scales.dtype)
        * scales.T.unsqueeze(-1)
    ).reshape(out_f, in_f)


# -------- pack / unpack convention --------------------------------------

def test_pack_unpack_roundtrip_cpu():
    torch.manual_seed(0)
    x = torch.randint(-8, 8, (6, 32), dtype=torch.int8)
    packed = pack_int4_gptq(x)
    assert packed.dtype == torch.int32
    # shape: (in_f // 8, out_f) = (4, 6)
    assert packed.shape == (32 // 8, 6)
    assert torch.equal(unpack_int4_gptq(packed), x)


def test_int4_layout_convention():
    """Packing order: value at in-position `i` of out-column `n` goes to the
    nibble at bit offset `(i % 8) * 4` of `packed[i // 8, n]`. Each value
    is stored as unsigned `v + 8` so the mid-point of int4 is zero point 8.

    This is the GPTQ symmetric convention; inference loaders (vllm's
    GPTQ Marlin) depend on it. Regressing this silently would break
    checkpoint interop.
    """
    # Single out column, 8 in positions: values -8, -7, ..., -1
    row = torch.tensor([[-8, -7, -6, -5, -4, -3, -2, -1]], dtype=torch.int8)
    packed = pack_int4_gptq(row)
    assert packed.dtype == torch.int32
    assert packed.shape == (1, 1)  # (in_f//8, out_f)
    # Uint stream: 0, 1, 2, 3, 4, 5, 6, 7 at shifts 0, 4, 8, 12, 16, 20, 24, 28.
    expected = sum(i << (i * 4) for i in range(8))
    assert packed[0, 0].item() == expected
    assert torch.equal(unpack_int4_gptq(packed), row)


# -------- python quantize -----------------------------------------------

def test_quantize_int4_groupwise_shape_and_dtype():
    torch.manual_seed(0)
    w = torch.randn(64, 256, dtype=torch.float32) * 0.05
    qweight, scales = quantize_int4_groupwise(w, group_size=128)
    assert qweight.shape == (256 // 8, 64)
    assert qweight.dtype == torch.int32
    assert scales.shape == (2, 64)
    assert scales.dtype == w.dtype
    # Unpacked values are in [-8, 7]
    unpacked = unpack_int4_gptq(qweight)
    assert unpacked.min().item() >= -8 and unpacked.max().item() <= 7


# -------- dequant triton parity -----------------------------------------

@CUDA
@pytest.mark.parametrize("out_f,in_f,group_size", [
    (128, 256, 128),
    (4096, 4096, 128),
    (12288, 4096, 128),
    (4096, 11008, 128),
    (35, 256, 128),     # BLOCK_M-misaligned rows
])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dequant_triton_parity(out_f, in_f, group_size, dtype):
    torch.manual_seed(0)
    w = torch.randn(out_f, in_f, dtype=dtype, device="cuda") * 0.05
    # Pin scale dtype to the input so we actually cover the fp16 path;
    # `from_float` defaults to bf16 otherwise.
    qw = Int4QWeight.from_float(w, group_size=group_size, scale_dtype=dtype)
    ref = _python_dequant(qw.qweight, qw.scales, group_size)
    got = _dequant_int4_triton(qw.qweight, qw.scales, group_size)
    assert got.shape == ref.shape
    assert got.dtype == ref.dtype
    assert torch.equal(got, ref), (
        f"shape=({out_f},{in_f}) g={group_size} dtype={dtype}: "
        f"max_abs={(got.float()-ref.float()).abs().max().item()}"
    )


@CUDA
def test_dequantize_fp32_fallback():
    """dtype=torch.float32 forces the python fallback path."""
    torch.manual_seed(0)
    w = torch.randn(64, 256, dtype=torch.bfloat16, device="cuda") * 0.05
    qw = Int4QWeight.from_float(w, group_size=128)
    got = qw.dequantize(torch.float32)
    expected = (
        unpack_int4_gptq(qw.qweight)
        .reshape(64, 2, 128)
        .to(torch.float32)
        * qw.scales.T.to(torch.float32).unsqueeze(-1)
    ).reshape(64, 256)
    assert got.dtype == torch.float32
    assert torch.equal(got, expected)


# -------- F.linear dispatch ---------------------------------------------

@CUDA
def test_linear_autograd_matches_dequant_reference():
    torch.manual_seed(0)
    in_f, out_f, batch = 256, 64, 8
    w = torch.randn(out_f, in_f, dtype=torch.bfloat16, device="cuda") * 0.05
    b = torch.randn(out_f, dtype=torch.bfloat16, device="cuda") * 0.05
    qw = Int4QWeight.from_float(w, group_size=128)
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
@pytest.mark.parametrize("out_f,in_f,group_size", [
    (128, 256, 128),
    (4096, 4096, 128),
    (12288, 4096, 128),
    (4096, 11008, 128),
    (35, 256, 128),
])
@pytest.mark.parametrize("src_dtype", [torch.bfloat16, torch.float32])
def test_quantize_triton_parity(out_f, in_f, group_size, src_dtype):
    torch.manual_seed(0)
    w = torch.randn(out_f, in_f, dtype=src_dtype, device="cuda") * 0.05
    ref_qweight, ref_scales = quantize_int4_groupwise(w, group_size)
    got_qweight, got_scales = _quantize_pack_int4_triton(w, group_size, src_dtype)
    assert got_qweight.shape == ref_qweight.shape
    assert got_qweight.dtype == torch.int32
    assert got_scales.shape == ref_scales.shape
    assert got_scales.dtype == src_dtype
    assert torch.equal(got_qweight, ref_qweight), (
        f"qweight mismatch: shape=({out_f},{in_f}) g={group_size} dtype={src_dtype}"
    )
    assert torch.equal(got_scales, ref_scales), (
        f"scales mismatch: max_abs="
        f"{(got_scales.float()-ref_scales.float()).abs().max()}"
    )


@CUDA
def test_copy_fp32_to_int4_matches_fresh_quantize():
    """aten.copy_(int4, fp32) re-quantizes through the triton kernel."""
    torch.manual_seed(0)
    w = torch.randn(64, 256, dtype=torch.bfloat16, device="cuda") * 0.05
    qw = Int4QWeight.from_float(w, group_size=128)

    new_w = torch.randn(64, 256, dtype=torch.float32, device="cuda")
    qw.copy_(new_w)

    ref_qweight, ref_scales = quantize_int4_groupwise(new_w, 128)
    assert torch.equal(qw.qweight, ref_qweight)
    # Stored scales.dtype = bf16; fresh quantize returns fp32.
    assert torch.allclose(
        qw.scales.float(), ref_scales.float(), atol=1e-2, rtol=1e-2,
    )
