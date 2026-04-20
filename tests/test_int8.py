"""Correctness tests for int8 per-channel symmetric quantization.

Covers:
- python reference quantize: shape / dtype / value range
- layout anchor: explicit int8 values + scale for a hand-crafted input
- triton dequant / quantize parity vs python
- F.linear autograd through the dispatch
- aten.copy_(int8, fp32) triggers re-quantize that matches fresh quantize
"""

import pytest
import torch
import torch.nn.functional as F

from offload_adam.qweight.int8 import (
    Int8QWeight,
    _dequant_int8_triton,
    _quantize_int8_triton,
    dequantize_int8_per_channel,
    quantize_int8_per_channel,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# -------- python quantize -----------------------------------------------


def test_quantize_int8_shape_and_dtype():
    torch.manual_seed(0)
    w = torch.randn(64, 256, dtype=torch.float32) * 0.05
    weight, scales = quantize_int8_per_channel(w)
    assert weight.shape == (64, 256)
    assert weight.dtype == torch.int8
    assert scales.shape == (64,)
    assert scales.dtype == w.dtype
    assert weight.min().item() >= -128 and weight.max().item() <= 127
    # Exactly one row should hit the +127 or -127 rail per construction.
    row_abs_max = weight.abs().amax(dim=1)
    assert (row_abs_max == 127).all()


def test_int8_layout_convention():
    """Per-row amax lands at int8 value ±127, scale = amax / 127.

    Regressing this silently would break checkpoint interop with
    compressed-tensors W8A16 loaders.
    """
    # Known row with amax at position 2.
    row = torch.tensor(
        [[0.0, 0.5, -2.0, 1.0]],
        dtype=torch.float32,
    )
    weight, scales = quantize_int8_per_channel(row)
    assert weight.shape == (1, 4)
    assert weight.dtype == torch.int8
    assert scales.shape == (1,)
    # scale = 2.0 / 127
    assert torch.isclose(scales, torch.tensor([2.0 / 127.0]))
    # -2.0 / (2/127) = -127 → int8 -127
    assert weight[0, 2].item() == -127
    # Round-half-to-even on (0.5 / (2/127)) = 31.75 → int8 32
    assert weight[0, 1].item() == round(0.5 / (2.0 / 127.0))


# -------- dequant triton parity -----------------------------------------


@CUDA
@pytest.mark.parametrize(
    "out_f,in_f",
    [
        (128, 256),
        (4096, 4096),
        (12288, 4096),
        (4096, 11008),
        (35, 256),  # BLOCK_M-misaligned rows
        (64, 37),  # BLOCK_N-misaligned cols
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dequant_triton_parity(out_f, in_f, dtype):
    torch.manual_seed(0)
    w = torch.randn(out_f, in_f, dtype=dtype, device="cuda") * 0.05
    # Pin scale dtype to the input so we actually cover the fp16 path;
    # `from_float` defaults to bf16 otherwise.
    qw = Int8QWeight.from_float(w, scale_dtype=dtype)
    ref = dequantize_int8_per_channel(qw.weight, qw.scales)
    got = _dequant_int8_triton(qw.weight, qw.scales)
    assert got.shape == ref.shape
    assert got.dtype == ref.dtype
    assert torch.equal(got, ref), (
        f"shape=({out_f},{in_f}) dtype={dtype}: "
        f"max_abs={(got.float() - ref.float()).abs().max().item()}"
    )


@CUDA
def test_dequantize_fp32_fallback():
    """dtype=torch.float32 forces the python fallback path."""
    torch.manual_seed(0)
    w = torch.randn(64, 256, dtype=torch.bfloat16, device="cuda") * 0.05
    qw = Int8QWeight.from_float(w)
    got = qw.dequantize(torch.float32)
    expected = qw.weight.to(torch.float32) * qw.scales.to(torch.float32).unsqueeze(1)
    assert got.dtype == torch.float32
    assert torch.equal(got, expected)


# -------- F.linear dispatch ---------------------------------------------


@CUDA
def test_linear_autograd_matches_dequant_reference():
    torch.manual_seed(0)
    in_f, out_f, batch = 256, 64, 8
    w = torch.randn(out_f, in_f, dtype=torch.bfloat16, device="cuda") * 0.05
    b = torch.randn(out_f, dtype=torch.bfloat16, device="cuda") * 0.05
    qw = Int8QWeight.from_float(w)
    qw.requires_grad_(True)
    b.requires_grad_(True)
    x = torch.randn(
        batch,
        in_f,
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
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
@pytest.mark.parametrize(
    "out_f,in_f",
    [
        (128, 256),
        (4096, 4096),
        (12288, 4096),
        (4096, 11008),
        (35, 256),
        (64, 37),
    ],
)
@pytest.mark.parametrize("src_dtype", [torch.bfloat16, torch.float32])
def test_quantize_triton_parity(out_f, in_f, src_dtype):
    torch.manual_seed(0)
    w = torch.randn(out_f, in_f, dtype=src_dtype, device="cuda") * 0.05
    ref_weight, ref_scales = quantize_int8_per_channel(w)
    got_weight, got_scales = _quantize_int8_triton(w, src_dtype)
    assert got_weight.shape == ref_weight.shape
    assert got_weight.dtype == torch.int8
    assert got_scales.shape == ref_scales.shape
    assert got_scales.dtype == src_dtype
    assert torch.equal(got_weight, ref_weight), (
        f"weight mismatch: shape=({out_f},{in_f}) dtype={src_dtype}"
    )
    assert torch.equal(got_scales, ref_scales), (
        f"scales mismatch: max_abs="
        f"{(got_scales.float() - ref_scales.float()).abs().max()}"
    )


@CUDA
def test_copy_fp32_to_int8_matches_fresh_quantize():
    """aten.copy_(int8, fp32) re-quantizes through the triton kernel."""
    torch.manual_seed(0)
    w = torch.randn(64, 256, dtype=torch.bfloat16, device="cuda") * 0.05
    qw = Int8QWeight.from_float(w)

    new_w = torch.randn(64, 256, dtype=torch.float32, device="cuda")
    qw.copy_(new_w)

    ref_weight, ref_scales = quantize_int8_per_channel(new_w)
    assert torch.equal(qw.weight, ref_weight)
    # Stored scales.dtype = bf16; fresh quantize returns fp32.
    assert torch.allclose(
        qw.scales.float(),
        ref_scales.float(),
        atol=1e-2,
        rtol=1e-2,
    )


def test_can_quantize():
    assert Int8QWeight.can_quantize(torch.empty(4, 4))
    assert Int8QWeight.can_quantize(torch.empty(33, 16))  # in_f % 4 == 0
    assert not Int8QWeight.can_quantize(torch.empty(4, 17))  # in_f not % 4
    assert not Int8QWeight.can_quantize(torch.empty(4))
    assert not Int8QWeight.can_quantize(torch.empty(4, 4, 4))
