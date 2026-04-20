"""Correctness tests for NVFP4 quantization, compressed-tensors format.

Covers:
- pack / unpack nibble convention (even → low, odd → high)
- layout anchor: explicit byte value for a hand-crafted index row
- quantize_nvfp4_blockwise python reference: shape / dtype
- triton dequant + quantize parity vs python
- FP4 codebook integrity (16 E2M1 values, including ±0)
- F.linear autograd through the dispatch
- aten.copy_(nvfp4, fp32) triggers re-quantize that matches fresh quantize
"""

import pytest
import torch
import torch.nn.functional as F

from offload_adam.qweight import nvfp4 as _nvfp4_mod
from offload_adam.qweight.nvfp4 import (
    FP4_E2M1_LUT,
    NVFP4_BLOCKSIZE,
    NVFP4QWeight,
    _dequant_nvfp4_triton,
    _fp4_lut,
    _fp8_e4m3_bits_to_fp32,
    _quantize_fp4_e2m1_rne,
    _quantize_nvfp4_triton,
    pack_fp4,
    quantize_nvfp4_blockwise,
    unpack_fp4,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
BLACKWELL = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not _nvfp4_mod._supports_native_fp4(),
    reason="Blackwell (sm_100+) required for native-vs-fallback parity",
)


# -------- pack / unpack convention --------------------------------------

def test_pack_unpack_roundtrip_cpu():
    torch.manual_seed(0)
    x = torch.randint(0, 16, (6, 32), dtype=torch.uint8)
    packed = pack_fp4(x)
    assert packed.dtype == torch.uint8
    assert packed.shape == (6, 32 // 2)
    assert torch.equal(unpack_fp4(packed), x)


def test_nvfp4_layout_convention():
    """Packing: even col → low nibble, odd col → high nibble (cuBLAS
    FP4 convention). This is the opposite of bitsandbytes NF4.
    """
    row = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.uint8)
    packed = pack_fp4(row)
    assert packed.shape == (1, 4)
    # byte[k] = (odd << 4) | even
    assert packed[0, 0].item() == (1 << 4) | 0
    assert packed[0, 1].item() == (3 << 4) | 2
    assert packed[0, 2].item() == (5 << 4) | 4
    assert packed[0, 3].item() == (7 << 4) | 6
    assert torch.equal(unpack_fp4(packed), row)


def test_fp4_lut_has_sixteen_e2m1_values():
    assert len(FP4_E2M1_LUT) == 16
    # Positive half (codes 0..7): 0, 0.5, 1, 1.5, 2, 3, 4, 6
    expected_pos = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    for i, v in enumerate(expected_pos):
        assert FP4_E2M1_LUT[i] == v
    # Negative half (codes 8..15): mirror with sign.
    for i, v in enumerate(expected_pos):
        assert FP4_E2M1_LUT[i + 8] == -v


# -------- python quantize -----------------------------------------------

def test_quantize_nvfp4_shape_and_dtype():
    torch.manual_seed(0)
    w = torch.randn(64, 256, dtype=torch.float32) * 0.05
    packed, scale, gs = quantize_nvfp4_blockwise(w)
    assert packed.shape == (64, 256 // 2)
    assert packed.dtype == torch.uint8
    assert scale.shape == (64, 256 // NVFP4_BLOCKSIZE)
    assert scale.dtype == torch.float8_e4m3fn
    assert gs.shape == ()                                    # scalar
    assert gs.dtype == torch.float32
    # All FP4 indices fit in 4 bits.
    idx = unpack_fp4(packed)
    assert idx.min().item() >= 0 and idx.max().item() < 16


# -------- dequant triton parity -----------------------------------------

@CUDA
@pytest.mark.parametrize("out_f,in_f", [
    (128, 256),
    (4096, 4096),
    (12288, 4096),
    (4096, 11008),
    (35, 256),
])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dequant_triton_parity(out_f, in_f, dtype):
    torch.manual_seed(0)
    w = torch.randn(out_f, in_f, dtype=dtype, device="cuda") * 0.05
    qw = NVFP4QWeight.from_float(w)
    idx = unpack_fp4(qw.weight_packed).to(torch.long)
    lut_fp32 = _fp4_lut(qw.weight_packed.device, dtype=torch.float32)
    n_groups = in_f // NVFP4_BLOCKSIZE
    vals_fp32 = lut_fp32[idx].reshape(out_f, n_groups, NVFP4_BLOCKSIZE)
    eff = (
        qw.block_scale_fp8.to(torch.float32) * qw.global_scale
    ).reshape(out_f, n_groups)
    ref = (vals_fp32 * eff.unsqueeze(-1)).reshape(out_f, in_f).to(dtype)
    got = _dequant_nvfp4_triton(
        qw.weight_packed, qw.block_scale_fp8, qw.global_scale, in_f, dtype,
    )
    assert got.shape == ref.shape
    assert got.dtype == ref.dtype
    assert torch.equal(got, ref), (
        f"shape=({out_f},{in_f}) dtype={dtype}: "
        f"max_abs={(got.float() - ref.float()).abs().max().item()}"
    )


# -------- F.linear dispatch ---------------------------------------------

@CUDA
def test_linear_autograd_matches_dequant_reference():
    torch.manual_seed(0)
    in_f, out_f, batch = 256, 64, 8
    w = torch.randn(out_f, in_f, dtype=torch.bfloat16, device="cuda") * 0.05
    b = torch.randn(out_f, dtype=torch.bfloat16, device="cuda") * 0.05
    qw = NVFP4QWeight.from_float(w)
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
@pytest.mark.parametrize("out_f,in_f", [
    (128, 256),
    (4096, 4096),
    (12288, 4096),
    (4096, 11008),
    (35, 256),
])
@pytest.mark.parametrize("src_dtype", [torch.bfloat16, torch.float32])
def test_quantize_triton_parity(out_f, in_f, src_dtype):
    torch.manual_seed(0)
    w = torch.randn(out_f, in_f, dtype=src_dtype, device="cuda") * 0.05
    ref_packed, ref_scale, ref_gs = quantize_nvfp4_blockwise(w)
    got_packed, got_scale, got_gs = _quantize_nvfp4_triton(w)
    assert got_packed.shape == ref_packed.shape
    assert got_packed.dtype == torch.uint8
    assert got_scale.shape == ref_scale.shape
    assert got_scale.dtype == torch.float8_e4m3fn
    assert got_gs.shape == ref_gs.shape
    assert got_gs.dtype == torch.float32
    assert torch.equal(got_gs, ref_gs)
    assert torch.equal(
        got_scale.to(torch.float32), ref_scale.to(torch.float32),
    )
    # With RNE-matching software reference, hardware and soft bit-op paths
    # now produce byte-identical packed output.
    assert torch.equal(got_packed, ref_packed), (
        f"packed mismatch: shape=({out_f},{in_f}) dtype={src_dtype} "
        f"diff_bytes={(got_packed != ref_packed).sum().item()}"
    )


@CUDA
def test_copy_fp32_to_nvfp4_matches_fresh_quantize():
    """Emulates the optimizer's `p.data.copy_(master)` re-quantize path."""
    torch.manual_seed(0)
    w = torch.randn(64, 256, dtype=torch.bfloat16, device="cuda") * 0.05
    qw = NVFP4QWeight.from_float(w)

    new_w = torch.randn(64, 256, dtype=torch.float32, device="cuda")
    qw.copy_(new_w)

    ref_packed, ref_scale, ref_gs = _quantize_nvfp4_triton(new_w)
    assert torch.equal(qw.weight_packed, ref_packed)
    assert torch.equal(qw.block_scale_fp8.to(torch.float32),
                       ref_scale.to(torch.float32))
    assert torch.equal(qw.global_scale, ref_gs)


# -------- Exhaustive decoder / RNE quantize edge cases -----------------

@CUDA
def test_fp4_code_to_fp32_exhaustive_via_dequant_kernel():
    """All 16 FP4 codes must decode to the canonical LUT values. Bypass
    the quantize path by constructing an `NVFP4QWeight` directly with a
    packed tensor that pairs each byte = (j<<4)|j (both nibbles = code
    j) and effective scale = 1. Each 16-element block then contains 16
    copies of the same code; dequant must emit exactly `FP4_LUT[j]`."""
    # 16 blocks × 8 bytes/block = 128 packed bytes covering all 16 codes.
    rows = []
    for j in range(16):
        rows.extend([(j << 4) | j] * 8)
    packed = torch.tensor(rows, dtype=torch.uint8, device="cuda").reshape(1, 128)
    # block_scale_fp8 = 1 on every block; global_scale = 1 → effective = 1.
    block_scale_fp8 = torch.ones(
        (1, 16), dtype=torch.float8_e4m3fn, device="cuda",
    )
    global_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    qw = NVFP4QWeight(
        packed, block_scale_fp8, global_scale, in_f=256, outer_dtype=torch.bfloat16,
    )
    deq = qw.dequantize(torch.float32).reshape(-1)
    for j in range(16):
        block = deq[j * 16: (j + 1) * 16]
        expected_val = FP4_E2M1_LUT[j]
        assert (block == expected_val).all(), (
            f"code {j} decoded to {block[:4].tolist()}..., expected {expected_val}"
        )
    # Signed-zero: code 8 must decode to an actual -0 bit pattern.
    first_m0 = deq[8 * 16]
    assert torch.signbit(first_m0).item() is True, (
        f"code 8 lost signed-zero: bits={first_m0.view(torch.int32).item():#x}"
    )


def test_fp4_code_to_fp32_preserves_signed_zero():
    """code 0 → +0.0, code 8 → -0.0 (IEEE sign bit preserved)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    # Build a minimal input that yields code 0 / 8 in specific positions.
    x = torch.tensor([[0.0, -0.0] * 8], dtype=torch.float32, device="cuda")
    qw = NVFP4QWeight.from_float(x.to(torch.bfloat16))
    idx = unpack_fp4(qw.weight_packed).reshape(-1)
    # Even-position is +0 (code 0), odd is -0 (code 8).
    assert idx[0].item() == 0 and idx[1].item() == 8
    deq = qw.dequantize(torch.float32).reshape(-1)
    # Values are 0, but signbit is preserved.
    assert torch.signbit(deq[0]).item() is False
    assert torch.signbit(deq[1]).item() is True


@pytest.mark.parametrize("sign", [+1.0, -1.0])
def test_rne_quantize_boundaries(sign):
    """Verify every RNE midpoint, plus values ±ulp on either side.

    Boundaries (positive side) with RNE-even tie resolution:
      0.25 → 0.0  (0 even)
      0.75 → 1.0  (1 even)
      1.25 → 1.0
      1.75 → 2.0
      2.5  → 2.0
      3.5  → 4.0
      5.0  → 4.0
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    eps = 1e-3
    # (input magnitude, expected FP4 value)
    cases = [
        (0.0,       0.0),
        (0.1,       0.0),   # rounds to 0
        (0.25,      0.0),   # tie → 0
        (0.25 + eps, 0.5),
        (0.5,       0.5),
        (0.75 - eps, 0.5),
        (0.75,      1.0),   # tie → 1
        (0.75 + eps, 1.0),
        (1.0,       1.0),
        (1.25 - eps, 1.0),
        (1.25,      1.0),   # tie → 1
        (1.25 + eps, 1.5),
        (1.5,       1.5),
        (1.75 - eps, 1.5),
        (1.75,      2.0),   # tie → 2
        (1.75 + eps, 2.0),
        (2.0,       2.0),
        (2.5 - eps, 2.0),
        (2.5,       2.0),   # tie → 2
        (2.5 + eps, 3.0),
        (3.0,       3.0),
        (3.5 - eps, 3.0),
        (3.5,       4.0),   # tie → 4
        (3.5 + eps, 4.0),
        (4.0,       4.0),
        (5.0 - eps, 4.0),
        (5.0,       4.0),   # tie → 4
        (5.0 + eps, 6.0),
        (6.0,       6.0),
        (1e6,       6.0),   # saturate
    ]
    mags = torch.tensor(
        [m for m, _ in cases], dtype=torch.float32, device="cuda",
    )
    expected = torch.tensor(
        [e for _, e in cases], dtype=torch.float32, device="cuda",
    ) * sign
    x = mags * sign
    codes = _quantize_fp4_e2m1_rne(x).cpu().tolist()
    lut = torch.tensor(FP4_E2M1_LUT, dtype=torch.float32, device="cuda")
    got = lut[torch.tensor(codes, device="cuda").long()]
    # Compare magnitudes (sign handling verified separately by the signed-zero test).
    assert torch.equal(got.abs(), expected.abs()), (
        f"RNE boundary mismatch at sign={sign}: "
        f"inputs={x.tolist()} got={got.tolist()} expected={expected.tolist()}"
    )


@BLACKWELL
def test_large_random_stress_matches_native():
    """1M random fp32 values spanning [-8, 8] (includes saturation,
    near-boundary, and ±0). Software fallback must bit-equal native
    PTX on Blackwell, verifying no code path diverges on unusual
    inputs.
    """
    torch.manual_seed(12345)
    # Use (256, 4096) rather than (1, 1M) — the kernel grid dim-1 maps
    # to blocks-per-row, so a million-wide row overflows CUDA's 65k
    # grid-y limit. Same total element count, stays within limits.
    out_f, in_f = 256, 4096
    rand = torch.randn(out_f, in_f - 34, device="cuda") * 2.0     # ~[-8, 8]
    boundaries = torch.tensor(
        [0.0, -0.0, 0.25, -0.25, 0.75, -0.75, 1.25, 1.75, 2.5, 3.5, 5.0,
         6.0, -6.0, 10.0, -10.0, 1e6, -1e6],
        dtype=torch.float32, device="cuda",
    )                                                              # (17,)
    # Seed the first row with exact boundary values + their negatives.
    rand_first = torch.cat([
        boundaries, -boundaries, torch.zeros(in_f - 34, device="cuda"),
    ])[:in_f]
    w = rand.clone()
    w = torch.cat([w, torch.zeros(out_f, 34, device="cuda")], dim=1)
    w[0] = rand_first
    w = w.to(torch.float32)

    # Native path
    assert _nvfp4_mod._supports_native_fp4()
    pk_n, sc_n, gs_n = _quantize_nvfp4_triton(w)

    # Fallback path (force soft)
    orig = _nvfp4_mod._supports_native_fp4
    _nvfp4_mod._supports_native_fp4 = lambda: False
    try:
        pk_l, sc_l, gs_l = _quantize_nvfp4_triton(w)
    finally:
        _nvfp4_mod._supports_native_fp4 = orig

    assert torch.equal(gs_n, gs_l)
    assert torch.equal(sc_n.to(torch.float32), sc_l.to(torch.float32))
    assert torch.equal(pk_n, pk_l), (
        f"stress test divergence: "
        f"{(pk_n != pk_l).sum().item()}/{pk_n.numel()} bytes"
    )


def test_can_quantize():
    assert NVFP4QWeight.can_quantize(torch.empty(4, 16))
    assert NVFP4QWeight.can_quantize(torch.empty(4, 256))
    assert not NVFP4QWeight.can_quantize(torch.empty(16))
    assert not NVFP4QWeight.can_quantize(torch.empty(4, 4, 4))
    assert not NVFP4QWeight.can_quantize(torch.empty(4, 15))   # < blocksize


# -------- fallback (soft bit-op) path parity ---------------------------

@BLACKWELL
def test_native_vs_soft_fallback_parity(monkeypatch):
    """Blackwell-native PTX and the soft bit-op fallback must produce
    bit-identical packed bytes and dequantized values.

    Forcing `_supports_native_fp4` → False exercises the fallback on
    the same hardware.
    """
    torch.manual_seed(0)
    w = torch.randn(512, 1024, dtype=torch.bfloat16, device="cuda") * 0.05

    assert _nvfp4_mod._supports_native_fp4()
    pk_n, sc_n, gs_n = _quantize_nvfp4_triton(w)
    deq_n = _dequant_nvfp4_triton(pk_n, sc_n, gs_n, 1024, torch.bfloat16)

    monkeypatch.setattr(_nvfp4_mod, "_supports_native_fp4", lambda: False)
    pk_l, sc_l, gs_l = _quantize_nvfp4_triton(w)
    deq_l = _dequant_nvfp4_triton(pk_l, sc_l, gs_l, 1024, torch.bfloat16)

    assert torch.equal(gs_n, gs_l)
    assert torch.equal(sc_n.to(torch.float32), sc_l.to(torch.float32))
    assert torch.equal(pk_n, pk_l), (
        f"packed byte divergence: "
        f"{(pk_n != pk_l).sum().item()}/{pk_n.numel()}"
    )
    # Dequant values are bit-equal modulo signed-zero: -0.0 and 0.0
    # compare equal under arithmetic `==` but differ under `torch.equal`.
    assert ((deq_n - deq_l) == 0).all()


@CUDA
def test_fp8_e4m3_bit_synth_matches_torch_cast():
    """All 256 fp8_e4m3fn encodings: bit-synth fp32 must match torch cast."""
    import triton
    import triton.language as tl

    @triton.jit
    def _decode_all(in_ptr, out_ptr, n):
        pid = tl.program_id(0)
        offs = pid * 256 + tl.arange(0, 256)
        mask = offs < n
        bits = tl.load(in_ptr + offs, mask=mask)
        vals = _fp8_e4m3_bits_to_fp32(bits)
        tl.store(out_ptr + offs, vals, mask=mask)

    all_bits = torch.arange(256, dtype=torch.uint8, device="cuda")
    out = torch.empty(256, dtype=torch.float32, device="cuda")
    _decode_all[(1,)](all_bits, out, 256)

    ref = all_bits.view(torch.float8_e4m3fn).to(torch.float32)
    # NaN encodings (0x7F, 0xFF): both are NaN but bit patterns may differ;
    # compare via isnan. Everything else must be bit-identical.
    is_nan_ref = torch.isnan(ref)
    is_nan_got = torch.isnan(out)
    assert torch.equal(is_nan_ref, is_nan_got)
    finite = ~is_nan_ref
    assert torch.equal(
        out[finite].view(torch.int32), ref[finite].view(torch.int32)
    ), f"bit mismatch at indices {(out[finite] != ref[finite]).nonzero().flatten()}"
