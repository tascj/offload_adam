"""state_dict save/load round-trip tests for all quantized dtypes.

Contract:
- `model.state_dict()` returns only plain `torch.Tensor` values (no custom
  subclass). Compatible with safetensors and `torch.load(weights_only=True)`.
- `model.load_state_dict()` reconstructs the subclass from those plain
  tensors via the pre-hook attached by `quantize_linears`, and the
  round-tripped weight dequantizes to a bit-identical tensor.
"""

import io
import os
import tempfile

import pytest
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from offload_adam.qweight import (
    Int4QWeight,
    Int8QWeight,
    NF4QWeight,
    NVFP4QWeight,
    quantize_linears,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DTYPE_CASES = [
    pytest.param(Int4QWeight, {"group_size": 128}, id="int4"),
    pytest.param(Int8QWeight, {}, id="int8"),
    pytest.param(NF4QWeight, {"blocksize": 64}, id="nf4"),
    pytest.param(NVFP4QWeight, {}, id="nvfp4"),
]


class _Tiny(nn.Module):
    """Two-layer model — exercises multiple quant params and non-quant tensors."""

    def __init__(self, in_f=256, mid_f=256, out_f=128):
        super().__init__()
        self.l1 = nn.Linear(in_f, mid_f, bias=False)
        self.ln = nn.LayerNorm(mid_f)
        self.l2 = nn.Linear(mid_f, out_f, bias=True)


def _build_model(weight_cls, kw):
    torch.manual_seed(0)
    m = _Tiny().to(torch.bfloat16).cuda()
    quantize_linears(m, weight_cls, **kw)
    return m


# -------- state_dict contract ------------------------------------------

@CUDA
@pytest.mark.parametrize("weight_cls,kw", DTYPE_CASES)
def test_state_dict_returns_plain_tensors(weight_cls, kw):
    m = _build_model(weight_cls, kw)
    sd = m.state_dict()
    for k, v in sd.items():
        assert type(v) is torch.Tensor, (
            f"state_dict['{k}'] is {type(v).__name__}, expected plain Tensor"
        )


@CUDA
@pytest.mark.parametrize("weight_cls,kw", DTYPE_CASES)
def test_state_dict_keys_canonical(weight_cls, kw):
    m = _build_model(weight_cls, kw)
    sd = m.state_dict()
    suffixes = weight_cls.canonical_key_suffixes()
    for linear_name in ("l1", "l2"):
        for suf in suffixes:
            assert f"{linear_name}.{suf}" in sd, (
                f"missing quant key {linear_name}.{suf}"
            )
        # `weight` may legitimately appear (e.g. bnb-NF4 uses it as the
        # packed-uint8 key) — what must not leak is the QWeightBase
        # subclass itself; the plain-tensor contract is covered by
        # `test_state_dict_returns_plain_tensors`.
    # Non-quant parameters are untouched.
    assert "ln.weight" in sd and "ln.bias" in sd
    assert "l2.bias" in sd


# -------- safetensors round-trip ---------------------------------------

@CUDA
@pytest.mark.parametrize("weight_cls,kw", DTYPE_CASES)
def test_safetensors_round_trip(weight_cls, kw):
    m = _build_model(weight_cls, kw)
    sd = m.state_dict()
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        path = f.name
    try:
        save_file(sd, path)
        loaded = load_file(path)
        assert set(sd.keys()) == set(loaded.keys())
        for k in sd:
            assert torch.equal(sd[k].cpu(), loaded[k].cpu()), f"{k} mismatch"
    finally:
        os.unlink(path)


# -------- torch.load weights_only=True ---------------------------------

@CUDA
@pytest.mark.parametrize("weight_cls,kw", DTYPE_CASES)
def test_torch_save_weights_only_load(weight_cls, kw):
    """torch.save → torch.load(weights_only=True) — default since torch 2.6.

    Proves the state_dict carries no custom-class metadata that would
    require the user to register globals for safe deserialization.
    """
    m = _build_model(weight_cls, kw)
    sd = m.state_dict()
    buf = io.BytesIO()
    torch.save(sd, buf)
    buf.seek(0)
    loaded = torch.load(buf, weights_only=True)
    assert set(sd.keys()) == set(loaded.keys())
    for k in sd:
        assert torch.equal(sd[k].cpu(), loaded[k].cpu())


# -------- load_state_dict round-trip -----------------------------------

@CUDA
@pytest.mark.parametrize("weight_cls,kw", DTYPE_CASES)
def test_load_state_dict_round_trip(weight_cls, kw):
    src = _build_model(weight_cls, kw)
    sd = src.state_dict()

    dst = _Tiny().to(torch.bfloat16).cuda()
    quantize_linears(dst, weight_cls, **kw)
    result = dst.load_state_dict(sd)
    assert result.missing_keys == []
    assert result.unexpected_keys == []

    # Round-tripped weights dequantize to bit-identical tensors.
    for name in ("l1", "l2"):
        w_src = getattr(src, name).weight.dequantize()
        w_dst = getattr(dst, name).weight.dequantize()
        assert torch.equal(w_src, w_dst), f"{name} dequant mismatch"
    # Non-quant tensors round-trip too.
    assert torch.equal(src.ln.weight, dst.ln.weight)
    assert torch.equal(src.l2.bias, dst.l2.bias)


@CUDA
@pytest.mark.parametrize("weight_cls,kw", DTYPE_CASES)
def test_load_state_dict_partial_keys_reports_missing(weight_cls, kw):
    """Partial quant keys in state_dict should be reported as missing —
    the pre-hook must not silently drop incomplete entries."""
    src = _build_model(weight_cls, kw)
    sd = src.state_dict()

    suffixes = weight_cls.canonical_key_suffixes()
    # Drop one canonical key from l1.
    dropped = f"l1.{suffixes[0]}"
    del sd[dropped]

    dst = _Tiny().to(torch.bfloat16).cuda()
    quantize_linears(dst, weight_cls, **kw)
    result = dst.load_state_dict(sd, strict=False)
    remaining = tuple(f"l1.{s}" for s in suffixes if s != suffixes[0])
    assert dropped in result.missing_keys or any(
        k in result.missing_keys for k in remaining + (dropped,)
    ), (
        f"expected missing to include {dropped} or one of {remaining}; "
        f"got {result.missing_keys}"
    )


# -------- subclass-level API (unit check, no module) -------------------

@CUDA
@pytest.mark.parametrize("weight_cls,kw", DTYPE_CASES)
def test_to_from_plain_state_dict(weight_cls, kw):
    torch.manual_seed(0)
    w = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda") * 0.05
    qw = weight_cls.from_float(w, **kw)

    plain = qw.to_plain_state_dict()
    assert set(plain.keys()) == set(weight_cls.canonical_key_suffixes())
    assert all(type(v) is torch.Tensor for v in plain.values())

    restored = weight_cls.from_plain_state_dict(plain, reference=qw)
    assert isinstance(restored, weight_cls)
    assert torch.equal(qw.dequantize(), restored.dequantize())


# -------- plain-weight fallback (re-quantize via aten.copy_) -----------

@CUDA
@pytest.mark.parametrize("weight_cls,kw", DTYPE_CASES)
def test_load_plain_weight_fallback(weight_cls, kw):
    """Loading a plain (unquantized) checkpoint into a quantized model
    must fall through to the `aten.copy_` dispatch, which re-quantizes
    the float weight into the subclass's packed storage.

    For NF4 this is a regression guard: its canonical suffix set
    contains a bare `"weight"` key (bnb convention), which previously
    collided with the plain-tensor key and caused the pre-hook to
    silently drop the weight while reporting the 3 aux keys missing.
    """
    torch.manual_seed(0)
    src = _Tiny().to(torch.bfloat16).cuda()
    plain_sd = src.state_dict()
    assert plain_sd["l1.weight"].dtype is torch.bfloat16

    dst = _Tiny().to(torch.bfloat16).cuda()
    quantize_linears(dst, weight_cls, **kw)
    result = dst.load_state_dict(plain_sd, strict=False)

    # The plain `weight` key must drive re-quantize, not be reported missing.
    assert "l1.weight" not in result.missing_keys
    assert "l2.weight" not in result.missing_keys
    # Re-quantize was applied: dequant matches a fresh triton-path
    # quantize of the same source (from_float uses the python reference,
    # so emulate the aten.copy_ path with an explicit copy_).
    for name in ("l1", "l2"):
        orig = getattr(src, name).weight.data
        ref = weight_cls.from_float(orig, **kw)
        ref.copy_(orig)
        got = getattr(dst, name).weight
        assert torch.equal(got.dequantize(), ref.dequantize()), (
            f"{name} dequant mismatch"
        )
