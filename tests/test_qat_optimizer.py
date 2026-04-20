"""Optimizer-side QAT contract.

- Canonical flow is two lines: `quantize_linears(model, ...)` then
  `Adam(model.parameters(), mode='fp32_master')`. Optimizer init seeds
  `state[p]["master_params"]` from `p.data`; for a quant param the
  copy dispatches into the QWeight handler and dequantizes (lossy).
  `master_filled=False` flags that to introspection and the one-shot
  step-time warning.
- `optim.load_master_from_pretrained(path_or_repo, model)` streams
  tensors from the same bf16/fp16 checkpoint `from_pretrained`
  consumed and widens to fp32 as it fills master; peak host memory ≈
  one layer.
- Per-state `master_filled` rides along in `Optimizer.state_dict()` so
  resume does not re-warn on already-lossless masters.
"""

import json
import warnings
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from offload_adam import Adam, OffloadAdam
from offload_adam.qweight import (
    Int4QWeight,
    QuantizeReport,
    quantize_linears,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(128, 128, bias=False)

    def forward(self, x):
        return self.l1(x)


class _Two(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(128, 128, bias=False)
        self.l2 = nn.Linear(128, 128, bias=False)

    def forward(self, x):
        return self.l2(self.l1(x))


def _build(quantize: bool, cls=_Tiny):
    torch.manual_seed(0)
    m = cls().to(torch.bfloat16).cuda()
    if quantize:
        quantize_linears(m, Int4QWeight, group_size=128)
    return m


def _count_master_warnings(caught):
    return sum(1 for w in caught if "lossy dequant-seeded" in str(w.message))


def _save_safetensors(model, path: Path, keys=None):
    """Save {name: tensor} from a model's `named_parameters()` to a
    safetensors file. Optionally restrict to `keys`.
    """
    from safetensors.torch import save_file

    state = {
        n: p.detach().cpu().contiguous()
        for n, p in model.named_parameters()
        if keys is None or n in keys
    }
    save_file(state, str(path))


# -------- quantize_linears report ------------------------------------


@CUDA
def test_quantize_linears_returns_report():
    torch.manual_seed(0)
    m = _Two().to(torch.bfloat16).cuda()
    report = quantize_linears(
        m,
        Int4QWeight,
        group_size=128,
        skip_patterns=("l2",),
    )
    assert isinstance(report, QuantizeReport)
    assert report.quantized == ["l1"]
    assert report.excluded == ["l2"]
    assert report.incompatible == []


@CUDA
def test_quantize_linears_strict_raises_on_incompatible():
    torch.manual_seed(0)
    m = nn.Linear(100, 128, bias=False).to(torch.bfloat16).cuda()
    with pytest.raises(ValueError, match="strict=True"):
        quantize_linears(m, Int4QWeight, group_size=128, strict=True)


# -------- init seeds master (lossy for quant, lossless for non-quant) --


@CUDA
def test_adam_qat_init_seeds_dequant_master_with_flag():
    m = _build(quantize=True)
    opt = Adam(m.parameters(), mode="fp32_master", lr=1e-3)
    state = opt.state[m.l1.weight]
    opt._init_state_if_empty(m.l1.weight, state)
    # Master is the dequantized weight (lossy), not zero.
    assert state["master_filled"] is False
    expected = m.l1.weight.data.dequantize().to(torch.float32)
    assert torch.equal(state["master_params"], expected)


@CUDA
def test_adam_nonquant_init_fills_master_from_data():
    m = _build(quantize=False)
    opt = Adam(m.parameters(), mode="fp32_master", lr=1e-3)
    state = opt.state[m.l1.weight]
    opt._init_state_if_empty(m.l1.weight, state)
    assert torch.equal(
        state["master_params"],
        m.l1.weight.data.to(torch.float32),
    )
    assert "master_filled" not in state


@CUDA
def test_offload_adam_qat_alloc_marks_master_unfilled():
    m = _build(quantize=True)
    opt = OffloadAdam(m, mode="fp32_master", lr=1e-3)
    state = opt.state[m.l1.weight]
    assert state["master_filled"] is False
    assert state["master_params"].dtype == torch.float32


# -------- step-time warning -----------------------------------------


@CUDA
def test_adam_qat_warns_once_when_master_lossy():
    m = _build(quantize=True)
    opt = Adam(m.parameters(), mode="fp32_master", lr=1e-3)
    x = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")
    m(x).sum().backward()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        opt.step()
    assert _count_master_warnings(caught) == 1

    m.zero_grad(set_to_none=True)
    m(x).sum().backward()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        opt.step()
    assert _count_master_warnings(caught) == 0


@CUDA
def test_adam_qat_empty_step_does_not_burn_warning():
    m = _build(quantize=True)
    opt = Adam(m.parameters(), mode="fp32_master", lr=1e-3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        opt.step()
    assert _count_master_warnings(caught) == 0

    x = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")
    m(x).sum().backward()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        opt.step()
    assert _count_master_warnings(caught) == 1


@CUDA
def test_adam_nonquant_never_warns():
    m = _build(quantize=False)
    opt = Adam(m.parameters(), mode="fp32_master", lr=1e-3)
    x = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")
    m(x).sum().backward()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        opt.step()
    assert _count_master_warnings(caught) == 0


# -------- load_master_from_pretrained -------------------------------


@CUDA
def test_load_master_from_pretrained_single_file(tmp_path):
    """Save fp32 master from a pre-quantize model, quantize, construct
    optim (lossy), then refill losslessly from the saved safetensors."""
    torch.manual_seed(0)
    pre = _Tiny().to(torch.bfloat16).cuda()
    expected = pre.l1.weight.data.clone().to(torch.float32)

    path = tmp_path / "model.safetensors"
    _save_safetensors(pre, path)

    quantize_linears(pre, Int4QWeight, group_size=128)
    opt = Adam(pre.parameters(), mode="fp32_master", lr=1e-3)

    missing = opt.load_master_from_pretrained(path, pre)
    assert missing == []
    state = opt.state[pre.l1.weight]
    assert state["master_filled"] is True
    assert torch.equal(state["master_params"], expected)


@CUDA
def test_load_master_from_pretrained_skips_non_quant(tmp_path):
    """Non-quant params (LayerNorm, biases) are skipped — they already
    have a lossless master from optim init."""

    class _Mixed(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(128, 128, bias=False)
            self.ln = nn.LayerNorm(128)

        def forward(self, x):
            return self.ln(self.l1(x))

    torch.manual_seed(0)
    m = _Mixed().to(torch.bfloat16).cuda()
    path = tmp_path / "model.safetensors"
    _save_safetensors(m, path)
    ln_weight_pre = m.ln.weight.data.clone().to(torch.float32)

    quantize_linears(m, Int4QWeight, group_size=128)
    opt = Adam(m.parameters(), mode="fp32_master", lr=1e-3)

    missing = opt.load_master_from_pretrained(path, m)
    assert missing == []
    assert opt.state[m.l1.weight]["master_filled"] is True
    # LN weight was not touched by from_pretrained (non-quant). Its
    # master is only populated when `_init_state_if_empty` runs, which
    # happens at step time. Force init and verify it matches `p.data`.
    opt._init_state_if_empty(m.ln.weight, opt.state[m.ln.weight])
    assert torch.equal(
        opt.state[m.ln.weight]["master_params"],
        ln_weight_pre,
    )
    assert "master_filled" not in opt.state[m.ln.weight]


@CUDA
def test_load_master_from_pretrained_strict_raises_on_missing(tmp_path):
    """A quant param whose name isn't in the checkpoint raises when
    strict=True."""
    torch.manual_seed(0)
    m = _Two().to(torch.bfloat16).cuda()
    path = tmp_path / "model.safetensors"
    _save_safetensors(m, path, keys={"l1.weight"})  # drop l2

    quantize_linears(m, Int4QWeight, group_size=128)
    opt = Adam(m.parameters(), mode="fp32_master", lr=1e-3)
    with pytest.raises(ValueError, match="strict=True"):
        opt.load_master_from_pretrained(path, m, strict=True)
    # Without strict, returns the missing list.
    missing = opt.load_master_from_pretrained(path, m)
    assert missing == ["l2.weight"]


@CUDA
def test_load_master_from_pretrained_sharded_dir(tmp_path):
    """Resolve a sharded checkpoint via `model.safetensors.index.json`."""
    from safetensors.torch import save_file

    torch.manual_seed(0)
    m = _Two().to(torch.bfloat16).cuda()
    expected_l1 = m.l1.weight.data.clone().to(torch.float32)
    expected_l2 = m.l2.weight.data.clone().to(torch.float32)

    # Split the weights across two shards.
    shard_a = {"l1.weight": m.l1.weight.detach().cpu().contiguous()}
    shard_b = {"l2.weight": m.l2.weight.detach().cpu().contiguous()}
    save_file(shard_a, str(tmp_path / "model-00001-of-00002.safetensors"))
    save_file(shard_b, str(tmp_path / "model-00002-of-00002.safetensors"))
    index = {
        "metadata": {"total_size": 0},
        "weight_map": {
            "l1.weight": "model-00001-of-00002.safetensors",
            "l2.weight": "model-00002-of-00002.safetensors",
        },
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    quantize_linears(m, Int4QWeight, group_size=128)
    opt = Adam(m.parameters(), mode="fp32_master", lr=1e-3)
    missing = opt.load_master_from_pretrained(tmp_path, m, strict=True)
    assert missing == []
    assert torch.equal(
        opt.state[m.l1.weight]["master_params"],
        expected_l1,
    )
    assert torch.equal(
        opt.state[m.l2.weight]["master_params"],
        expected_l2,
    )


@CUDA
def test_load_master_from_pretrained_flips_master_filled_and_suppresses_warn(
    tmp_path,
):
    m = _build(quantize=False)
    path = tmp_path / "model.safetensors"
    _save_safetensors(m, path)

    quantize_linears(m, Int4QWeight, group_size=128)
    opt = Adam(m.parameters(), mode="fp32_master", lr=1e-3)
    opt.load_master_from_pretrained(path, m, strict=True)

    x = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")
    m(x).sum().backward()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        opt.step()
    assert _count_master_warnings(caught) == 0


@CUDA
def test_offload_adam_load_master_from_pretrained(tmp_path):
    torch.manual_seed(0)
    m = _Tiny().to(torch.bfloat16).cuda()
    expected = m.l1.weight.data.clone().to(torch.float32)
    path = tmp_path / "model.safetensors"
    _save_safetensors(m, path)

    quantize_linears(m, Int4QWeight, group_size=128)
    opt = OffloadAdam(m, mode="fp32_master", lr=1e-3)
    missing = opt.load_master_from_pretrained(path, m, strict=True)
    assert missing == []
    state = opt.state[m.l1.weight]
    assert state["master_filled"] is True
    assert torch.equal(state["master_params"].cpu(), expected.cpu())


# -------- advanced load_master_state_dict ----------------------------


@CUDA
def test_adam_load_master_state_dict_strict_rejects_typo():
    m = _build(quantize=True)
    opt = Adam(m.parameters(), mode="fp32_master", lr=1e-3)
    bogus_sd = {"l1.weigth": torch.zeros(128, 128)}  # typo
    with pytest.raises(ValueError, match="strict=True"):
        opt.load_master_state_dict(bogus_sd, m, strict=True)


@CUDA
def test_adam_load_master_state_dict_returns_unexpected():
    m = _build(quantize=True)
    opt = Adam(m.parameters(), mode="fp32_master", lr=1e-3)
    sd = {"l1.weight": torch.zeros(128, 128), "ghost.weight": torch.zeros(4)}
    (unexpected,) = opt.load_master_state_dict(sd, m)
    assert unexpected == ["ghost.weight"]


# -------- resume round-trip ------------------------------------------


@CUDA
def test_adam_qat_state_dict_round_trip_preserves_fp32_and_flag(tmp_path):
    torch.manual_seed(0)
    pre = _Tiny().to(torch.bfloat16).cuda()
    path = tmp_path / "model.safetensors"
    _save_safetensors(pre, path)

    m = _build(quantize=True)
    opt1 = Adam(m.parameters(), mode="fp32_master", lr=1e-3)
    opt1.load_master_from_pretrained(path, m, strict=True)
    x = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")
    m(x).sum().backward()
    opt1.step()
    saved_master = opt1.state[m.l1.weight]["master_params"].clone()
    sd = opt1.state_dict()

    m2 = _build(quantize=True)
    opt2 = Adam(m2.parameters(), mode="fp32_master", lr=1e-3)
    opt2.load_state_dict(sd)

    state = opt2.state[m2.l1.weight]
    assert state["master_filled"] is True
    assert state["master_params"].dtype == torch.float32
    assert torch.equal(state["master_params"], saved_master)

    m2.zero_grad(set_to_none=True)
    m2(x).sum().backward()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        opt2.step()
    assert _count_master_warnings(caught) == 0


@CUDA
def test_offload_adam_load_state_dict_raises():
    m = _build(quantize=True)
    opt = OffloadAdam(m, mode="fp32_master", lr=1e-3)
    sd = opt.state_dict()
    with pytest.raises(NotImplementedError, match="not supported"):
        opt.load_state_dict(sd)


# -------- key shape regression ---------------------------------------


@CUDA
def test_quantize_linears_top_level_linear_key_has_no_leading_dot():
    torch.manual_seed(0)
    m = nn.Linear(128, 128, bias=False).to(torch.bfloat16).cuda()
    quantize_linears(m, Int4QWeight, group_size=128)
    # named_parameters has "weight", not ".weight".
    assert "weight" in dict(m.named_parameters())
