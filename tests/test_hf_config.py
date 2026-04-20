"""Tests for `build_hf_quantization_config`, the `update_hf_config`
flag on ``quantize_linears``, and ``save_quantized_pretrained``.
"""

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from offload_adam.qweight import (
    Int4QWeight,
    Int8QWeight,
    NF4QWeight,
    NVFP4QWeight,
    quantize_linears,
    save_quantized_pretrained,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# -------- per-dtype config shape ---------------------------------------


def test_int4_config_is_gptq():
    cfg = Int4QWeight.build_hf_quantization_config(
        skip_patterns=("lm_head",),
        group_size=128,
    )
    assert cfg["quant_method"] == "gptq"
    assert cfg["bits"] == 4
    assert cfg["group_size"] == 128
    assert cfg["sym"] is True
    assert cfg["desc_act"] is False
    assert cfg["modules_to_not_convert"] == ["lm_head"]


def test_int8_config_is_compressed_tensors():
    cfg = Int8QWeight.build_hf_quantization_config(skip_patterns=("lm_head",))
    assert cfg["quant_method"] == "compressed-tensors"
    assert cfg["format"] == "pack-quantized"
    weights = cfg["config_groups"]["group_0"]["weights"]
    assert weights["num_bits"] == 8
    assert weights["type"] == "int"
    assert weights["strategy"] == "channel"
    assert weights["symmetric"] is True
    assert cfg["ignore"] == ["lm_head"]


def test_nf4_config_is_bitsandbytes():
    cfg = NF4QWeight.build_hf_quantization_config(skip_patterns=("lm_head",))
    assert cfg["quant_method"] == "bitsandbytes"
    assert cfg["load_in_4bit"] is True
    assert cfg["bnb_4bit_quant_type"] == "nf4"
    assert cfg["bnb_4bit_use_double_quant"] is False
    assert cfg["bnb_4bit_compute_dtype"] == "bfloat16"
    assert cfg["llm_int8_skip_modules"] == ["lm_head"]


def test_nf4_config_compute_dtype_override():
    cfg = NF4QWeight.build_hf_quantization_config(compute_dtype="float16")
    assert cfg["bnb_4bit_compute_dtype"] == "float16"


def test_nvfp4_config_is_compressed_tensors():
    cfg = NVFP4QWeight.build_hf_quantization_config(skip_patterns=("lm_head",))
    assert cfg["quant_method"] == "compressed-tensors"
    assert cfg["format"] == "nvfp4-pack-quantized"
    weights = cfg["config_groups"]["group_0"]["weights"]
    assert weights["num_bits"] == 4
    assert weights["type"] == "float"
    assert weights["group_size"] == 16
    assert cfg["ignore"] == ["lm_head"]


# -------- update_hf_config flag ----------------------------------------


class _ModelWithConfig(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(128, 128, bias=False)
        self.config = SimpleNamespace()


@CUDA
def test_update_hf_config_sets_attribute():
    m = _ModelWithConfig().to(torch.bfloat16).cuda()
    quantize_linears(
        m,
        Int4QWeight,
        group_size=128,
        skip_patterns=("lm_head",),
        update_hf_config=True,
    )
    assert hasattr(m.config, "quantization_config")
    assert m.config.quantization_config["quant_method"] == "gptq"
    assert m.config.quantization_config["modules_to_not_convert"] == ["lm_head"]


@CUDA
def test_update_hf_config_false_is_noop():
    m = _ModelWithConfig().to(torch.bfloat16).cuda()
    quantize_linears(m, Int4QWeight, group_size=128)
    assert not hasattr(m.config, "quantization_config")


@CUDA
def test_update_hf_config_skipped_when_no_config():
    # Plain nn.Module has no `.config`; flag is a no-op rather than an error.
    m = nn.Linear(128, 128, bias=False).to(torch.bfloat16).cuda()
    quantize_linears(m, Int4QWeight, group_size=128, update_hf_config=True)
    assert not hasattr(m, "config")


# -------- save_quantized_pretrained ------------------------------------


class _FakeHFModel(nn.Module):
    """Mimics just enough of a HF model for save_quantized_pretrained:
    a `save_pretrained(save_dir)` that writes a `config.json`."""

    def __init__(self, base_config):
        super().__init__()
        self.l1 = nn.Linear(128, 128, bias=False)
        self._base_config = dict(base_config)

    def save_pretrained(self, save_dir):
        from pathlib import Path

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "config.json").write_text(json.dumps(self._base_config))


@CUDA
def test_save_quantized_pretrained_patches_config(tmp_path):
    m = (
        _FakeHFModel({"model_type": "tiny", "hidden_size": 128})
        .to(
            torch.bfloat16,
        )
        .cuda()
    )
    quantize_linears(m, Int4QWeight, group_size=128)
    save_quantized_pretrained(
        m,
        tmp_path,
        Int4QWeight,
        skip_patterns=("lm_head",),
        group_size=128,
    )
    saved = json.loads((tmp_path / "config.json").read_text())
    # Original fields preserved.
    assert saved["model_type"] == "tiny"
    assert saved["hidden_size"] == 128
    # Patched field is the GPTQ config.
    qc = saved["quantization_config"]
    assert qc["quant_method"] == "gptq"
    assert qc["bits"] == 4
    assert qc["group_size"] == 128
    assert qc["modules_to_not_convert"] == ["lm_head"]
