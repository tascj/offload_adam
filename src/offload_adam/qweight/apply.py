"""Replace eligible ``nn.Linear`` weights with ``QWeightBase`` subclass
instances, and attach state_dict hooks so ``model.state_dict()`` emits
plain tensors (safetensors- / ``weights_only=True``-friendly) and
``load_state_dict`` reconstructs the subclass on the way back in.
"""

import json
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

from .base import QWeightBase


class QuantizeReport(NamedTuple):
    """Summary of a `quantize_linears` pass.

    Fields are lists of module names (same convention as
    `model.named_modules()`):

    - ``quantized``: layers whose weight was replaced by a
      ``weight_cls`` subclass instance.
    - ``incompatible``: layers rejected by ``weight_cls.can_quantize``.
      Each entry is ``(name, reason)``.
    - ``excluded``: layers whose name matched one of ``skip_patterns``.
    """

    quantized: List[str]
    incompatible: List[Tuple[str, str]]
    excluded: List[str]


def _register_qweight_hooks(module: nn.Module, weight_cls: Type[QWeightBase]) -> None:
    """Attach save/load hooks that round-trip the subclass weight through
    plain tensors, keyed by ``weight_cls.canonical_key_suffixes()``."""

    def save_hook(mod, destination, prefix, local_metadata):
        key = prefix + "weight"
        w = destination.get(key)
        if not isinstance(w, weight_cls):
            return
        del destination[key]
        for suf, t in w.to_plain_state_dict().items():
            destination[prefix + suf] = t

    def load_pre_hook(
        state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        suffixes = weight_cls.canonical_key_suffixes()
        plain = {}
        for suf in suffixes:
            key = prefix + suf
            if key in state_dict:
                plain[suf] = state_dict.pop(key)
        # Bare "weight" at non-uint8 dtype = plain checkpoint; let aten.copy_ re-quantize.
        if "weight" in plain and plain["weight"].dtype is not torch.uint8:
            state_dict[prefix + "weight"] = plain.pop("weight")
        if not plain:
            return
        if len(plain) != len(suffixes):
            missing = [prefix + s for s in suffixes if s not in plain]
            missing_keys.extend(missing)
            return
        reference = (
            module.weight.data if isinstance(module.weight.data, weight_cls) else None
        )
        state_dict[prefix + "weight"] = weight_cls.from_plain_state_dict(
            plain, reference=reference,
        )

    module._register_state_dict_hook(save_hook)
    module._register_load_state_dict_pre_hook(load_pre_hook)


@torch.no_grad()
def quantize_linears(
    model: nn.Module,
    weight_cls: Type[QWeightBase],
    skip_patterns: Tuple[str, ...] = (),
    device: Optional[str] = None,
    strict: bool = False,
    update_hf_config: bool = False,
    **weight_kwargs,
) -> QuantizeReport:
    """Replace eligible Linear weights with `weight_cls` instances in place.

    Args:
        model: Module to scan for `nn.Linear` children.
        weight_cls: A `QWeightBase` subclass with `from_float(tensor,
            **weight_kwargs)` and `can_quantize(tensor, **weight_kwargs)`.
        skip_patterns: Substring patterns matched against module names —
            layers whose name contains any pattern are listed under
            ``QuantizeReport.excluded`` without being quantized (e.g.
            ``("lm_head",)`` for language models).
        device: If set, move weight tensors to this device before
            quantization.
        strict: If True, raise ``ValueError`` when any layer is rejected
            by ``weight_cls.can_quantize`` (shape / dtype mismatch).
            Excluded layers (matched ``skip_patterns``) are always OK
            since they are an explicit user choice.
        update_hf_config: If True and ``model`` has a ``config``
            attribute (HuggingFace ``PretrainedConfig``), set
            ``model.config.quantization_config`` so a later
            ``model.save_pretrained`` writes the right routing hint
            into ``config.json``.
        **weight_kwargs: Passed through to ``weight_cls.from_float`` and
            ``weight_cls.can_quantize`` (e.g. ``group_size=128``).

    Returns:
        ``QuantizeReport`` listing quantized / incompatible / excluded
        layers.

    Raises:
        ValueError: If ``strict=True`` and one or more layers would be
            quantized but were rejected by ``can_quantize``.
    """
    quantized: List[str] = []
    incompatible: List[Tuple[str, str]] = []
    excluded: List[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(p in name for p in skip_patterns):
            excluded.append(name)
            continue
        if not weight_cls.can_quantize(module.weight, **weight_kwargs):
            w = module.weight
            incompatible.append(
                (name, f"shape={tuple(w.shape)} dtype={w.dtype}")
            )
            continue
        w = module.weight
        src = w.data.to(device) if device is not None else w.data
        new_weight = weight_cls.from_float(src, **weight_kwargs)
        module.weight = nn.Parameter(new_weight, requires_grad=w.requires_grad)
        _register_qweight_hooks(module, weight_cls)
        quantized.append(name)
    if strict and incompatible:
        reasons = "; ".join(f"{n} ({r})" for n, r in incompatible)
        raise ValueError(
            f"quantize_linears(strict=True) rejected "
            f"{len(incompatible)} layer(s): {reasons}"
        )
    if update_hf_config and hasattr(model, "config"):
        model.config.quantization_config = weight_cls.build_hf_quantization_config(
            skip_patterns=skip_patterns, **weight_kwargs,
        )
    return QuantizeReport(
        quantized=quantized, incompatible=incompatible, excluded=excluded,
    )


def save_quantized_pretrained(
    model: nn.Module,
    save_directory: Union[str, Path],
    weight_cls: Type[QWeightBase],
    skip_patterns: Tuple[str, ...] = (),
    **weight_kwargs,
) -> None:
    """Save a quantized HF model and patch ``config.json`` with the
    matching ``quantization_config``.

    Calls ``model.save_pretrained(save_directory)``, then merges
    ``weight_cls.build_hf_quantization_config(skip_patterns=..., **weight_kwargs)``
    into the saved ``config.json``. Use this instead of a plain
    ``save_pretrained`` when you have not passed ``update_hf_config=True``
    to ``quantize_linears``."""
    save_dir = Path(save_directory)
    model.save_pretrained(save_dir)
    cfg_path = save_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["quantization_config"] = weight_cls.build_hf_quantization_config(
        skip_patterns=skip_patterns, **weight_kwargs,
    )
    cfg_path.write_text(json.dumps(cfg, indent=2))
