"""Shared streaming loader for ``{Adam,OffloadAdam}.load_master_from_pretrained``.

Resolves a ``from_pretrained``-style identifier (single file / sharded
directory / HF Hub repo ID) to local safetensors paths and streams each
tensor once into the optimizer's fp32 master. Hub resolution uses
``huggingface_hub.snapshot_download`` as a soft dependency.
"""

import json
from pathlib import Path
from typing import List, Union

import torch

from .qweight.base import QWeightBase


@torch.no_grad()
def stream_master_from_pretrained(
    optim, pretrained_name_or_path, model, *, strict: bool,
) -> List[str]:
    """Stream lossless fp32 master tensors into ``optim.state`` for every
    param whose ``.data`` is a ``QWeightBase`` subclass. Peak memory per
    step ≈ one tensor (read buffer released after each copy).
    """
    from safetensors import safe_open

    paths = resolve_safetensors_paths(pretrained_name_or_path)
    name2param = {
        n: p for n, p in model.named_parameters() if p.requires_grad
    }
    # Non-quant params already got a lossless widen at optimizer init.
    quant_names = {
        n for n, p in name2param.items()
        if isinstance(p.data, QWeightBase)
    }
    # Adam inits state lazily; OffloadAdam eagerly. Route through the
    # initializer if it exists.
    init_state = getattr(optim, "_init_state_if_empty", None)
    matched = set()
    for path in paths:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            file_keys = set(f.keys())
            for key in file_keys & quant_names:
                p = name2param[key]
                state = optim.state[p]
                if init_state is not None:
                    init_state(p, state)
                if "master_params" not in state:
                    continue
                tensor = f.get_tensor(key)
                state["master_params"].copy_(tensor)
                if "master_filled" in state:
                    state["master_filled"] = True
                matched.add(key)
                del tensor
    missing = sorted(quant_names - matched)
    if strict and missing:
        raise ValueError(
            f"load_master_from_pretrained(strict=True): {len(missing)} "
            f"quant param(s) have no matching key in the checkpoint: "
            f"{missing}"
        )
    return missing


def resolve_safetensors_paths(
    pretrained_name_or_path: Union[str, Path],
) -> List[Path]:
    """Expand a ``from_pretrained`` identifier to local safetensors paths.

    Raises ``FileNotFoundError`` for a local path with no safetensors
    artefact, or ``ImportError`` if a HF Hub repo ID is given but
    ``huggingface_hub`` is unavailable.
    """
    candidate = Path(pretrained_name_or_path)
    if candidate.is_file():
        return [candidate]
    if candidate.is_dir():
        return _resolve_directory(candidate)
    # Not a local path — try HF Hub.
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            f"'{pretrained_name_or_path}' is not a local file or directory "
            "and `huggingface_hub` is not installed. Install it or pass a "
            "local path."
        ) from e
    local_dir = snapshot_download(
        repo_id=str(pretrained_name_or_path),
        allow_patterns=["*.safetensors", "*.safetensors.index.json"],
    )
    return _resolve_directory(Path(local_dir))


def _resolve_directory(directory: Path) -> List[Path]:
    index_file = directory / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            index = json.load(f)
        shards = sorted(set(index["weight_map"].values()))
        return [directory / s for s in shards]
    single = directory / "model.safetensors"
    if single.exists():
        return [single]
    # Fall back: any *.safetensors in the directory, alphabetical.
    found = sorted(directory.glob("*.safetensors"))
    if found:
        return found
    raise FileNotFoundError(
        f"No safetensors files under {directory} "
        f"(looked for model.safetensors, model.safetensors.index.json, "
        f"or any *.safetensors)."
    )
