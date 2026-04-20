import warnings

import torch
from torch.optim.optimizer import Optimizer

from ._pretrained import stream_master_from_pretrained
from .kernels import (
    adam_step_fp31_master,
    adam_step_fp32_master,
    adam_step_fp32_master_custom_rounding,
    adam_step_stochastic_rounding,
)
from .qweight.base import QWeightBase


class Adam(Optimizer):
    configs = {
        "stochastic_rounding": {
            "step": adam_step_stochastic_rounding,
            "states": {
                "exp_avg": torch.bfloat16,
                "exp_avg_sq": torch.bfloat16,
            },
        },
        "fp32_master": {
            "step": adam_step_fp32_master,
            "states": {
                "exp_avg": torch.bfloat16,
                "exp_avg_sq": torch.bfloat16,
                "master_params": torch.float32,
            },
        },
        "fp31_master": {
            "step": adam_step_fp31_master,
            "states": {
                "exp_avg": torch.bfloat16,
                "exp_avg_sq": torch.bfloat16,
                "rounding_error": torch.int16,
            },
        },
        "fp32_master_custom_rounding": {
            "step": adam_step_fp32_master_custom_rounding,
            "states": {
                "exp_avg": torch.bfloat16,
                "exp_avg_sq": torch.bfloat16,
                "rounding_error": torch.int16,
            },
        },
    }

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        mode="stochastic_rounding",
        decoupled_weight_decay=False,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Adam, self).__init__(params, defaults)

        assert mode in self.configs, f"Invalid mode: {mode}"
        self.mode = mode
        self.config = self.configs[self.mode]
        self.step_fn = self.config["step"]
        self.decoupled_weight_decay = decoupled_weight_decay

        quant_count = sum(
            1
            for g in self.param_groups
            for p in g["params"]
            if isinstance(p.data, QWeightBase)
        )
        if quant_count > 0 and mode != "fp32_master":
            raise ValueError(
                f"Adam found {quant_count} quantized param(s) but "
                f"mode={mode!r}; QAT requires mode='fp32_master'."
            )
        # One-shot warning flag, not persisted. `True` when there are no
        # quant params so `.step()` skips the per-step scan. The per-state
        # `master_filled` bool in `self.state[p]` is what survives
        # `state_dict()` / `load_state_dict()` round-trips.
        self._quant_master_warned = quant_count == 0

    def _init_state_if_empty(self, p, state):
        if len(state) != 0:
            return
        state["step"] = 0
        for name, dtype in self.config["states"].items():
            state[name] = torch.zeros_like(p, dtype=dtype)
        if "master_params" in state:
            # Quant param: `copy_` dispatches to dequantize — lossy seed.
            # Plain bf16/fp16: lossless widen to fp32.
            # `master_filled=False` flags the lossy case for
            # `load_master_from_pretrained`; persists via state_dict.
            state["master_params"].copy_(p.data)
            if isinstance(p.data, QWeightBase):
                state["master_filled"] = False

    @torch.no_grad()
    def step(self, closure=None):
        if not self._quant_master_warned:
            self._maybe_warn_lossy_master()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Sparse gradient is not supported")

                state = self.state[p]
                self._init_state_if_empty(p, state)
                state["step"] += 1

                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                step = state["step"]

                is_quant = isinstance(p.data, QWeightBase)
                # Quant: kernel's bf16 master write-back into params_ptr
                # would smash packed storage. Redirect to grad (dummy sink
                # of matching shape/dtype) and re-quantize via dispatch below.
                param_view = grad if is_quant else p.data

                device_states = {n: state[n] for n in self.config["states"]}
                device_states["grad"] = grad
                self.step_fn(
                    param_view,
                    device_states,
                    lr,
                    weight_decay,
                    beta1,
                    beta2,
                    eps,
                    step,
                    decoupled_weight_decay=self.decoupled_weight_decay,
                )
                if is_quant:
                    p.data.copy_(state["master_params"])
                    # Grad held the bf16 master write-back; drop so a
                    # missed zero_grad can't leak into next backward.
                    p.grad = None

        return loss

    def _maybe_warn_lossy_master(self):
        """One-shot FYI warning when a quant param is about to step with a
        lossy dequant-seeded master. No state mutation."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if not isinstance(p.data, QWeightBase):
                    continue
                state = self.state.get(p, {})
                # `True` → explicit lossless load ran; else lossy.
                if state.get("master_filled") is True:
                    continue
                warnings.warn(
                    "Adam: quantized params running with a lossy "
                    "dequant-seeded fp32 master. For higher precision, "
                    "call `optim.load_master_from_pretrained("
                    "pretrained_name_or_path, model)` with the same "
                    "bf16/fp16 checkpoint `from_pretrained` used before "
                    "`quantize_linears` ran.",
                    stacklevel=3,
                )
                self._quant_master_warned = True
                return

    @torch.no_grad()
    def load_master_state_dict(self, state_dict, model, strict=False):
        """Overwrite `master_params` buffers from an in-memory
        ``{name: tensor}`` mapping.

        Advanced / test-time entry point — the canonical way to upgrade
        the lossy dequant-seeded master to a lossless source is
        ``load_master_from_pretrained``, which streams from disk.

        Source tensors may be any float dtype (bf16 / fp16 / fp32);
        they are copied into the optimizer's fp32 ``master_params`` —
        narrower dtypes widen losslessly.

        Args:
            state_dict: ``{param_name: tensor}`` mapping. Names are
                resolved via ``model.named_parameters()``.
            model: Used to resolve param names.
            strict: If True, raise ``ValueError`` on any name in
                ``state_dict`` that doesn't match a trainable parameter.

        Returns:
            ``(unexpected_keys,)`` tuple — list of names in
            ``state_dict`` that no param matched.
        """
        if "master_params" not in self.config["states"]:
            raise RuntimeError(
                "load_master_state_dict requires mode='fp32_master'; "
                f"current mode is {self.mode!r}."
            )
        name2param = {n: p for n, p in model.named_parameters() if p.requires_grad}
        unexpected = []
        for name, tensor in state_dict.items():
            p = name2param.get(name)
            if p is None:
                unexpected.append(name)
                continue
            state = self.state[p]
            self._init_state_if_empty(p, state)
            self._copy_master(p, tensor)
            if "master_filled" in state:
                state["master_filled"] = True
        if strict and unexpected:
            raise ValueError(
                "load_master_state_dict(strict=True): no param matches "
                f"{len(unexpected)} key(s): {unexpected}"
            )
        return (unexpected,)

    def _copy_master(self, p, tensor):
        """Copy a full-shape source tensor into ``p``'s master buffer."""
        self.state[p]["master_params"].copy_(tensor)

    @torch.no_grad()
    def load_master_from_pretrained(
        self,
        pretrained_name_or_path,
        model,
        *,
        strict=False,
    ):
        """Refill the fp32 master from a ``from_pretrained``-style
        checkpoint (single file, sharded directory, or HF Hub repo ID).

        Reads each tensor once from the on-disk bf16/fp16 checkpoint,
        widens to fp32 as it copies into ``state[p]["master_params"]``
        (lossless), and releases the read buffer before moving on —
        per-layer peak memory stays bounded. Only acts on params whose
        ``p.data`` is a ``QWeightBase`` subclass; non-quant params
        already got the same widen at optimizer init.

        Args:
            pretrained_name_or_path: Same shape
                ``AutoModelForCausalLM.from_pretrained`` accepts — a
                local file, a local directory (single or sharded), or a
                HuggingFace Hub repo ID (resolved via
                ``huggingface_hub``).
            model: Used to resolve key → ``Parameter``.
            strict: If True, raise ``ValueError`` when one or more quant
                params have no matching key across the provided files.

        Returns:
            Sorted list of quant param names that had no match across
            the provided files. Empty list means every quant master was
            refilled losslessly.
        """
        if "master_params" not in self.config["states"]:
            raise RuntimeError(
                "load_master_from_pretrained requires mode='fp32_master'; "
                f"current mode is {self.mode!r}."
            )
        return stream_master_from_pretrained(
            self,
            pretrained_name_or_path,
            model,
            strict=strict,
        )

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        """Preserve fp32 ``master_params`` across resume.

        Default ``Optimizer.load_state_dict`` casts state tensors to
        ``param.dtype`` (bf16 for QAT), demoting our fp32 master. We
        stash the fp32 tensors before super and rewrite them back.
        """
        saved_master = {}
        for pid, pstate in state_dict.get("state", {}).items():
            if isinstance(pstate, dict):
                t = pstate.get("master_params")
                if isinstance(t, torch.Tensor) and t.dtype == torch.float32:
                    saved_master[pid] = t.detach()
        super().load_state_dict(state_dict)
        if not saved_master:
            return
        params = [p for g in self.param_groups for p in g["params"]]
        for pid, fp32_t in saved_master.items():
            p = params[pid]
            st = self.state.get(p)
            if st is not None and "master_params" in st:
                st["master_params"] = fp32_t.to(device=p.device)
