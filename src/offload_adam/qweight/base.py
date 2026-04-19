"""Minimal base class for quantized weight tensor subclasses used in QAT.

Provides a wrapper-subclass Tensor that dispatches a small set of ATen ops
and torch functions to per-subclass handler tables. Subclasses register
handlers via the `@cls.implements(...)` / `@cls.implements_torch_function(...)`
decorators.
"""

import torch
from torch import Tensor


class QWeightBase(Tensor):
    """Base wrapper-subclass for quantized weights.

    Subclasses must:
      * Override `__new__` to declare outer (shape, dtype, device).
      * Override `__init__` to stash their payload tensors (e.g. int_data, scale).
      * Implement `__tensor_flatten__` / `__tensor_unflatten__`.
      * Register dispatch handlers with `@cls.implements(aten_op)` and
        `@cls.implements_torch_function(torch_fn)`.
    """

    _ATEN_OP_TABLE: dict
    _TORCH_FN_TABLE: dict

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._ATEN_OP_TABLE = {}
        cls._TORCH_FN_TABLE = {}

    @classmethod
    def implements(cls, aten_ops):
        """Register an ATen op handler on this subclass."""
        if not isinstance(aten_ops, (list, tuple)):
            aten_ops = [aten_ops]

        def decorator(fn):
            for op in aten_ops:
                cls._ATEN_OP_TABLE[op] = fn
            return fn

        return decorator

    @classmethod
    def implements_torch_function(cls, torch_fns):
        """Register a torch function handler on this subclass."""
        if not isinstance(torch_fns, (list, tuple)):
            torch_fns = [torch_fns]

        def decorator(fn):
            for f in torch_fns:
                cls._TORCH_FN_TABLE[f] = fn
            return fn

        return decorator

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if func in cls._ATEN_OP_TABLE:
            return cls._ATEN_OP_TABLE[func](func, types, args, kwargs)
        raise NotImplementedError(
            f"{cls.__name__} does not support {func}. "
            f"Register a handler via @{cls.__name__}.implements(...)."
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in cls._TORCH_FN_TABLE:
            return cls._TORCH_FN_TABLE[func](func, types, args, kwargs)
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    def dequantize(self, dtype=None) -> Tensor:
        """Return a plain Tensor reconstruction at the requested dtype."""
        raise NotImplementedError

    @classmethod
    def can_quantize(cls, tensor: Tensor, **kwargs) -> bool:
        """Advisory hook for `quantize_linears` — return False to skip a
        weight whose shape or dtype this subclass cannot handle. Default
        is permissive; subclasses with shape constraints should override.
        """
        return True

    @classmethod
    def canonical_key_suffixes(cls) -> tuple:
        """Plain-tensor key suffixes (no prefix) that `to_plain_state_dict`
        emits and `from_plain_state_dict` consumes. Drives the state_dict
        hook's save/load round-trip and constrains checkpoint format."""
        raise NotImplementedError

    def to_plain_state_dict(self) -> dict:
        """Expand this subclass into plain `torch.Tensor`s keyed by
        `canonical_key_suffixes()`. Auxiliary tensors (e.g. symmetric
        qzeros for GPTQ) are synthesized here."""
        raise NotImplementedError

    @classmethod
    def from_plain_state_dict(cls, plain: dict, reference=None) -> "QWeightBase":
        """Reconstruct from the dict produced by `to_plain_state_dict()`.

        `reference`, if provided, is the existing subclass instance
        already attached to the module (set up by `quantize_linears`).
        Subclasses that carry non-tensor state not derivable from the
        plain tensors (e.g. `outer_dtype` for nf4/nvfp4) pull that info
        from `reference`."""
        raise NotImplementedError

    @classmethod
    def build_hf_quantization_config(
        cls, skip_patterns=(), **weight_kwargs,
    ) -> dict:
        """HF-style ``quantization_config`` dict for this subclass.

        Placed in ``config.json`` so vllm / HF transformers route the
        state_dict through the matching loader (gptq / bitsandbytes /
        compressed-tensors). ``skip_patterns`` map to each method's
        skip field (``modules_to_not_convert`` / ``ignore`` / ...).
        ``weight_kwargs`` are the same ones passed to ``from_float``."""
        raise NotImplementedError
