"""BitFit-style bias-only tuning with an explicit, testable policy."""
from __future__ import annotations

import torch.nn as nn


_HEAD_TOKENS = ("head.", "heads.", "classifier.", "fc.")


def _is_head(name: str) -> bool:
    return name.startswith(_HEAD_TOKENS) or any(f".{token}" in name for token in _HEAD_TOKENS)


def set_bitfit_trainability(
    model: nn.Module,
    *,
    train_head: bool = True,
    bias_scope: str = "all",
) -> set[str]:
    """Freeze the model and enable the requested bias-only BitFit policy.

    ``all`` trains every bias. ``transformer`` excludes task-head biases unless
    ``train_head`` is enabled. ``attention`` trains only biases whose parameter
    path belongs to an attention module. The task head may be fully trained, as
    is standard for downstream adaptation with a newly initialized classifier.
    """
    scope = str(bias_scope).lower()
    if scope not in {"all", "transformer", "attention"}:
        raise ValueError("bias_scope must be all, transformer, or attention")
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    selected: set[str] = set()
    for name, parameter in model.named_parameters():
        is_bias = name.endswith(".bias")
        allowed = False
        if is_bias:
            if scope == "all":
                allowed = True
            elif scope == "transformer":
                allowed = not _is_head(name)
            else:
                lower = name.lower()
                allowed = any(token in lower for token in ("attn", "attention", "qkv", "query", "key", "value"))
        if train_head and _is_head(name):
            allowed = True
        parameter.requires_grad_(allowed)
        if allowed:
            selected.add(name)

    if not selected:
        raise RuntimeError("BitFit policy selected no trainable parameters")
    return selected


__all__ = ["set_bitfit_trainability"]
