# models/tuning_modules/residual_adapter.py
from __future__ import annotations

from typing import Iterable
import torch
import torch.nn as nn


_ACTS = {
    "relu": lambda: nn.ReLU(inplace=True),
    "gelu": lambda: nn.GELU(),
    "silu": lambda: nn.SiLU(inplace=True),
    "none": lambda: nn.Identity(),
}
_NORMS = {
    "bn": lambda c: nn.BatchNorm2d(c),
    "ln": lambda c: nn.GroupNorm(1, c),
    "none": lambda c: nn.Identity(),
}


def _make_core(channels: int, reduction: int, norm: str, act: str) -> nn.Sequential:
    hidden = max(1, int(channels) // max(1, int(reduction)))
    if norm not in _NORMS:
        raise ValueError(f"Unknown norm: {norm}")
    if act not in _ACTS:
        raise ValueError(f"Unknown activation: {act}")
    return nn.Sequential(
        nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
        _NORMS[norm](hidden),
        _ACTS[act](),
        nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
    )


class ParallelResidualAdapter(nn.Module):
    """Output-space parallel residual adapter for frozen ResNet blocks.

    Original residual adapters introduce a small residual branch around a frozen
    block. For post-hoc wrapping of torchvision ResNet blocks, using the block
    output ``y`` avoids channel mismatch in downsample blocks while preserving a
    residual-branch adaptation protocol:

        y   = Block(x)
        out = y + gate * Core(y)
    """

    def __init__(
        self,
        block: nn.Module,
        channels: int,
        reduction: int = 16,
        norm: str = "bn",
        act: str = "relu",
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.block = block
        self.core = _make_core(channels, reduction, norm, act)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.is_residual_adapter = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        return y + self.gate * self.core(y)


class SeriesResidualAdapter(nn.Module):
    """Series/post-block residual adapter: y = Block(x); out = y + gate*Core(y)."""

    def __init__(
        self,
        block: nn.Module,
        channels: int,
        reduction: int = 16,
        norm: str = "bn",
        act: str = "relu",
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.block = block
        self.core = _make_core(channels, reduction, norm, act)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.is_residual_adapter = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        return y + self.gate * self.core(y)


def build_residual_adapter(
    block: nn.Module,
    channels: int,
    mode: str = "parallel",
    reduction: int = 16,
    norm: str = "bn",
    act: str = "relu",
    gate_init: float = 0.0,
) -> nn.Module:
    if mode == "parallel":
        return ParallelResidualAdapter(block, channels, reduction, norm, act, gate_init)
    if mode == "series":
        return SeriesResidualAdapter(block, channels, reduction, norm, act, gate_init)
    raise ValueError(f"Unknown residual adapter mode: {mode}")


def residual_adapter_trainable_parameters(module: nn.Module):
    """Yield only adapter-core/gate params, excluding the wrapped frozen block."""
    for name, p in module.named_parameters(recurse=True):
        if name.startswith("core.") or name == "gate":
            yield p


def _wrap_resnet_layer(layer: nn.Sequential, mode: str, reduction: int, norm: str, act: str, gate_init: float):
    for i, block in enumerate(layer):
        if hasattr(block, "conv3"):
            channels = block.conv3.out_channels
        elif hasattr(block, "conv2"):
            channels = block.conv2.out_channels
        else:
            raise TypeError(f"Cannot infer output channels for block: {type(block).__name__}")
        layer[i] = build_residual_adapter(block, channels, mode, reduction, norm, act, gate_init)


def attach_residual_adapters_resnet(
    model: nn.Module,
    stages: Iterable[int] = (1, 2, 3, 4),
    mode: str = "parallel",
    reduction: int = 16,
    norm: str = "bn",
    act: str = "relu",
    gate_init: float = 0.0,
) -> nn.Module:
    stages = set(int(s) for s in stages)
    if 1 in stages:
        _wrap_resnet_layer(model.layer1, mode, reduction, norm, act, gate_init)
    if 2 in stages:
        _wrap_resnet_layer(model.layer2, mode, reduction, norm, act, gate_init)
    if 3 in stages:
        _wrap_resnet_layer(model.layer3, mode, reduction, norm, act, gate_init)
    if 4 in stages:
        _wrap_resnet_layer(model.layer4, mode, reduction, norm, act, gate_init)
    return model


__all__ = [
    "ParallelResidualAdapter",
    "SeriesResidualAdapter",
    "build_residual_adapter",
    "attach_residual_adapters_resnet",
    "residual_adapter_trainable_parameters",
]
