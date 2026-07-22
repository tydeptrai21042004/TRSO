"""Residual Adapters in the official reduced-resolution ResNet-26 domain.

This module is a single-task extraction of Rebuffi et al.'s released model. It
preserves the official ResNet-26 stem, [4,4,4] blocks, option-A shortcuts,
parallel/series adapter ordering, task-specific BatchNorm and classifier.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeriesAdapter(nn.Module):
    """Official series adapter: ``x + Conv1x1(BN(x))``."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(int(channels))
        self.conv = nn.Conv2d(int(channels), int(channels), kernel_size=1, bias=False)
        self.conv.is_residual_adapter = True
        nn.init.zeros_(self.conv.weight)
        self.is_residual_adapter = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(self.bn(x))


class ConvTask(nn.Module):
    """Shared 3x3 convolution plus one task-specific adapter/BatchNorm path."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, mode: str = "parallel") -> None:
        super().__init__()
        mode = str(mode).lower()
        if mode not in {"parallel", "series"}:
            raise ValueError(f"Unknown residual-adapter mode: {mode}")
        self.mode = mode
        self.conv = nn.Conv2d(
            int(in_channels), int(out_channels), kernel_size=3, stride=int(stride), padding=1, bias=False
        )
        if mode == "parallel":
            self.adapter = nn.Conv2d(
                int(in_channels), int(out_channels), kernel_size=1, stride=int(stride), bias=False
            )
            self.adapter.is_residual_adapter = True
            nn.init.zeros_(self.adapter.weight)
        else:
            self.adapter = SeriesAdapter(int(out_channels))
        # In the official series path, SeriesAdapter is applied before this BN.
        self.bn = nn.BatchNorm2d(int(out_channels))
        self.is_residual_adapter = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.mode == "parallel":
            y = y + self.adapter(x)
        else:
            y = self.adapter(y)
        return self.bn(y)


class ResidualAdapterBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int, mode: str) -> None:
        super().__init__()
        self.conv1 = ConvTask(in_channels, out_channels, stride=stride, mode=mode)
        # Official code applies ReLU immediately before the second ConvTask.
        self.conv2 = ConvTask(out_channels, out_channels, stride=1, mode=mode)
        self.relu = nn.ReLU(inplace=True)
        self.stride = int(stride)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

    def _shortcut(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride != 1:
            x = F.avg_pool2d(x, kernel_size=2, stride=self.stride)
        if self.out_channels == x.shape[1]:
            return x
        if self.out_channels != 2 * x.shape[1]:
            raise RuntimeError(
                "Official Residual-Adapter option-A shortcut expects unchanged or doubled channels"
            )
        # Exact released code: concatenate the downsampled tensor with zeros.
        return torch.cat((x, torch.zeros_like(x)), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self._shortcut(x)
        y = self.conv1(x)
        y = self.conv2(self.relu(y))
        return self.relu(y + residual)


@dataclass(frozen=True)
class SharedCheckpointCoverage:
    loaded_shared: int
    expected_shared: int
    missing_shared: tuple[str, ...]
    unexpected: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing_shared


class ResidualAdapterResNet26(nn.Module):
    """Official 26-layer reduced-resolution residual-adapter backbone."""

    def __init__(self, num_classes: int, mode: str = "parallel") -> None:
        super().__init__()
        mode = str(mode).lower()
        if mode not in {"parallel", "series"}:
            raise ValueError("mode must be 'parallel' or 'series'")
        self.mode = mode
        # Official stem is ConvTask without an immediate ReLU.
        self.pre_layers_conv = ConvTask(3, 32, stride=1, mode=mode)
        self.in_channels = 32
        self.layer1 = self._make_layer(64, blocks=4, stride=2)
        self.layer2 = self._make_layer(128, blocks=4, stride=2)
        self.layer3 = self._make_layer(256, blocks=4, stride=2)
        self.final_bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, int(num_classes))
        self.backbone_family = "resnet"
        self._initialize()
        self.last_checkpoint_coverage: SharedCheckpointCoverage | None = None

    @property
    def conv1(self):
        """Compatibility view of the shared stem convolution."""
        return self.pre_layers_conv.conv

    @property
    def bn1(self):
        return self.pre_layers_conv.bn

    def _make_layer(self, channels: int, blocks: int, stride: int) -> nn.Sequential:
        modules = [ResidualAdapterBasicBlock(self.in_channels, channels, stride, self.mode)]
        self.in_channels = channels
        modules.extend(
            ResidualAdapterBasicBlock(self.in_channels, channels, 1, self.mode)
            for _ in range(1, int(blocks))
        )
        return nn.Sequential(*modules)

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if getattr(module, "is_residual_adapter", False):
                    continue
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                nn.init.normal_(module.weight, 0.0, (2.0 / n) ** 0.5)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_layers_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.relu(self.final_bn(x))
        return self.avgpool(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.forward_features(x))

    @staticmethod
    def _strip_prefix(key: str) -> str:
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "model.", "backbone."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        return key

    def _map_official_key(self, key: str) -> str:
        key = self._strip_prefix(key)
        key = key.replace("linears.0.", "fc.")
        key = key.replace("end_bns.0.0.", "final_bn.")
        key = key.replace(".parallel_conv.0.conv.", ".adapter.")
        key = key.replace(".bns.0.", ".bn.")
        # Official series: bns.0.0.conv.0(BN), bns.0.0.conv.1(1x1), bns.0.1(final BN)
        key = key.replace(".bn.0.conv.0.", ".adapter.bn.")
        key = key.replace(".bn.0.conv.1.", ".adapter.conv.")
        key = key.replace(".bn.1.", ".bn.")
        return key

    def shared_parameter_keys(self) -> set[str]:
        return {name for name, _ in self.named_parameters() if name.endswith("conv.weight") and ".adapter." not in name}

    def load_shared_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = False,
        require_shared_coverage: bool = True,
    ):
        """Load official/shared weights and verify all shared 3x3 filters were found."""
        model_state = self.state_dict()
        cleaned: Dict[str, torch.Tensor] = {}
        for raw_key, value in state_dict.items():
            key = self._map_official_key(raw_key)
            if key in model_state and hasattr(value, "shape") and value.shape == model_state[key].shape:
                cleaned[key] = value

        incompat = self.load_state_dict(cleaned, strict=strict)
        expected = self.shared_parameter_keys()
        loaded = expected.intersection(cleaned)
        missing = tuple(sorted(expected - loaded))
        coverage = SharedCheckpointCoverage(
            loaded_shared=len(loaded),
            expected_shared=len(expected),
            missing_shared=missing,
            unexpected=tuple(sorted(incompat.unexpected_keys)),
        )
        self.last_checkpoint_coverage = coverage
        if require_shared_coverage and missing:
            preview = ", ".join(missing[:5])
            raise RuntimeError(
                f"Residual-Adapter checkpoint covers {len(loaded)}/{len(expected)} shared filters; "
                f"missing: {preview}{' ...' if len(missing) > 5 else ''}"
            )
        return incompat


def set_residual_adapter_trainability(model: ResidualAdapterResNet26) -> None:
    """Freeze shared filters; train task BN, adapters and classifier."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        if isinstance(module, SeriesAdapter):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        if isinstance(module, ConvTask) and module.mode == "parallel":
            for parameter in module.adapter.parameters():
                parameter.requires_grad_(True)
    for parameter in model.fc.parameters():
        parameter.requires_grad_(True)


def residual_adapter_trainable_parameters(module: nn.Module) -> Iterator[nn.Parameter]:
    yield from (parameter for parameter in module.parameters() if parameter.requires_grad)


ParallelResidualAdapter = ResidualAdapterResNet26
SeriesResidualAdapter = ResidualAdapterResNet26


def attach_residual_adapters_resnet(*args, **kwargs):
    raise RuntimeError(
        "Strict reproduction uses ResidualAdapterResNet26(mode='parallel'|'series'), "
        "not arbitrary torchvision ResNet wrappers."
    )


__all__ = [
    "SeriesAdapter",
    "ConvTask",
    "ResidualAdapterBasicBlock",
    "ResidualAdapterResNet26",
    "SharedCheckpointCoverage",
    "set_residual_adapter_trainability",
    "residual_adapter_trainable_parameters",
    "ParallelResidualAdapter",
    "SeriesResidualAdapter",
    "attach_residual_adapters_resnet",
]
