"""Residual Adapters in the original reduced-resolution ResNet-26 domain.

This is a single-task extraction of the official multi-domain architecture by
Rebuffi, Bilen, and Vedaldi. Shared 3x3 convolutions are frozen during transfer;
task-specific BatchNorm, 1x1 residual adapters, and the task classifier are
trained. Both published series and parallel parameterizations are available.
"""
from __future__ import annotations

from typing import Dict, Iterable, Iterator, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeriesAdapter(nn.Module):
    """Published series residual mapping: y = x + A(BN(x))."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.conv.is_residual_adapter = True
        nn.init.zeros_(self.conv.weight)
        self.is_residual_adapter = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(self.bn(x))


class ConvTask(nn.Module):
    """Shared 3x3 convolution with task BN and optional residual adapter."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        mode: str = "parallel",
    ) -> None:
        super().__init__()
        mode = str(mode).lower()
        if mode not in {"parallel", "series"}:
            raise ValueError(f"Unknown residual-adapter mode: {mode}")
        self.mode = mode
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        if mode == "parallel":
            self.adapter = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
            self.adapter.is_residual_adapter = True
            nn.init.zeros_(self.adapter.weight)
        else:
            self.adapter = SeriesAdapter(out_channels)
        self.is_residual_adapter = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.mode == "parallel":
            y = y + self.adapter(x)
        y = self.bn(y)
        if self.mode == "series":
            y = self.adapter(y)
        return y


class ResidualAdapterBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int, mode: str) -> None:
        super().__init__()
        self.conv1 = ConvTask(in_channels, out_channels, stride=stride, mode=mode)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = ConvTask(out_channels, out_channels, stride=1, mode=mode)
        self.stride = int(stride)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

    def _shortcut(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride != 1:
            x = F.avg_pool2d(x, kernel_size=2, stride=self.stride)
        channel_delta = self.out_channels - x.shape[1]
        if channel_delta < 0:
            raise RuntimeError("ResidualAdapterBasicBlock does not support channel reduction")
        if channel_delta:
            left = channel_delta // 2
            right = channel_delta - left
            x = torch.cat(
                (
                    x.new_zeros(x.shape[0], left, x.shape[2], x.shape[3]),
                    x,
                    x.new_zeros(x.shape[0], right, x.shape[2], x.shape[3]),
                ),
                dim=1,
            )
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self._shortcut(x)
        y = self.relu(self.conv1(x))
        y = self.conv2(y)
        return self.relu(y + residual)


class ResidualAdapterResNet26(nn.Module):
    """Official 26-layer reduced-resolution residual-adapter backbone.

    Stem: 3x3, 32 channels. Stages: [4,4,4] BasicBlocks with channels
    [64,128,256], with stride 2 at the beginning of each stage.
    """

    def __init__(self, num_classes: int, mode: str = "parallel") -> None:
        super().__init__()
        mode = str(mode).lower()
        if mode not in {"parallel", "series"}:
            raise ValueError("mode must be 'parallel' or 'series'")
        self.mode = mode
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.in_channels = 32
        self.layer1 = self._make_layer(64, blocks=4, stride=2)
        self.layer2 = self._make_layer(128, blocks=4, stride=2)
        self.layer3 = self._make_layer(256, blocks=4, stride=2)
        self.final_bn = nn.BatchNorm2d(256)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, int(num_classes))
        self.backbone_family = "resnet"
        self._initialize()

    def _make_layer(self, channels: int, blocks: int, stride: int) -> nn.Sequential:
        modules = [ResidualAdapterBasicBlock(self.in_channels, channels, stride, self.mode)]
        self.in_channels = channels
        for _ in range(1, int(blocks)):
            modules.append(ResidualAdapterBasicBlock(self.in_channels, channels, 1, self.mode))
        return nn.Sequential(*modules)

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if getattr(module, "is_residual_adapter", False):
                    continue
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.relu(self.final_bn(x))
        return self.avgpool(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.forward_features(x))

    def load_shared_state_dict(self, state_dict: Mapping[str, torch.Tensor], strict: bool = False):
        """Load a pretrained official/shared checkpoint.

        Adapter, task-BN, and classifier keys may be absent. Common wrapper
        prefixes are stripped to support official checkpoint layouts.
        """
        cleaned: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            for prefix in ("module.", "model.", "backbone."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
            cleaned[key] = value
        return self.load_state_dict(cleaned, strict=strict)


def set_residual_adapter_trainability(model: ResidualAdapterResNet26) -> None:
    """Freeze shared filters and train task-specific BN/adapters/head."""
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
    for parameter in module.parameters():
        if parameter.requires_grad:
            yield parameter


# Legacy names retained only for import compatibility. The old arbitrary
# torchvision BasicBlock wrappers are intentionally removed from strict mode.
ParallelResidualAdapter = ResidualAdapterResNet26
SeriesResidualAdapter = ResidualAdapterResNet26


def attach_residual_adapters_resnet(*args, **kwargs):
    raise RuntimeError(
        "Strict paper reproduction no longer wraps arbitrary torchvision ResNets. "
        "Build ResidualAdapterResNet26(mode='parallel'|'series') instead."
    )


__all__ = [
    "SeriesAdapter",
    "ConvTask",
    "ResidualAdapterBasicBlock",
    "ResidualAdapterResNet26",
    "set_residual_adapter_trainability",
    "residual_adapter_trainable_parameters",
    "ParallelResidualAdapter",
    "SeriesResidualAdapter",
    "attach_residual_adapters_resnet",
]
