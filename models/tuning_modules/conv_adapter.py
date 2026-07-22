"""Conv-Adapter for the official ResNet-50 implementation.

The default ``conv_parallel`` path mirrors the released Conv-Adapter code:

    z = ReLU(BN1(Conv1(x)))
    a = Conv1x1(ReLU(GroupedConv3x3(z))) * scale
    z = ReLU(BN2(Conv2(z))) + a

The remaining insertion modes are retained as explicitly named ablations. They
use the same published adapter parameterization but are not silently presented
as the official default.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn


def _valid_groups(in_channels: int, width: int, requested: int | None = None) -> int:
    if requested is None:
        requested = width
    groups = math.gcd(int(in_channels), math.gcd(int(width), max(1, int(requested))))
    return max(1, groups)


class ConvAdapter(nn.Module):
    """Official Conv-Adapter Design-2/v4 module.

    The spatial convolution maps ``inplanes`` to a reduced ``width`` using
    grouped convolution, followed by the activation and a pointwise projection.
    The output is multiplied by a learnable per-channel scale initialized to
    ``adapt_scale``.
    """

    def __init__(
        self,
        inplanes: int,
        outplanes: int,
        width: int | None = None,
        kernel_size: int = 3,
        padding: int | None = None,
        stride: int = 1,
        groups: int | None = None,
        dilation: int = 1,
        norm_layer=None,
        act_layer=None,
        adapt_scale: float = 1.0,
        **_: object,
    ) -> None:
        super().__init__()
        del norm_layer  # The released Design-2 module does not use normalization.
        inplanes, outplanes = int(inplanes), int(outplanes)
        width = inplanes if width is None else int(width)
        if min(inplanes, outplanes, width) <= 0:
            raise ValueError("Conv-Adapter channel dimensions must be positive")
        kernel_size, stride, dilation = int(kernel_size), int(stride), int(dilation)
        if padding is None:
            padding = ((kernel_size - 1) // 2) * dilation
        groups = _valid_groups(inplanes, width, groups)
        if act_layer is None:
            act_layer = nn.Identity

        self.conv1 = nn.Conv2d(
            inplanes,
            width,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=int(padding),
            dilation=dilation,
            bias=True,
        )
        self.act = act_layer()
        self.conv2 = nn.Conv2d(width, outplanes, kernel_size=1, stride=1, bias=True)
        self.se = nn.Parameter(
            torch.full((1, outplanes, 1, 1), float(adapt_scale)),
            requires_grad=True,
        )
        self.is_conv_adapter = True
        self.width = width
        self.groups = groups

    # Compatibility aliases for older local code and external inspection.
    @property
    def depthwise(self):
        return self.conv1

    @property
    def pointwise(self):
        return self.conv2

    @property
    def activation(self):
        return self.act

    @property
    def scale(self):
        return self.se

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.act(self.conv1(x))) * self.se


class ConvAdapterBottleneck(nn.Module):
    """Wrap a torchvision ResNet Bottleneck with Conv-Adapter.

    ``conv_parallel`` is the released paper path. Other modes are controlled
    ablations that share the same adapter implementation.
    """

    MODES = {"conv_parallel", "conv_sequential", "residual_parallel", "residual_sequential"}

    def __init__(
        self,
        block: nn.Module,
        mode: str = "conv_parallel",
        kernel_size: int = 3,
        reduction: float = 8.0,
        adapt_scale: float = 1.0,
    ) -> None:
        super().__init__()
        required = ("conv1", "bn1", "conv2", "bn2", "conv3", "bn3", "relu")
        if not all(hasattr(block, name) for name in required):
            raise TypeError("Conv-Adapter reproduction requires torchvision ResNet Bottleneck blocks")
        mode = str(mode).lower()
        if mode not in self.MODES:
            raise ValueError(f"Unknown Conv-Adapter mode: {mode}")
        if float(reduction) <= 0:
            raise ValueError("Conv-Adapter reduction must be positive")

        self.block = block
        self.mode = mode
        self.reduction = float(reduction)
        for parameter in self.block.parameters():
            parameter.requires_grad_(False)

        bottleneck_channels = int(block.conv2.in_channels)
        stride = int(block.conv2.stride[0] if isinstance(block.conv2.stride, tuple) else block.conv2.stride)
        dilation = int(block.conv2.dilation[0] if isinstance(block.conv2.dilation, tuple) else block.conv2.dilation)

        if mode == "conv_parallel":
            # Exact released ResNet-50 construction.
            in_channels = out_channels = bottleneck_channels
            adapter_stride = stride
            adapter_width = max(1, int(bottleneck_channels // self.reduction))
            groups = adapter_width
            activation = nn.ReLU
        elif mode == "conv_sequential":
            in_channels = out_channels = bottleneck_channels
            adapter_stride = 1
            adapter_width = max(1, int(bottleneck_channels // self.reduction))
            groups = adapter_width
            activation = nn.ReLU
        elif mode == "residual_parallel":
            in_channels = int(block.conv1.in_channels)
            out_channels = int(block.conv3.out_channels)
            adapter_stride = stride
            adapter_width = max(1, int(min(in_channels, out_channels) // self.reduction))
            groups = _valid_groups(in_channels, adapter_width)
            activation = nn.ReLU
        else:
            in_channels = out_channels = int(block.conv3.out_channels)
            adapter_stride = 1
            adapter_width = max(1, int(in_channels // self.reduction))
            groups = _valid_groups(in_channels, adapter_width)
            activation = nn.ReLU

        self.adapter = ConvAdapter(
            in_channels,
            out_channels,
            width=adapter_width,
            kernel_size=kernel_size,
            stride=adapter_stride,
            groups=groups,
            dilation=dilation if mode == "conv_parallel" else 1,
            act_layer=activation,
            adapt_scale=adapt_scale,
        )
        self.is_conv_adapter_wrapper = True
        self.is_official_conv_adapter_path = mode == "conv_parallel"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.block
        identity = x

        out = b.relu(b.bn1(b.conv1(x)))
        if self.mode == "conv_parallel":
            out_adapt = self.adapter(out)
        else:
            out_adapt = None

        out = b.relu(b.bn2(b.conv2(out)))
        if self.mode == "conv_parallel":
            # Official code adds after BN2 and ReLU.
            out = out + out_adapt
        elif self.mode == "conv_sequential":
            out = out + self.adapter(out)

        out = b.bn3(b.conv3(out))
        if self.mode == "residual_parallel":
            out = out + self.adapter(x)

        if b.downsample is not None:
            identity = b.downsample(x)
        out = b.relu(out + identity)
        if self.mode == "residual_sequential":
            out = out + self.adapter(out)
        return out


def apply_conv_adapter_resnet50(
    model: nn.Module,
    mode: str = "conv_parallel",
    kernel_size: int = 3,
    stages: Iterable[int] = (1, 2, 3, 4),
    reduction: float = 8.0,
    adapt_scale: float = 1.0,
) -> int:
    """Insert Conv-Adapter into every bottleneck in selected ResNet-50 stages."""
    count = 0
    for stage in stages:
        layer = getattr(model, f"layer{int(stage)}", None)
        if not isinstance(layer, nn.Sequential):
            raise TypeError("Conv-Adapter strict reproduction requires standard ResNet stages")
        for index, block in enumerate(layer):
            if isinstance(block, ConvAdapterBottleneck):
                continue
            layer[index] = ConvAdapterBottleneck(
                block,
                mode=mode,
                kernel_size=kernel_size,
                reduction=reduction,
                adapt_scale=adapt_scale,
            )
            count += 1
    if count == 0:
        raise RuntimeError("No ResNet Bottleneck was adapted")
    return count


def set_conv_adapter_trainability(model: nn.Module) -> None:
    """Freeze the ResNet and train only Conv-Adapters plus the task classifier."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, ConvAdapter):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    for name, parameter in model.named_parameters():
        if name.startswith("fc.") or ".fc." in name:
            parameter.requires_grad_(True)


# Legacy names retained for import compatibility.
ConvAdapterDesign1 = ConvAdapter
LinearAdapter = nn.Identity


__all__ = [
    "ConvAdapter",
    "ConvAdapterBottleneck",
    "apply_conv_adapter_resnet50",
    "set_conv_adapter_trainability",
    "ConvAdapterDesign1",
    "LinearAdapter",
]
