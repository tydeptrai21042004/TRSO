"""Conv-Adapter reproduction for ResNet-50 bottleneck blocks.

Implements the published depthwise-then-pointwise adapter with a learnable
output-channel scaling vector and the four insertion schemes studied in the
paper: convolution parallel/sequential and residual parallel/sequential.
"""
from __future__ import annotations

from typing import Iterable, Iterator

import torch
import torch.nn as nn


class ConvAdapter(nn.Module):
    """Published Conv-Adapter module.

    A depthwise KxK convolution preserves locality, followed by nonlinearity,
    pointwise channel projection, and channel-wise learnable scaling initialized
    to one.
    """

    def __init__(
        self,
        inplanes: int,
        outplanes: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        act_layer=nn.GELU,
    ) -> None:
        super().__init__()
        padding = ((int(kernel_size) - 1) // 2) * int(dilation)
        self.depthwise = nn.Conv2d(
            int(inplanes),
            int(inplanes),
            kernel_size=int(kernel_size),
            stride=int(stride),
            padding=padding,
            dilation=int(dilation),
            groups=int(inplanes),
            bias=True,
        )
        self.activation = act_layer()
        self.pointwise = nn.Conv2d(int(inplanes), int(outplanes), kernel_size=1, bias=True)
        self.scale = nn.Parameter(torch.ones(1, int(outplanes), 1, 1))
        self.is_conv_adapter = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.activation(self.depthwise(x))) * self.scale


class ConvAdapterBottleneck(nn.Module):
    """Wrap one frozen ResNet-50 Bottleneck with a published insertion scheme."""

    MODES = {"conv_parallel", "conv_sequential", "residual_parallel", "residual_sequential"}

    def __init__(self, block: nn.Module, mode: str, kernel_size: int = 3) -> None:
        super().__init__()
        if not all(hasattr(block, name) for name in ("conv1", "bn1", "conv2", "bn2", "conv3", "bn3", "relu")):
            raise TypeError("Conv-Adapter paper reproduction requires ResNet Bottleneck blocks")
        mode = str(mode).lower()
        if mode not in self.MODES:
            raise ValueError(f"Unknown Conv-Adapter mode: {mode}")
        self.block = block
        self.mode = mode
        for parameter in self.block.parameters():
            parameter.requires_grad_(False)

        if mode.startswith("conv_"):
            self.adapter = ConvAdapter(
                block.conv2.in_channels,
                block.conv2.out_channels,
                kernel_size=kernel_size,
                stride=block.conv2.stride[0] if isinstance(block.conv2.stride, tuple) else block.conv2.stride,
                dilation=block.conv2.dilation[0] if isinstance(block.conv2.dilation, tuple) else block.conv2.dilation,
            )
        elif mode == "residual_parallel":
            self.adapter = ConvAdapter(
                block.conv1.in_channels,
                block.conv3.out_channels,
                kernel_size=kernel_size,
                stride=block.conv2.stride[0] if isinstance(block.conv2.stride, tuple) else block.conv2.stride,
            )
        else:
            self.adapter = ConvAdapter(
                block.conv3.out_channels,
                block.conv3.out_channels,
                kernel_size=kernel_size,
                stride=1,
            )
        self.is_conv_adapter_wrapper = True

    def _frozen_residual_branch(self, x: torch.Tensor):
        b = self.block
        out = b.relu(b.bn1(b.conv1(x)))
        conv2_input = out
        conv2_output = b.conv2(conv2_input)
        if self.mode == "conv_parallel":
            conv2_output = conv2_output + self.adapter(conv2_input)
        elif self.mode == "conv_sequential":
            conv2_output = conv2_output + self.adapter(conv2_output)
        out = b.relu(b.bn2(conv2_output))
        out = b.bn3(b.conv3(out))
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        residual = self._frozen_residual_branch(x)
        if self.mode == "residual_parallel":
            residual = residual + self.adapter(x)
        if self.block.downsample is not None:
            identity = self.block.downsample(x)
        out = self.block.relu(residual + identity)
        if self.mode == "residual_sequential":
            out = out + self.adapter(out)
        return out


def apply_conv_adapter_resnet50(
    model: nn.Module,
    mode: str = "conv_parallel",
    kernel_size: int = 3,
    stages: Iterable[int] = (1, 2, 3, 4),
) -> int:
    """Insert Conv-Adapter into every Bottleneck of selected ResNet-50 stages."""
    count = 0
    for stage in stages:
        layer = getattr(model, f"layer{int(stage)}", None)
        if not isinstance(layer, nn.Sequential):
            raise TypeError("Conv-Adapter strict reproduction requires standard ResNet stages")
        for index, block in enumerate(layer):
            if isinstance(block, ConvAdapterBottleneck):
                continue
            layer[index] = ConvAdapterBottleneck(block, mode=mode, kernel_size=kernel_size)
            count += 1
    if count == 0:
        raise RuntimeError("No ResNet Bottleneck was adapted")
    return count


def set_conv_adapter_trainability(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, ConvAdapter):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    for name, parameter in model.named_parameters():
        if name.startswith("fc.") or ".fc." in name:
            parameter.requires_grad_(True)


# Legacy classes retained as aliases to avoid broken imports.
ConvAdapterDesign1 = ConvAdapter
LinearAdapter = nn.Identity


__all__ = [
    "ConvAdapter", "ConvAdapterBottleneck", "apply_conv_adapter_resnet50",
    "set_conv_adapter_trainability", "ConvAdapterDesign1", "LinearAdapter",
]
