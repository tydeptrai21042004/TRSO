"""Bottleneck Attention Module (BAM), paper-faithful ResNet implementation.

Reference formulation:
    M(F) = sigmoid(M_c(F) * M_s(F))
    F'   = (1 + M(F)) * F

The strict baseline is an architectural module trained end-to-end with a
ResNet-50. It is not treated as a frozen-backbone PEFT adapter.
"""
from __future__ import annotations

import copy
from typing import Sequence

import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channel: int, reduction_ratio: int = 16, num_layers: int = 1) -> None:
        super().__init__()
        gate_channel = int(gate_channel)
        hidden = max(1, gate_channel // int(reduction_ratio))
        modules = [nn.AdaptiveAvgPool2d(1), Flatten()]
        dims = [gate_channel] + [hidden] * int(num_layers) + [gate_channel]
        for index in range(len(dims) - 2):
            modules.extend(
                [
                    nn.Linear(dims[index], dims[index + 1]),
                    nn.BatchNorm1d(dims[index + 1]),
                    nn.ReLU(inplace=True),
                ]
            )
        modules.append(nn.Linear(dims[-2], dims[-1]))
        self.gate_c = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate_c(x).unsqueeze(2).unsqueeze(3).expand_as(x)


class SpatialGate(nn.Module):
    def __init__(
        self,
        gate_channel: int,
        reduction_ratio: int = 16,
        dilation_conv_num: int = 2,
        dilation_val: int = 4,
    ) -> None:
        super().__init__()
        hidden = max(1, int(gate_channel) // int(reduction_ratio))
        modules = [
            nn.Conv2d(gate_channel, hidden, kernel_size=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        ]
        for _ in range(int(dilation_conv_num)):
            modules.extend(
                [
                    nn.Conv2d(
                        hidden,
                        hidden,
                        kernel_size=3,
                        padding=int(dilation_val),
                        dilation=int(dilation_val),
                    ),
                    nn.BatchNorm2d(hidden),
                    nn.ReLU(inplace=True),
                ]
            )
        modules.append(nn.Conv2d(hidden, 1, kernel_size=1))
        self.gate_s = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate_s(x).expand_as(x)


class BAM(nn.Module):
    """Exact BAM attention equation from Park et al."""

    def __init__(
        self,
        gate_channel: int,
        reduction_ratio: int = 16,
        dilation_conv_num: int = 2,
        dilation_val: int = 4,
    ) -> None:
        super().__init__()
        self.channel_att = ChannelGate(gate_channel, reduction_ratio)
        self.spatial_att = SpatialGate(
            gate_channel,
            reduction_ratio=reduction_ratio,
            dilation_conv_num=dilation_conv_num,
            dilation_val=dilation_val,
        )
        self.is_bam = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"BAM expects NCHW input, got {tuple(x.shape)}")
        attention = 1.0 + torch.sigmoid(self.channel_att(x) * self.spatial_att(x))
        return attention * x


class BAMResNet50(nn.Module):
    """ResNet-50 with BAM modules at the three published stage transitions.

    The wrapped backbone must expose the standard torchvision/ResNet stem,
    layers 1--4, average pool, and fully-connected classifier. The BAM modules
    are applied after layer1, layer2, and layer3, before the next downsampling
    stage.
    """

    def __init__(
        self,
        base_resnet50: nn.Module,
        num_classes: int,
        reduction_ratio: int = 16,
        dilation_conv_num: int = 2,
        dilation_val: int = 4,
    ) -> None:
        super().__init__()
        required = ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc")
        if not all(hasattr(base_resnet50, name) for name in required):
            raise TypeError("BAMResNet50 requires a standard ResNet-50 backbone")
        first = base_resnet50.layer1[0]
        if not hasattr(first, "conv3"):
            raise TypeError("BAM paper reproduction requires ResNet-50 Bottleneck blocks")

        # Reuse pretrained backbone modules without changing their weights.
        self.conv1 = base_resnet50.conv1
        self.bn1 = base_resnet50.bn1
        self.relu = base_resnet50.relu
        self.maxpool = base_resnet50.maxpool
        self.layer1 = base_resnet50.layer1
        self.layer2 = base_resnet50.layer2
        self.layer3 = base_resnet50.layer3
        self.layer4 = base_resnet50.layer4
        self.avgpool = base_resnet50.avgpool
        self.bam1 = BAM(256, reduction_ratio, dilation_conv_num, dilation_val)
        self.bam2 = BAM(512, reduction_ratio, dilation_conv_num, dilation_val)
        self.bam3 = BAM(1024, reduction_ratio, dilation_conv_num, dilation_val)
        in_features = int(base_resnet50.fc.in_features)
        self.fc = nn.Linear(in_features, int(num_classes))
        self.backbone_family = "resnet"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.bam1(x)
        x = self.layer2(x)
        x = self.bam2(x)
        x = self.layer3(x)
        x = self.bam3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# Backward-compatible import name. It intentionally aliases the true BAM
# module, not the former frozen-backbone gated approximation.
BAMAdapter = BAM


__all__ = ["Flatten", "ChannelGate", "SpatialGate", "BAM", "BAMAdapter", "BAMResNet50"]
