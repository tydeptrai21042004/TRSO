# models/tuning_modules/bam_adapter.py
"""BAM-Tuning baseline for frozen-backbone CNN PEFT.

This module adapts the Bottleneck Attention Module (BAM) idea into the same
protocol used by the other baselines in this repository:

    frozen CNN backbone + trainable BAM modules + trainable classifier head.

The module is identity-safe at initialization when ``gate_init=0.0``:
``forward(x) == x`` up to floating-point roundoff. This makes it fair as a
plug-in PEFT baseline because it does not perturb the pretrained backbone before
training.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BAMAdapter(nn.Module):
    """BAM-style lightweight attention adapter for NCHW feature maps.

    Args:
        channels: Number of input/output channels.
        reduction: Bottleneck reduction ratio for channel/spatial branches.
        dilation: Dilation used in the spatial branch 3x3 convolution.
        gate_init: Initial scalar residual gate. Use 0.0 for identity-safe PEFT.
        use_bn: Whether to use BatchNorm in the spatial branch.

    Formula:
        A(x) = sigmoid(ChannelAtt(x) + SpatialAtt(x))
        out  = x + gate * x * A(x)
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        dilation: int = 4,
        gate_init: float = 0.0,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        channels = int(channels)
        hidden = max(1, channels // max(1, int(reduction)))
        dilation = max(1, int(dilation))

        self.channels = channels
        self.reduction = int(reduction)
        self.dilation = dilation
        self.use_bn = bool(use_bn)

        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

        spatial_layers = [
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden) if self.use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(hidden) if self.use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=False),
        ]
        self.spatial_att = nn.Sequential(*spatial_layers)

        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.is_bam_adapter = True

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"BAMAdapter expects NCHW tensor, got shape {tuple(x.shape)}")
        return torch.sigmoid(self.channel_att(x) + self.spatial_att(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        att = self.attention(x)
        return x + self.gate * x * att


__all__ = ["BAMAdapter"]
