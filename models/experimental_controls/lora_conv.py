# models/tuning_modules/lora_conv.py
from __future__ import annotations

import torch
import torch.nn as nn


class LoRAConv2d(nn.Module):
    """LoRA wrapper for Conv2d layers.

    This baseline freezes the original convolution and trains a low-rank update.
    The update is initialized to zero, so the wrapped layer is initially exactly
    equivalent to the frozen base convolution.
    """

    def __init__(self, base: nn.Conv2d, r: int = 4, alpha: float = 1.0):
        super().__init__()
        if not isinstance(base, nn.Conv2d):
            raise TypeError("LoRAConv2d expects an nn.Conv2d base layer.")

        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(1, self.r)

        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        # This simple, paper-baseline-safe implementation only wraps normal
        # convolutions. Depthwise/grouped layers are kept frozen.
        if self.base.groups != 1 or self.r <= 0:
            self.lora_down = None
            self.lora_up = None
            return

        in_ch = self.base.in_channels
        out_ch = self.base.out_channels
        k_h, k_w = self.base.kernel_size

        self.lora_down = nn.Conv2d(in_ch, self.r, kernel_size=1, bias=False)
        self.lora_up = nn.Conv2d(
            self.r,
            out_ch,
            kernel_size=(k_h, k_w),
            stride=self.base.stride,
            padding=self.base.padding,
            dilation=self.base.dilation,
            groups=1,
            bias=False,
        )

        # Zero up-projection gives exact frozen-base behavior at initialization.
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.lora_down is None or self.lora_up is None:
            return y
        return y + self.scaling * self.lora_up(self.lora_down(x))


def _target_matches(conv: nn.Conv2d, target: str) -> bool:
    target = str(target or "all").lower()
    if target == "all":
        return conv.groups == 1
    if target == "1x1":
        return conv.groups == 1 and conv.kernel_size == (1, 1)
    if target == "3x3":
        return conv.groups == 1 and conv.kernel_size == (3, 3)
    if target in ("dw", "depthwise"):
        # This lightweight implementation intentionally does not wrap depthwise
        # convs; return False to keep the baseline well-defined.
        return False
    raise ValueError(f"Unknown LoRA Conv2d target: {target}")


def apply_lora_conv2d(model: nn.Module, r: int = 4, alpha: float = 1.0, target: str = "all") -> nn.Module:
    """Replace selected Conv2d layers by LoRAConv2d wrappers in-place.

    Important correctness fix: we snapshot parent modules before replacing
    children and skip existing LoRAConv2d wrappers, preventing recursive wrapping
    of LoRA's own internal convolutions.
    """
    parents = list(model.modules())
    for parent in parents:
        if isinstance(parent, LoRAConv2d):
            continue
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRAConv2d):
                continue
            if isinstance(child, nn.Conv2d) and _target_matches(child, target):
                setattr(parent, name, LoRAConv2d(child, r=r, alpha=alpha))
    return model


__all__ = ["LoRAConv2d", "apply_lora_conv2d"]
