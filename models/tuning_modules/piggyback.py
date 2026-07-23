"""Piggyback binary-mask adaptation for CNNs (ECCV 2018).

Frozen pretrained convolution/linear weights are multiplied by learned binary
masks.  Real-valued mask scores are optimized with a straight-through
estimator; deployed masks require one bit per masked weight.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryMaskSTE(torch.autograd.Function):
    """Binarize to {0,1}; pass the mask-score gradient straight through."""

    @staticmethod
    def forward(ctx, scores: torch.Tensor, threshold: float):
        return (scores >= float(threshold)).to(dtype=scores.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


def binary_mask(scores: torch.Tensor, threshold: float = 5e-3) -> torch.Tensor:
    return BinaryMaskSTE.apply(scores, float(threshold))


def _initialize_scores(parameter: nn.Parameter, mode: str, scale: float, threshold: float) -> None:
    if mode == "ones":
        # The released Piggyback ``1s`` initialization multiplies an all-one
        # tensor by mask_scale. With the paper defaults this is 1e-2, above the
        # 5e-3 threshold, so the inserted model initially equals the pretrained
        # network while retaining the intended mask-score scale.
        nn.init.constant_(parameter, float(scale))
    elif mode == "near_threshold":
        nn.init.normal_(parameter, mean=float(threshold), std=float(scale))
    else:
        raise ValueError("Piggyback mask_init must be 'ones' or 'near_threshold'")


class PiggybackConv2d(nn.Module):
    def __init__(
        self,
        base: nn.Conv2d,
        *,
        threshold: float = 5e-3,
        mask_init: str = "ones",
        mask_scale: float = 1e-2,
    ) -> None:
        super().__init__()
        self.in_channels = base.in_channels
        self.out_channels = base.out_channels
        self.kernel_size = base.kernel_size
        self.stride = base.stride
        self.padding = base.padding
        self.dilation = base.dilation
        self.groups = base.groups
        self.padding_mode = base.padding_mode
        self.threshold = float(threshold)
        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        if base.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)
        self.mask_scores = nn.Parameter(torch.empty_like(self.weight))
        _initialize_scores(self.mask_scores, mask_init, mask_scale, threshold)
        self.is_piggyback = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding_mode != "zeros":
            x = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = (0, 0)
        else:
            padding = self.padding
        return F.conv2d(
            x,
            self.weight * binary_mask(self.mask_scores, self.threshold),
            self.bias,
            self.stride,
            padding,
            self.dilation,
            self.groups,
        )

    @property
    def _reversed_padding_repeated_twice(self):
        p = self.padding if isinstance(self.padding, tuple) else (self.padding, self.padding)
        return [p[1], p[1], p[0], p[0]]


class PiggybackLinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        *,
        threshold: float = 5e-3,
        mask_init: str = "ones",
        mask_scale: float = 1e-2,
    ) -> None:
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.threshold = float(threshold)
        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        if base.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)
        self.mask_scores = nn.Parameter(torch.empty_like(self.weight))
        _initialize_scores(self.mask_scores, mask_init, mask_scale, threshold)
        self.is_piggyback = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * binary_mask(self.mask_scores, self.threshold), self.bias)


@dataclass(frozen=True)
class PiggybackStorage:
    masked_weights: int
    deployed_mask_bits: int
    deployed_mask_megabytes: float
    training_score_megabytes_fp32: float


def apply_piggyback(
    model: nn.Module,
    *,
    threshold: float = 5e-3,
    mask_init: str = "ones",
    mask_scale: float = 1e-2,
    mask_linear: bool = False,
    exclude_task_head: bool = True,
) -> List[str]:
    """Replace eligible frozen operations by Piggyback masked operations."""
    records: List[str] = []

    def visit(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            path = f"{prefix}.{child_name}" if prefix else child_name
            is_head = path.startswith(("head", "heads", "classifier", "fc")) or any(
                token in path for token in (".head.", ".heads.", ".classifier.")
            )
            if isinstance(child, PiggybackConv2d) or isinstance(child, PiggybackLinear):
                continue
            if isinstance(child, nn.Conv2d) and not (exclude_task_head and is_head):
                setattr(
                    parent,
                    child_name,
                    PiggybackConv2d(
                        child,
                        threshold=threshold,
                        mask_init=mask_init,
                        mask_scale=mask_scale,
                    ),
                )
                records.append(path)
            elif mask_linear and isinstance(child, nn.Linear) and not (exclude_task_head and is_head):
                setattr(
                    parent,
                    child_name,
                    PiggybackLinear(
                        child,
                        threshold=threshold,
                        mask_init=mask_init,
                        mask_scale=mask_scale,
                    ),
                )
                records.append(path)
            else:
                visit(child, path)

    visit(model)
    if not records:
        raise TypeError("Piggyback found no eligible CNN convolutional weights")
    return records


def set_piggyback_trainability(model: nn.Module, train_head: bool = True) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if name.endswith("mask_scores"):
            parameter.requires_grad_(True)
        elif train_head and (
            name.startswith(("head.", "heads.", "classifier.", "fc."))
            or any(token in name for token in (".head.", ".heads.", ".classifier."))
        ):
            parameter.requires_grad_(True)


def piggyback_storage(model: nn.Module) -> PiggybackStorage:
    count = sum(module.mask_scores.numel() for module in model.modules() if hasattr(module, "mask_scores"))
    return PiggybackStorage(
        masked_weights=int(count),
        deployed_mask_bits=int(count),
        deployed_mask_megabytes=float(count / 8 / 1024**2),
        training_score_megabytes_fp32=float(count * 4 / 1024**2),
    )


@torch.no_grad()
def export_binary_masks(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: binary_mask(module.mask_scores, module.threshold).to(torch.uint8).cpu()
        for name, module in model.named_modules()
        if hasattr(module, "mask_scores")
    }


__all__ = [
    "BinaryMaskSTE",
    "PiggybackConv2d",
    "PiggybackLinear",
    "PiggybackStorage",
    "apply_piggyback",
    "set_piggyback_trainability",
    "piggyback_storage",
    "export_binary_masks",
]
