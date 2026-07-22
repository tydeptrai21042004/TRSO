"""Visual Prompting from Bahng et al. (ECCV 2022).

The pretrained image classifier, including its original output head, remains
frozen. Only an image-space prompt is optimized. The three prompt families from
the released implementation are provided: padding, fixed patch and random
patch. Downstream labels select a fixed subset of pretrained output logits.
"""
from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn


class _PromptBase(nn.Module):
    is_visual_prompt = True

    def _validate(self, x: torch.Tensor, image_size: int) -> None:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Visual prompts expect BCHW RGB input, got {tuple(x.shape)}")
        if x.shape[-2:] != (image_size, image_size):
            raise ValueError(
                f"Prompt was built for {image_size}x{image_size}, got {tuple(x.shape[-2:])}"
            )


class PadPrompter(_PromptBase):
    """Learn four independent border tensors around an unchanged center."""

    def __init__(self, prompt_size: int = 30, image_size: int = 224) -> None:
        super().__init__()
        prompt_size, image_size = int(prompt_size), int(image_size)
        if prompt_size <= 0:
            raise ValueError("prompt_size must be positive")
        if 2 * prompt_size >= image_size:
            raise ValueError("prompt_size must leave a positive image center")
        self.prompt_size = prompt_size
        self.image_size = image_size
        self.base_size = image_size - 2 * prompt_size
        self.pad_up = nn.Parameter(torch.randn(1, 3, prompt_size, image_size))
        self.pad_down = nn.Parameter(torch.randn(1, 3, prompt_size, image_size))
        self.pad_left = nn.Parameter(torch.randn(1, 3, self.base_size, prompt_size))
        self.pad_right = nn.Parameter(torch.randn(1, 3, self.base_size, prompt_size))

    def prompt(self, x: torch.Tensor) -> torch.Tensor:
        self._validate(x, self.image_size)
        center = x.new_zeros(1, 3, self.base_size, self.base_size)
        middle = torch.cat((self.pad_left, center, self.pad_right), dim=3)
        prompt = torch.cat((self.pad_up, middle, self.pad_down), dim=2)
        return prompt.expand(x.shape[0], -1, -1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.prompt(x)


class FixedPatchPrompter(_PromptBase):
    """Learn one square patch at a fixed location (bottom-right by default)."""

    def __init__(
        self,
        prompt_size: int = 30,
        image_size: int = 224,
        location: str = "bottom_right",
    ) -> None:
        super().__init__()
        prompt_size, image_size = int(prompt_size), int(image_size)
        if not (0 < prompt_size <= image_size):
            raise ValueError("prompt_size must be in [1, image_size]")
        if location not in {"bottom_right", "top_left", "center"}:
            raise ValueError("location must be bottom_right, top_left, or center")
        self.prompt_size = prompt_size
        self.image_size = image_size
        self.location = location
        self.patch = nn.Parameter(torch.randn(1, 3, prompt_size, prompt_size))

    def coordinates(self) -> tuple[int, int]:
        p, size = self.prompt_size, self.image_size
        if self.location == "top_left":
            return 0, 0
        if self.location == "center":
            start = (size - p) // 2
            return start, start
        return size - p, size - p

    def prompt(self, x: torch.Tensor) -> torch.Tensor:
        self._validate(x, self.image_size)
        top, left = self.coordinates()
        canvas = x.new_zeros(1, 3, self.image_size, self.image_size)
        canvas[:, :, top : top + self.prompt_size, left : left + self.prompt_size] = self.patch
        return canvas.expand(x.shape[0], -1, -1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.prompt(x)


class RandomPatchPrompter(FixedPatchPrompter):
    """Place the learned patch at one uniformly sampled location per forward."""

    def __init__(self, prompt_size: int = 30, image_size: int = 224) -> None:
        super().__init__(prompt_size=prompt_size, image_size=image_size, location="top_left")
        self.register_buffer("last_location", torch.zeros(2, dtype=torch.long), persistent=False)

    def coordinates(self) -> tuple[int, int]:
        maximum = self.image_size - self.prompt_size
        if maximum <= 0:
            top = left = 0
        else:
            top = int(torch.randint(maximum + 1, (1,), device=self.patch.device).item())
            left = int(torch.randint(maximum + 1, (1,), device=self.patch.device).item())
        self.last_location.copy_(torch.tensor((top, left), device=self.last_location.device))
        return top, left


PROMPTERS = {
    "padding": PadPrompter,
    "pad": PadPrompter,
    "fixed_patch": FixedPatchPrompter,
    "random_patch": RandomPatchPrompter,
}


def build_prompter(prompt_type: str, prompt_size: int, image_size: int) -> nn.Module:
    key = str(prompt_type).lower()
    if key not in PROMPTERS:
        raise ValueError(f"Unknown visual prompt type {prompt_type!r}; choose {sorted(PROMPTERS)}")
    return PROMPTERS[key](prompt_size=prompt_size, image_size=image_size)


class VisualPromptingClassifier(nn.Module):
    """Frozen pretrained classifier plus a learned image-space prompt."""

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        prompt_size: int = 30,
        image_size: int = 224,
        output_indices: Optional[Iterable[int]] = None,
        prompt_type: str = "padding",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.tuning_module = build_prompter(prompt_type, prompt_size, image_size)
        self.prompt_type = str(prompt_type).lower()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

        num_classes = int(num_classes)
        if num_classes <= 1:
            raise ValueError("Visual prompting requires at least two classes")
        if output_indices is None:
            output_indices = range(num_classes)
        indices = torch.as_tensor(list(output_indices), dtype=torch.long)
        if indices.numel() != num_classes:
            raise ValueError("output_indices length must equal num_classes")
        if indices.min().item() < 0:
            raise ValueError("output_indices must be non-negative")
        if torch.unique(indices).numel() != indices.numel():
            raise ValueError("output_indices must be unique")
        self.register_buffer("output_indices", indices, persistent=True)
        self.num_classes = num_classes
        self.backbone_family = getattr(backbone, "backbone_family", None)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        self.tuning_module.train(mode)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(self.tuning_module(x))
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if logits.ndim != 2:
            raise RuntimeError(f"Expected pretrained class logits [B,K], got {tuple(logits.shape)}")
        max_index = int(self.output_indices.max().item())
        if logits.shape[1] <= max_index:
            raise RuntimeError(
                f"Pretrained classifier exposes {logits.shape[1]} outputs but mapping requests "
                f"index {max_index}. Preserve the pretrained output head and choose valid indices."
            )
        return logits.index_select(1, self.output_indices)


__all__ = [
    "PadPrompter",
    "FixedPatchPrompter",
    "RandomPatchPrompter",
    "build_prompter",
    "VisualPromptingClassifier",
]
