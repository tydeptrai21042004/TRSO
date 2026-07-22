"""Visual prompting from Bahng et al., ECCV 2022.

The strict reproduction keeps the pretrained model, including its original
classifier, frozen. Only the image-space prompt is optimized. Downstream labels
are mapped to a fixed subset of the pretrained output coordinates, matching the
published visual-prompting protocol rather than training a new classifier.
"""
from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn


class PadPrompter(nn.Module):
    """Learn a border prompt of width ``prompt_size``.

    This follows the official padding-prompt construction: four independent
    trainable tensors surround an unprompted zero-valued center. The complete
    prompt is added to every input image.
    """

    def __init__(self, prompt_size: int = 30, image_size: int = 224) -> None:
        super().__init__()
        prompt_size = int(prompt_size)
        image_size = int(image_size)
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
        self.is_visual_prompt = True

    def prompt(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"PadPrompter expects BCHW RGB input, got {tuple(x.shape)}")
        if x.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"PadPrompter was built for {self.image_size}x{self.image_size}, "
                f"got {tuple(x.shape[-2:])}"
            )
        center = x.new_zeros(1, 3, self.base_size, self.base_size)
        middle = torch.cat((self.pad_left, center, self.pad_right), dim=3)
        prompt = torch.cat((self.pad_up, middle, self.pad_down), dim=2)
        return prompt.expand(x.shape[0], -1, -1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.prompt(x)


class VisualPromptingClassifier(nn.Module):
    """Frozen pretrained classifier plus a learned image-space prompt.

    ``output_indices`` implements the fixed output-label mapping used by visual
    prompting. No new downstream classifier is created or trained.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        prompt_size: int = 30,
        image_size: int = 224,
        output_indices: Optional[Iterable[int]] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.tuning_module = PadPrompter(prompt_size=prompt_size, image_size=image_size)
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
        # The published method keeps the pretrained classifier in evaluation
        # mode while optimizing only the prompt.
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
                f"Pretrained classifier exposes {logits.shape[1]} outputs but "
                f"label mapping requests index {max_index}. Keep the original "
                "pretrained output head and choose valid prompt output indices."
            )
        return logits.index_select(1, self.output_indices)


__all__ = ["PadPrompter", "VisualPromptingClassifier"]
