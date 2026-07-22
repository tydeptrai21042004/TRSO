"""Side-Tuning with copy initialization from the frozen base network.

The strict reproduction uses a side network with the same ResNet architecture,
initialized from the pretrained base. The base remains frozen; the side,
mixture coefficient, and downstream classifier are optimized.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


class SideTuningClassifier(nn.Module):
    def __init__(
        self,
        base_model: nn.Module = None,
        num_classes: int = 1000,
        learn_alpha: bool = True,
        alpha_init: float = 0.5,
        **legacy,
    ) -> None:
        super().__init__()
        if base_model is None:
            base_model = legacy.pop("base_backbone", None)
        if base_model is None:
            raise TypeError("SideTuningClassifier requires base_model")
        required = ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc")
        if not all(hasattr(base_model, name) for name in required):
            raise TypeError("Strict Side-Tuning reproduction currently supports ResNet backbones")

        self.base = base_model
        self.side = copy.deepcopy(base_model)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()
        for parameter in self.side.parameters():
            parameter.requires_grad_(True)

        feature_dim = int(base_model.fc.in_features)
        self.base.fc = nn.Identity()
        self.side.fc = nn.Identity()
        alpha_init = min(max(float(alpha_init), 1e-6), 1.0 - 1e-6)
        value = torch.tensor(alpha_init, dtype=torch.float32)
        self.alpha_logit = nn.Parameter(torch.log(value / (1.0 - value)), requires_grad=bool(learn_alpha))
        self.head = nn.Linear(feature_dim, int(num_classes))
        self.num_features = feature_dim
        self.backbone_family = "resnet"
        self.is_side_tuning = True

    @property
    def alpha(self) -> torch.Tensor:
        """Weight assigned to the frozen base, matching the paper notation."""
        return torch.sigmoid(self.alpha_logit)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        self.side.train(mode)
        return self

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            base_features = self.base(x)
        side_features = self.side(x)
        alpha = self.alpha
        return alpha * base_features + (1.0 - alpha) * side_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


__all__ = ["SideTuningClassifier"]
