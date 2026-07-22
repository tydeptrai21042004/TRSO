"""Side-Tuning via an additive lightweight side network.

The frozen base and trainable side produce compatible representations that are
alpha-blended before the downstream head:

    h(x) = alpha * B(x) + (1 - alpha) * S(x).

The default side is a small fully convolutional network, matching the paper's
classification/vision use of a lightweight 4/5-layer FCN. A full copied side is
available only as an explicitly named capacity ablation.
"""
from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn as nn


class ConvSideNetwork(nn.Module):
    """Small fully convolutional side network with a projected feature output."""

    def __init__(self, feature_dim: int, width: int = 64, depth: int = 4) -> None:
        super().__init__()
        width, depth, feature_dim = int(width), int(depth), int(feature_dim)
        if width <= 0 or depth < 2 or feature_dim <= 0:
            raise ValueError("side width/feature_dim must be positive and depth must be at least 2")
        layers: list[nn.Module] = []
        in_channels = 3
        channels = width
        for index in range(depth):
            stride = 2 if index < depth - 1 else 1
            layers.extend(
                [
                    nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = channels
            channels = min(channels * 2, width * 4)
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(in_channels, feature_dim)
        self.num_features = feature_dim
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.pool(self.features(x)).flatten(1))


class SideTuningClassifier(nn.Module):
    def __init__(
        self,
        base_model: nn.Module | None = None,
        num_classes: int = 1000,
        learn_alpha: bool = True,
        alpha_init: float = 0.5,
        side_arch: str = "lightweight",
        side_width: int = 64,
        side_depth: int = 4,
        side_checkpoint: str = "",
        **legacy,
    ) -> None:
        super().__init__()
        if base_model is None:
            base_model = legacy.pop("base_backbone", None)
        if base_model is None:
            raise TypeError("SideTuningClassifier requires base_model")
        required = ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc")
        if not all(hasattr(base_model, name) for name in required):
            raise TypeError("This Side-Tuning classification implementation supports ResNet backbones")

        feature_dim = int(base_model.fc.in_features)
        self.base = base_model
        self.base.fc = nn.Identity()
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()

        side_arch = str(side_arch).lower()
        if side_arch == "lightweight":
            self.side = ConvSideNetwork(feature_dim, width=side_width, depth=side_depth)
        elif side_arch == "copy":
            # Explicit high-capacity ablation, not the default paper baseline.
            self.side = copy.deepcopy(base_model)
            self.side.fc = nn.Identity()
            for parameter in self.side.parameters():
                parameter.requires_grad_(True)
        else:
            raise ValueError("side_arch must be 'lightweight' or 'copy'")
        self.side_arch = side_arch

        if side_checkpoint:
            checkpoint = torch.load(Path(side_checkpoint), map_location="cpu", weights_only=False)
            state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            incompat = self.side.load_state_dict(state, strict=False)
            if incompat.unexpected_keys:
                raise RuntimeError(f"Unexpected Side-Tuning checkpoint keys: {incompat.unexpected_keys[:5]}")

        alpha_init = min(max(float(alpha_init), 1e-6), 1.0 - 1e-6)
        value = torch.tensor(alpha_init, dtype=torch.float32)
        self.alpha_logit = nn.Parameter(
            torch.log(value / (1.0 - value)), requires_grad=bool(learn_alpha)
        )
        self.head = nn.Linear(feature_dim, int(num_classes))
        self.num_features = feature_dim
        self.backbone_family = "resnet"
        self.is_side_tuning = True

    @property
    def alpha(self) -> torch.Tensor:
        """Weight assigned to the frozen base, matching paper notation."""
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
        if side_features.shape != base_features.shape:
            raise RuntimeError(
                f"Base/side features must match, got {tuple(base_features.shape)} and {tuple(side_features.shape)}"
            )
        alpha = self.alpha
        return alpha * base_features + (1.0 - alpha) * side_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


__all__ = ["ConvSideNetwork", "SideTuningClassifier"]
