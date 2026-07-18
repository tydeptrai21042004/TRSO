# models/tuning_modules/side_tuning.py
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class _SideConvStem(nn.Module):
    def __init__(self, in_ch: int, mid: int, out_ch: int, depth: int = 3):
        super().__init__()
        layers = [nn.Conv2d(in_ch, mid, 3, padding=1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True)]
        for _ in range(max(0, int(depth) - 2)):
            layers += [nn.Conv2d(mid, mid, 3, padding=1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(mid, out_ch, 1, bias=False)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SideTuningClassifier(nn.Module):
    """Side-tuning baseline with frozen backbone + trainable side network.

    Robust to torchvision ResNet-style backbones that expose ``fc.in_features``
    but not ``num_features``.
    """

    def __init__(
        self,
        base_model: nn.Module = None,
        num_classes: int = 1000,
        side_width: int = 64,
        side_depth: int = 3,
        learn_alpha: bool = True,
        alpha_init: float = 0.5,
        use_checkpoint: bool = False,
        **legacy,
    ):
        super().__init__()
        if base_model is None:
            base_model = legacy.pop("base_backbone", None)
        if base_model is None:
            raise TypeError("SideTuningClassifier requires base_model or legacy base_backbone.")

        self.base = base_model
        self.use_checkpoint = bool(use_checkpoint)
        for p in self.base.parameters():
            p.requires_grad = False

        channels = self._infer_feature_dim(self.base)
        self.num_features = channels
        self.side_net = _SideConvStem(3, side_width, channels, depth=side_depth)

        a0 = min(max(float(alpha_init), 1e-4), 1.0 - 1e-4)
        a0_t = torch.tensor(a0)
        self.alpha_logit = nn.Parameter(torch.log(a0_t / (1.0 - a0_t)), requires_grad=learn_alpha)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(channels, num_classes)

    @staticmethod
    def _infer_feature_dim(base: nn.Module) -> int:
        for attr in ("num_features", "feature_dim", "embed_dim"):
            val = getattr(base, attr, None)
            if isinstance(val, int) and val > 0:
                return val
        if hasattr(base, "fc") and isinstance(base.fc, nn.Linear):
            return int(base.fc.in_features)
        if hasattr(base, "classifier"):
            clf = base.classifier
            if isinstance(clf, nn.Linear):
                return int(clf.in_features)
            if isinstance(clf, nn.Sequential):
                for m in reversed(clf):
                    if isinstance(m, nn.Linear):
                        return int(m.in_features)
        if hasattr(base, "head") and isinstance(base.head, nn.Linear):
            return int(base.head.in_features)
        raise RuntimeError("Cannot infer base feature dimension for side-tuning.")

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_logit)

    def _base_feats_resnet(self, x: torch.Tensor) -> torch.Tensor:
        b = self.base
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)
        x = b.layer1(x)
        x = b.layer2(x)
        x = b.layer3(x)
        x = b.layer4(x)
        x = b.avgpool(x)
        return torch.flatten(x, 1)

    def _base_feats(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if all(hasattr(self.base, a) for a in ("conv1", "bn1", "relu", "layer1", "layer2", "layer3", "layer4", "avgpool")):
                out = self._base_feats_resnet(x)
            elif hasattr(self.base, "forward_features"):
                out = self.base.forward_features(x)
            else:
                out = self.base(x)
        if out.dim() == 4:
            out = self.pool(out).flatten(1)
        return out

    def _side_feats(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.use_checkpoint:
            side = checkpoint(self.side_net, x)
        else:
            side = self.side_net(x)
        return self.pool(side).flatten(1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        base_vec = self._base_feats(x)
        side_vec = self._side_feats(x)
        a = self.alpha
        return (1.0 - a) * base_vec + a * side_vec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


__all__ = ["SideTuningClassifier"]
