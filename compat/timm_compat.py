"""Small fallbacks for the subset of timm used by the training pipeline.

The project still supports timm when it is installed. These fallbacks keep the
core torchvision pipeline and tests runnable in minimal environments.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Iterable, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTargetCrossEntropy(nn.Module):
    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1).mean()


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1) -> None:
        super().__init__()
        self.smoothing = float(smoothing)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(x, dim=-1)
        nll = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth = -log_probs.mean(dim=-1)
        return ((1.0 - self.smoothing) * nll + self.smoothing * smooth).mean()


class Mixup:
    """Minimal mixup/cutmix-compatible callable.

    The fallback implements mixup only. It preserves the public call contract
    used by this repository and is intentionally deterministic under PyTorch's
    RNG seed.
    """

    def __init__(
        self,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        cutmix_minmax=None,
        prob: float = 1.0,
        switch_prob: float = 0.5,
        mode: str = "batch",
        label_smoothing: float = 0.0,
        num_classes: int = 1000,
        **_: object,
    ) -> None:
        self.alpha = float(max(mixup_alpha, cutmix_alpha))
        self.prob = float(prob)
        self.label_smoothing = float(label_smoothing)
        self.num_classes = int(num_classes)

    def _one_hot(self, target: torch.Tensor) -> torch.Tensor:
        off = self.label_smoothing / max(1, self.num_classes)
        on = 1.0 - self.label_smoothing + off
        out = torch.full(
            (target.shape[0], self.num_classes),
            off,
            device=target.device,
            dtype=torch.float32,
        )
        return out.scatter_(1, target.long().view(-1, 1), on)

    def __call__(self, x: torch.Tensor, target: torch.Tensor):
        target = self._one_hot(target)
        if self.alpha <= 0 or torch.rand((), device=x.device).item() > self.prob:
            return x, target
        beta = torch.distributions.Beta(self.alpha, self.alpha)
        lam = beta.sample().to(device=x.device, dtype=x.dtype)
        perm = torch.randperm(x.shape[0], device=x.device)
        mixed_x = x * lam + x[perm] * (1.0 - lam)
        mixed_target = target * lam + target[perm] * (1.0 - lam)
        return mixed_x, mixed_target


class ModelEma:
    def __init__(self, model: nn.Module, decay: float = 0.9999, device: str = "", resume: str = "") -> None:
        self.ema = deepcopy(model).eval()
        self.decay = float(decay)
        if device:
            self.ema.to(device)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_state = model.state_dict()
        for key, ema_value in self.ema.state_dict().items():
            model_value = model_state[key].detach().to(ema_value.device)
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Sequence[int] = (1,)):
    with torch.no_grad():
        maxk = min(max(topk), output.shape[1])
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        result = []
        for k in topk:
            k = min(k, output.shape[1])
            correct_k = correct[:k].reshape(-1).float().sum(0)
            result.append(correct_k.mul_(100.0 / target.shape[0]))
        return result


def get_state_dict(model_ema: object):
    if hasattr(model_ema, "ema"):
        return model_ema.ema.state_dict()
    if hasattr(model_ema, "state_dict"):
        return model_ema.state_dict()
    raise TypeError("Object does not provide a state_dict.")
