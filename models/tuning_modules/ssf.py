"""Scaling & Shifting Your Features (SSF), architecture-specific reproduction.

The published method inserts a learned channel-wise affine transformation after
feature-producing operations throughout the frozen backbone. This module
provides exact insertion maps for torchvision VisionTransformer, SwinTransformer
and ConvNeXt implementations instead of applying one transform per block.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_ssf(scale: nn.Parameter, shift: nn.Parameter, std: float = 0.02) -> None:
    nn.init.normal_(scale, mean=1.0, std=float(std))
    nn.init.normal_(shift, mean=0.0, std=float(std))


def _feature_dim(module: nn.Module) -> int:
    if isinstance(module, nn.Linear):
        return int(module.out_features)
    if isinstance(module, nn.Conv2d):
        return int(module.out_channels)
    if isinstance(module, (nn.LayerNorm,)):
        shape = module.normalized_shape
        return int(shape[-1] if isinstance(shape, (tuple, list)) else shape)
    # torchvision LayerNorm2d derives from LayerNorm but keep fallback.
    shape = getattr(module, "normalized_shape", None)
    if shape is not None:
        return int(shape[-1] if isinstance(shape, (tuple, list)) else shape)
    raise TypeError(f"Cannot infer SSF feature dimension for {type(module).__name__}")


def ssf_ada(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Apply SSF to last-dimension tokens or channel-first feature maps."""
    if x.shape[-1] == scale.numel():
        view = [1] * (x.ndim - 1) + [scale.numel()]
        return x * scale.view(*view) + shift.view(*view)
    if x.ndim >= 2 and x.shape[1] == scale.numel():
        view = [1, scale.numel()] + [1] * (x.ndim - 2)
        return x * scale.view(*view) + shift.view(*view)
    raise RuntimeError(
        f"SSF feature dimension {scale.numel()} matches neither last nor channel axis of {tuple(x.shape)}"
    )


class SSFPost(nn.Module):
    """Wrap one frozen operation and apply its published post-operation SSF."""

    def __init__(self, base: nn.Module, init_std: float = 0.02) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        dim = _feature_dim(base)
        self.ssf_scale = nn.Parameter(torch.empty(dim))
        self.ssf_shift = nn.Parameter(torch.empty(dim))
        _init_ssf(self.ssf_scale, self.ssf_shift, init_std)
        self.is_ssf = True

    @property
    def weight(self):
        weight = getattr(self.base, "weight", None)
        if weight is None:
            raise AttributeError("wrapped module has no weight")
        if isinstance(self.base, nn.Linear):
            return weight * self.ssf_scale[:, None]
        if isinstance(self.base, nn.Conv2d):
            return weight * self.ssf_scale[:, None, None, None]
        if isinstance(self.base, nn.LayerNorm) or hasattr(self.base, "normalized_shape"):
            return weight * self.ssf_scale
        return weight

    @property
    def bias(self):
        bias = getattr(self.base, "bias", None)
        if bias is None:
            return self.ssf_shift
        return bias * self.ssf_scale + self.ssf_shift

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, *args, **kwargs):
        output = self.base(*args, **kwargs)
        if isinstance(output, tuple):
            first = ssf_ada(output[0], self.ssf_scale, self.ssf_shift)
            return (first, *output[1:])
        return ssf_ada(output, self.ssf_scale, self.ssf_shift)


class SSFMultiheadAttention(nn.Module):
    """torchvision ViT attention with SSF after packed QKV and projection."""

    def __init__(self, base: nn.MultiheadAttention, init_std: float = 0.02) -> None:
        super().__init__()
        if base.in_proj_weight is None:
            raise TypeError("SSFMultiheadAttention requires packed QKV weights")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        embed_dim = int(base.embed_dim)
        self.qkv_scale = nn.Parameter(torch.empty(3 * embed_dim))
        self.qkv_shift = nn.Parameter(torch.empty(3 * embed_dim))
        self.proj_scale = nn.Parameter(torch.empty(embed_dim))
        self.proj_shift = nn.Parameter(torch.empty(embed_dim))
        _init_ssf(self.qkv_scale, self.qkv_shift, init_std)
        _init_ssf(self.proj_scale, self.proj_shift, init_std)
        self.is_ssf = True

    def _effective_qkv(self):
        scale = self.qkv_scale[:, None]
        weight = self.base.in_proj_weight * scale
        bias = self.base.in_proj_bias
        if bias is None:
            bias = self.qkv_shift
        else:
            bias = bias * self.qkv_scale + self.qkv_shift
        return weight, bias

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask=None,
        need_weights: bool = True,
        attn_mask=None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):
        base = self.base
        qkv_weight, qkv_bias = self._effective_qkv()
        is_batched = query.dim() == 3
        if base.batch_first and is_batched:
            query, key, value = (tensor.transpose(0, 1) for tensor in (query, key, value))
        output, weights = F.multi_head_attention_forward(
            query,
            key,
            value,
            base.embed_dim,
            base.num_heads,
            qkv_weight,
            qkv_bias,
            base.bias_k,
            base.bias_v,
            base.add_zero_attn,
            base.dropout,
            base.out_proj.weight,
            base.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=False,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )
        output = ssf_ada(output, self.proj_scale, self.proj_shift)
        if base.batch_first and is_batched:
            output = output.transpose(0, 1)
        return output, weights


def _wrap(parent: nn.Module, name: str, init_std: float, records: List[str], prefix: str) -> None:
    child = getattr(parent, name)
    if isinstance(child, (SSFPost, SSFMultiheadAttention)):
        return
    setattr(parent, name, SSFPost(child, init_std))
    records.append(f"{prefix}.{name}" if prefix else name)


def _inject_vit(model: nn.Module, init_std: float) -> List[str]:
    records: List[str] = []
    _wrap(model, "conv_proj", init_std, records, "")
    encoder = model.encoder
    for layer_name, layer in encoder.layers.named_children():
        _wrap(layer, "ln_1", init_std, records, f"encoder.layers.{layer_name}")
        layer.self_attention = SSFMultiheadAttention(layer.self_attention, init_std)
        records.append(f"encoder.layers.{layer_name}.self_attention[qkv,proj]")
        _wrap(layer, "ln_2", init_std, records, f"encoder.layers.{layer_name}")
        layer.mlp[0] = SSFPost(layer.mlp[0], init_std)
        layer.mlp[3] = SSFPost(layer.mlp[3], init_std)
        records.extend((f"encoder.layers.{layer_name}.mlp.0", f"encoder.layers.{layer_name}.mlp.3"))
    _wrap(encoder, "ln", init_std, records, "encoder")
    return records


def _inject_swin(model: nn.Module, init_std: float) -> List[str]:
    records: List[str] = []
    # Patch embedding Conv2d and norm.
    model.features[0][0] = SSFPost(model.features[0][0], init_std)
    model.features[0][2] = SSFPost(model.features[0][2], init_std)
    records.extend(("features.0.0", "features.0.2"))
    for name, module in model.named_modules():
        if name == "":
            continue
        if hasattr(module, "attn") and hasattr(module, "norm1") and hasattr(module, "norm2") and hasattr(module, "mlp"):
            if not isinstance(module.norm1, SSFPost):
                module.norm1 = SSFPost(module.norm1, init_std)
                module.attn.qkv = SSFPost(module.attn.qkv, init_std)
                module.attn.proj = SSFPost(module.attn.proj, init_std)
                module.norm2 = SSFPost(module.norm2, init_std)
                module.mlp[0] = SSFPost(module.mlp[0], init_std)
                module.mlp[3] = SSFPost(module.mlp[3], init_std)
                records.extend(
                    (
                        f"{name}.norm1", f"{name}.attn.qkv", f"{name}.attn.proj",
                        f"{name}.norm2", f"{name}.mlp.0", f"{name}.mlp.3",
                    )
                )
        if hasattr(module, "reduction") and hasattr(module, "norm"):
            if isinstance(module.reduction, nn.Linear) and not isinstance(module.reduction, SSFPost):
                module.norm = SSFPost(module.norm, init_std)
                module.reduction = SSFPost(module.reduction, init_std)
                records.extend((f"{name}.norm", f"{name}.reduction"))
    if hasattr(model, "norm") and not isinstance(model.norm, SSFPost):
        model.norm = SSFPost(model.norm, init_std)
        records.append("norm")
    return records


def _inject_convnext(model: nn.Module, init_std: float) -> List[str]:
    records: List[str] = []
    # Official SSF insertion follows stem/downsample operations and each
    # ConvNeXt block's depthwise conv, normalization, and two MLP linears.
    for name, module in list(model.named_modules()):
        if name == "":
            continue
        if hasattr(module, "block") and isinstance(module.block, nn.Sequential) and len(module.block) >= 6:
            if not isinstance(module.block[0], SSFPost):
                for index in (0, 2, 3, 5):
                    module.block[index] = SSFPost(module.block[index], init_std)
                    records.append(f"{name}.block.{index}")
    # Stem and downsample Sequential pairs.
    for index, stage in enumerate(model.features):
        if isinstance(stage, nn.Sequential) and len(stage) == 2:
            if isinstance(stage[0], (nn.Conv2d, nn.LayerNorm)) and not isinstance(stage[0], SSFPost):
                stage[0] = SSFPost(stage[0], init_std)
                records.append(f"features.{index}.0")
            if isinstance(stage[1], (nn.Conv2d, nn.LayerNorm)) and not isinstance(stage[1], SSFPost):
                stage[1] = SSFPost(stage[1], init_std)
                records.append(f"features.{index}.1")
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        if isinstance(model.classifier[0], nn.LayerNorm) and not isinstance(model.classifier[0], SSFPost):
            model.classifier[0] = SSFPost(model.classifier[0], init_std)
            records.append("classifier.0")
    return records


def apply_ssf(model: nn.Module, init_std: float = 0.02) -> List[str]:
    """Insert SSF at the paper-specified operations for supported backbones."""
    text = f"{model.__class__.__module__}.{model.__class__.__name__}".lower()
    if "vision_transformer" in text or model.__class__.__name__.lower() == "visiontransformer":
        records = _inject_vit(model, init_std)
    elif "swin" in text:
        records = _inject_swin(model, init_std)
    elif "convnext" in text:
        records = _inject_convnext(model, init_std)
    else:
        raise TypeError(
            "Strict SSF reproduction supports torchvision VisionTransformer, "
            "SwinTransformer, and ConvNeXt architectures only."
        )
    if not records:
        raise RuntimeError("SSF insertion produced no adapted operations")
    return records


def set_ssf_trainability(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, (SSFPost, SSFMultiheadAttention)):
            for name, parameter in module.named_parameters(recurse=False):
                if "ssf" in name or name in {"qkv_scale", "qkv_shift", "proj_scale", "proj_shift"}:
                    parameter.requires_grad_(True)
    # Downstream classifier follows the paper's transfer-learning protocol.
    for name, parameter in model.named_parameters():
        if name.startswith(("head.", "heads.", "classifier.")) or ".head." in name:
            parameter.requires_grad_(True)


@torch.no_grad()
def _merge_affine_post(wrapper: SSFPost) -> nn.Module:
    """Fold one SSFPost transform into its frozen affine base operation."""
    base = wrapper.base
    scale = wrapper.ssf_scale.detach()
    shift = wrapper.ssf_shift.detach()
    if isinstance(base, nn.Linear):
        base.weight.mul_(scale[:, None])
    elif isinstance(base, nn.Conv2d):
        base.weight.mul_(scale[:, None, None, None])
    elif isinstance(base, nn.LayerNorm) or hasattr(base, "normalized_shape"):
        if getattr(base, "weight", None) is None:
            base.weight = nn.Parameter(scale.clone(), requires_grad=False)
        else:
            base.weight.mul_(scale)
    else:
        raise TypeError(f"Cannot merge SSF into {type(base).__name__}")

    bias = getattr(base, "bias", None)
    if bias is None:
        base.bias = nn.Parameter(shift.clone(), requires_grad=False)
    else:
        bias.mul_(scale).add_(shift)
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    return base


@torch.no_grad()
def _merge_attention(wrapper: SSFMultiheadAttention) -> nn.MultiheadAttention:
    base = wrapper.base
    base.in_proj_weight.mul_(wrapper.qkv_scale.detach()[:, None])
    if base.in_proj_bias is None:
        base.in_proj_bias = nn.Parameter(wrapper.qkv_shift.detach().clone(), requires_grad=False)
    else:
        base.in_proj_bias.mul_(wrapper.qkv_scale.detach()).add_(wrapper.qkv_shift.detach())
    base.out_proj.weight.mul_(wrapper.proj_scale.detach()[:, None])
    if base.out_proj.bias is None:
        base.out_proj.bias = nn.Parameter(wrapper.proj_shift.detach().clone(), requires_grad=False)
    else:
        base.out_proj.bias.mul_(wrapper.proj_scale.detach()).add_(wrapper.proj_shift.detach())
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    return base


def merge_ssf_(model: nn.Module) -> List[str]:
    """Merge all SSF transforms in-place and remove their wrappers.

    The returned model has ordinary Linear/Conv/LayerNorm/MHA modules and is
    numerically equivalent in evaluation mode. The list contains merged paths.
    """
    records: List[str] = []

    def visit(parent: nn.Module, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, SSFPost):
                setattr(parent, name, _merge_affine_post(child))
                records.append(path)
            elif isinstance(child, SSFMultiheadAttention):
                setattr(parent, name, _merge_attention(child))
                records.append(path)
            else:
                visit(child, path)

    visit(model)
    return records


# Legacy name retained for tests/imports; this is an individual SSF transform,
# not the complete baseline. New runs use apply_ssf().
class SSF(nn.Module):
    def __init__(self, C: int, init_scale: float = 1.0, init_shift: float = 0.0, init_std: float = 0.02):
        super().__init__()
        self.scale = nn.Parameter(torch.empty(int(C)))
        self.shift = nn.Parameter(torch.empty(int(C)))
        if init_std > 0:
            nn.init.normal_(self.scale, mean=float(init_scale), std=float(init_std))
            nn.init.normal_(self.shift, mean=float(init_shift), std=float(init_std))
        else:
            nn.init.constant_(self.scale, float(init_scale))
            nn.init.constant_(self.shift, float(init_shift))
        self.is_ssf = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ssf_ada(x, self.scale, self.shift)


__all__ = [
    "SSF", "SSFPost", "SSFMultiheadAttention", "ssf_ada", "apply_ssf", "set_ssf_trainability", "merge_ssf_"
]
