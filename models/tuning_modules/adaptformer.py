"""AdaptFormer baseline for Vision Transformers.

This implementation follows the public NeurIPS 2022 implementation:
  x <- x + attention(norm1(x))
  x <- x + mlp(norm2(x)) + s * up(ReLU(down(x)))

The frozen Transformer block is preserved and only the parallel bottleneck
branch plus the downstream task head are trainable.  The up projection is
zero-initialized, so insertion is exactly identity-safe.
"""
from __future__ import annotations

import math
import types
from typing import List

import torch
import torch.nn as nn


class AdaptFormerAdapter(nn.Module):
    """Official-style parallel FFN adapter."""

    def __init__(
        self,
        dim: int,
        bottleneck: int = 16,
        dropout: float = 0.0,
        scale: float = 0.1,
        layernorm_option: str = "none",
    ) -> None:
        super().__init__()
        if int(dim) <= 0 or int(bottleneck) <= 0:
            raise ValueError("AdaptFormer dimensions must be positive")
        if layernorm_option not in {"none", "in", "out"}:
            raise ValueError("layernorm_option must be one of: none, in, out")
        self.dim = int(dim)
        self.bottleneck = int(bottleneck)
        self.layernorm_option = layernorm_option
        self.adapter_layer_norm = (
            nn.LayerNorm(self.dim) if layernorm_option in {"in", "out"} else None
        )
        self.down_proj = nn.Linear(self.dim, self.bottleneck)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.bottleneck, self.dim)
        self.dropout = float(dropout)
        self.scale = float(scale)
        self.is_adaptformer = True

        # Matches the released AdaptFormer "lora" initialization.
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layernorm_option == "in":
            x = self.adapter_layer_norm(x)
        x = self.down_proj(x)
        x = self.non_linear_func(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.up_proj(x) * self.scale
        if self.layernorm_option == "out":
            x = self.adapter_layer_norm(x)
        return x


def _apply_optional(module: nn.Module, name: str, x: torch.Tensor) -> torch.Tensor:
    child = getattr(module, name, None)
    return child(x) if child is not None else x


def _adaptformer_timm_forward(self, x: torch.Tensor, *args, **kwargs):
    """Forward for timm-style ViT blocks, retaining optional layer-scale/drop-path."""
    attn = self.attn(self.norm1(x))
    attn = _apply_optional(self, "ls1", attn)
    if hasattr(self, "drop_path1"):
        attn = self.drop_path1(attn)
    elif hasattr(self, "drop_path"):
        attn = self.drop_path(attn)
    x = x + attn

    adapt_x = self.adaptformer_adapter(x)
    mlp_x = self.mlp(self.norm2(x))
    mlp_x = _apply_optional(self, "ls2", mlp_x)
    if hasattr(self, "drop_path2"):
        mlp_x = self.drop_path2(mlp_x)
    elif hasattr(self, "drop_path"):
        mlp_x = self.drop_path(mlp_x)
    return x + mlp_x + adapt_x


def _adaptformer_torchvision_forward(self, input: torch.Tensor):
    """Forward for torchvision EncoderBlock."""
    torch._assert(input.dim() == 3, f"Expected (batch, seq, hidden), got {input.shape}")
    x = self.ln_1(input)
    x, _ = self.self_attention(x, x, x, need_weights=False)
    x = self.dropout(x)
    x = x + input
    adapt_x = self.adaptformer_adapter(x)
    y = self.ln_2(x)
    y = self.mlp(y)
    return x + y + adapt_x


def _block_hidden_dim(block: nn.Module) -> int:
    for norm_name in ("norm2", "ln_2"):
        norm = getattr(block, norm_name, None)
        shape = getattr(norm, "normalized_shape", None)
        if shape is not None:
            return int(shape[-1] if isinstance(shape, (tuple, list)) else shape)
    mlp = getattr(block, "mlp", None)
    if mlp is not None:
        for module in mlp.modules():
            if isinstance(module, nn.Linear):
                return int(module.in_features)
    raise TypeError(f"Cannot infer Transformer width for {type(block).__name__}")


def apply_adaptformer(
    model: nn.Module,
    bottleneck: int = 16,
    dropout: float = 0.0,
    scale: float = 0.1,
    layernorm_option: str = "none",
) -> List[str]:
    """Insert AdaptFormer in every recognized ViT block.

    Supported block contracts are the timm-style ``norm1/attn/norm2/mlp``
    block and torchvision's ``ln_1/self_attention/ln_2/mlp`` EncoderBlock.
    Swin blocks are deliberately rejected because their window/shift forward
    path is architecture-specific and should not be silently approximated.
    """
    records: List[str] = []
    for name, block in list(model.named_modules()):
        if name == "" or hasattr(block, "adaptformer_adapter"):
            continue
        is_timm_vit = all(hasattr(block, key) for key in ("norm1", "attn", "norm2", "mlp"))
        is_tv_vit = all(hasattr(block, key) for key in ("ln_1", "self_attention", "ln_2", "mlp", "dropout"))
        # Avoid Swin/window-attention blocks; AdaptFormer is registered here as a ViT baseline.
        block_text = f"{block.__class__.__module__}.{block.__class__.__name__}".lower()
        if "swin" in block_text or "window" in block_text:
            continue
        if not (is_timm_vit or is_tv_vit):
            continue
        dim = _block_hidden_dim(block)
        block.add_module(
            "adaptformer_adapter",
            AdaptFormerAdapter(
                dim=dim,
                bottleneck=bottleneck,
                dropout=dropout,
                scale=scale,
                layernorm_option=layernorm_option,
            ),
        )
        if is_tv_vit:
            block.forward = types.MethodType(_adaptformer_torchvision_forward, block)
        else:
            block.forward = types.MethodType(_adaptformer_timm_forward, block)
        records.append(name)
    if not records:
        raise TypeError(
            "AdaptFormer requires a recognized Vision Transformer with "
            "timm-style or torchvision EncoderBlock modules."
        )
    return records


def set_adaptformer_trainability(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if "adaptformer_adapter" in name:
            parameter.requires_grad_(True)
        elif name.startswith(("head.", "heads.", "classifier.")) or ".head." in name:
            parameter.requires_grad_(True)


__all__ = [
    "AdaptFormerAdapter",
    "apply_adaptformer",
    "set_adaptformer_trainability",
]
