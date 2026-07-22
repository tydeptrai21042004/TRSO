"""LoRA reproduction for Transformer attention Q/V projections.

The implementation follows loralib semantics: frozen pretrained weights,
Kaiming-initialized A, zero-initialized B, alpha/r scaling, optional input
dropout, and merge/unmerge of the low-rank update in evaluation/training modes.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _LoRAMixin:
    merged: bool
    merge_weights: bool

    def _set_merged(self, mode: bool) -> None:
        raise NotImplementedError

    def train(self, mode: bool = True):
        super().train(mode)
        if self.merge_weights:
            self._set_merged(not mode)
        return self


class LoRALinear(_LoRAMixin, nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        merge_weights: bool = True,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.lora_b = nn.Parameter(torch.empty(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.merge_weights = bool(merge_weights)
        self.merged = False
        self.is_lora_transformer = True

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def delta_weight(self) -> torch.Tensor:
        return (self.lora_b @ self.lora_a) * self.scaling

    @torch.no_grad()
    def _set_merged(self, mode: bool) -> None:
        if mode and not self.merged:
            self.base.weight.add_(self.delta_weight())
            self.merged = True
        elif not mode and self.merged:
            self.base.weight.sub_(self.delta_weight())
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        if not self.merged:
            result = result + F.linear(F.linear(self.lora_dropout(x), self.lora_a), self.lora_b) * self.scaling
        return result


class LoRAQKVLinear(_LoRAMixin, nn.Module):
    """LoRA on only Q and V slices of one packed qkv projection."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        merge_weights: bool = True,
    ) -> None:
        super().__init__()
        if base.out_features % 3:
            raise ValueError("Packed qkv output dimension must be divisible by three")
        self.base = base
        self.embed_dim = base.out_features // 3
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.q_a = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.q_b = nn.Parameter(torch.empty(self.embed_dim, self.rank))
        self.v_a = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.v_b = nn.Parameter(torch.empty(self.embed_dim, self.rank))
        nn.init.kaiming_uniform_(self.q_a, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v_a, a=math.sqrt(5))
        nn.init.zeros_(self.q_b)
        nn.init.zeros_(self.v_b)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.merge_weights = bool(merge_weights)
        self.merged = False
        self.is_lora_transformer = True

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight if self.merged else self.base.weight + self.delta_weight()

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def delta_weight(self) -> torch.Tensor:
        q = (self.q_b @ self.q_a) * self.scaling
        v = (self.v_b @ self.v_a) * self.scaling
        return torch.cat((q, torch.zeros_like(q), v), dim=0)

    @torch.no_grad()
    def _set_merged(self, mode: bool) -> None:
        if mode and not self.merged:
            self.base.weight.add_(self.delta_weight())
            self.merged = True
        elif not mode and self.merged:
            self.base.weight.sub_(self.delta_weight())
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        if not self.merged:
            dropped = self.lora_dropout(x)
            q = F.linear(F.linear(dropped, self.q_a), self.q_b)
            v = F.linear(F.linear(dropped, self.v_a), self.v_b)
            result = result + torch.cat((q, torch.zeros_like(q), v), dim=-1) * self.scaling
        return result


class LoRAMultiheadAttention(_LoRAMixin, nn.Module):
    """Q/V LoRA for torchvision ``nn.MultiheadAttention``."""

    def __init__(
        self,
        base: nn.MultiheadAttention,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        merge_weights: bool = True,
    ) -> None:
        super().__init__()
        if base.in_proj_weight is None:
            raise ValueError("Packed in_proj_weight is required")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        dim = int(base.embed_dim)
        self.q_a = nn.Parameter(torch.empty(self.rank, dim))
        self.q_b = nn.Parameter(torch.empty(dim, self.rank))
        self.v_a = nn.Parameter(torch.empty(self.rank, dim))
        self.v_b = nn.Parameter(torch.empty(dim, self.rank))
        nn.init.kaiming_uniform_(self.q_a, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v_a, a=math.sqrt(5))
        nn.init.zeros_(self.q_b)
        nn.init.zeros_(self.v_b)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.merge_weights = bool(merge_weights)
        self.merged = False
        self.is_lora_transformer = True

    def delta_weight(self) -> torch.Tensor:
        q = (self.q_b @ self.q_a) * self.scaling
        v = (self.v_b @ self.v_a) * self.scaling
        return torch.cat((q, torch.zeros_like(q), v), dim=0)

    @torch.no_grad()
    def _set_merged(self, mode: bool) -> None:
        if mode and not self.merged:
            self.base.in_proj_weight.add_(self.delta_weight())
            self.merged = True
        elif not mode and self.merged:
            self.base.in_proj_weight.sub_(self.delta_weight())
            self.merged = False

    def _weight(self) -> torch.Tensor:
        return self.base.in_proj_weight if self.merged else self.base.in_proj_weight + self.delta_weight()

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
        # loralib dropout acts on the low-rank branch. For self-attention the
        # equivalent weight update cannot encode stochastic dropout, so when
        # dropout is active compute explicit Q/V deltas on query/value.
        weight = self._weight()
        if self.training and not isinstance(self.lora_dropout, nn.Identity):
            # MultiheadAttention's packed functional path cannot inject only
            # stochastic Q/V deltas cleanly. Original LoRA defaults to zero
            # dropout; reject a nonzero value for this wrapper at runtime.
            raise RuntimeError("LoRA dropout must be 0 for packed nn.MultiheadAttention")
        is_batched = query.dim() == 3
        if base.batch_first and is_batched:
            query, key, value = (tensor.transpose(0, 1) for tensor in (query, key, value))
        output, weights = F.multi_head_attention_forward(
            query, key, value,
            base.embed_dim, base.num_heads,
            weight, base.in_proj_bias,
            base.bias_k, base.bias_v,
            base.add_zero_attn, base.dropout,
            base.out_proj.weight, base.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=False,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )
        if base.batch_first and is_batched:
            output = output.transpose(0, 1)
        return output, weights


def apply_lora_transformer(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    merge_weights: bool = True,
) -> int:
    replacements: List[Tuple[nn.Module, str, nn.Module]] = []
    for parent in list(model.modules()):
        if isinstance(parent, (LoRALinear, LoRAQKVLinear, LoRAMultiheadAttention)):
            continue
        for name, child in list(parent.named_children()):
            lower = name.lower()
            if isinstance(child, nn.MultiheadAttention):
                replacements.append((parent, name, LoRAMultiheadAttention(child, rank, alpha, dropout, merge_weights)))
            elif isinstance(child, nn.Linear) and lower in {"qkv", "query_key_value"} and child.out_features % 3 == 0:
                replacements.append((parent, name, LoRAQKVLinear(child, rank, alpha, dropout, merge_weights)))
            elif isinstance(child, nn.Linear) and lower in {"q_proj", "v_proj", "query", "value"}:
                replacements.append((parent, name, LoRALinear(child, rank, alpha, dropout, merge_weights)))
    for parent, name, replacement in replacements:
        setattr(parent, name, replacement)
    if not replacements:
        raise RuntimeError("No Q/V Transformer attention projection was found for LoRA")
    return len(replacements)


def mark_only_lora_as_trainable(model: nn.Module, train_bias: str = "none") -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(any(token in name for token in ("lora_a", "lora_b", "q_a", "q_b", "v_a", "v_b")))
    if train_bias == "all":
        for name, parameter in model.named_parameters():
            if name.endswith(".bias"):
                parameter.requires_grad_(True)
    elif train_bias == "lora_only":
        for module in model.modules():
            if isinstance(module, (LoRALinear, LoRAQKVLinear, LoRAMultiheadAttention)) and module.base.bias is not None:
                module.base.bias.requires_grad_(True)


__all__ = [
    "LoRALinear", "LoRAQKVLinear", "LoRAMultiheadAttention",
    "apply_lora_transformer", "mark_only_lora_as_trainable",
]
