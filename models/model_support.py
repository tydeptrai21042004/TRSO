"""Backbone-family detection and baseline compatibility contracts.

The compatibility table is deliberately conservative. A baseline is not
silently rewritten to operate on a different representation family. Unsupported
method/backbone pairs fail before training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, Optional

import torch.nn as nn


ALL_FAMILIES = frozenset({"resnet", "cnn", "vit", "swin", "transformer", "mlp", "clip_cnn", "clip_transformer"})
CNN_FAMILIES = frozenset({"resnet", "cnn", "clip_cnn"})
TRANSFORMER_FAMILIES = frozenset({"vit", "swin", "transformer", "clip_transformer"})


@dataclass(frozen=True)
class MethodSupport:
    families: FrozenSet[str]
    paper_scope: str
    implementation_scope: str


METHOD_SUPPORT: Dict[str, MethodSupport] = {
    "full": MethodSupport(ALL_FAMILIES, "Architecture-agnostic optimization baseline.", "Any supported classifier/regressor."),
    "linear": MethodSupport(ALL_FAMILIES, "Architecture-agnostic frozen-feature baseline.", "Any backbone with a replaceable task head."),
    "prompt": MethodSupport(
        frozenset({"resnet", "cnn", "vit", "swin", "transformer"}),
        "Visual prompting learns an image-space border while the pretrained classifier remains frozen.",
        "Single-label classification with an unchanged pretrained output head and fixed label mapping."
    ),
    "conv": MethodSupport(
        frozenset({"resnet"}),
        "Conv-Adapter is reproduced in ResNet-50 Bottleneck blocks.",
        "ResNet-50 only; four paper insertion schemes are available."
    ),
    "adapter": MethodSupport(
        frozenset({"resnet"}),
        "Alias of the strict Conv-Adapter reproduction.",
        "ResNet-50 only; do not report conv and adapter as separate rows."
    ),
    "trso": MethodSupport(
        frozenset({"resnet", "cnn", "vit", "swin", "transformer"}),
        "The proposal defines explicit BCHW, BNC, and BHWC layouts.",
        "Recognized torchvision/timm CNN, ViT, and Swin blocks."
    ),
    "bam": MethodSupport(
        frozenset({"resnet"}),
        "BAM is an end-to-end ResNet-50 architectural attention module.",
        "ResNet-50 only, inserted at the three published stage transitions."
    ),
    "residual": MethodSupport(
        frozenset({"resnet"}),
        "Residual Adapters use the original reduced-resolution ResNet-26 with task BN.",
        "Dedicated ResNet-26 series/parallel model; arbitrary torchvision ResNets are rejected."
    ),
    "ssf": MethodSupport(
        frozenset({"cnn", "vit", "swin"}),
        "SSF inserts affine scale/shift after published internal operations.",
        "torchvision ConvNeXt, VisionTransformer, and SwinTransformer only."
    ),
    "lora": MethodSupport(
        frozenset({"vit", "swin", "transformer"}),
        "LoRA adapts Transformer query/value matrices.",
        "Recognized Transformer Q/V or packed QKV attention projections."
    ),
    "bitfit": MethodSupport(
        TRANSFORMER_FAMILIES,
        "BitFit trains Transformer bias terms and the task classifier.",
        "Transformer biases plus task head."
    ),
    "sidetune": MethodSupport(
        frozenset({"resnet"}),
        "Side-Tuning combines a frozen base with a copied trainable side network.",
        "ResNet only; side is initialized from the pretrained base."
    ),
}



def canonical_method(name: str) -> str:
    value = str(name or "").strip().lower().replace("-", "_")
    aliases = {
        "task_response": "trso",
        "task_response_adapter": "trso",
        "trso_adapter": "trso",
        "conv_adapter": "conv",
        "conv_adapt": "conv",
        "bam_adapter": "bam",
        "bam_tuning": "bam",
        "residual_adapter": "residual",
        "residual_adapters": "residual",
        "ra": "residual",
        "ssf_adapter": "ssf",
        "lora_conv2d": "lora_conv",
        "side_tuning": "sidetune",
        "sidetuning": "sidetune",
        "side_tune": "sidetune",
        "linear_probe": "linear",
        "finetune": "full",
    }
    return aliases.get(value, value)


def detect_backbone_family(model: nn.Module, backbone_name: str = "", source: str = "") -> str:
    """Infer a conservative representation family from type/name metadata."""
    name = str(backbone_name or "").lower()
    cls_name = model.__class__.__name__.lower()
    module_name = model.__class__.__module__.lower()
    text = " ".join((name, cls_name, module_name, str(source).lower()))

    if "clip" in text:
        if any(token in text for token in ("vit", "visiontransformer")):
            return "clip_transformer"
        return "clip_cnn"
    if any(token in text for token in ("swin", "shiftedwindow")):
        return "swin"
    if any(token in text for token in ("visiontransformer", "vision_transformer", "vit_", "deit", "beit", "eva", "cait")):
        return "vit"
    if any(token in text for token in ("transformer", "maxvit")):
        return "transformer"
    if "resnet" in text or all(hasattr(model, attr) for attr in ("layer1", "layer2", "layer3", "layer4")):
        return "resnet"
    if any(token in text for token in (
        "convnext", "efficientnet", "mobilenet", "densenet", "regnet", "vgg", "alexnet",
        "mnasnet", "shufflenet", "squeezenet", "inception", "googlenet", "nasnet",
    )):
        return "cnn"
    if any(token in text for token in ("mlpmixer", "mixer", "resmlp", "gmlp")):
        return "mlp"

    # Structural fallback: presence of spatial convolutions is not enough to
    # classify hybrid Transformers, so use it only after name/type checks.
    if any(isinstance(module, nn.Conv2d) for module in model.modules()):
        return "cnn"
    return "transformer" if any(isinstance(module, nn.MultiheadAttention) for module in model.modules()) else "unknown"


def validate_method_backbone(method: str, family: str) -> None:
    method = canonical_method(method)
    if method not in METHOD_SUPPORT:
        raise ValueError(f"Unknown tuning method '{method}'.")
    support = METHOD_SUPPORT[method]
    if family not in support.families:
        allowed = ", ".join(sorted(support.families))
        raise ValueError(
            f"Method '{method}' is not supported on backbone family '{family}'. "
            f"Allowed families: {allowed}. Paper scope: {support.paper_scope} "
            f"Implementation scope: {support.implementation_scope}"
        )


def compatibility_rows():
    for method, support in METHOD_SUPPORT.items():
        yield {
            "method": method,
            "families": sorted(support.families),
            "paper_scope": support.paper_scope,
            "implementation_scope": support.implementation_scope,
        }


__all__ = [
    "ALL_FAMILIES",
    "CNN_FAMILIES",
    "TRANSFORMER_FAMILIES",
    "METHOD_SUPPORT",
    "canonical_method",
    "detect_backbone_family",
    "validate_method_backbone",
    "compatibility_rows",
]
