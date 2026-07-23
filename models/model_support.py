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
ALL_TASKS = frozenset({"single_label", "multilabel", "regression"})
CLASSIFICATION_TASKS = frozenset({"single_label", "multilabel"})
CNN_FAMILIES = frozenset({"resnet", "cnn", "clip_cnn"})
TRANSFORMER_FAMILIES = frozenset({"vit", "swin", "transformer", "clip_transformer"})


@dataclass(frozen=True)
class MethodSupport:
    families: FrozenSet[str]
    paper_scope: str
    implementation_scope: str
    tasks: FrozenSet[str] = ALL_TASKS
    task_scope: str = "Task-head agnostic implementation."


METHOD_SUPPORT: Dict[str, MethodSupport] = {
    "full": MethodSupport(ALL_FAMILIES, "Architecture-agnostic optimization baseline.", "Any supported classifier/regressor."),
    "linear": MethodSupport(ALL_FAMILIES, "Architecture-agnostic frozen-feature baseline.", "Any backbone with a replaceable task head."),
    "prompt": MethodSupport(
        frozenset({"resnet", "cnn", "vit", "swin", "transformer"}),
        "Visual prompting learns an image-space border while the pretrained classifier remains frozen.",
        "Single-label classification with an unchanged pretrained output head and fixed label mapping.",
        tasks=frozenset({"single_label"}),
        task_scope="The source-to-target label mapping is defined only for single-label classification."
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
    "adaptformer": MethodSupport(
        frozenset({"vit"}),
        "AdaptFormer adds a parallel bottleneck branch beside each frozen ViT MLP.",
        "Recognized timm/torchvision Vision Transformer blocks; Swin is rejected rather than approximated."
    ),
    "piggyback": MethodSupport(
        frozenset({"resnet", "cnn"}),
        "Piggyback learns binary masks over frozen pretrained CNN weights.",
        "CNN Conv2d weights with a trainable downstream classifier; deployment storage is reported in bits."
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
        "adapt_former": "adaptformer",
        "piggy_back": "piggyback",
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



def normalize_task(task: str) -> str:
    value = str(task or "auto").strip().lower().replace("-", "_")
    aliases = {
        "classification": "single_label",
        "singlelabel": "single_label",
        "multi_label": "multilabel",
        "multi_label_classification": "multilabel",
    }
    return aliases.get(value, value)


def validate_method_task(method: str, task: str, *, allow_auto: bool = True) -> None:
    method = canonical_method(method)
    task = normalize_task(task)
    if allow_auto and task == "auto":
        return
    if method not in METHOD_SUPPORT:
        raise ValueError(f"Unknown tuning method '{method}'.")
    support = METHOD_SUPPORT[method]
    if task not in support.tasks:
        allowed = ", ".join(sorted(support.tasks))
        raise ValueError(
            f"Method '{method}' is not supported for task '{task}'. "
            f"Allowed tasks: {allowed}. Task scope: {support.task_scope}"
        )


def method_compatibility(method: str, family: str, task: str) -> tuple[bool, str]:
    """Return a non-throwing capability decision for experiment planners."""
    try:
        validate_method_backbone(method, family)
        validate_method_task(method, task)
    except ValueError as exc:
        return False, str(exc)
    return True, "supported"


def infer_backbone_family_name(backbone_name: str, source: str = "") -> str:
    """Conservative family inference without constructing/downloading a model."""
    name = str(backbone_name or "").lower()
    text = f"{name} {str(source or '').lower()}"
    if "clip" in text:
        return "clip_transformer" if any(token in text for token in ("vit", "transformer")) else "clip_cnn"
    if "swin" in text:
        return "swin"
    if any(token in text for token in ("vit", "deit", "beit", "eva", "cait", "vision_transformer")):
        return "vit"
    if any(token in text for token in ("maxvit", "transformer")):
        return "transformer"
    if "resnet" in text or "resnext" in text or "wide_resnet" in text:
        return "resnet"
    if any(token in text for token in (
        "convnext", "efficientnet", "mobilenet", "densenet", "regnet", "vgg",
        "alexnet", "mnasnet", "shufflenet", "squeezenet", "inception",
        "googlenet", "nasnet", "xception", "rexnet", "coatnet",
    )):
        return "cnn"
    if any(token in text for token in ("mixer", "resmlp", "gmlp")):
        return "mlp"
    return "unknown"


def static_method_compatibility(
    method: str, backbone_name: str, task: str, *, source: str = "auto",
    residual_checkpoint: str = "",
) -> tuple[bool, str, str]:
    """Plan-time compatibility including paper-specific backbone contracts."""
    method = canonical_method(method)
    family = infer_backbone_family_name(backbone_name, source)
    if method == "residual":
        family = "resnet"
    ok, reason = method_compatibility(method, family, task)
    if not ok:
        return False, reason, family
    normalized = str(backbone_name or "").lower().replace("-", "_")
    if method in {"conv", "adapter", "bam"} and "resnet50" not in normalized:
        return False, f"{method} strict reproduction requires ResNet-50, got {backbone_name!r}.", family
    if method == "residual":
        if normalized not in {"resnet26_adapter", "resnet26", "residual_adapter_resnet26"}:
            return False, "Residual Adapter uses the dedicated ResNet-26; add resnet26_adapter@auto as a separate backbone group.", family
        if not residual_checkpoint:
            return False, "Residual Adapter requires --ra_pretrained_checkpoint for a transfer-learning comparison.", family
    if method == "adaptformer" and family != "vit":
        return False, "AdaptFormer requires a recognized ViT backbone.", family
    return True, "supported", family

def compatibility_rows():
    for method, support in METHOD_SUPPORT.items():
        yield {
            "method": method,
            "families": sorted(support.families),
            "paper_scope": support.paper_scope,
            "implementation_scope": support.implementation_scope,
            "tasks": sorted(support.tasks),
            "task_scope": support.task_scope,
        }


__all__ = [
    "ALL_FAMILIES",
    "ALL_TASKS",
    "CLASSIFICATION_TASKS",
    "CNN_FAMILIES",
    "TRANSFORMER_FAMILIES",
    "METHOD_SUPPORT",
    "canonical_method",
    "detect_backbone_family",
    "infer_backbone_family_name",
    "static_method_compatibility",
    "validate_method_backbone",
    "validate_method_task",
    "method_compatibility",
    "normalize_task",
    "compatibility_rows",
]
