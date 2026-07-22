# main.py — TRSO parameter-efficient fine-tuning entry point
"""
Revision-focused training script.

Main fixes for reviewer/editor comments:
1. Adds full fine-tuning and linear-probe baselines.
2. Adds Task-Response Spatial Operator (TRSO) calibration and layer selection.
3. Uses train/val/test semantics: best checkpoint is selected on validation, then
   evaluated once on test.
4. Adds profiling hooks for trainable params, total params, FLOPs, latency, and memory.
5. Logs per-epoch history and convergence summaries for mean/std aggregation.
6. Fixes side-tuning constructor and safer adapter trainability rules.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
import time
import random
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
try:
    from timm.data.mixup import Mixup
    from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
    from timm.utils import ModelEma
except Exception:
    from compat.timm_compat import Mixup, LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, ModelEma

from datasets import build_dataset
try:
    from datasets.build import (
        TASK_MULTILABEL, TASK_REGRESSION, TASK_SINGLE_LABEL,
        available_datasets, build_dataset_split,
    )
except Exception:  # backward fallback
    build_dataset_split = None
    TASK_SINGLE_LABEL, TASK_MULTILABEL, TASK_REGRESSION = "single_label", "multilabel", "regression"
    available_datasets = lambda: []

from engine import evaluate, train_one_epoch
from memory_utils import profile_memory_cost
from models import build_model
from models.model_support import (
    canonical_method, compatibility_rows, detect_backbone_family,
    validate_method_backbone,
)
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils


def str2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.lower()
        if s in ("yes", "true", "t", "y", "1"):
            return True
        if s in ("no", "false", "f", "n", "0"):
            return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args_parser():
    parser = argparse.ArgumentParser("Parameter Efficient Tuning", add_help=False)

    # Backbone
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--model_source", type=str, default="auto", choices=["auto", "torchvision", "timm", "hub"])
    parser.add_argument("--weights", type=str, default="DEFAULT")
    parser.add_argument("--list_backbones", action="store_true")
    parser.add_argument("--list_compatibility", action="store_true")
    parser.add_argument("--experiment_suite", type=str, default="", help="Manifest suite metadata.")
    parser.add_argument("--experiment_name", type=str, default="", help="Ablation or sweep variant metadata.")
    parser.add_argument("--experiment_run_id", type=str, default="", help="Deterministic manifest run identifier.")
    parser.add_argument("--legacy_auto_hparams", type=str2bool, default=False, help="Opt into the original repository hyperparameter override table.")
    parser.add_argument("--pretrained", type=str2bool, default=None)
    parser.add_argument("--keep_pretrained_head", type=str2bool, default=True)
    parser.add_argument("--cifar_hub", type=str, default="auto", choices=["auto", "chenyaofo", "akamaster"])

    # CLIP linear probe branch
    parser.add_argument("--clip_model", type=str, default=None, help="OpenAI CLIP/OpenCLIP visual backbone name, e.g. RN50 or ViT-B/16.")
    parser.add_argument("--clip_pretrained", type=str, default="openai")
    parser.add_argument("--freeze_backbone", type=str2bool, default=True)

    # Methods
    parser.add_argument(
        "--tuning_method",
        type=str,
        default="trso",
        help=(
            "full | linear | prompt | conv | adapter | trso | bam | residual | "
            "ssf | lora | bitfit | sidetune"
        ),
    )

    # Side-tuning
    parser.add_argument("--sidetune_alpha", type=float, default=0.5)
    parser.add_argument("--sidetune_learn_alpha", type=str2bool, default=True)
    parser.add_argument("--sidetune_width", type=int, default=64)
    parser.add_argument("--sidetune_depth", type=int, default=4)
    parser.add_argument("--sidetune_arch", type=str, default="lightweight", choices=["lightweight", "copy"])
    parser.add_argument("--sidetune_checkpoint", type=str, default="", help="Optional distilled/lightweight side-network checkpoint.")

    # Prompt / Conv-Adapter
    parser.add_argument("--prompt_size", default=30, type=int)
    parser.add_argument("--prompt_type", default="padding", choices=["padding", "fixed_patch", "random_patch"])
    parser.add_argument("--prompt_output_indices", default="", type=str, help="Comma-separated fixed pretrained output indices; empty uses 0..C-1.")
    parser.add_argument("--kernel_size", default=3, type=int)
    parser.add_argument("--adapt_size", default=8, type=float)
    parser.add_argument("--adapt_scale", default=1.0, type=float)
    parser.add_argument("--conv_adapter_mode", default="conv_parallel", choices=["conv_parallel", "conv_sequential", "residual_parallel", "residual_sequential"])

    # Task-Response Spatial Operator (TRSO)
    parser.add_argument("--trso_kernel_size", type=int, default=5, help="Odd support of each depthwise spatial atom.")
    parser.add_argument("--trso_spatial_rank", type=int, default=2, help="Maximum rank of the flattened C x k^2 channel-kernel bank.")
    parser.add_argument("--trso_channel_ratio", type=int, default=16, help="Legacy compatibility argument; Scientific TRSO has no channel bottleneck.")
    parser.add_argument("--trso_operator_radius", type=float, default=1.0)
    parser.add_argument("--trso_gate_init", type=float, default=1e-2)
    parser.add_argument("--trso_basis_init_scale", type=float, default=5e-2)
    parser.add_argument("--trso_basis_trainable", type=str2bool, default=False)
    parser.add_argument("--trso_calibration", type=str2bool, default=True)
    parser.add_argument("--trso_calibration_batches", type=int, default=8)
    parser.add_argument("--trso_calibration_scale", type=float, default=1.0)
    parser.add_argument("--trso_head_warmup_steps", type=int, default=0)
    parser.add_argument("--trso_keep_ratio", type=float, default=1.0)
    parser.add_argument("--trso_max_adapters", type=int, default=0, help="0 keeps all candidates allowed by keep_ratio.")
    parser.add_argument("--trso_parameter_budget", type=int, default=0, help="Exact global budget for active TRSO coefficients/gates; 0 disables the budget.")
    parser.add_argument("--trso_config", type=str, default="", help="Load a previously calibrated TRSO JSON config.")
    parser.add_argument("--trso_save_config", type=str, default="", help="Optional explicit output path for the calibrated TRSO JSON config.")
    parser.add_argument("--trso_basis_source", type=str, default="response", choices=["response", "random", "dct"], help="Controlled basis ablation; response is the proposed SVD basis.")
    parser.add_argument("--trso_allocation", type=str, default="exact", choices=["exact", "greedy", "uniform"], help="Rank-allocation ablation under the same budget.")
    parser.add_argument("--trso_score_mode", type=str, default="energy", choices=["energy", "energy_per_param", "energy_per_channel", "noise_adjusted"])
    parser.add_argument("--trso_noise_beta", type=float, default=0.0)

    # BAM-Tuning baseline (Q1/IJCV CNN attention module adapted to frozen-backbone PEFT)
    parser.add_argument("--bam_reduction", type=int, default=16)
    parser.add_argument("--bam_dilation", type=int, default=4)
    parser.add_argument("--bam_gate_init", type=float, default=0.0)
    parser.add_argument("--bam_use_bn", type=str2bool, default=True)
    parser.add_argument("--bam_insert", type=str, default="stage", choices=["stage", "all"])
    parser.add_argument("--bam_stages", type=str, default="1,2,3,4")

    # Residual Adapter
    parser.add_argument("--ra_mode", type=str, default="parallel", choices=["parallel", "series"])
    parser.add_argument("--ra_reduction", type=int, default=1, help="Legacy compatibility option; paper-style residual adapters use full-width 1x1 mappings.")
    parser.add_argument("--ra_norm", type=str, default="bn", choices=["bn", "ln", "none"], help="Legacy compatibility option retained for old command lines.")
    parser.add_argument("--ra_act", type=str, default="none", choices=["relu", "gelu", "silu", "none"], help="Legacy compatibility option retained for old command lines.")
    parser.add_argument("--ra_gate_init", type=float, default=0.0, help="Legacy compatibility option; paper-style adapters are identity-safe through zero-initialized 1x1 weights.")
    parser.add_argument("--ra_stages", type=str, default="1,2,3,4")
    parser.add_argument("--ra_pretrained_checkpoint", type=str, default="", help="Official/shared ResNet-26 checkpoint for residual-adapter transfer.")

    # SSF / LoRA / BitFit
    parser.add_argument("--ssf_init_scale", type=float, default=1.0)
    parser.add_argument("--ssf_init_shift", type=float, default=0.0)
    parser.add_argument("--ssf_init_std", type=float, default=0.02)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_merge_weights", type=str2bool, default=True)
    parser.add_argument("--lora_target", type=str, default="all", choices=["all", "1x1", "3x3", "dw"])
    parser.add_argument("--bitfit_train_head", type=str2bool, default=True)
    parser.add_argument("--bitfit_bias_scope", type=str, default="all", choices=["all", "transformer", "attention"])

    # Batch / epochs
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--update_freq", default=1, type=int)
    parser.add_argument("--fs_shot", default=16, type=int)

    # Generic model params
    parser.add_argument("--model", default="resnet50_clip", type=str)
    parser.add_argument("--drop_path", type=float, default=0)
    parser.add_argument("--input_size", default=224, type=int)
    parser.add_argument("--crop_ratio", default=0.875, type=float)

    # EMA
    parser.add_argument("--model_ema", type=str2bool, default=False)
    parser.add_argument("--model_ema_decay", type=float, default=0.9999)
    parser.add_argument("--model_ema_force_cpu", type=str2bool, default=False)
    parser.add_argument("--model_ema_eval", type=str2bool, default=False)

    # Optimization
    parser.add_argument("--optimizer", default="auto", choices=["auto", "adamw", "sgd"])
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--paper_hparams", type=str2bool, default=False, help="Apply the original paper default optimizer schedule where a single canonical setting exists.")
    parser.add_argument("--opt_eps", default=1e-8, type=float)
    parser.add_argument("--clip_grad", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--weight_decay_adapter", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--warmup_steps", type=int, default=-1)
    parser.add_argument("--weight_decay_end", type=float, default=None)

    # Augmentation / preprocessing
    parser.add_argument("--color_jitter", type=float, default=0.0)
    parser.add_argument("--aa", type=str, default="rand-m9-mstd0.5-inc1")
    parser.add_argument("--smoothing", type=float, default=0.0)
    parser.add_argument("--regression_loss", type=str, default="mse", choices=["mse", "smooth_l1"])
    parser.add_argument("--train_interpolation", type=str, default="bicubic")
    parser.add_argument("--crop_pct", type=float, default=None)
    parser.add_argument("--reprob", type=float, default=0.0)
    parser.add_argument("--remode", type=str, default="pixel")
    parser.add_argument("--recount", type=int, default=1)
    parser.add_argument("--resplit", type=str2bool, default=False)
    parser.add_argument("--imagenet_norm", type=str2bool, default=True)
    parser.add_argument("--train_aug", type=str, default="standard", choices=["standard", "resize"])

    # Mixup/Cutmix
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--cutmix", type=float, default=0.0)
    parser.add_argument("--cutmix_minmax", type=float, nargs="+", default=None)
    parser.add_argument("--mixup_prob", type=float, default=1.0)
    parser.add_argument("--mixup_switch_prob", type=float, default=0.5)
    parser.add_argument("--mixup_mode", type=str, default="batch")

    # Finetuning/checkpoint params
    parser.add_argument("--finetune", default="")
    parser.add_argument("--head_from", default="", type=str)
    parser.add_argument("--head_init_scale", default=1.0, type=float)
    parser.add_argument("--model_key", default="model|module", type=str)
    parser.add_argument("--model_prefix", default="", type=str)

    # Dataset
    parser.add_argument("--is_tuning", default=False, type=str2bool)
    parser.add_argument("--dataset", default="dtd", type=str)
    parser.add_argument("--task", default="auto", choices=["auto", "single_label", "multilabel", "regression"])
    parser.add_argument("--download", type=str2bool, default=False)
    parser.add_argument("--allow_val_as_test", type=str2bool, default=False)
    parser.add_argument("--val_ratio", default=0.1, type=float)
    parser.add_argument("--dtd_partition", default=1, type=int)
    parser.add_argument("--places_small", type=str2bool, default=True)
    parser.add_argument("--inat_target_type", default="full", type=str)
    parser.add_argument("--coco_task", default="multilabel", choices=["multilabel", "majority"])
    parser.add_argument("--celeba_task", default="attributes", choices=["attributes", "landmarks"], help="CelebA attributes is multi-label; landmarks is 10-output regression.")
    parser.add_argument("--train_csv", default="", type=str)
    parser.add_argument("--val_csv", default="", type=str)
    parser.add_argument("--test_csv", default="", type=str)
    parser.add_argument("--image_column", default="image", type=str)
    parser.add_argument("--label_column", default="label", type=str)
    parser.add_argument("--label_separator", default=";", type=str)
    parser.add_argument("--data_path", default="./data", type=str)
    parser.add_argument("--eval_data_path", default=None, type=str)
    parser.add_argument("--nb_classes", default=1000, type=int)
    parser.add_argument("--fake_train_size", default=32, type=int)
    parser.add_argument("--fake_val_size", default=16, type=int)
    parser.add_argument("--fake_test_size", default=16, type=int)
    parser.add_argument("--imagenet_default_mean_and_std", type=str2bool, default=True)
    parser.add_argument("--output_dir", default="./experiments/")
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--deterministic", type=str2bool, default=False, help="Use deterministic PyTorch algorithms when available.")

    parser.add_argument("--resume", default="")
    parser.add_argument("--auto_resume", type=str2bool, default=False)
    parser.add_argument("--save_ckpt", type=str2bool, default=True)
    parser.add_argument("--save_ckpt_freq", default=1, type=int)
    parser.add_argument("--save_ckpt_num", default=2, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--eval", type=str2bool, default=False)
    parser.add_argument("--dist_eval", type=str2bool, default=True)
    parser.add_argument("--disable_eval", type=str2bool, default=False)
    parser.add_argument("--final_test", type=str2bool, default=True)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--pin_mem", type=str2bool, default=True)

    # distributed
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", type=str2bool, default=False)
    parser.add_argument("--dist_url", default="env://")

    # AMP / profiling / logging
    parser.add_argument("--use_amp", type=str2bool, default=True)
    parser.add_argument("--profile_efficiency", type=str2bool, default=True)
    parser.add_argument("--profile_batch_size", type=int, default=32)
    parser.add_argument("--save_history", type=str2bool, default=True)
    parser.add_argument("--measure_eval_latency", type=str2bool, default=False)

    # W&B kept for compatibility. Actual logging is optional.
    parser.add_argument("--enable_wandb", type=str2bool, default=False)
    parser.add_argument("--project", default="parameter_efficient_tuning_cv", type=str)
    parser.add_argument("--wandb_ckpt", type=str2bool, default=False)

    return parser


def canonicalize_args(args):
    args.tuning_method = canonical_method(args.tuning_method)

    if args.tuning_method == "full":
        args.freeze_backbone = False
    else:
        args.freeze_backbone = True

    if args.optimizer == "auto":
        args.optimizer = "sgd" if args.tuning_method in {"prompt", "bam", "residual", "sidetune"} else "adamw"
    if args.paper_hparams:
        if args.tuning_method == "prompt":
            args.optimizer = "sgd"
            args.lr = 40.0
            args.weight_decay = 0.0
            args.weight_decay_adapter = 0.0
            args.epochs = 1000
            args.warmup_steps = 1000
            args.prompt_size = 30
        elif args.tuning_method == "bam":
            args.optimizer = "sgd"
            args.lr = 0.1
            args.momentum = 0.9
            args.weight_decay = 1e-4
            args.epochs = 100

    if args.trso_kernel_size <= 0 or args.trso_kernel_size % 2 == 0:
        raise ValueError("--trso_kernel_size must be a positive odd integer.")
    if args.trso_spatial_rank <= 0 or args.trso_spatial_rank > args.trso_kernel_size ** 2:
        raise ValueError("--trso_spatial_rank must be in [1, trso_kernel_size^2].")
    if args.trso_calibration_batches < 1:
        raise ValueError("--trso_calibration_batches must be at least 1.")
    if not (0 < args.trso_keep_ratio <= 1.0):
        raise ValueError("--trso_keep_ratio must be in (0, 1].")
    if args.trso_parameter_budget < 0:
        raise ValueError("--trso_parameter_budget cannot be negative.")
    if args.trso_noise_beta < 0:
        raise ValueError("--trso_noise_beta cannot be negative.")
    if not (0.0 < float(args.val_ratio) < 1.0):
        raise ValueError("--val_ratio must be in (0, 1).")
    if args.task in {"multilabel", "regression"} and (args.mixup > 0 or args.cutmix > 0 or args.cutmix_minmax is not None):
        raise ValueError("Mixup/CutMix is currently supported only for single-label classification.")
    paper_classification_only = {"prompt", "conv", "adapter", "bam", "residual", "ssf", "lora", "bitfit", "sidetune"}
    if args.tuning_method in paper_classification_only and args.task not in {"auto", "classification", "single_label"}:
        raise ValueError(f"Strict paper baseline '{args.tuning_method}' supports single-label classification only.")
    return args


def save_json_on_master(obj: Dict, path: str):
    if utils.is_main_process():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)


def _resolve_weights_multiapi(backbone: str, weights_str: str):
    try:
        from torchvision.models import get_model_weights, get_weight  # noqa: F401
        has_new_api = True
    except Exception:
        has_new_api = False

    if not weights_str or str(weights_str).lower() in ("none", "scratch", "random"):
        return None, has_new_api
    if not has_new_api:
        return "legacy_pretrained", False
    if "." in weights_str:
        from torchvision.models import get_weight
        return get_weight(weights_str), True
    from torchvision.models import get_model_weights
    try:
        enum_cls = get_model_weights(backbone)
        member = weights_str.upper()
        if member == "DEFAULT":
            return enum_cls.DEFAULT, True
        if hasattr(enum_cls, member):
            return getattr(enum_cls, member), True
    except Exception:
        pass
    return None, True


def _infer_clip_input_size(preprocess):
    size = None
    try:
        ts = getattr(preprocess, "transforms", None) or []
        for t in ts:
            name = t.__class__.__name__.lower()
            if "centercrop" in name and hasattr(t, "size"):
                size = t.size[0] if isinstance(t.size, (tuple, list)) else int(t.size)
        if size is None:
            for t in ts:
                name = t.__class__.__name__.lower()
                if "resize" in name and hasattr(t, "size"):
                    val = t.size
                    size = max(val) if isinstance(val, (tuple, list)) else int(val)
    except Exception:
        size = None
    return size or 224


def _strip_wrapper_prefixes(name: str) -> str:
    prefixes = ("module.", "backbone.", "visual.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
                changed = True
    return name


def _is_head_key(k: str) -> bool:
    name = _strip_wrapper_prefixes(k)
    return name.startswith(("head.", "heads.", "fc.", "classifier.", "linear."))


def _is_head_param(name: str) -> bool:
    return _is_head_key(name)


def _extract_checkpoint_model(ckpt: dict, model_key: str):
    for mk in str(model_key).split("|"):
        if mk in ckpt:
            return ckpt[mk]
    return ckpt


def safe_torch_load(path, map_location="cpu"):
    """Load a trusted PyTorch checkpoint across PyTorch versions.

    PyTorch 2.6 changed the default behavior of torch.load to
    weights_only=True. The checkpoints saved by this training script include
    argparse.Namespace and sometimes NumPy scalar objects in `args`, so loading
    them with the new default can fail with WeightsUnpickler errors.

    Use this only for checkpoints you created or otherwise trust.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older PyTorch versions do not expose the weights_only argument.
        return torch.load(path, map_location=map_location)


class CLIPLinearProbe(nn.Module):
    def __init__(self, visual_module: nn.Module, feat_dim: int, num_classes: int, freeze_backbone: bool = True):
        super().__init__()
        self.visual = visual_module
        if freeze_backbone:
            for p in self.visual.parameters():
                p.requires_grad = False
            self.visual.eval()
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        feats = self.visual(x)
        return self.head(feats)


class PixelPromptWrapper(nn.Module):
    """Apply the repository's original border prompt before any image backbone."""

    def __init__(self, backbone: nn.Module, prompt_size: int, image_size: int):
        super().__init__()
        from models.tuning_modules.prompter import PadPrompter
        self.tuning_module = PadPrompter(prompt_size=prompt_size, image_size=image_size)
        self.backbone = backbone
        self.backbone_family = getattr(backbone, "backbone_family", None)

    def forward(self, x):
        return self.backbone(self.tuning_module(x))


def _replace_classifier_head(model_backbone: nn.Module, num_classes: int, keep_pretrained_head: bool = True):
    """Replace a classification head across torchvision, timm, and local models.

    Returns True when a compatible head is found, including when its output
    dimension already matches. A missing head is an error at the call site.
    """
    num_classes = int(num_classes)

    if hasattr(model_backbone, "reset_classifier") and callable(model_backbone.reset_classifier):
        try:
            current = model_backbone.get_classifier() if hasattr(model_backbone, "get_classifier") else None
            if not (keep_pretrained_head and isinstance(current, nn.Linear) and current.out_features == num_classes):
                model_backbone.reset_classifier(num_classes)
            return True
        except Exception:
            pass

    def replace_attr(parent, name) -> bool:
        layer = getattr(parent, name, None)
        if isinstance(layer, nn.Linear):
            if not (keep_pretrained_head and layer.out_features == num_classes):
                setattr(parent, name, nn.Linear(layer.in_features, num_classes, bias=layer.bias is not None))
            return True
        if isinstance(layer, nn.Conv2d):
            if not (keep_pretrained_head and layer.out_channels == num_classes):
                setattr(parent, name, nn.Conv2d(layer.in_channels, num_classes, layer.kernel_size, layer.stride, layer.padding, bias=layer.bias is not None))
            return True
        if isinstance(layer, nn.Sequential):
            items = list(layer)
            for index in reversed(range(len(items))):
                child = items[index]
                if isinstance(child, nn.Linear):
                    if not (keep_pretrained_head and child.out_features == num_classes):
                        items[index] = nn.Linear(child.in_features, num_classes, bias=child.bias is not None)
                        setattr(parent, name, nn.Sequential(*items))
                    return True
                if isinstance(child, nn.Conv2d):
                    if not (keep_pretrained_head and child.out_channels == num_classes):
                        items[index] = nn.Conv2d(child.in_channels, num_classes, child.kernel_size, child.stride, child.padding, bias=child.bias is not None)
                        setattr(parent, name, nn.Sequential(*items))
                    return True
        return False

    for attribute in ("fc", "classifier", "linear", "head"):
        if replace_attr(model_backbone, attribute):
            return True

    # Torchvision VisionTransformer stores the classifier at heads.head.
    heads = getattr(model_backbone, "heads", None)
    if heads is not None:
        if replace_attr(heads, "head"):
            return True
        if isinstance(heads, nn.Sequential):
            items = list(heads)
            for index in reversed(range(len(items))):
                if isinstance(items[index], nn.Linear):
                    child = items[index]
                    if not (keep_pretrained_head and child.out_features == num_classes):
                        items[index] = nn.Linear(child.in_features, num_classes, bias=child.bias is not None)
                        model_backbone.heads = nn.Sequential(*items)
                    return True
    return False


def _freeze_batchnorm(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
            if m.affine:
                if m.weight is not None:
                    m.weight.requires_grad = False
                if m.bias is not None:
                    m.bias.requires_grad = False




def _torchvision_classification_backbones():
    """Return torchvision image-classification builders only."""
    import torchvision
    blocked_modules = (".detection.", ".segmentation.", ".video.", ".optical_flow.", ".quantization.")
    names = []
    try:
        from torchvision.models import get_model_builder
        for name in torchvision.models.list_models():
            try:
                module_name = get_model_builder(name).__module__
            except Exception:
                continue
            if not any(token in module_name for token in blocked_modules):
                names.append(name)
    except Exception:
        names = [name for name in torchvision.models.list_models() if not name.startswith("video")]
    return sorted(set(names))

def _build_torchvision_or_hub_backbone(args):
    """Build from torchvision, timm, or supported CIFAR hubs."""
    import torchvision

    if args.pretrained is None:
        pretrained_flag = args.weights is not None and str(args.weights).lower() not in ("none", "scratch", "random")
    else:
        pretrained_flag = bool(args.pretrained)

    source = str(getattr(args, "model_source", "auto")).lower()
    hub_pattern = re.match(r"^cifar(10|100)_.+$", args.backbone) or args.backbone in (
        "cifar_resnet56", "resnet56_cifar", "akamaster_resnet56", "resnet56_cifar10"
    ) or re.match(r"^akamaster_resnet(20|32|44|56|110)$", args.backbone)

    if source in ("auto", "hub") and hub_pattern:
        if re.match(r"^cifar(10|100)_.+$", args.backbone) and args.cifar_hub in ("auto", "chenyaofo"):
            provider, entry = "chenyaofo/pytorch-cifar-models", args.backbone
            model_backbone = torch.hub.load(provider, entry, pretrained=pretrained_flag)
        else:
            provider = "akamaster/pytorch_resnet_cifar10"
            entry = args.backbone.replace("akamaster_", "") if args.backbone.startswith("akamaster_") else "resnet56"
            model_backbone = torch.hub.load(provider, entry)
        args.input_size = 32
        args.resolved_model_source = "hub"
        print(f"[Info] Loaded {entry} from {provider} (input_size=32).")
        return model_backbone
    if source == "hub":
        raise RuntimeError(f"Backbone '{args.backbone}' is not registered in the supported hub patterns.")

    try:
        tv_names = set(_torchvision_classification_backbones())
    except Exception:
        tv_names = {name for name in dir(torchvision.models) if callable(getattr(torchvision.models, name, None))}

    if source in ("auto", "torchvision") and args.backbone in tv_names:
        tv_weights, has_new_api = _resolve_weights_multiapi(args.backbone, args.weights)
        kwargs = {}
        # Randomly initialized torchvision ViTs can be constructed for a custom image size.
        if tv_weights is None and any(token in args.backbone.lower() for token in ("vit_", "vision_transformer")):
            kwargs["image_size"] = int(args.input_size)
        if has_new_api:
            from torchvision.models import get_model
            try:
                model_backbone = get_model(args.backbone, weights=tv_weights, **kwargs)
            except TypeError:
                model_backbone = get_model(args.backbone, weights=tv_weights)
        else:
            function = getattr(torchvision.models, args.backbone)
            try:
                model_backbone = function(pretrained=tv_weights == "legacy_pretrained", **kwargs)
            except TypeError:
                model_backbone = function(pretrained=tv_weights == "legacy_pretrained")
        if tv_weights is not None:
            try:
                crop = tv_weights.transforms().crop_size
                args.input_size = int(crop[0] if isinstance(crop, (tuple, list)) else crop)
            except Exception:
                pass
        args.resolved_model_source = "torchvision"
        print(f"[Info] Loaded {args.backbone} from torchvision with weights={tv_weights}.")
        return model_backbone
    if source == "torchvision":
        raise RuntimeError(f"Backbone '{args.backbone}' is not available in torchvision.")

    if source in ("auto", "timm"):
        try:
            import timm
        except Exception as exc:
            raise RuntimeError(
                f"Backbone '{args.backbone}' was not found in torchvision and timm is unavailable. "
                "Install the requirements or choose --model_source torchvision."
            ) from exc
        if args.backbone not in set(timm.list_models(pretrained=False)):
            raise RuntimeError(f"Backbone '{args.backbone}' is not available in timm.")
        create_kwargs = {"pretrained": pretrained_flag, "num_classes": int(args.nb_classes)}
        try:
            model_backbone = timm.create_model(args.backbone, img_size=int(args.input_size), **create_kwargs)
        except TypeError:
            model_backbone = timm.create_model(args.backbone, **create_kwargs)
        args.resolved_model_source = "timm"
        print(f"[Info] Loaded {args.backbone} from timm (pretrained={pretrained_flag}).")
        return model_backbone

    raise RuntimeError(f"Unable to resolve backbone '{args.backbone}' from source '{source}'.")


def _attach_hook_adapters(model_backbone: nn.Module, args, make_adapter):
    """Attach output-space adapters and return the number of insertion points."""
    attached = []
    try:
        from torchvision.models.resnet import BasicBlock, Bottleneck
    except Exception:
        BasicBlock = Bottleneck = tuple()
    try:
        from torchvision.models.convnext import CNBlock
    except Exception:
        CNBlock = tuple()
    try:
        from torchvision.models.efficientnet import MBConv, FusedMBConv
    except Exception:
        MBConv = FusedMBConv = tuple()
    try:
        from torchvision.models.mobilenetv3 import InvertedResidual
    except Exception:
        InvertedResidual = tuple()
    try:
        from torchvision.models.vision_transformer import EncoderBlock as TVEncoderBlock
    except Exception:
        TVEncoderBlock = tuple()
    try:
        from torchvision.models.swin_transformer import SwinTransformerBlock as TVSwinBlock
    except Exception:
        TVSwinBlock = tuple()
    try:
        from models.backbones.swin_transformer import SwinTransformerBlock as LocalSwinBlock
    except Exception:
        LocalSwinBlock = tuple()

    def build_adapter(out_ch: int, layout: str = "auto", grid_size=None):
        try:
            return make_adapter(out_ch, layout=layout, grid_size=grid_size)
        except TypeError:
            return make_adapter(out_ch)

    def attach(module: nn.Module, out_ch: int, layout: str = "auto", grid_size=None):
        if hasattr(module, "pet_adapter"):
            return
        module.add_module("pet_adapter", build_adapter(out_ch, layout=layout, grid_size=grid_size))

        def hook(mod, inputs, out):
            adapter = mod.pet_adapter
            if getattr(adapter, "is_trso_adapter", False):
                return adapter(out)
            if getattr(adapter, "is_bam_adapter", False):
                return adapter(out)
            if args.tuning_method in ("conv", "adapter"):
                return out + args.adapt_scale * adapter(out)
            if args.tuning_method == "residual":
                return out + adapter(out)
            if args.tuning_method == "ssf":
                return adapter(out)
            return out

        module.register_forward_hook(hook)
        attached.append(module)

    # Snapshot modules before adding adapters to avoid recursive insertion.
    for m in list(model_backbone.modules()):
        if BasicBlock and isinstance(m, BasicBlock):
            attach(m, m.conv2.out_channels, layout="bchw")
        elif Bottleneck and isinstance(m, Bottleneck):
            attach(m, m.conv3.out_channels, layout="bchw")
        elif hasattr(m, "conv3") and isinstance(getattr(m, "conv3"), nn.Conv2d):
            attach(m, m.conv3.out_channels, layout="bchw")
        elif hasattr(m, "conv2") and isinstance(getattr(m, "conv2"), nn.Conv2d):
            attach(m, m.conv2.out_channels, layout="bchw")
        elif CNBlock and isinstance(m, CNBlock):
            if hasattr(m, "block") and len(getattr(m, "block")) > 0 and isinstance(m.block[0], nn.Conv2d):
                attach(m, m.block[0].out_channels, layout="bchw")
        elif MBConv and isinstance(m, (MBConv, FusedMBConv)):
            last_conv = next((c for c in reversed(list(m.modules())) if isinstance(c, nn.Conv2d)), None)
            if last_conv is not None:
                attach(m, last_conv.out_channels, layout="bchw")
        elif InvertedResidual and isinstance(m, InvertedResidual):
            last_conv = next((c for c in reversed(list(m.modules())) if isinstance(c, nn.Conv2d)), None)
            if last_conv is not None:
                attach(m, last_conv.out_channels, layout="bchw")
        elif TVEncoderBlock and isinstance(m, TVEncoderBlock):
            dim = int(m.ln_1.normalized_shape[0])
            attach(m, dim, layout="bnc")
        elif TVSwinBlock and isinstance(m, TVSwinBlock):
            dim = int(m.norm1.normalized_shape[0])
            attach(m, dim, layout="bhwc")
        elif LocalSwinBlock and isinstance(m, LocalSwinBlock):
            attach(m, int(m.dim), layout="bnc", grid_size=tuple(m.input_resolution))
        elif args.tuning_method == "trso":
            module_path = m.__class__.__module__.lower()
            class_name = m.__class__.__name__.lower()
            # timm Vision Transformer blocks expose norm1 with the embedding size.
            if "vision_transformer" in module_path and class_name == "block" and hasattr(m, "norm1"):
                shape = getattr(m.norm1, "normalized_shape", None)
                if shape:
                    attach(m, int(shape[0]), layout="bnc")
            # timm Swin blocks use either BHWC or flattened BNC depending on version.
            elif "swin_transformer" in module_path and "block" in class_name and hasattr(m, "norm1"):
                shape = getattr(m.norm1, "normalized_shape", None)
                if shape:
                    attach(m, int(shape[0]), layout="auto")
            # timm ConvNeXt blocks remain NCHW.
            elif "convnext" in module_path and "block" in class_name:
                channels = None
                for candidate in (getattr(m, "conv_dw", None), getattr(m, "conv1", None)):
                    if isinstance(candidate, nn.Conv2d):
                        channels = int(candidate.out_channels)
                        break
                if channels is not None:
                    attach(m, channels, layout="bchw")
    return len(attached)


def _add_adapters(model_backbone: nn.Module, args):
    method = args.tuning_method
    adapter_param_ids = set()

    if method in ("full", "linear", "bitfit", "prompt", "bam", "sidetune", "residual"):
        return model_backbone, adapter_param_ids

    if method in ("conv", "adapter"):
        if "resnet50" not in str(args.backbone).lower():
            raise ValueError("Conv-Adapter paper reproduction requires ResNet-50.")
        from models.tuning_modules.conv_adapter import apply_conv_adapter_resnet50
        count = apply_conv_adapter_resnet50(
            model_backbone,
            mode=args.conv_adapter_mode,
            kernel_size=args.kernel_size,
            stages=(1, 2, 3, 4),
            reduction=args.adapt_size,
            adapt_scale=args.adapt_scale,
        )
        print(f"[Conv-Adapter] inserted {count} adapters using {args.conv_adapter_mode}.")

    elif method == "trso":
        from models.task_response_adapter import TaskResponseSpatialAdapter

        def make_adapter(ch, layout="auto", grid_size=None):
            return TaskResponseSpatialAdapter(
                channels=ch,
                kernel_size=args.trso_kernel_size,
                spatial_rank=args.trso_spatial_rank,
                channel_ratio=args.trso_channel_ratio,
                operator_radius=args.trso_operator_radius,
                gate_init=args.trso_gate_init,
                basis_init_scale=args.trso_basis_init_scale,
                calibration_scale=args.trso_calibration_scale,
                layout=layout,
                grid_size=grid_size,
                basis_trainable=args.trso_basis_trainable,
            )

        count = _attach_hook_adapters(model_backbone, args, make_adapter)
        if count == 0:
            raise RuntimeError("TRSO found no compatible CNN/ViT/Swin blocks in this backbone.")

    elif method == "ssf":
        from models.tuning_modules.ssf import apply_ssf
        records = apply_ssf(model_backbone, init_std=args.ssf_init_std)
        print(f"[SSF] inserted {len(records)} paper-specified affine modules.")

    elif method == "lora":
        from models.tuning_modules.lora_transformer import apply_lora_transformer
        count = apply_lora_transformer(
            model_backbone,
            rank=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            merge_weights=bool(args.lora_merge_weights),
        )
        print(f"[LoRA] wrapped {count} Transformer Q/V attention projections.")

    else:
        raise ValueError(
            f"Unsupported strict paper baseline: {method}. "
            "LoRA-Conv and former hook approximations were removed from the paper-reproduction path."
        )

    return model_backbone, adapter_param_ids


def set_trainability_policy(model: nn.Module, args, extra_adapter_param_ids: Optional[set] = None):
    method = args.tuning_method
    extra_adapter_param_ids = extra_adapter_param_ids or set()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    if method == "full":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return model

    if method == "linear":
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(_is_head_param(name))
        _freeze_batchnorm(model)
        return model

    if method == "prompt":
        # Original visual prompting optimizes only the prompt. The pretrained
        # output classifier remains frozen and is accessed through fixed labels.
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("tuning_module."))
        if not any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("Visual prompting produced no trainable prompt parameters")
        model.backbone.eval()
        args.weight_decay = 0.0
        return model

    if method == "bam":
        # BAM is trained jointly with ResNet-50 in the original architecture.
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return model

    if method in ("conv", "adapter"):
        from models.tuning_modules.conv_adapter import set_conv_adapter_trainability
        set_conv_adapter_trainability(model)
        _freeze_batchnorm(model)
        return model

    if method == "residual":
        from models.tuning_modules.residual_adapter import set_residual_adapter_trainability
        set_residual_adapter_trainability(model)
        return model

    if method == "ssf":
        from models.tuning_modules.ssf import set_ssf_trainability
        set_ssf_trainability(model)
        return model

    if method == "lora":
        from models.tuning_modules.lora_transformer import mark_only_lora_as_trainable
        mark_only_lora_as_trainable(model, train_bias="none")
        for name, parameter in model.named_parameters():
            if _is_head_param(name):
                parameter.requires_grad_(True)
        return model

    if method == "bitfit":
        from models.tuning_modules.bitfit import set_bitfit_trainability
        set_bitfit_trainability(
            model,
            train_head=bool(args.bitfit_train_head),
            bias_scope=getattr(args, "bitfit_bias_scope", "all"),
        )
        args.weight_decay = 0.0
        return model

    if method == "sidetune":
        for name, parameter in model.named_parameters():
            trainable = name.startswith("side.") or name.startswith("head.") or name == "alpha_logit"
            parameter.requires_grad_(trainable)
        model.base.eval()
        return model

    if method == "trso":
        tokens = ("pet_adapter", "trso", "basis_atoms", "coefficients", "gate")
        for name, parameter in model.named_parameters():
            if name.endswith("probe_kernel"):
                parameter.requires_grad_(False)
            elif _is_head_param(name) or any(token in name for token in tokens) or id(parameter) in extra_adapter_param_ids:
                parameter.requires_grad_(True)
        _freeze_batchnorm(model)
        if not any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("TRSO produced no trainable parameters")
        return model

    raise ValueError(f"Unsupported strict paper baseline: {method}")



def build_task_criterion(args, mixup_active: bool = False):
    task_type = getattr(args, "task_type", TASK_SINGLE_LABEL)
    if task_type == TASK_MULTILABEL:
        return nn.BCEWithLogitsLoss()
    if task_type == TASK_REGRESSION:
        return nn.SmoothL1Loss() if getattr(args, "regression_loss", "mse") == "smooth_l1" else nn.MSELoss()
    if mixup_active:
        return SoftTargetCrossEntropy()
    smoothing = float(getattr(args, "smoothing", 0.0) or 0.0)
    if smoothing > 0.0:
        return LabelSmoothingCrossEntropy(smoothing=smoothing)
    return nn.CrossEntropyLoss()


def primary_metric(task_type: str):
    if task_type == TASK_MULTILABEL:
        return "map", True
    if task_type == TASK_REGRESSION:
        return "mae", False
    return "acc1", True


def format_primary(stats: Dict, task_type: str) -> str:
    name, _ = primary_metric(task_type)
    value = float(stats.get(name, float("nan")))
    label = {"acc1": "Acc@1", "map": "mAP", "mae": "MAE"}[name]
    return f"{label}={value:.5f}"


def _classification_logits(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    if isinstance(output, dict):
        for key in ("logits", "out", "pred"):
            if key in output and isinstance(output[key], torch.Tensor):
                return output[key]
    raise TypeError(f"Cannot extract classification logits from output type {type(output)!r}.")


def _move_training_batch(batch, device):
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise TypeError("TRSO calibration expects data-loader batches shaped as (images, labels, ...).")
    images, labels = batch[0], batch[1]
    return images.to(device, non_blocking=True), labels.to(device, non_blocking=True)


def calibrate_trso_model(model: nn.Module, data_loader, device: torch.device, args) -> list[str]:
    """Discover task-response bases and select layers before normal training."""
    from models.task_response_adapter import (
        iter_trso_adapters,
        load_trso_config,
        save_trso_config,
        select_trso_layers,
    )

    adapters = list(iter_trso_adapters(model))
    if not adapters:
        raise RuntimeError("TRSO was requested, but no compatible CNN or Transformer blocks were found.")

    if args.trso_config:
        selected = load_trso_config(model, args.trso_config, strict=True)
        print(f"[TRSO] Loaded calibration config from {args.trso_config}; selected={len(selected)}")
        return selected

    if not args.trso_calibration:
        for _, adapter in adapters:
            adapter.set_enabled(True)
        print("[TRSO] Calibration disabled; using deterministic fallback bases in all candidate layers.")
        return [name for name, _ in adapters]

    was_training = model.training
    criterion = build_task_criterion(args, mixup_active=False)

    # Optional small head warm-up makes task-response gradients less dependent on a
    # randomly initialized classifier while preserving the frozen backbone.
    head_params = [p for name, p in model.named_parameters() if _is_head_param(name) and p.requires_grad]
    if args.trso_head_warmup_steps > 0 and head_params:
        for _, adapter in adapters:
            adapter.stop_calibration()
            adapter.set_enabled(False)
        warmup_opt = torch.optim.AdamW(head_params, lr=args.lr, weight_decay=args.weight_decay)
        model.train()
        _freeze_batchnorm(model)
        warmup_done = 0
        for step, batch in enumerate(data_loader):
            if step >= args.trso_head_warmup_steps:
                break
            images, labels = _move_training_batch(batch, device)
            warmup_opt.zero_grad(set_to_none=True)
            loss = criterion(_classification_logits(model(images)), labels)
            loss.backward()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                for parameter in head_params:
                    if parameter.grad is not None:
                        torch.distributed.all_reduce(parameter.grad, op=torch.distributed.ReduceOp.SUM)
                        parameter.grad.div_(world_size)
            warmup_opt.step()
            warmup_done += 1
        print(f"[TRSO] Completed {warmup_done} head warm-up steps.")

    for _, adapter in adapters:
        adapter.start_calibration(reset=True)

    model.eval()
    used_batches = 0
    for batch_index, batch in enumerate(data_loader):
        if batch_index >= args.trso_calibration_batches:
            break
        images, labels = _move_training_batch(batch, device)
        model.zero_grad(set_to_none=True)
        logits = _classification_logits(model(images))
        loss = criterion(logits, labels)
        loss.backward()
        for _, adapter in adapters:
            adapter.accumulate_probe_gradient()
        used_batches += 1

    if used_batches == 0:
        raise RuntimeError("TRSO calibration received no batches.")

    # Calibration occurs before DDP wrapping. Aggregate response statistics here
    # so every rank derives the same SVD basis and the same layer/rank allocation.
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        for _, adapter in adapters:
            torch.distributed.all_reduce(adapter.gradient_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(adapter.gradient_square_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(adapter.gradient_samples, op=torch.distributed.ReduceOp.SUM)

    score_rows = []
    for name, adapter in adapters:
        score = adapter.finalize_calibration(
            init_scale=args.trso_basis_init_scale,
            basis_source=getattr(args, "trso_basis_source", "response"),
            random_seed=getattr(args, "seed", 0),
        )
        score_rows.append((name, score))
    selected = select_trso_layers(
        model,
        max_adapters=args.trso_max_adapters,
        keep_ratio=args.trso_keep_ratio,
        parameter_budget=args.trso_parameter_budget,
        allocation=getattr(args, "trso_allocation", "exact"),
        score_mode=getattr(args, "trso_score_mode", "energy"),
        noise_beta=getattr(args, "trso_noise_beta", 0.0),
    )

    # Clear any calibration gradients on the classifier and backbone.
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    if was_training:
        _freeze_batchnorm(model)

    score_rows.sort(key=lambda item: item[1], reverse=True)
    print(f"[TRSO] Calibration batches={used_batches}; candidates={len(adapters)}; selected={len(selected)}")
    adapter_map = dict(adapters)
    for name, score in score_rows[: min(10, len(score_rows))]:
        marker = "*" if name in selected else " "
        adapter = adapter_map[name]
        print(
            f"[TRSO] {marker} {name}: response_score={score:.6e}; "
            f"active_rank={adapter.active_rank}; "
            f"trainable_cost={adapter.parameter_count_breakdown()['active_trainable']}"
        )

    output_path = args.trso_save_config
    if not output_path and args.output_dir:
        output_path = os.path.join(args.output_dir, "trso_calibration.json")
    if output_path:
        save_trso_config(model, output_path, selected_layers=selected)
        print(f"[TRSO] Saved calibration config to {output_path}")
    return selected

def build_datasets(args):
    if build_dataset_split is None:
        dataset_train, args.nb_classes = build_dataset(args=args, is_train=True)
        dataset_val, _ = build_dataset(args=args, is_train=False) if not args.disable_eval else (None, args.nb_classes)
        dataset_test = None
        return dataset_train, dataset_val, dataset_test

    dataset_train, args.nb_classes = build_dataset_split(args=args, split="train")
    dataset_val = None
    dataset_test = None
    if not args.disable_eval:
        dataset_val, _ = build_dataset_split(args=args, split="val")
    if args.final_test:
        try:
            dataset_test, _ = build_dataset_split(args=args, split="test")
        except Exception as e:
            if args.allow_val_as_test:
                print(f"[Warn] test split unavailable for {args.dataset}: {e}. --allow_val_as_test=True, so validation is reused.")
                dataset_test = dataset_val
            else:
                raise RuntimeError(
                    f"A distinct test split is unavailable for dataset '{args.dataset}'. "
                    "Provide a test split or explicitly set --allow_val_as_test True."
                ) from e
    return dataset_train, dataset_val, dataset_test


def _seed_data_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_samplers(args, dataset_train, dataset_val, dataset_test):
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    if getattr(args, "distributed", False):
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed)
    else:
        sampler_generator = torch.Generator().manual_seed(int(args.seed) + int(global_rank))
        sampler_train = torch.utils.data.RandomSampler(dataset_train, generator=sampler_generator)

    def eval_sampler(ds):
        if ds is None:
            return None
        if getattr(args, "distributed", False) and args.dist_eval:
            return torch.utils.data.DistributedSampler(ds, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        return torch.utils.data.SequentialSampler(ds)

    return sampler_train, eval_sampler(dataset_val), eval_sampler(dataset_test)


def _parse_prompt_indices(value: str, num_classes: int):
    text = str(value or "").strip()
    if not text:
        return list(range(int(num_classes)))
    indices = [int(item.strip()) for item in text.split(",") if item.strip()]
    if len(indices) != int(num_classes):
        raise ValueError("--prompt_output_indices must contain exactly nb_classes indices")
    return indices


def build_model_for_experiment(args, clip_visual=None, clip_feat_dim=None):
    # The original visual-prompting baseline requires a frozen pretrained
    # classifier output space. A feature-only CLIP visual encoder is therefore
    # not a faithful implementation and is rejected by the compatibility table.
    if args.clip_model:
        if clip_visual is None or clip_feat_dim is None:
            raise RuntimeError("CLIP requested but not initialized.")
        family = "clip_transformer" if str(args.clip_model).upper().startswith("VIT") else "clip_cnn"
        validate_method_backbone(args.tuning_method, family)
        if args.tuning_method not in ("full", "linear", "bitfit"):
            raise ValueError("Strict CLIP branch supports full, linear, and BitFit only.")
        model = CLIPLinearProbe(
            clip_visual,
            int(clip_feat_dim),
            args.nb_classes,
            freeze_backbone=bool(args.freeze_backbone),
        )
        model.backbone_family = family
        return model, set()

    # Residual Adapters use their original dedicated ResNet-26 rather than
    # wrapping an arbitrary torchvision model.
    if args.tuning_method == "residual":
        from models.tuning_modules.residual_adapter import ResidualAdapterResNet26
        model = ResidualAdapterResNet26(num_classes=args.nb_classes, mode=args.ra_mode)
        if args.ra_pretrained_checkpoint:
            checkpoint = safe_torch_load(args.ra_pretrained_checkpoint, map_location="cpu")
            state = _extract_checkpoint_model(checkpoint, args.model_key) if isinstance(checkpoint, dict) else checkpoint
            incompat = model.load_shared_state_dict(state, strict=False, require_shared_coverage=True)
            print(f"[Residual Adapter] loaded shared checkpoint; missing={len(incompat.missing_keys)} unexpected={len(incompat.unexpected_keys)}")
        elif str(args.weights).lower() not in {"none", "scratch", "random"}:
            raise ValueError(
                "Strict Residual Adapter transfer requires --ra_pretrained_checkpoint. "
                "Use --weights none only for architecture smoke tests."
            )
        args.backbone_family = "resnet"
        return model, set()

    model_backbone = _build_torchvision_or_hub_backbone(args)
    family = detect_backbone_family(
        model_backbone,
        backbone_name=args.backbone,
        source=getattr(args, "resolved_model_source", args.model_source),
    )
    if family == "unknown":
        raise RuntimeError(
            f"Could not determine the architecture family of '{args.backbone}'. "
            "Use a registered torchvision/timm model or add an explicit family detector."
        )
    validate_method_backbone(args.tuning_method, family)
    args.backbone_family = family
    model_backbone.backbone_family = family
    print(f"[Compatibility] method={args.tuning_method} | family={family} | source={getattr(args, 'resolved_model_source', 'unknown')}")

    if args.tuning_method == "prompt":
        from models.tuning_modules.prompter import VisualPromptingClassifier
        indices = _parse_prompt_indices(args.prompt_output_indices, args.nb_classes)
        return VisualPromptingClassifier(
            backbone=model_backbone,
            num_classes=args.nb_classes,
            prompt_size=args.prompt_size,
            image_size=args.input_size,
            output_indices=indices,
            prompt_type=args.prompt_type,
        ), set()

    if args.tuning_method == "sidetune":
        from models.tuning_modules.side_tuning import SideTuningClassifier
        model = SideTuningClassifier(
            base_model=model_backbone,
            num_classes=args.nb_classes,
            learn_alpha=bool(args.sidetune_learn_alpha),
            alpha_init=float(args.sidetune_alpha),
            side_arch=args.sidetune_arch,
            side_width=args.sidetune_width,
            side_depth=args.sidetune_depth,
            side_checkpoint=args.sidetune_checkpoint,
        )
        model.backbone_family = family
        return model, set()

    if args.tuning_method == "bam":
        if "resnet50" not in str(args.backbone).lower():
            raise ValueError("BAM paper reproduction requires a ResNet-50 backbone.")
        from models.tuning_modules.bam_adapter import BAMResNet50
        return BAMResNet50(
            model_backbone,
            num_classes=args.nb_classes,
            reduction_ratio=args.bam_reduction,
            dilation_conv_num=2,
            dilation_val=args.bam_dilation,
        ), set()

    if not _replace_classifier_head(model_backbone, args.nb_classes, keep_pretrained_head=args.keep_pretrained_head):
        raise RuntimeError(
            f"No replaceable task head was found for backbone '{args.backbone}'. "
            "This run is stopped to avoid training with an incorrect output dimension."
        )

    model_backbone, adapter_param_ids = _add_adapters(model_backbone, args)
    return model_backbone, adapter_param_ids


def main(args):
    args = canonicalize_args(args)

    if args.list_compatibility:
        for row in compatibility_rows():
            print(f"{row['method']:12s} | {','.join(row['families']):45s} | {row['implementation_scope']}")
        return

    if args.list_backbones:
        import torchvision
        print("[torchvision]")
        for name in _torchvision_classification_backbones():
            print(name)
        try:
            import timm
            print("[timm]")
            for name in timm.list_models(pretrained=False):
                print(name)
        except Exception:
            print("[timm unavailable: install timm to enable its backbone catalogue]")
        print("[datasets]")
        for name in available_datasets():
            print(name)
        return

    utils.init_distributed_mode(args)

    if str(args.device).lower().startswith("cuda") and not torch.cuda.is_available():
        print("[Info] CUDA not available — falling back to CPU.")
        args.device = "cpu"
        args.use_amp = False
    elif str(args.device).lower() == "cpu":
        args.use_amp = False
        args.pin_mem = False

    device = torch.device(args.device)
    print(args)
    print(f"[Info] Using device: {device}  (AMP={'on' if args.use_amp else 'off'})")

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = device.type == "cuda"

    clip_preprocess = None
    clip_visual = None
    clip_feat_dim = None
    if args.clip_model:
        ok = False
        try:
            import clip
            clip_model_full, clip_preprocess = clip.load(args.clip_model, device="cpu", jit=False)
            clip_visual = clip_model_full.visual
            clip_feat_dim = getattr(clip_visual, "output_dim", None)
            if clip_feat_dim is None:
                size = _infer_clip_input_size(clip_preprocess)
                with torch.no_grad():
                    clip_feat_dim = clip_visual(torch.zeros(1, 3, size, size)).shape[1]
            ok = True
            print(f"[CLIP] Loaded OpenAI {args.clip_model}")
        except Exception as e:
            print(f"[CLIP] OpenAI CLIP load failed ({e}). Trying OpenCLIP ...")
            try:
                import open_clip
                model_full, _, clip_preprocess = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_pretrained)
                clip_visual = model_full.visual
                clip_feat_dim = getattr(clip_visual, "output_dim", None)
                if clip_feat_dim is None:
                    size = _infer_clip_input_size(clip_preprocess)
                    with torch.no_grad():
                        clip_feat_dim = clip_visual(torch.zeros(1, 3, size, size)).shape[1]
                ok = True
                print(f"[CLIP] Loaded OpenCLIP {args.clip_model}")
            except Exception as e2:
                raise RuntimeError(f"Failed to load CLIP via OpenAI and OpenCLIP: {e2}")
        if ok:
            args.input_size = _infer_clip_input_size(clip_preprocess)

    dataset_train, dataset_val, dataset_test = build_datasets(args)

    if args.clip_model and clip_preprocess is not None:
        for ds in (dataset_train, dataset_val, dataset_test):
            if ds is None:
                continue
            target = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
            if hasattr(target, "transform"):
                target.transform = clip_preprocess
            if hasattr(target, "transforms"):
                target.transforms = clip_preprocess

    sampler_train, sampler_val, sampler_test = build_samplers(args, dataset_train, dataset_val, dataset_test)

    if utils.get_rank() == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    loader_generator = torch.Generator().manual_seed(seed)
    drop_last = len(dataset_train) >= args.batch_size
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=drop_last,
        worker_init_fn=_seed_data_worker,
        generator=loader_generator,
    )
    data_loader_val = None
    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=min(max(1, args.batch_size), 256),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            worker_init_fn=_seed_data_worker,
            generator=torch.Generator().manual_seed(seed + 1000),
        )
    data_loader_test = None
    if dataset_test is not None:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            sampler=sampler_test,
            batch_size=min(max(1, args.batch_size), 256),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            worker_init_fn=_seed_data_worker,
            generator=torch.Generator().manual_seed(seed + 2000),
        )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active and args.task_type != TASK_SINGLE_LABEL:
        raise ValueError("Mixup/CutMix is supported only for single-label classification.")
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes,
        )

    model, adapter_param_ids = build_model_for_experiment(args, clip_visual=clip_visual, clip_feat_dim=clip_feat_dim)
    model = set_trainability_policy(model, args, extra_adapter_param_ids=adapter_param_ids)
    model.to(device)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        save_json_on_master(vars(args), os.path.join(args.output_dir, "args.json"))

    # Optional finetune checkpoint.
    if args.finetune:
        checkpoint = torch.hub.load_state_dict_from_url(args.finetune, map_location="cpu", check_hash=True) if args.finetune.startswith("https") else safe_torch_load(args.finetune, map_location="cpu")
        checkpoint_model = _extract_checkpoint_model(checkpoint, args.model_key)
        state_dict = model.state_dict()
        for k in list(checkpoint_model.keys()):
            if _is_head_key(k) and k in state_dict and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint (shape mismatch)")
                del checkpoint_model[k]
        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)


    # Load model state before TRSO calibration and before optimizer construction.
    # This is required because active TRSO ranks determine the trainable parameter set.
    resume_checkpoint = utils.load_model_for_resume(args, model, strict=True)

    if args.tuning_method == "trso":
        if resume_checkpoint is None:
            calibrate_trso_model(model, data_loader_train, device, args)
        else:
            from models.task_response_adapter import sync_trso_trainability
            selected = sync_trso_trainability(model)
            print(f"[TRSO] Resume restored {len(selected)} active adapters; calibration skipped.")

    # Old approximate memory profile retained for compatibility.
    try:
        memory_cost, detailed_info = profile_memory_cost(
            model,
            (1, 3, args.input_size, args.input_size),
            True,
            activation_bits=32,
            trainable_param_bits=32,
            frozen_param_bits=8,
            batch_size=min(8, args.batch_size),
        )
        print(f"memory_cost_MB: {memory_cost / 1e6:.3f}")
        print(f"param_size_MB: {detailed_info['param_size'] / 1e6:.3f}")
        print(f"act_size_MB: {detailed_info['act_size'] / 1e6:.3f}")
    except Exception as e:
        print(f"[Warn] memory_utils profile failed: {e}")

    if args.profile_efficiency:
        try:
            from tools.profile_efficiency import profile_model, save_profile
            profile = profile_model(
                model=model,
                device=device,
                input_size=args.input_size,
                batch_size=min(args.profile_batch_size, args.batch_size),
                use_amp=args.use_amp,
            )
            print("[Efficiency Profile]", profile)
            if args.output_dir:
                save_profile(profile, os.path.join(args.output_dir, "efficiency_profile.json"))
        except Exception as e:
            print(f"[Warn] efficiency profiling failed: {e}")

    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(model, decay=args.model_ema_decay, device="cpu" if args.model_ema_force_cpu else "", resume="")
        print(f"Using EMA with decay = {args.model_ema_decay:.8f}")

    model_without_ddp = model
    if getattr(args, "distributed", False):
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu] if device.type == "cuda" else None)
        model_without_ddp = model.module

    n_trainable = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model_without_ddp.parameters())
    print(f"Number of trainable params: {n_trainable:,}")
    print(f"Number of total params: {n_total:,}")

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = max(
        1, math.ceil(len(data_loader_train) / max(1, args.update_freq))
    )
    print(f"LR = {args.lr:.8f}")
    print(f"Batch size = {total_batch_size}")
    print(f"Number of training examples = {len(dataset_train)}")
    print(f"Number of training steps per epoch = {num_training_steps_per_epoch}")

    # Optimizer param groups: adapter-like params use the adapter-specific decay.
    adapter_like, other = [], []
    for name, p in model_without_ddp.named_parameters():
        if not p.requires_grad:
            continue
        is_adapter_like = any(tok in name for tok in ("pet_adapter", "trso", "basis_atoms", "coefficients", "gate", "bam", "ssf", "lora", "adapter", "side", "tuning_module", "prompt")) or id(p) in adapter_param_ids
        (adapter_like if is_adapter_like else other).append(p)
    print(f"[ParamGroups] adapter_like={sum(p.numel() for p in adapter_like):,} others={sum(p.numel() for p in other):,}")

    parameter_groups = []
    if adapter_like:
        parameter_groups.append({"params": adapter_like, "lr": args.lr, "weight_decay": args.weight_decay_adapter})
    if other:
        parameter_groups.append({"params": other, "lr": args.lr, "weight_decay": args.weight_decay})
    if not parameter_groups:
        raise RuntimeError("No trainable parameters were supplied to the optimizer")
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(parameter_groups, lr=args.lr, momentum=args.momentum)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(parameter_groups, lr=args.lr, betas=(0.9, 0.999), eps=args.opt_eps)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    print(f"Optimizer = {optimizer.__class__.__name__}")
    loss_scaler = NativeScaler()

    lr_schedule_values = utils.cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)

    criterion = build_task_criterion(args, mixup_active=mixup_fn is not None)
    eval_criterion = build_task_criterion(args, mixup_active=False)
    print(f"criterion = {criterion} | task={args.task_type}")

    restored_training_state = utils.restore_optimizer_state(
        args, resume_checkpoint, optimizer, loss_scaler, model_ema
    )

    if args.head_from:
        head_ckpt = safe_torch_load(args.head_from, map_location="cpu")
        head_model = _extract_checkpoint_model(head_ckpt, args.model_key)
        model_sd = model_without_ddp.state_dict()
        to_load = {
            k: v for k, v in head_model.items()
            if _is_head_key(k) and k in model_sd and hasattr(v, "shape") and v.shape == model_sd[k].shape
        }
        if to_load:
            model_without_ddp.load_state_dict(to_load, strict=False)
            print(f"[Info] Loaded {len(to_load)} head params from {args.head_from}")
        else:
            print(f"[Warn] --head_from provided but no matching head keys were found: {args.head_from}")

    metric_name, maximize_metric = primary_metric(args.task_type)

    if args.eval:
        print("Eval only mode")
        eval_loader = data_loader_test if data_loader_test is not None else data_loader_val
        eval_ds = dataset_test if data_loader_test is not None else dataset_val
        if eval_loader is None:
            raise RuntimeError("No evaluation loader available.")
        stats = evaluate(
            eval_loader,
            model,
            device,
            use_amp=args.use_amp,
            measure_latency=args.measure_eval_latency,
            task_type=args.task_type,
            criterion=eval_criterion,
        )
        print(f"Evaluation on {len(eval_ds)} samples: {format_primary(stats, args.task_type)}")
        if args.output_dir:
            save_json_on_master(stats, os.path.join(args.output_dir, "eval_summary.json"))
        return

    default_best = float("-inf") if maximize_metric else float("inf")
    best_val_metric = float(restored_training_state.get("best_val_metric", default_best))
    best_epoch = int(restored_training_state.get("best_epoch", -1))
    history = list(restored_training_state.get("history", []))
    start_time = time.time()

    print(f"Start training for {args.epochs} epochs")
    for epoch in range(args.start_epoch, args.epochs):
        if getattr(args, "distributed", False) and hasattr(data_loader_train.sampler, "set_epoch"):
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)

        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            model_ema,
            mixup_fn,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            update_freq=args.update_freq,
            use_amp=args.use_amp,
            task_type=args.task_type,
        )

        val_stats = {}
        is_better = False
        if data_loader_val is not None:
            val_stats = evaluate(
                data_loader_val,
                model,
                device,
                use_amp=args.use_amp,
                task_type=args.task_type,
                criterion=eval_criterion,
            )
            current_metric = float(val_stats[metric_name])
            is_better = current_metric > best_val_metric if maximize_metric else current_metric < best_val_metric
            print(f"Validation on {len(dataset_val)} samples: {format_primary(val_stats, args.task_type)}")
            if is_better:
                best_val_metric = current_metric
                best_epoch = epoch
            best_stats = {metric_name: best_val_metric}
            print(f"Best validation {format_primary(best_stats, args.task_type)} at epoch {best_epoch}")

        if log_writer is not None and val_stats:
            for key, value in val_stats.items():
                if isinstance(value, (int, float)):
                    log_writer.update(**{f"val_{key}": value}, head="perf", step=epoch)

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_stats.items()},
            "epoch": epoch,
            "n_trainable_parameters": n_trainable,
            "n_total_parameters": n_total,
            f"best_val_{metric_name}": best_val_metric,
            "best_epoch": best_epoch,
            "primary_metric": metric_name,
            "maximize_primary_metric": maximize_metric,
        }
        history.append(log_stats)

        checkpoint_training_state = {
            "best_val_metric": best_val_metric,
            "best_epoch": best_epoch,
            "history": history,
            "primary_metric": metric_name,
            "maximize_primary_metric": maximize_metric,
            "global_update_step": (epoch + 1) * num_training_steps_per_epoch,
        }
        if args.output_dir and args.save_ckpt and is_better:
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch="best",
                model_ema=model_ema, training_state=checkpoint_training_state,
            )
        if args.output_dir and args.save_ckpt and (
            (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs
        ):
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch,
                model_ema=model_ema, training_state=checkpoint_training_state,
            )

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats, default=str) + "\n")
            if args.save_history:
                save_json_on_master(history, os.path.join(args.output_dir, "history.json"))

        if args.model_ema and args.model_ema_eval and data_loader_val is not None:
            ema_stats = evaluate(
                data_loader_val,
                model_ema.ema,
                device,
                use_amp=args.use_amp,
                task_type=args.task_type,
                criterion=eval_criterion,
            )
            print(f"EMA validation: {format_primary(ema_stats, args.task_type)}")

    total_time = time.time() - start_time
    print(f"Training time {str(datetime.timedelta(seconds=int(total_time)))}")

    convergence_summary = {
        f"best_val_{metric_name}": best_val_metric,
        "primary_metric": metric_name,
        "maximize_primary_metric": maximize_metric,
        "best_epoch": best_epoch,
        "total_train_time_sec": total_time,
        "epochs": args.epochs,
        "n_trainable_parameters": n_trainable,
        "n_total_parameters": n_total,
    }
    if history and best_epoch >= 0:
        if maximize_metric and best_val_metric > 0:
            target = 0.95 * best_val_metric
            epochs_to_target = next(
                (int(row["epoch"]) + 1 for row in history if row.get(f"val_{metric_name}", float("-inf")) >= target),
                None,
            )
            convergence_summary["epochs_to_95pct_best"] = epochs_to_target
        elif not maximize_metric and np.isfinite(best_val_metric):
            # For an error metric, reaching within 5% of the best error is the analogous threshold.
            target = 1.05 * best_val_metric
            epochs_to_target = next(
                (int(row["epoch"]) + 1 for row in history if row.get(f"val_{metric_name}", float("inf")) <= target),
                None,
            )
            convergence_summary["epochs_to_within_5pct_best"] = epochs_to_target

        convergence_summary["mean_epoch_time_sec"] = float(
            np.mean([row.get("train_epoch_time", 0.0) for row in history])
        )
        mem_values = [
            row.get("train_peak_train_memory_mb")
            for row in history
            if row.get("train_peak_train_memory_mb") is not None
        ]
        if mem_values:
            convergence_summary["peak_train_memory_mb"] = float(max(mem_values))
    if args.output_dir:
        save_json_on_master(convergence_summary, os.path.join(args.output_dir, "convergence_summary.json"))

    # Final test: restore the best validation checkpoint and evaluate once.
    if data_loader_test is not None:
        best_ckpt = os.path.join(args.output_dir, "checkpoint-best.pth") if args.output_dir else ""
        if not best_ckpt or not os.path.exists(best_ckpt):
            raise FileNotFoundError(
                "Final test requires checkpoint-best.pth produced by validation selection."
            )
        ckpt = safe_torch_load(best_ckpt, map_location="cpu")
        if not isinstance(ckpt, dict) or "model" not in ckpt:
            raise RuntimeError(f"Best checkpoint is malformed: {best_ckpt}")
        model_without_ddp.load_state_dict(ckpt["model"], strict=True)
        if args.tuning_method == "trso":
            from models.task_response_adapter import sync_trso_trainability
            sync_trso_trainability(model_without_ddp)
        print(f"[Info] Strictly loaded best checkpoint for final test: {best_ckpt}")
        test_stats = evaluate(
            data_loader_test,
            model,
            device,
            use_amp=args.use_amp,
            measure_latency=args.measure_eval_latency,
            task_type=args.task_type,
            criterion=eval_criterion,
        )
        print(f"Final test on {len(dataset_test)} samples: {format_primary(test_stats, args.task_type)}")
        if args.output_dir:
            save_json_on_master(test_stats, os.path.join(args.output_dir, "test_summary.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("TRSO training and evaluation script", parents=[get_args_parser()])
    args = parser.parse_args()
    args = canonicalize_args(args)

    if args.list_backbones or args.list_compatibility:
        main(args)
        raise SystemExit(0)

    # The original table can overwrite explicit CLI choices. It is now opt-in.
    if args.legacy_auto_hparams and not args.is_tuning and args.tuning_method not in ("full", "linear"):
        try:
            args = utils.auto_load_optim_param(args, args.model, args.tuning_method, args.dataset)
            args = canonicalize_args(args)
        except Exception as exc:
            print(f"[Warn] legacy auto_load_optim_param failed or unavailable: {exc}")
    elif args.is_tuning:
        args.save_ckpt = False

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.data = Path(args.data_path).name
    main(args)
