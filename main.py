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
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma

from datasets import build_dataset
try:
    from datasets.build import build_dataset_split
except Exception:  # backward fallback if user did not replace datasets/build.py yet
    build_dataset_split = None

from engine import evaluate, train_one_epoch
from memory_utils import profile_memory_cost
from models import build_model
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
    parser.add_argument("--weights", type=str, default="DEFAULT")
    parser.add_argument("--list_backbones", action="store_true")
    parser.add_argument("--pretrained", type=str2bool, default=None)
    parser.add_argument("--keep_pretrained_head", type=str2bool, default=True)
    parser.add_argument("--cifar_hub", type=str, default="auto", choices=["auto", "chenyaofo", "akamaster"])

    # CLIP linear probe branch
    parser.add_argument("--clip_model", type=str, default=None, choices=["RN50", "RN50x4"])
    parser.add_argument("--clip_pretrained", type=str, default="openai")
    parser.add_argument("--freeze_backbone", type=str2bool, default=True)

    # Methods
    parser.add_argument(
        "--tuning_method",
        type=str,
        default="trso",
        help=(
            "full | linear | prompt | conv | adapter | trso | bam | residual | "
            "ssf | lora_conv | bitfit | sidetune"
        ),
    )

    # Side-tuning
    parser.add_argument("--sidetune_alpha", type=float, default=0.5)
    parser.add_argument("--sidetune_learn_alpha", type=str2bool, default=True)
    parser.add_argument("--sidetune_width", type=int, default=64)
    parser.add_argument("--sidetune_depth", type=int, default=3)

    # Prompt / Conv-Adapter
    parser.add_argument("--prompt_size", default=10, type=int)
    parser.add_argument("--kernel_size", default=3, type=int)
    parser.add_argument("--adapt_size", default=8, type=float)
    parser.add_argument("--adapt_scale", default=1.0, type=float)

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

    # BAM-Tuning baseline (Q1/IJCV CNN attention module adapted to frozen-backbone PEFT)
    parser.add_argument("--bam_reduction", type=int, default=16)
    parser.add_argument("--bam_dilation", type=int, default=4)
    parser.add_argument("--bam_gate_init", type=float, default=0.0)
    parser.add_argument("--bam_use_bn", type=str2bool, default=True)
    parser.add_argument("--bam_insert", type=str, default="stage", choices=["stage", "all"])
    parser.add_argument("--bam_stages", type=str, default="1,2,3,4")

    # Residual Adapter
    parser.add_argument("--ra_mode", type=str, default="parallel", choices=["parallel", "series"])
    parser.add_argument("--ra_reduction", type=int, default=16)
    parser.add_argument("--ra_norm", type=str, default="bn", choices=["bn", "ln", "none"])
    parser.add_argument("--ra_act", type=str, default="relu", choices=["relu", "gelu", "silu", "none"])
    parser.add_argument("--ra_gate_init", type=float, default=0.0)
    parser.add_argument("--ra_stages", type=str, default="1,2,3,4")

    # SSF / LoRA / BitFit
    parser.add_argument("--ssf_init_scale", type=float, default=1.0)
    parser.add_argument("--ssf_init_shift", type=float, default=0.0)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_target", type=str, default="all", choices=["all", "1x1", "3x3", "dw"])
    parser.add_argument("--bitfit_train_head", type=str2bool, default=True)

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
    aliases = {
        "task_response": "trso",
        "task-response": "trso",
        "task_response_spatial_operator": "trso",
        "trso_adapter": "trso",
        "bam_adapter": "bam",
        "bam-tuning": "bam",
        "bam_tuning": "bam",
        "lora": "lora_conv",
        "lora_conv2d": "lora_conv",
        "ssf_adapter": "ssf",
        "ra": "residual",
        "residual_adapter": "residual",
        "side-tuning": "sidetune",
        "sidetuning": "sidetune",
        "side_tune": "sidetune",
        "full_finetune": "full",
        "finetune_full": "full",
        "linear_probe": "linear",
    }
    args.tuning_method = aliases.get(str(args.tuning_method).lower(), str(args.tuning_method).lower())

    if args.tuning_method == "full":
        args.freeze_backbone = False
    elif args.tuning_method in (
        "linear", "conv", "adapter", "trso", "bam", "residual", "ssf",
        "lora_conv", "bitfit", "sidetune",
    ):
        args.freeze_backbone = True

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


def _is_head_key(k: str) -> bool:
    return k.startswith(("head.", "fc.", "classifier.", "linear."))


def _is_head_param(name: str) -> bool:
    return name.startswith(("head.", "fc.", "classifier.", "linear."))


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


def _replace_classifier_head(model_backbone: nn.Module, num_classes: int, keep_pretrained_head: bool = True):
    def maybe_replace_linear(parent, name, lin):
        if not isinstance(lin, nn.Linear):
            return False
        if keep_pretrained_head and lin.out_features == num_classes:
            return False
        setattr(parent, name, nn.Linear(lin.in_features, num_classes))
        return True

    replaced = False
    if hasattr(model_backbone, "fc") and isinstance(model_backbone.fc, nn.Linear):
        replaced = maybe_replace_linear(model_backbone, "fc", model_backbone.fc)
    elif hasattr(model_backbone, "classifier"):
        head = model_backbone.classifier
        if isinstance(head, nn.Linear):
            replaced = maybe_replace_linear(model_backbone, "classifier", head)
        elif isinstance(head, nn.Sequential):
            new_seq = list(head)
            for i in reversed(range(len(new_seq))):
                if isinstance(new_seq[i], nn.Linear):
                    if not (keep_pretrained_head and new_seq[i].out_features == num_classes):
                        new_seq[i] = nn.Linear(new_seq[i].in_features, num_classes)
                        model_backbone.classifier = nn.Sequential(*new_seq)
                        replaced = True
                    break
    if not replaced and hasattr(model_backbone, "linear") and isinstance(model_backbone.linear, nn.Linear):
        replaced = maybe_replace_linear(model_backbone, "linear", model_backbone.linear)
    if not replaced and hasattr(model_backbone, "head") and isinstance(model_backbone.head, nn.Linear):
        replaced = maybe_replace_linear(model_backbone, "head", model_backbone.head)
    return replaced


def _freeze_batchnorm(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            if m.affine:
                if m.weight is not None:
                    m.weight.requires_grad = False
                if m.bias is not None:
                    m.bias.requires_grad = False


def _build_torchvision_or_hub_backbone(args):
    import torchvision
    if args.pretrained is None:
        pretrained_flag = args.weights is not None and str(args.weights).lower() not in ("none", "scratch", "random")
    else:
        pretrained_flag = bool(args.pretrained)

    tv_weights, has_new_api = _resolve_weights_multiapi(args.backbone, args.weights)
    print(f"[Info] Backbone={args.backbone} | tv_weights={args.weights} -> {tv_weights} | hub.pretrained={pretrained_flag}")

    model_backbone = None
    if re.match(r"^cifar(10|100)_.+$", args.backbone) and args.cifar_hub in ("auto", "chenyaofo"):
        provider = "chenyaofo/pytorch-cifar-models"
        model_backbone = torch.hub.load(provider, args.backbone, pretrained=pretrained_flag)
        args.input_size = 32
        print(f"[Info] Loaded {args.backbone} from {provider} (input_size=32).")
    elif args.backbone in ("cifar_resnet56", "resnet56_cifar", "akamaster_resnet56", "resnet56_cifar10") or (
        re.match(r"^akamaster_resnet(20|32|44|56|110)$", args.backbone) is not None
    ):
        provider = "akamaster/pytorch_resnet_cifar10"
        entry = args.backbone.replace("akamaster_", "") if args.backbone.startswith("akamaster_") else "resnet56"
        model_backbone = torch.hub.load(provider, entry)
        args.input_size = 32
        print(f"[Info] Loaded {entry} from {provider} (input_size=32).")

    if model_backbone is None:
        try:
            if has_new_api:
                from torchvision.models import get_model
                model_backbone = get_model(args.backbone, weights=tv_weights)
            else:
                fn = getattr(torchvision.models, args.backbone)
                pretrained = tv_weights == "legacy_pretrained"
                try:
                    model_backbone = fn(pretrained=pretrained, num_classes=1000)
                except TypeError:
                    model_backbone = fn(pretrained=pretrained)
            print(f"[Info] Loaded {args.backbone} from torchvision.")
        except AttributeError as e:
            raise RuntimeError(f"Backbone '{args.backbone}' is not available in torchvision or supported hubs.") from e

    return model_backbone


def _attach_hook_adapters(model_backbone: nn.Module, args, make_adapter):
    """Attach output-space adapters to common CNN and Transformer blocks."""
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


def _add_adapters(model_backbone: nn.Module, args):
    method = args.tuning_method
    adapter_param_ids = set()

    if method in ("full", "linear", "bitfit"):
        return model_backbone, adapter_param_ids

    if method in ("conv", "adapter"):
        class ConvAdapter(nn.Module):
            def __init__(self, c, k=args.kernel_size):
                super().__init__()
                hidden = max(1, int(c // max(1, int(args.adapt_size))))
                self.net = nn.Sequential(
                    nn.Conv2d(c, hidden, 1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden, hidden, k, padding=k // 2, groups=hidden, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden, c, 1, bias=False),
                )

            def forward(self, x):
                return self.net(x)

        _attach_hook_adapters(model_backbone, args, lambda ch: ConvAdapter(ch))

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

        _attach_hook_adapters(model_backbone, args, make_adapter)

    elif method == "bam":
        from models.tuning_modules.bam_adapter import BAMAdapter

        def make_adapter(ch):
            m = BAMAdapter(
                channels=ch,
                reduction=args.bam_reduction,
                dilation=args.bam_dilation,
                gate_init=args.bam_gate_init,
                use_bn=args.bam_use_bn,
            )
            m.is_bam_adapter = True
            return m

        def attach_bam(module: nn.Module, out_ch: int):
            if hasattr(module, "pet_adapter"):
                return
            module.add_module("pet_adapter", make_adapter(out_ch))
            module.register_forward_hook(lambda mod, _inputs, out: mod.pet_adapter(out))

        if args.bam_insert == "stage" and all(hasattr(model_backbone, f"layer{i}") for i in range(1, 5)):
            stages = [int(s.strip()) for s in args.bam_stages.split(",") if s.strip().isdigit()]
            for stage in stages:
                layer = getattr(model_backbone, f"layer{stage}")
                block = layer[-1]
                if hasattr(block, "conv3"):
                    out_ch = block.conv3.out_channels
                elif hasattr(block, "conv2"):
                    out_ch = block.conv2.out_channels
                else:
                    raise RuntimeError(f"Cannot infer BAM channels for ResNet stage {stage}")
                attach_bam(block, out_ch)
        else:
            # all-block insertion or non-ResNet fallback
            _attach_hook_adapters(model_backbone, args, make_adapter)

    elif method == "ssf":
        from models.tuning_modules.ssf import SSF
        _attach_hook_adapters(
            model_backbone,
            args,
            lambda ch: SSF(ch, init_scale=args.ssf_init_scale, init_shift=args.ssf_init_shift),
        )

    elif method == "lora_conv":
        from models.tuning_modules.lora_conv import apply_lora_conv2d
        apply_lora_conv2d(model_backbone, r=args.lora_r, alpha=args.lora_alpha, target=args.lora_target)

    elif method == "residual":
        try:
            from models.tuning_modules.residual_adapter import (
                ParallelResidualAdapter,
                SeriesResidualAdapter,
                attach_residual_adapters_resnet,
            )
            if args.backbone.startswith("resnet"):
                stages = [int(s.strip()) for s in args.ra_stages.split(",") if s.strip().isdigit()]
                model_backbone = attach_residual_adapters_resnet(
                    model_backbone,
                    mode=args.ra_mode,
                    reduction=args.ra_reduction,
                    norm=args.ra_norm,
                    act=args.ra_act,
                    gate_init=args.ra_gate_init,
                    stages=stages,
                )
                from models.tuning_modules.residual_adapter import residual_adapter_trainable_parameters
                for mod in model_backbone.modules():
                    if isinstance(mod, (ParallelResidualAdapter, SeriesResidualAdapter)):
                        for p in residual_adapter_trainable_parameters(mod):
                            adapter_param_ids.add(id(p))
                return model_backbone, adapter_param_ids
        except Exception as e:
            print(f"[Warn] residual_adapter wrapper failed ({e}); falling back to hook-based residual adapters.")

        class ResidualCore(nn.Module):
            def __init__(self, c):
                super().__init__()
                hidden = max(1, c // max(1, args.ra_reduction))
                norm = nn.BatchNorm2d(hidden) if args.ra_norm == "bn" else nn.GroupNorm(1, hidden) if args.ra_norm == "ln" else nn.Identity()
                act = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "none": nn.Identity}[args.ra_act]()
                self.core = nn.Sequential(nn.Conv2d(c, hidden, 1, bias=False), norm, act, nn.Conv2d(hidden, c, 1, bias=False))
                self.gate = nn.Parameter(torch.tensor(float(args.ra_gate_init)))

            def forward(self, x):
                return self.core(x) * self.gate

        _attach_hook_adapters(model_backbone, args, lambda ch: ResidualCore(ch))

    else:
        raise ValueError(f"Unsupported PET method for torchvision branch: {method}")

    return model_backbone, adapter_param_ids


def set_trainability_policy(model: nn.Module, args, extra_adapter_param_ids: Optional[set] = None):
    method = args.tuning_method
    extra_adapter_param_ids = extra_adapter_param_ids or set()

    for p in model.parameters():
        p.requires_grad = False

    if method == "full":
        for p in model.parameters():
            p.requires_grad = True
        return model

    if method == "linear":
        for name, p in model.named_parameters():
            if _is_head_param(name):
                p.requires_grad = True
        _freeze_batchnorm(model)
        return model

    if method == "bitfit":
        for name, p in model.named_parameters():
            if name.endswith(".bias"):
                p.requires_grad = True
            if args.bitfit_train_head and _is_head_param(name):
                p.requires_grad = True
        args.weight_decay = 0.0
        _freeze_batchnorm(model)
        return model

    # PEFT: adapters and task head only.
    tokens = ("pet_adapter", "trso", "basis_atoms", "coefficients", "gate", "bam", "ssf", "lora", "adapter", "side", "lora_down", "lora_up")
    for name, p in model.named_parameters():
        if name.endswith("probe_kernel"):
            p.requires_grad = False
        elif _is_head_param(name) or name == "alpha_logit" or any(tok in name for tok in tokens) or id(p) in extra_adapter_param_ids:
            p.requires_grad = True

    _freeze_batchnorm(model)
    return model



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
    criterion = nn.CrossEntropyLoss()

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
        score = adapter.finalize_calibration(init_scale=args.trso_basis_init_scale)
        score_rows.append((name, score))
    selected = select_trso_layers(
        model,
        max_adapters=args.trso_max_adapters,
        keep_ratio=args.trso_keep_ratio,
        parameter_budget=args.trso_parameter_budget,
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
            print(f"[Warn] test split unavailable for {args.dataset}: {e}. Final test will use validation split.")
            dataset_test = dataset_val
    return dataset_train, dataset_val, dataset_test


def build_samplers(args, dataset_train, dataset_val, dataset_test):
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    if getattr(args, "distributed", False):
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    def eval_sampler(ds):
        if ds is None:
            return None
        if getattr(args, "distributed", False) and args.dist_eval:
            return torch.utils.data.DistributedSampler(ds, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        return torch.utils.data.SequentialSampler(ds)

    return sampler_train, eval_sampler(dataset_val), eval_sampler(dataset_test)


def build_model_for_experiment(args, clip_visual=None, clip_feat_dim=None):
    if args.clip_model:
        if clip_visual is None or clip_feat_dim is None:
            raise RuntimeError("CLIP requested but not initialized.")
        model = CLIPLinearProbe(clip_visual, int(clip_feat_dim), args.nb_classes, freeze_backbone=bool(args.freeze_backbone))
        return model, set()

    tv_methods = ("full", "linear", "conv", "adapter", "trso", "bam", "residual", "ssf", "lora_conv", "bitfit", "sidetune")
    if args.tuning_method in tv_methods:
        model_backbone = _build_torchvision_or_hub_backbone(args)
        if args.tuning_method == "sidetune":
            from models.tuning_modules.side_tuning import SideTuningClassifier
            model = SideTuningClassifier(
                base_model=model_backbone,
                num_classes=args.nb_classes,
                side_width=args.sidetune_width,
                side_depth=args.sidetune_depth,
                learn_alpha=bool(args.sidetune_learn_alpha),
                alpha_init=float(args.sidetune_alpha),
            )
            return model, set()

        _replace_classifier_head(model_backbone, args.nb_classes, keep_pretrained_head=args.keep_pretrained_head)
        model_backbone, adapter_param_ids = _add_adapters(model_backbone, args)
        return model_backbone, adapter_param_ids

    # Fallback to original repo builder for prompt/VPT or other custom models.
    model = build_model(args.model, pretrained=True, num_classes=args.nb_classes, tuning_method=args.tuning_method, args=args)
    return model, set()


def main(args):
    args = canonicalize_args(args)

    if args.list_backbones:
        try:
            from torchvision.models import list_models
            for n in sorted(n for n in list_models() if not n.startswith("video")):
                print(n)
        except Exception as e:
            print(f"[Warn] list_backbones failed: {e}")
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

    drop_last = len(dataset_train) >= args.batch_size
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=drop_last,
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
        )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
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


    if args.tuning_method == "trso":
        calibrate_trso_model(model, data_loader_train, device, args)

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
    num_training_steps_per_epoch = max(1, len(data_loader_train) // max(1, args.update_freq))
    print(f"LR = {args.lr:.8f}")
    print(f"Batch size = {total_batch_size}")
    print(f"Number of training examples = {len(dataset_train)}")
    print(f"Number of training steps per epoch = {num_training_steps_per_epoch}")

    # Optimizer param groups: adapter-like params use the adapter-specific decay.
    adapter_like, other = [], []
    for name, p in model_without_ddp.named_parameters():
        if not p.requires_grad:
            continue
        is_adapter_like = any(tok in name for tok in ("pet_adapter", "trso", "basis_atoms", "coefficients", "gate", "bam", "ssf", "lora", "adapter", "side")) or id(p) in adapter_param_ids
        (adapter_like if is_adapter_like else other).append(p)
    print(f"[ParamGroups] adapter_like={sum(p.numel() for p in adapter_like):,} others={sum(p.numel() for p in other):,}")

    optimizer = torch.optim.AdamW(
        [
            {"params": adapter_like, "lr": args.lr, "weight_decay": args.weight_decay_adapter},
            {"params": other, "lr": args.lr, "weight_decay": args.weight_decay},
        ],
        betas=(0.9, 0.999),
        eps=args.opt_eps,
    )
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

    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    print(f"criterion = {criterion}")

    utils.auto_load_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

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

    if args.eval:
        print("Eval only mode")
        eval_loader = data_loader_test if data_loader_test is not None else data_loader_val
        eval_ds = dataset_test if data_loader_test is not None else dataset_val
        if eval_loader is None:
            raise RuntimeError("No evaluation loader available.")
        stats = evaluate(eval_loader, model, device, use_amp=args.use_amp, measure_latency=args.measure_eval_latency)
        print(f"Accuracy of the network on {len(eval_ds)} images: {stats['acc1']:.5f}%")
        if args.output_dir:
            save_json_on_master(stats, os.path.join(args.output_dir, "eval_summary.json"))
        return

    best_val_acc = -1.0
    best_epoch = -1
    history = []
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
        )

        if args.output_dir and args.save_ckpt and ((epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs):
            utils.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)

        val_stats = {}
        if data_loader_val is not None:
            val_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
            print(f"Validation accuracy on {len(dataset_val)} images: {val_stats['acc1']:.3f}%")
            if val_stats["acc1"] > best_val_acc:
                best_val_acc = val_stats["acc1"]
                best_epoch = epoch
                if args.output_dir and args.save_ckpt:
                    utils.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
            print(f"Best validation accuracy: {best_val_acc:.3f}% at epoch {best_epoch}")

        if log_writer is not None and val_stats:
            log_writer.update(val_acc1=val_stats["acc1"], head="perf", step=epoch)
            if "acc5" in val_stats:
                log_writer.update(val_acc5=val_stats["acc5"], head="perf", step=epoch)
            log_writer.update(val_loss=val_stats["loss"], head="perf", step=epoch)

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_stats.items()},
            "epoch": epoch,
            "n_trainable_parameters": n_trainable,
            "n_total_parameters": n_total,
            "best_val_acc1": best_val_acc,
            "best_epoch": best_epoch,
        }
        history.append(log_stats)

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats, default=str) + "\n")
            if args.save_history:
                save_json_on_master(history, os.path.join(args.output_dir, "history.json"))

        if args.model_ema and args.model_ema_eval and data_loader_val is not None:
            ema_stats = evaluate(data_loader_val, model_ema.ema, device, use_amp=args.use_amp)
            print(f"EMA validation accuracy: {ema_stats['acc1']:.3f}%")

    total_time = time.time() - start_time
    print(f"Training time {str(datetime.timedelta(seconds=int(total_time)))}")

    # Convergence summary.
    convergence_summary = {
        "best_val_acc1": best_val_acc,
        "best_epoch": best_epoch,
        "total_train_time_sec": total_time,
        "epochs": args.epochs,
        "n_trainable_parameters": n_trainable,
        "n_total_parameters": n_total,
    }
    if history and best_val_acc > 0:
        target = 0.95 * best_val_acc
        epochs_to_95 = None
        for h in history:
            if h.get("val_acc1", -1) >= target:
                epochs_to_95 = int(h["epoch"]) + 1
                break
        convergence_summary["epochs_to_95pct_best"] = epochs_to_95
        convergence_summary["mean_epoch_time_sec"] = float(np.mean([h.get("train_epoch_time", 0.0) for h in history]))
        mem_values = [h.get("train_peak_train_memory_mb") for h in history if h.get("train_peak_train_memory_mb") is not None]
        if mem_values:
            convergence_summary["peak_train_memory_mb"] = float(max(mem_values))
    if args.output_dir:
        save_json_on_master(convergence_summary, os.path.join(args.output_dir, "convergence_summary.json"))

    # Final test: load best validation checkpoint and evaluate once.
    if data_loader_test is not None:
        best_ckpt = os.path.join(args.output_dir, "checkpoint-best.pth") if args.output_dir else ""
        if best_ckpt and os.path.exists(best_ckpt):
            try:
                ckpt = safe_torch_load(best_ckpt, map_location="cpu")
                state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
                model_without_ddp.load_state_dict(state, strict=False)
                print(f"[Info] Loaded best checkpoint for final test: {best_ckpt}")
            except Exception as e:
                print(f"[Warn] Failed to reload best checkpoint ({best_ckpt}): {e}")
                print("[Warn] Continuing final test with the current in-memory model.")
        test_stats = evaluate(data_loader_test, model, device, use_amp=args.use_amp, measure_latency=args.measure_eval_latency)
        print(f"Final test accuracy on {len(dataset_test)} images: {test_stats['acc1']:.3f}%")
        if args.output_dir:
            save_json_on_master(test_stats, os.path.join(args.output_dir, "test_summary.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("TRSO training and evaluation script", parents=[get_args_parser()])
    args = parser.parse_args()
    args = canonicalize_args(args)

    if args.list_backbones:
        main(args)
        raise SystemExit(0)

    # Keep compatibility with original repo auto hyperparameter loader, but avoid overwriting revision baselines.
    if not args.is_tuning and args.tuning_method not in ("full", "linear"):
        try:
            args = utils.auto_load_optim_param(args, args.model, args.tuning_method, args.dataset)
            args = canonicalize_args(args)
        except Exception as e:
            print(f"[Warn] auto_load_optim_param failed or unavailable: {e}")
    else:
        args.save_ckpt = False if args.is_tuning else args.save_ckpt

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.data = Path(args.data_path).name
    main(args)
