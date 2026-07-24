"""Controlled all-baseline DTD suite.

Every PEFT method and TRSO uses the same AdamW/cosine recipe. Full fine-tuning
and linear probing use only the explicitly permitted different learning rates.
A shared head-only checkpoint is trained once per backbone and seed, then loaded
before method construction/calibration so all compatible PEFT methods start
from the same task-aware classifier.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from typing import Iterable

from tools.experiment_grid import (
    RunSpec,
    build_specs,
    execute_specs,
    execute_specs_parallel,
    parse_csv_values,
    write_manifest,
)


CNN_METHODS = (
    "full", "linear", "prompt", "conv", "bam", "sidetune", "piggyback", "trso",
)
TRANSFORMER_METHODS = (
    "full", "linear", "prompt", "ssf", "lora", "bitfit", "adaptformer", "trso",
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="./data")
    p.add_argument("--output_root", default="outputs_fair_dtd")
    p.add_argument("--manifest", default="experiments/fair_dtd_manifest.json")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--head_epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--input_size", type=int, default=224)
    p.add_argument("--cnn_backbone", default="resnet50")
    p.add_argument("--transformer_backbone", default="vit_tiny_patch16_224")
    p.add_argument("--peft_lr", type=float, default=1e-3)
    p.add_argument("--full_lr", type=float, default=1e-4)
    p.add_argument("--linear_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--trso_budget_cnn", type=int, default=0)
    p.add_argument("--trso_budget_transformer", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu_ids", default="0", help="Comma-separated GPUs for independent parallel runs, e.g. 0,1.")
    p.add_argument("--parallel_runs", type=int, default=1, help="Maximum concurrent GPU workers.")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--max_runs", type=int, default=0)
    p.add_argument("--ra_pretrained_checkpoint", default="")
    return p


def base_common(args) -> dict:
    return {
        "dataset": "dtd",
        "data_path": args.data_path,
        "download": True,
        "dtd_partition": 1,
        "input_size": args.input_size,
        "nb_classes": 47,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": args.device,
        "use_amp": True,
        "profile_efficiency": True,
        "measure_eval_latency": True,
        "save_ckpt": True,
        "save_ckpt_freq": 1,
        "save_history": True,
        "final_test": True,
        "auto_resume": False,
        "fair_protocol": True,
        "fair_optimizer": "adamw",
        "fair_peft_lr": args.peft_lr,
        "fair_full_lr": args.full_lr,
        "fair_linear_lr": args.linear_lr,
        "fair_weight_decay": args.weight_decay,
        "fair_warmup_epochs": args.warmup_epochs,
        "fair_min_lr": 1e-6,
        "paper_hparams": False,
        "legacy_auto_hparams": False,
        "train_aug": "standard",
        "aa": "rand-m9-mstd0.5-inc1",
        "color_jitter": 0.2,
        "mixup": 0.2,
        "cutmix": 0.0,
        "smoothing": 0.1,
        "reprob": 0.1,
        "keep_pretrained_head": False,
        "peft_freeze_head": False,
        "peft_head_lr_scale": 0.5,
        "clip_grad": 1.0,
        "no_decay_bias_norm": True,
        "split_seed": 2026,
    }


def build_head_specs(args, seeds: list[int]) -> tuple[list[RunSpec], dict[tuple[str, int], str]]:
    specs: list[RunSpec] = []
    paths: dict[tuple[str, int], str] = {}
    for family, backbone, source in (
        ("cnn", args.cnn_backbone, "torchvision"),
        ("transformer", args.transformer_backbone, "timm"),
    ):
        common = {
            **base_common(args),
            "backbone": backbone,
            "model_source": source,
            "epochs": args.head_epochs,
        }
        variants = [
            (f"{family}_shared_head", {"tuning_method": "linear", "seed": seed})
            for seed in seeds
        ]
        family_specs = build_specs(
            suite="fair_head_preparation",
            variants=variants,
            common=common,
            output_root=args.output_root,
        )
        specs.extend(family_specs)
        for spec in family_specs:
            seed = int(spec.parameters["seed"])
            paths[(family, seed)] = str(Path(spec.output_dir) / "checkpoint-best.pth")
    return specs, paths


def _method_variant(method: str, family: str, seed: int, head_path: str, args) -> dict:
    row = {
        "tuning_method": method,
        "seed": seed,
        "keep_pretrained_head": method == "prompt",
    }
    if method == "prompt":
        row.update({"prompt_mapping": "frequency", "prompt_mapping_batches": 0})
    if method not in {"full", "linear", "prompt"}:
        row["head_from"] = head_path
    if method == "trso":
        row.update({
            "trso_variant": "v3",
            "trso_basis_source": "response",
            "trso_allocation": "exact",
            "trso_score_mode": "normalized_stable_energy_per_param",
            "trso_noise_beta": 0.0,
            "trso_parameter_budget": args.trso_budget_cnn if family == "cnn" else args.trso_budget_transformer,
            "trso_auto_budget_ratio": 0.35,
            "trso_auto_sparse": True,
            "trso_calibration_batches": 16,
            "trso_kernel_size": 5,
            "trso_spatial_rank": 4,
            "trso_basis_trainable": True,
            "trso_coefficient_mode": "grouped",
            "trso_channel_groups": 0,
            "trso_grouping_mode": "response",
            "trso_input_norm": "rms",
            "trso_calibration_grad_norm": "global_rms",
            "trso_residual_norm": "rms",
            "trso_residual_target": 0.05,
            "trso_residual_budget_mode": "global",
            "trso_gate_search": True,
            "trso_gate_search_values": "0.0,0.05,0.1,0.25,0.5,1.0",
            "trso_gate_search_batches": 16,
            "trso_head_warmup_steps": 0,
        })
    elif method == "conv":
        row.update({"adapt_size": 8, "kernel_size": 3, "conv_adapter_mode": "conv_parallel"})
    elif method == "sidetune":
        row.update({"sidetune_arch": "lightweight", "sidetune_width": 64, "sidetune_depth": 4})
    elif method == "piggyback":
        row.update({
            "piggyback_threshold": 5e-3,
            "piggyback_mask_init": "ones",
            "piggyback_mask_linear": False,
        })
    elif method == "ssf":
        row.update({"ssf_init_std": 0.02})
    elif method == "lora":
        row.update({"lora_r": 8, "lora_alpha": 16.0, "lora_dropout": 0.0})
    elif method == "bitfit":
        row.update({"bitfit_bias_scope": "all", "bitfit_train_head": True})
    elif method == "adaptformer":
        row.update({
            "adaptformer_dim": 16,
            "adaptformer_scale": 0.1,
            "adaptformer_dropout": 0.0,
            "adaptformer_layernorm": "none",
        })
    return row


def build_comparison_specs(args, seeds: list[int], head_paths: dict[tuple[str, int], str]) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for family, methods, backbone, source in (
        ("cnn", CNN_METHODS, args.cnn_backbone, "torchvision"),
        ("transformer", TRANSFORMER_METHODS, args.transformer_backbone, "timm"),
    ):
        common = {
            **base_common(args),
            "backbone": backbone,
            "model_source": source,
            "epochs": args.epochs,
        }
        variants = []
        for seed in seeds:
            for method in methods:
                variants.append((method, _method_variant(method, family, seed, head_paths[(family, seed)], args)))
        specs.extend(build_specs(
            suite=f"fair_dtd_{family}",
            variants=variants,
            common=common,
            output_root=args.output_root,
        ))

    # Residual Adapter is architecture-specific and needs the released shared
    # ResNet-26 checkpoint. It is included automatically when supplied, but is
    # kept in a separate table rather than mixed with ResNet-50 results.
    if args.ra_pretrained_checkpoint:
        common = {
            **base_common(args),
            "backbone": "resnet26_adapter",
            "model_source": "auto",
            "epochs": args.epochs,
            "ra_pretrained_checkpoint": args.ra_pretrained_checkpoint,
        }
        variants = [
            (f"residual_{mode}", {
                "tuning_method": "residual",
                "ra_mode": mode,
                "seed": seed,
                "weights": "DEFAULT",
            })
            for seed in seeds for mode in ("series", "parallel")
        ]
        specs.extend(build_specs(
            suite="fair_dtd_residual_adapter",
            variants=variants,
            common=common,
            output_root=args.output_root,
        ))
    return specs


def main() -> None:
    args = parser().parse_args()
    seeds = parse_csv_values(args.seeds, int)
    head_specs, head_paths = build_head_specs(args, seeds)
    comparison_specs = build_comparison_specs(args, seeds, head_paths)
    all_specs = [*head_specs, *comparison_specs]
    manifest, csv_manifest = write_manifest(all_specs, args.manifest)
    protocol = {
        "dataset": "DTD complete official partition 1",
        "seeds": seeds,
        "shared_peft_optimizer": "AdamW",
        "shared_peft_lr": args.peft_lr,
        "shared_scheduler": "cosine",
        "shared_warmup_epochs": args.warmup_epochs,
        "shared_weight_decay": args.weight_decay,
        "full_lr": args.full_lr,
        "linear_lr": args.linear_lr,
        "input_size": args.input_size,
        "cnn_methods": CNN_METHODS,
        "transformer_methods": TRANSFORMER_METHODS,
        "residual_adapter_included": bool(args.ra_pretrained_checkpoint),
        "residual_adapter_requirement": "--ra_pretrained_checkpoint",
    }
    protocol_path = Path(args.manifest).with_name("fair_dtd_protocol.json")
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Wrote {len(all_specs)} runs to {manifest} and {csv_manifest}")
    print(f"Protocol: {protocol_path}")
    if not args.ra_pretrained_checkpoint:
        print("[Info] Residual Adapter is listed as a separate architecture-specific prerequisite; "
              "provide --ra_pretrained_checkpoint to execute its series/parallel runs.")
    if args.execute:
        # Shared heads must exist before PEFT/TRSO runs begin. Independent head
        # and comparison runs can use one worker per selected GPU.
        gpu_ids = parse_csv_values(args.gpu_ids, int)[: max(1, int(args.parallel_runs))]
        if len(gpu_ids) > 1:
            execute_specs_parallel(head_specs, execute=True, gpu_ids=gpu_ids, max_runs=0)
            execute_specs_parallel(
                comparison_specs,
                execute=True,
                gpu_ids=gpu_ids,
                max_runs=args.max_runs,
            )
        else:
            execute_specs(head_specs, execute=True, max_runs=0)
            execute_specs(comparison_specs, execute=True, max_runs=args.max_runs)
    else:
        execute_specs(all_specs, execute=False, max_runs=args.max_runs)


if __name__ == "__main__":
    main()
