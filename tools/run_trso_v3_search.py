"""Generate or execute a compact universal TRSO-v3 search.

The same task-aware head, optimizer, scheduler, split, augmentation, and training
length are reused for every variant. The search changes only one TRSO factor at
a time and works with any dataset/task/backbone accepted by ``main.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.experiment_grid import (
    build_specs,
    execute_specs,
    execute_specs_parallel,
    parse_csv_values,
    write_manifest,
)


def str2bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}: return True
    if text in {"0", "false", "no", "n", "off"}: return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {value!r}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="fake")
    p.add_argument("--task", default="auto", choices=["auto", "single_label", "multilabel", "regression"])
    p.add_argument("--dataset_args_json", default="{}")
    p.add_argument("--data_path", default="./data")
    p.add_argument("--download", type=str2bool, default=False)
    p.add_argument("--weights", default="DEFAULT")
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--model_source", default="auto")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--input_size", type=int, default=0)
    p.add_argument("--peft_lr", type=float, default=1e-3)
    p.add_argument("--linear_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--output_root", default="outputs_trso_v3_search")
    p.add_argument("--manifest", default="experiments/trso_v3_search.json")
    p.add_argument("--gpu_ids", default="0")
    p.add_argument("--parallel_runs", type=int, default=1)
    p.add_argument("--execute", action="store_true")
    return p


def main() -> None:
    args = parser().parse_args()
    dataset_extra = json.loads(args.dataset_args_json)
    if not isinstance(dataset_extra, dict):
        raise ValueError("--dataset_args_json must decode to an object")
    seeds = parse_csv_values(args.seeds, int)
    common = {
        "dataset": args.dataset,
        "task": args.task,
        "data_path": args.data_path,
        "download": args.download,
        "weights": args.weights,
        "backbone": args.backbone,
        "model_source": args.model_source,
        "input_size": args.input_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": "cuda",
        "fair_protocol": True,
        "fair_optimizer": "adamw",
        "fair_peft_lr": args.peft_lr,
        "fair_linear_lr": args.linear_lr,
        "fair_full_lr": 1e-4,
        "fair_weight_decay": args.weight_decay,
        "fair_warmup_epochs": min(args.warmup_epochs, max(0, args.epochs - 1)),
        "fair_min_lr": 1e-6,
        "train_aug": "standard",
        "aa": "rand-m9-mstd0.5-inc1" if args.task in {"auto", "single_label"} else "none",
        "color_jitter": 0.2 if args.task in {"auto", "single_label"} else 0.0,
        "mixup": 0.2 if args.task in {"auto", "single_label"} else 0.0,
        "cutmix": 0.0,
        "smoothing": 0.1 if args.task in {"auto", "single_label"} else 0.0,
        "reprob": 0.1 if args.task in {"auto", "single_label"} else 0.0,
        "split_seed": 2026,
        "profile_efficiency": True,
        "measure_eval_latency": True,
        "evaluate_before_training": True,
        "save_history": True,
        "save_ckpt": True,
        "final_test": True,
        "auto_resume": False,
        "peft_freeze_head": False,
        "peft_head_lr_scale": 0.5,
        "clip_grad": 1.0,
        "no_decay_bias_norm": True,
        **dataset_extra,
    }
    heads = build_specs(
        suite="trso_v3_head",
        variants=[("linear", {"tuning_method": "linear", "seed": seed}) for seed in seeds],
        common=common,
        output_root=args.output_root,
    )
    head_paths = {
        int(spec.parameters["seed"]): str(Path(spec.output_dir) / "checkpoint-best.pth")
        for spec in heads
    }
    base = {
        "tuning_method": "trso",
        "trso_variant": "v3",
        "trso_basis_source": "response",
        "trso_allocation": "exact",
        "trso_score_mode": "normalized_stable_energy_per_param",
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
        "trso_channel_response": True,
        "trso_prefix_coupling": True,
        "trso_prefix_coupling_mode": "all",
        "trso_v2_gate_init": 1.0,
        "trso_gate_search": True,
        "trso_gate_search_values": "0.0,0.05,0.1,0.25,0.5,1.0",
        "trso_gate_search_batches": 16,
        "trso_parameter_budget": 0,
        "trso_auto_budget_ratio": 0.35,
        "trso_auto_sparse": True,
    }
    changes = {
        "proposed_v3": {},
        "v2_previous": {
            "trso_variant": "v2", "trso_channel_groups": 8,
            "trso_grouping_mode": "contiguous", "trso_calibration_grad_norm": "none",
            "trso_residual_norm": "none", "trso_score_mode": "snr_per_param",
            "trso_parameter_budget": 512, "trso_spatial_rank": 2,
            "trso_basis_trainable": False, "trso_residual_budget_mode": "per_layer",
        },
        "v1_original": {
            "trso_variant": "v1", "trso_coefficient_mode": "full",
            "trso_input_norm": "none", "trso_channel_response": False,
            "trso_prefix_coupling": False, "trso_gate_search": False,
            "trso_gate_init": 1e-2, "trso_score_mode": "energy",
            "trso_calibration_grad_norm": "none", "trso_residual_norm": "none",
            "trso_grouping_mode": "contiguous",
            "trso_parameter_budget": 12000, "trso_spatial_rank": 2,
            "trso_basis_trainable": False, "trso_residual_budget_mode": "per_layer",
        },
        "per_layer_gradient_rms": {"trso_calibration_grad_norm": "rms"},
        "no_response_grouping": {"trso_grouping_mode": "contiguous"},
        "no_residual_norm": {"trso_residual_norm": "none"},
        "per_layer_residual_budget": {"trso_residual_budget_mode": "per_layer"},
        "no_channel_response": {"trso_channel_response": False},
        "no_prefix_coupling": {"trso_prefix_coupling": False},
        "rank_2": {"trso_spatial_rank": 2},
        "rank_6": {"trso_spatial_rank": 6},
        "basis_frozen": {"trso_basis_trainable": False},
        "groups_4": {"trso_channel_groups": 4},
        "groups_16": {"trso_channel_groups": 16},
        "capacity_20pct": {"trso_auto_budget_ratio": 0.20},
        "capacity_50pct": {"trso_auto_budget_ratio": 0.50},
        "max_4_adapters": {"trso_max_adapters": 4},
        "max_8_adapters": {"trso_max_adapters": 8},
        "target_0025": {"trso_residual_target": 0.025},
        "target_010": {"trso_residual_target": 0.10},
        "score_snr": {"trso_score_mode": "snr_per_param"},
        "score_stability": {"trso_score_mode": "stability_per_param"},
        "frozen_shared_head": {"peft_freeze_head": True},
    }
    variants = []
    for seed in seeds:
        for name, update in changes.items():
            variants.append((name, {**base, **update, "seed": seed, "head_from": head_paths[seed]}))
    runs = build_specs(
        suite="trso_v3_search",
        variants=variants,
        common=common,
        output_root=args.output_root,
    )
    write_manifest([*heads, *runs], args.manifest)
    if args.execute:
        gpus = parse_csv_values(args.gpu_ids, int)[: max(1, args.parallel_runs)]
        if len(gpus) > 1:
            execute_specs_parallel(heads, execute=True, gpu_ids=gpus)
            execute_specs_parallel(runs, execute=True, gpu_ids=gpus)
        else:
            execute_specs(heads, execute=True)
            execute_specs(runs, execute=True)
    else:
        execute_specs([*heads, *runs], execute=False)


if __name__ == "__main__":
    main()
