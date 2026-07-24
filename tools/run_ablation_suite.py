"""Generate or execute a fair, task-aware TRSO ablation suite.

Every ablation starts from the same best linear-probe head for a given
backbone/seed, preventing random-head calibration from dominating the result.
The runner supports every dataset/task accepted by the main dataset router.
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


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="fake")
    parser.add_argument("--task", default="auto", choices=["auto", "single_label", "multilabel", "regression"])
    parser.add_argument("--dataset_args_json", default="{}")
    parser.add_argument("--data_path", default="./data")
    parser.add_argument("--download", type=str2bool, default=False)
    parser.add_argument("--weights", default="DEFAULT", help="Use none for offline architecture smoke tests.")
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--model_source", default="auto", choices=["auto", "torchvision", "timm", "hub"])
    parser.add_argument("--input_size", type=int, default=0, help="0 resolves native pretrained size.")
    parser.add_argument("--nb_classes", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--split_seed", type=int, default=2026)
    parser.add_argument("--parameter_budget", type=int, default=0, help="0 uses the automatic V3 candidate-capacity budget.")
    parser.add_argument("--peft_lr", type=float, default=1e-3)
    parser.add_argument("--linear_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--output_root", default="outputs_ablation")
    parser.add_argument("--manifest", default="experiments/generated_ablation_manifest.json")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max_runs", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu_ids", default="0")
    parser.add_argument("--parallel_runs", type=int, default=1)
    return parser


def ablation_variants(budget: int):
    proposed = {
        "tuning_method": "trso",
        "trso_variant": "v3",
        "trso_basis_source": "response",
        "trso_allocation": "exact",
        "trso_score_mode": "normalized_stable_energy_per_param",
        "trso_noise_beta": 0.0,
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
        "trso_parameter_budget": budget,
        "trso_auto_budget_ratio": 0.35,
        "trso_auto_sparse": True,
        "trso_calibration_batches": 16,
        "trso_kernel_size": 5,
        "trso_spatial_rank": 4,
        "trso_basis_trainable": True,
        "trso_gate_search_values": "0.0,0.05,0.1,0.25,0.5,1.0",
        "trso_gate_search_batches": 16,
        "trso_head_warmup_steps": 0,
    }
    rows = [("proposed", proposed)]
    controlled = {
        "v1_original": {
            "trso_variant": "v1", "trso_coefficient_mode": "full",
            "trso_input_norm": "none", "trso_channel_response": False,
            "trso_prefix_coupling": False, "trso_gate_init": 1e-2,
            "trso_calibration_grad_norm": "none", "trso_residual_norm": "none",
            "trso_grouping_mode": "contiguous", "trso_score_mode": "energy",
            "trso_parameter_budget": 12000, "trso_spatial_rank": 2,
            "trso_basis_trainable": False, "trso_residual_budget_mode": "per_layer",
        },
        "v2_previous": {
            "trso_variant": "v2", "trso_channel_groups": 8,
            "trso_grouping_mode": "contiguous", "trso_calibration_grad_norm": "none",
            "trso_residual_norm": "none", "trso_score_mode": "snr_per_param",
            "trso_spatial_rank": 2, "trso_basis_trainable": False,
            "trso_residual_budget_mode": "per_layer",
        },
        "v3_no_channel_response": {"trso_channel_response": False},
        "v3_no_input_norm": {"trso_input_norm": "none"},
        "v3_per_layer_gradient_rms": {"trso_calibration_grad_norm": "rms"},
        "v3_no_residual_norm": {"trso_residual_norm": "none"},
        "v3_per_layer_residual_budget": {"trso_residual_budget_mode": "per_layer"},
        "v3_contiguous_groups": {"trso_grouping_mode": "contiguous"},
        "v3_full_coefficients": {"trso_coefficient_mode": "full", "trso_parameter_budget": 12000},
        "v3_locked_coefficients": {"trso_coefficient_mode": "locked"},
        "v3_groups_1": {"trso_channel_groups": 1},
        "v3_groups_4": {"trso_channel_groups": 4},
        "v3_groups_16": {"trso_channel_groups": 16},
        "v3_no_prefix_coupling": {"trso_prefix_coupling": False},
        "v3_prefix_first": {"trso_prefix_coupling_mode": "first"},
        "basis_random": {"trso_basis_source": "random"},
        "basis_dct": {"trso_basis_source": "dct"},
        "allocation_greedy": {"trso_allocation": "greedy"},
        "allocation_uniform": {"trso_allocation": "uniform"},
        "score_energy": {"trso_score_mode": "energy", "trso_noise_beta": 0.0},
        "score_stability": {"trso_score_mode": "stability_per_param", "trso_noise_beta": 0.0},
        "score_noise_adjusted": {"trso_score_mode": "noise_adjusted", "trso_noise_beta": 0.1},
        "score_per_param": {"trso_score_mode": "energy_per_param", "trso_noise_beta": 0.0},
        "score_per_channel": {"trso_score_mode": "energy_per_channel", "trso_noise_beta": 0.0},
        "noise_beta_0": {"trso_noise_beta": 0.0},
        "noise_beta_05": {"trso_noise_beta": 0.5},
        "rank_1": {"trso_spatial_rank": 1},
        "rank_2": {"trso_spatial_rank": 2},
        "rank_6": {"trso_spatial_rank": 6},
        "basis_frozen": {"trso_basis_trainable": False},
        "kernel_3": {"trso_kernel_size": 3},
        "kernel_7": {"trso_kernel_size": 7},
        "calibration_1": {"trso_calibration_batches": 1},
        "calibration_4": {"trso_calibration_batches": 4},
        "calibration_32": {"trso_calibration_batches": 32},
        "capacity_20pct": {"trso_parameter_budget": 0, "trso_auto_budget_ratio": 0.20},
        "capacity_50pct": {"trso_parameter_budget": 0, "trso_auto_budget_ratio": 0.50},
        "budget_half": (
            {"trso_parameter_budget": max(1, budget // 2)}
            if budget > 0 else
            {"trso_parameter_budget": 0, "trso_auto_budget_ratio": 0.175}
        ),
        "max_4_adapters": {"trso_max_adapters": 4},
        "max_8_adapters": {"trso_max_adapters": 8},
        "frozen_shared_head": {"peft_freeze_head": True},
    }
    for name, update in controlled.items():
        variant = dict(proposed)
        variant.update(update)
        rows.append((name, variant))
    return rows


def common_args(args):
    extra = json.loads(args.dataset_args_json)
    if not isinstance(extra, dict):
        raise ValueError("--dataset_args_json must decode to an object")
    common = {
        "dataset": args.dataset,
        "task": args.task,
        "data_path": args.data_path,
        "download": args.download,
        "weights": args.weights,
        "backbone": args.backbone,
        "model_source": args.model_source,
        "input_size": args.input_size,
        "nb_classes": args.nb_classes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": args.device,
        "use_amp": True,
        "profile_efficiency": True,
        "measure_eval_latency": True,
        "save_ckpt": True,
        "save_ckpt_freq": 1,
        "save_history": True,
        "evaluate_before_training": True,
        "final_test": True,
        "auto_resume": False,
        "fair_protocol": True,
        "fair_optimizer": "adamw",
        "fair_peft_lr": args.peft_lr,
        "fair_full_lr": args.peft_lr,
        "fair_linear_lr": args.linear_lr,
        "fair_weight_decay": args.weight_decay,
        "fair_warmup_epochs": min(int(args.warmup_epochs), max(0, int(args.epochs) - 1)),
        "fair_min_lr": 1e-6,
        "paper_hparams": False,
        "legacy_auto_hparams": False,
        "split_seed": args.split_seed,
        "train_aug": "standard",
        "aa": "rand-m9-mstd0.5-inc1" if args.task in {"auto", "single_label"} else "none",
        "color_jitter": 0.2 if args.task in {"auto", "single_label"} else 0.0,
        "mixup": 0.2 if args.task in {"auto", "single_label"} else 0.0,
        "cutmix": 0.0,
        "smoothing": 0.1 if args.task in {"auto", "single_label"} else 0.0,
        "reprob": 0.1 if args.task in {"auto", "single_label"} else 0.0,
        "keep_pretrained_head": False,
        "peft_freeze_head": False,
        "peft_head_lr_scale": 0.5,
        "clip_grad": 1.0,
        "no_decay_bias_norm": True,
    }
    common.update(extra)
    return common


def main() -> None:
    args = get_parser().parse_args()
    seeds = parse_csv_values(args.seeds, int)
    common = common_args(args)

    head_specs = build_specs(
        suite="trso_ablation_head_preparation",
        variants=[("linear", {"tuning_method": "linear", "seed": seed}) for seed in seeds],
        common=common,
        output_root=args.output_root,
    )
    head_paths = {
        int(spec.parameters["seed"]): str(Path(spec.output_dir) / "checkpoint-best.pth")
        for spec in head_specs
    }

    variants = []
    for seed in seeds:
        for name, values in ablation_variants(args.parameter_budget):
            variants.append((name, {**values, "seed": seed, "head_from": head_paths[seed]}))
    ablation_specs = build_specs(
        suite="trso_ablation",
        variants=variants,
        common=common,
        output_root=args.output_root,
    )
    specs = [*head_specs, *ablation_specs]
    manifest, csv_manifest = write_manifest(specs, args.manifest)
    protocol = {
        "dataset": args.dataset,
        "task": args.task,
        "backbone": args.backbone,
        "model_source": args.model_source,
        "seeds": seeds,
        "shared_head": "best linear-probe checkpoint per seed",
        "shared_optimizer": "AdamW",
        "shared_lr": args.peft_lr,
        "shared_scheduler": "cosine",
        "shared_weight_decay": args.weight_decay,
        "shared_warmup_epochs": args.warmup_epochs,
        "variants": [name for name, _ in ablation_variants(args.parameter_budget)],
    }
    protocol_path = Path(args.manifest).with_name(Path(args.manifest).stem + "_protocol.json")
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Wrote {len(specs)} runs to {manifest} and {csv_manifest}")

    if args.execute:
        gpu_ids = parse_csv_values(args.gpu_ids, int)[: max(1, int(args.parallel_runs))]
        if len(gpu_ids) > 1:
            execute_specs_parallel(head_specs, execute=True, gpu_ids=gpu_ids)
            execute_specs_parallel(ablation_specs, execute=True, gpu_ids=gpu_ids, max_runs=args.max_runs)
        else:
            execute_specs(head_specs, execute=True)
            execute_specs(ablation_specs, execute=True, max_runs=args.max_runs)
    else:
        execute_specs(specs, execute=False, max_runs=args.max_runs)


if __name__ == "__main__":
    main()
