"""Generate or execute the paper-oriented TRSO ablation suite."""
from __future__ import annotations

import argparse
from pathlib import Path

from tools.experiment_grid import build_specs, execute_specs, parse_csv_values, write_manifest


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="fake")
    parser.add_argument("--data_path", default="./data")
    parser.add_argument("--weights", default="DEFAULT", help="Use none for offline architecture smoke tests.")
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--input_size", type=int, default=64)
    parser.add_argument("--nb_classes", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--parameter_budget", type=int, default=50000)
    parser.add_argument("--output_root", default="outputs_ablation")
    parser.add_argument("--manifest", default="experiments/generated_ablation_manifest.json")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max_runs", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser


def ablation_variants(budget: int):
    proposed = {
        "tuning_method": "trso",
        "trso_basis_source": "response",
        "trso_allocation": "exact",
        "trso_score_mode": "energy",
        "trso_parameter_budget": budget,
        "trso_calibration_batches": 8,
        "trso_kernel_size": 5,
        "trso_spatial_rank": 2,
        "trso_head_warmup_steps": 0,
    }
    rows = [("proposed", proposed)]
    controlled = {
        "basis_random": {"trso_basis_source": "random"},
        "basis_dct": {"trso_basis_source": "dct"},
        "allocation_greedy": {"trso_allocation": "greedy"},
        "allocation_uniform": {"trso_allocation": "uniform"},
        "score_per_param": {"trso_score_mode": "energy_per_param"},
        "score_per_channel": {"trso_score_mode": "energy_per_channel"},
        "score_noise_adjusted": {"trso_score_mode": "noise_adjusted", "trso_noise_beta": 0.1},
        "rank_1": {"trso_spatial_rank": 1},
        "rank_4": {"trso_spatial_rank": 4},
        "kernel_3": {"trso_kernel_size": 3},
        "kernel_7": {"trso_kernel_size": 7},
        "calibration_1": {"trso_calibration_batches": 1},
        "calibration_4": {"trso_calibration_batches": 4},
        "calibration_16": {"trso_calibration_batches": 16},
        "head_warmup_25": {"trso_head_warmup_steps": 25},
        "budget_half": {"trso_parameter_budget": max(1, budget // 2)},
        "budget_double": {"trso_parameter_budget": 2 * budget},
        "all_candidates": {"trso_parameter_budget": 0},
    }
    for name, update in controlled.items():
        variant = dict(proposed)
        variant.update(update)
        rows.append((name, variant))
    return rows


def main() -> None:
    args = get_parser().parse_args()
    common = {
        "dataset": args.dataset,
        "data_path": args.data_path,
        "weights": args.weights,
        "backbone": args.backbone,
        "input_size": args.input_size,
        "nb_classes": args.nb_classes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": args.device,
        "use_amp": True,
        "profile_efficiency": True,
        "save_ckpt": True,
        "final_test": True,
    }
    variants = []
    for seed in parse_csv_values(args.seeds, int):
        for name, values in ablation_variants(args.parameter_budget):
            variants.append((name, {**values, "seed": seed}))
    specs = build_specs(
        suite="trso_ablation",
        variants=variants,
        common=common,
        output_root=args.output_root,
    )
    manifest, csv_manifest = write_manifest(specs, args.manifest)
    print(f"Wrote {len(specs)} runs to {manifest} and {csv_manifest}")
    execute_specs(specs, execute=args.execute, max_runs=args.max_runs)


if __name__ == "__main__":
    main()
