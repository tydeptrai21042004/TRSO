"""Generate/execute a focused TRSO-v2 search with a shared frozen linear head.

The search is deliberately small: it changes one scientific factor at a time
while preserving the same optimizer, scheduler, data split, and task-aware head.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from tools.experiment_grid import build_specs, execute_specs, execute_specs_parallel, parse_csv_values, write_manifest


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="dtd")
    p.add_argument("--task", default="auto")
    p.add_argument("--data_path", default="./data")
    p.add_argument("--download", default="True")
    p.add_argument("--backbone", default="resnet50")
    p.add_argument("--model_source", default="torchvision")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--peft_lr", type=float, default=5e-3)
    p.add_argument("--output_root", default="outputs_trso_v2_search")
    p.add_argument("--manifest", default="experiments/trso_v2_search.json")
    p.add_argument("--gpu_ids", default="0")
    p.add_argument("--parallel_runs", type=int, default=1)
    p.add_argument("--execute", action="store_true")
    return p


def main():
    args = parser().parse_args()
    seeds = parse_csv_values(args.seeds, int)
    common = {
        "dataset": args.dataset, "task": args.task, "data_path": args.data_path,
        "download": args.download, "backbone": args.backbone,
        "model_source": args.model_source, "input_size": 0, "epochs": args.epochs,
        "batch_size": args.batch_size, "num_workers": 4, "device": "cuda",
        "fair_protocol": True, "fair_optimizer": "adamw",
        "fair_peft_lr": args.peft_lr, "fair_linear_lr": 1e-1,
        "fair_full_lr": 1e-4, "fair_weight_decay": 1e-4,
        "fair_warmup_epochs": min(5, max(0, args.epochs - 1)), "fair_min_lr": 1e-6,
        "train_aug": "standard", "aa": "none", "mixup": 0.0, "cutmix": 0.0,
        "smoothing": 0.0, "reprob": 0.0, "split_seed": 2026,
        "profile_efficiency": True, "measure_eval_latency": True,
        "evaluate_before_training": True, "save_history": True,
        "save_ckpt": True, "final_test": True, "auto_resume": False,
    }
    heads = build_specs(
        suite="trso_v2_head", variants=[("linear", {"tuning_method": "linear", "seed": s}) for s in seeds],
        common=common, output_root=args.output_root,
    )
    head_paths = {int(x.parameters["seed"]): str(Path(x.output_dir) / "checkpoint-best.pth") for x in heads}
    base = {
        "tuning_method": "trso", "trso_variant": "v2", "trso_basis_source": "response",
        "trso_allocation": "exact", "trso_score_mode": "snr_per_param",
        "trso_calibration_batches": 16, "trso_kernel_size": 5,
        "trso_spatial_rank": 2, "trso_coefficient_mode": "grouped",
        "trso_channel_groups": 8, "trso_input_norm": "rms",
        "trso_channel_response": True, "trso_prefix_coupling": True,
        "trso_v2_gate_init": 1.0, "trso_gate_search": True,
        "trso_parameter_budget": 512, "peft_freeze_head": True,
    }
    changes = {
        "proposed_v2": {},
        "v1_original": {"trso_variant": "v1", "trso_coefficient_mode": "full", "trso_input_norm": "none", "trso_channel_response": False, "trso_prefix_coupling": False, "trso_gate_search": False, "trso_gate_init": 1e-2, "trso_score_mode": "energy", "trso_parameter_budget": 12000},
        "groups_4": {"trso_channel_groups": 4},
        "groups_16": {"trso_channel_groups": 16},
        "rank_1": {"trso_spatial_rank": 1},
        "rank_4": {"trso_spatial_rank": 4},
        "budget_256": {"trso_parameter_budget": 256},
        "budget_1024": {"trso_parameter_budget": 1024},
        "no_channel": {"trso_channel_response": False},
        "no_rms": {"trso_input_norm": "none"},
        "locked": {"trso_coefficient_mode": "locked"},
    }
    variants = []
    for seed in seeds:
        for name, update in changes.items():
            variants.append((name, {**base, **update, "seed": seed, "head_from": head_paths[seed]}))
    runs = build_specs(suite="trso_v2_search", variants=variants, common=common, output_root=args.output_root)
    write_manifest([*heads, *runs], args.manifest)
    if args.execute:
        gpus = parse_csv_values(args.gpu_ids, int)[:max(1, args.parallel_runs)]
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
