"""Generate or execute reproducible baseline/TRSO hyperparameter sweeps.

The default strategy is one-factor-at-a-time (OFAT): it keeps a documented
reference setting and changes exactly one hyperparameter per run. This is much
more interpretable than an unconstrained Cartesian grid. Use ``--strategy grid``
only when the resulting run count is computationally affordable.
"""
from __future__ import annotations

import argparse
import json

from tools.experiment_grid import (
    build_specs,
    execute_specs,
    expand_grid,
    one_factor_grid,
    parse_csv_values,
    write_manifest,
)


METHOD_SPACES = {
    "full": (
        {"optimizer": "sgd", "lr": 0.1, "weight_decay": 1e-4, "momentum": 0.9},
        {"lr": [0.01, 0.03, 0.1, 0.3], "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
         "momentum": [0.85, 0.9, 0.95]},
    ),
    "linear": (
        {"optimizer": "sgd", "lr": 0.1, "weight_decay": 0.0, "momentum": 0.9},
        {"lr": [0.01, 0.03, 0.1, 0.3, 1.0], "weight_decay": [0.0, 1e-5, 1e-4],
         "momentum": [0.85, 0.9, 0.95]},
    ),
    "trso": (
        {"lr": 3e-4, "weight_decay": 0.05, "trso_kernel_size": 5, "trso_spatial_rank": 2,
         "trso_operator_radius": 1.0, "trso_gate_init": 1e-2, "trso_calibration_batches": 8},
        {"lr": [1e-4, 3e-4, 1e-3], "weight_decay": [0.0, 0.01, 0.05],
         "trso_kernel_size": [3, 5, 7], "trso_spatial_rank": [1, 2, 4],
         "trso_operator_radius": [0.5, 1.0, 2.0], "trso_gate_init": [1e-3, 1e-2, 1e-1],
         "trso_calibration_batches": [1, 4, 8, 16]},
    ),
    "conv": (
        {"lr": 1e-3, "weight_decay": 0.0, "adapt_size": 8, "adapt_scale": 1.0,
         "kernel_size": 3, "conv_adapter_mode": "conv_parallel"},
        {"lr": [1e-4, 3e-4, 1e-3, 3e-3], "adapt_size": [4, 8, 16, 32],
         "adapt_scale": [0.1, 0.5, 1.0]},
    ),
    "prompt": (
        {"optimizer": "sgd", "lr": 40.0, "weight_decay": 0.0,
         "prompt_size": 30, "prompt_type": "padding"},
        {"lr": [1.0, 10.0, 40.0], "prompt_size": [5, 10, 20, 30],
         "prompt_type": ["padding", "fixed_patch", "random_patch"]},
    ),
    "ssf": (
        {"lr": 1e-3, "weight_decay": 0.0, "ssf_init_std": 0.02},
        {"lr": [1e-4, 3e-4, 1e-3, 3e-3], "ssf_init_std": [0.0, 0.01, 0.02, 0.05]},
    ),
    "lora": (
        {"lr": 1e-3, "weight_decay": 0.0, "lora_r": 8, "lora_alpha": 16.0,
         "lora_dropout": 0.0},
        {"lr": [1e-4, 3e-4, 1e-3], "lora_r": [2, 4, 8, 16],
         "lora_alpha": [4.0, 8.0, 16.0, 32.0]},
    ),
    "bam": (
        {"optimizer": "sgd", "lr": 0.1, "weight_decay": 1e-4,
         "bam_reduction": 16, "bam_dilation": 4},
        {"lr": [0.03, 0.1, 0.3], "bam_reduction": [8, 16, 32],
         "bam_dilation": [2, 4, 6]},
    ),
    "residual": (
        {"optimizer": "sgd", "lr": 0.1, "weight_decay": 1e-4,
         "ra_mode": "parallel"},
        {"lr": [0.01, 0.03, 0.1], "ra_mode": ["parallel", "series"]},
    ),
    "bitfit": (
        {"lr": 1e-3, "weight_decay": 0.0, "bitfit_bias_scope": "all",
         "bitfit_train_head": True},
        {"lr": [1e-4, 3e-4, 1e-3, 3e-3],
         "bitfit_bias_scope": ["all", "transformer", "attention"],
         "bitfit_train_head": [True, False]},
    ),
    "adaptformer": (
        {"lr": 1e-3, "weight_decay": 0.0, "adaptformer_dim": 16,
         "adaptformer_scale": 0.1, "adaptformer_dropout": 0.0},
        {"lr": [3e-4, 1e-3, 3e-3, 5e-3], "adaptformer_dim": [4, 8, 16, 32],
         "adaptformer_scale": [0.05, 0.1, 0.5, 1.0]},
    ),
    "piggyback": (
        {"lr": 1e-3, "weight_decay": 1e-4, "piggyback_threshold": 5e-3,
         "piggyback_mask_init": "ones", "piggyback_mask_linear": False},
        {"lr": [3e-4, 1e-3, 3e-3, 5e-3],
         "piggyback_threshold": [0.0, 5e-3, 1e-2],
         "piggyback_mask_init": ["ones", "near_threshold"]},
    ),
    "sidetune": (
        {"optimizer": "sgd", "lr": 1e-3, "weight_decay": 1e-4,
         "sidetune_width": 64, "sidetune_depth": 4,
         "sidetune_alpha": 0.5, "sidetune_arch": "lightweight"},
        {"lr": [1e-4, 3e-4, 1e-3], "sidetune_width": [32, 64, 128],
         "sidetune_depth": [3, 4, 5], "sidetune_alpha": [0.25, 0.5, 0.75]},
    ),
}

# Backbones are selected to satisfy each paper-domain implementation by default.
DEFAULT_BACKBONES = {
    "full": "resnet18",
    "linear": "resnet18",
    "trso": "resnet18",
    "conv": "resnet50",
    "prompt": "resnet18",
    "ssf": "vit_b_16",
    "lora": "vit_b_16",
    "bam": "resnet50",
    "residual": "resnet26_adapter",
    "bitfit": "vit_b_16",
    "adaptformer": "vit_tiny_patch16_224",
    "piggyback": "resnet18",
    "sidetune": "resnet18",
}


def build_hparam_rows(method: str, strategy: str = "one_factor") -> list[dict]:
    """Build rows while retaining reference constants outside the swept axes."""
    if method not in METHOD_SPACES:
        raise KeyError(f"Unknown method {method!r}")
    defaults, candidates = METHOD_SPACES[method]
    if strategy == "one_factor":
        return one_factor_grid(defaults, candidates)
    if strategy == "grid":
        fixed = {key: value for key, value in defaults.items() if key not in candidates}
        return expand_grid(fixed, candidates)
    raise ValueError("strategy must be 'one_factor' or 'grid'")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=sorted(METHOD_SPACES))
    parser.add_argument("--strategy", default="one_factor", choices=["one_factor", "grid"])
    parser.add_argument("--dataset", default="fake")
    parser.add_argument("--task", default="auto", choices=["auto", "single_label", "multilabel", "regression"])
    parser.add_argument("--dataset_args_json", default="{}")
    parser.add_argument("--data_path", default="./data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--weights", default="DEFAULT", help="Use none for offline architecture smoke tests.")
    parser.add_argument(
        "--backbone",
        default="",
        help="Empty selects the method-compatible default; override explicitly for real experiments.",
    )
    parser.add_argument("--model_source", default="auto", choices=["auto", "torchvision", "timm", "hub"])
    parser.add_argument("--input_size", type=int, default=0, help="0 resolves native pretrained size.")
    parser.add_argument("--nb_classes", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--split_seed", type=int, default=2026)
    parser.add_argument("--output_root", default="outputs_hparams")
    parser.add_argument("--manifest", default="experiments/generated_hparam_manifest.json")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max_runs", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ra_pretrained_checkpoint", default="")
    return parser


def main() -> None:
    args = get_parser().parse_args()
    rows = build_hparam_rows(args.method, args.strategy)
    variants = []
    for seed in parse_csv_values(args.seeds, int):
        for index, row in enumerate(rows):
            variants.append((f"{args.method}_{index:04d}", {"tuning_method": args.method, "seed": seed, **row}))
    dataset_args = json.loads(args.dataset_args_json)
    if not isinstance(dataset_args, dict):
        raise ValueError("--dataset_args_json must decode to an object")
    common = {
        "dataset": args.dataset,
        "task": args.task,
        "data_path": args.data_path,
        "download": args.download,
        "weights": args.weights,
        "backbone": args.backbone or DEFAULT_BACKBONES[args.method],
        "model_source": args.model_source,
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
        "split_seed": args.split_seed,
        **dataset_args,
    }
    if args.method == "residual":
        common["ra_pretrained_checkpoint"] = args.ra_pretrained_checkpoint
        if not args.ra_pretrained_checkpoint:
            common["weights"] = "none"
    specs = build_specs(
        suite=f"hparams_{args.method}", variants=variants, common=common, output_root=args.output_root
    )
    manifest, csv_manifest = write_manifest(specs, args.manifest)
    print(f"Wrote {len(specs)} runs to {manifest} and {csv_manifest}")
    execute_specs(specs, execute=args.execute, max_runs=args.max_runs)


if __name__ == "__main__":
    main()
