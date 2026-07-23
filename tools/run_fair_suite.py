"""Dataset/task/backbone-aware controlled comparison runner.

The runner is intentionally capability driven:
- all task-compatible methods use one AdamW/cosine recipe;
- full fine-tuning and linear probing may use separate learning rates;
- a task-aware linear head is trained once per backbone/seed and reused by
  compatible PEFT methods before TRSO calibration;
- unsupported method/backbone/task combinations are written to an explicit
  compatibility report instead of disappearing from the result table.

Examples
--------
Single-label classification on two backbones::

    python -m tools.run_fair_suite \
      --dataset dtd --data_path ./data --download True \
      --backbones resnet50@torchvision,vit_tiny_patch16_224@timm \
      --seeds 0,1,2 --execute

Multi-label CSV data::

    python -m tools.run_fair_suite \
      --dataset csv --task multilabel --data_path ./dataset \
      --dataset_args_json '{"csv_train":"train.csv","csv_val":"val.csv",\
                            "csv_test":"test.csv","csv_target_columns":"a,b,c"}' \
      --backbones resnet50@torchvision,swin_t@torchvision
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.build import available_datasets
from models.model_support import (
    METHOD_SUPPORT,
    canonical_method,
    infer_backbone_family_name,
    normalize_task,
    static_method_compatibility,
)
from tools.experiment_grid import (
    RunSpec,
    build_specs,
    execute_specs,
    execute_specs_parallel,
    parse_csv_values,
    write_manifest,
)

DEFAULT_METHODS = tuple(METHOD_SUPPORT.keys())

SINGLE_LABEL_DATASETS = {
    "cifar10", "cifar100", "mnist", "fashion_mnist", "emnist", "kmnist",
    "qmnist", "usps", "svhn", "stl10", "food101", "oxfordiiitpet",
    "flowers102", "stanford_cars", "caltech101", "dtd", "eurosat",
    "fgvc_aircraft", "sun397", "gtsrb", "fer2013", "pcam", "country211",
    "rendered_sst2", "places365", "inaturalist", "imagefolder", "cub200",
    "nabirds", "stanford_dogs", "vtab", "fewshot",
}


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="dtd", choices=available_datasets())
    p.add_argument("--task", default="auto", choices=["auto", "single_label", "multilabel", "regression"])
    p.add_argument("--data_path", default="./data")
    p.add_argument("--download", type=str2bool, default=False)
    p.add_argument("--dataset_args_json", default="{}", help="Extra dataset CLI arguments as a JSON object.")
    p.add_argument("--output_root", default="outputs_fair")
    p.add_argument("--manifest", default="experiments/fair_manifest.json")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--split_seed", type=int, default=2026)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--input_size", type=int, default=0, help="0 resolves native pretrained size before dataset transforms.")
    p.add_argument(
        "--backbones",
        default="resnet50@torchvision,vit_tiny_patch16_224@timm",
        help="Comma-separated backbone@source entries.",
    )
    p.add_argument("--methods", default="auto", help="Comma-separated methods or 'auto' for every registered baseline.")
    p.add_argument("--peft_lr", type=float, default=5e-3)
    p.add_argument("--full_lr", type=float, default=1e-4)
    p.add_argument("--linear_lr", type=float, default=1e-1)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    p.add_argument("--trso_budget", type=int, default=12000)
    p.add_argument("--trso_calibration_batches", type=int, default=16)
    p.add_argument("--ra_pretrained_checkpoint", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu_ids", default="0")
    p.add_argument("--parallel_runs", type=int, default=1)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--max_runs", type=int, default=0)
    p.add_argument("--profile_efficiency", type=str2bool, default=True)
    p.add_argument("--measure_eval_latency", type=str2bool, default=True)
    p.add_argument("--allow_val_as_test", type=str2bool, default=False)
    return p


def resolve_task(args: argparse.Namespace, dataset_args: dict[str, Any]) -> str:
    requested = normalize_task(args.task)
    if requested != "auto":
        return requested
    dataset = args.dataset
    if dataset in SINGLE_LABEL_DATASETS:
        return "single_label"
    if dataset == "coco":
        return "single_label" if str(dataset_args.get("coco_task", "multilabel")).lower() == "majority" else "multilabel"
    if dataset == "voc2007":
        return "multilabel"
    if dataset == "celeba":
        return "regression" if str(dataset_args.get("celeba_task", "attributes")).lower() == "landmarks" else "multilabel"
    if dataset == "fake":
        return "single_label"
    if dataset == "csv":
        raise ValueError("--task must be explicit for --dataset csv.")
    raise ValueError(f"Could not infer task for dataset {dataset!r}; pass --task explicitly.")


def parse_backbones(text: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        if "@" in token:
            backbone, source = token.rsplit("@", 1)
        else:
            backbone, source = token, "auto"
        rows.append((backbone.strip(), source.strip()))
    if not rows:
        raise ValueError("At least one backbone is required.")
    return rows


def parse_methods(text: str) -> list[str]:
    if str(text).strip().lower() == "auto":
        # ``adapter`` is a legacy alias of Conv-Adapter, not a distinct paper row.
        return [method for method in DEFAULT_METHODS if method != "adapter"]
    methods = [canonical_method(item) for item in str(text).split(",") if item.strip()]
    unknown = [method for method in methods if method not in METHOD_SUPPORT]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Available: {sorted(METHOD_SUPPORT)}")
    # Conv-Adapter's legacy alias must not appear as a duplicate paper row.
    unique: list[str] = []
    for method in methods:
        method = "conv" if method == "adapter" else method
        if method not in unique:
            unique.append(method)
    return unique


def base_common(args: argparse.Namespace, task: str, dataset_args: dict[str, Any]) -> dict[str, Any]:
    common = {
        "dataset": args.dataset,
        "task": task,
        "data_path": args.data_path,
        "download": args.download,
        "input_size": args.input_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": args.device,
        "use_amp": True,
        "profile_efficiency": args.profile_efficiency,
        "measure_eval_latency": args.measure_eval_latency,
        "save_ckpt": True,
        "save_ckpt_freq": 1,
        "save_history": True,
        "final_test": True,
        "auto_resume": False,
        "fair_protocol": True,
        "fair_optimizer": args.optimizer,
        "fair_peft_lr": args.peft_lr,
        "fair_full_lr": args.full_lr,
        "fair_linear_lr": args.linear_lr,
        "fair_weight_decay": args.weight_decay,
        "fair_warmup_epochs": min(int(args.warmup_epochs), max(0, int(args.epochs) - 1)),
        "fair_min_lr": args.min_lr,
        "paper_hparams": False,
        "legacy_auto_hparams": False,
        "train_aug": "standard",
        "aa": "none",
        "color_jitter": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "smoothing": 0.0,
        "reprob": 0.0,
        "keep_pretrained_head": False,
        "split_seed": args.split_seed,
        "allow_val_as_test": args.allow_val_as_test,
    }
    common.update(dataset_args)
    return common


def method_variant(
    method: str,
    family: str,
    seed: int,
    head_path: str | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "tuning_method": method,
        "seed": seed,
        "keep_pretrained_head": method == "prompt",
    }
    if method == "prompt":
        row.update({"prompt_mapping": "frequency", "prompt_mapping_batches": 0})
    elif method not in {"full", "linear", "residual"} and head_path:
        row["head_from"] = head_path

    if method == "trso":
        row.update({
            "trso_basis_source": "response",
            "trso_allocation": "exact",
            "trso_score_mode": "noise_adjusted",
            "trso_noise_beta": 0.25,
            "trso_parameter_budget": args.trso_budget,
            "trso_calibration_batches": args.trso_calibration_batches,
            "trso_kernel_size": 5,
            "trso_spatial_rank": 2,
            "trso_head_warmup_steps": 0,
        })
    elif method == "conv":
        row.update({"adapt_size": 8, "kernel_size": 3, "conv_adapter_mode": "conv_parallel"})
    elif method == "sidetune":
        row.update({"sidetune_arch": "lightweight", "sidetune_width": 64, "sidetune_depth": 4})
    elif method == "piggyback":
        row.update({"piggyback_threshold": 5e-3, "piggyback_mask_init": "ones", "piggyback_mask_linear": False})
    elif method == "ssf":
        row.update({"ssf_init_std": 0.02})
    elif method == "lora":
        row.update({"lora_r": 8, "lora_alpha": 16.0, "lora_dropout": 0.0})
    elif method == "bitfit":
        row.update({"bitfit_bias_scope": "all", "bitfit_train_head": True})
    elif method == "adaptformer":
        row.update({"adaptformer_dim": 16, "adaptformer_scale": 0.1, "adaptformer_dropout": 0.0, "adaptformer_layernorm": "none"})
    elif method == "residual":
        row.update({"ra_pretrained_checkpoint": args.ra_pretrained_checkpoint, "ra_mode": "parallel"})
    return row


def build_suite(args: argparse.Namespace):
    dataset_args = json.loads(args.dataset_args_json)
    if not isinstance(dataset_args, dict):
        raise ValueError("--dataset_args_json must decode to a JSON object.")
    task = resolve_task(args, dataset_args)
    seeds = parse_csv_values(args.seeds, int)
    backbones = parse_backbones(args.backbones)
    methods = parse_methods(args.methods)
    common_base = base_common(args, task, dataset_args)

    head_specs: list[RunSpec] = []
    comparison_specs: list[RunSpec] = []
    compatibility: list[dict[str, Any]] = []

    for backbone, source in backbones:
        family = infer_backbone_family_name(backbone, source)
        supported_methods: list[str] = []
        for method in methods:
            ok, reason, resolved_family = static_method_compatibility(
                method,
                backbone,
                task,
                source=source,
                residual_checkpoint=args.ra_pretrained_checkpoint,
            )
            compatibility.append({
                "dataset": args.dataset,
                "task": task,
                "backbone": backbone,
                "model_source": source,
                "family": resolved_family,
                "method": method,
                "status": "scheduled" if ok else "skipped",
                "reason": reason,
            })
            if ok:
                supported_methods.append(method)

        # Linear probing is both a reported baseline and the common task-aware
        # head source. It is prepared first for each backbone/seed.
        linear_supported = "linear" in supported_methods
        head_path_by_seed: dict[int, str] = {}
        if linear_supported:
            head_common = {
                **common_base,
                "backbone": backbone,
                "model_source": source,
                "epochs": args.epochs,
            }
            variants = [("linear", {"tuning_method": "linear", "seed": seed}) for seed in seeds]
            built = build_specs(
                suite=f"fair_{args.dataset}_{task}_{family}_{backbone}",
                variants=variants,
                common=head_common,
                output_root=args.output_root,
            )
            head_specs.extend(built)
            for spec in built:
                head_path_by_seed[int(spec.parameters["seed"])] = str(Path(spec.output_dir) / "checkpoint-best.pth")

        compare_common = {
            **common_base,
            "backbone": backbone,
            "model_source": source,
            "epochs": args.epochs,
        }
        variants: list[tuple[str, dict[str, Any]]] = []
        for seed in seeds:
            for method in supported_methods:
                if method == "linear":
                    continue
                variants.append((method, method_variant(method, family, seed, head_path_by_seed.get(seed), args)))
        comparison_specs.extend(build_specs(
            suite=f"fair_{args.dataset}_{task}_{family}_{backbone}",
            variants=variants,
            common=compare_common,
            output_root=args.output_root,
        ))

    return task, seeds, head_specs, comparison_specs, compatibility, dataset_args


def write_compatibility(rows: list[dict[str, Any]], manifest_path: str | Path) -> tuple[Path, Path]:
    manifest_path = Path(manifest_path)
    json_path = manifest_path.with_name(manifest_path.stem + "_compatibility.json")
    csv_path = json_path.with_suffix(".csv")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["status"])
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def main() -> None:
    args = parser().parse_args()
    task, seeds, head_specs, comparison_specs, compatibility, dataset_args = build_suite(args)
    all_specs = [*head_specs, *comparison_specs]
    manifest, manifest_csv = write_manifest(all_specs, args.manifest)
    compatibility_json, compatibility_csv = write_compatibility(compatibility, args.manifest)
    scheduled = sum(row["status"] == "scheduled" for row in compatibility)
    skipped = len(compatibility) - scheduled
    protocol = {
        "dataset": args.dataset,
        "task": task,
        "dataset_args": dataset_args,
        "seeds": seeds,
        "backbones": parse_backbones(args.backbones),
        "same_peft_optimizer": args.optimizer,
        "same_peft_learning_rate": args.peft_lr,
        "same_scheduler": "cosine",
        "same_warmup_epochs": args.warmup_epochs,
        "same_weight_decay": args.weight_decay,
        "full_learning_rate": args.full_lr,
        "linear_learning_rate": args.linear_lr,
        "epochs": args.epochs,
        "input_size": args.input_size,
        "input_size_note": "0 means native pretrained size resolved before dataset transforms.",
        "scheduled_method_backbone_pairs": scheduled,
        "skipped_method_backbone_pairs": skipped,
        "shared_head_policy": "Best linear-probe head per backbone and seed is loaded before PEFT construction/TRSO calibration; Visual Prompting retains the source head by definition.",
        "unsupported_policy": "Explicit skip report; no silent architectural approximation.",
    }
    protocol_path = Path(args.manifest).with_name(Path(args.manifest).stem + "_protocol.json")
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Wrote {len(all_specs)} executable runs to {manifest} and {manifest_csv}")
    print(f"Compatibility: {compatibility_json} and {compatibility_csv}")
    print(f"Protocol: {protocol_path}")
    print(f"Scheduled pairs={scheduled}; explicit skips={skipped}")

    if args.execute:
        gpu_ids = parse_csv_values(args.gpu_ids, int)[: max(1, int(args.parallel_runs))]
        if len(gpu_ids) > 1:
            execute_specs_parallel(head_specs, execute=True, gpu_ids=gpu_ids, max_runs=0)
            execute_specs_parallel(comparison_specs, execute=True, gpu_ids=gpu_ids, max_runs=args.max_runs)
        else:
            execute_specs(head_specs, execute=True, max_runs=0)
            execute_specs(comparison_specs, execute=True, max_runs=args.max_runs)
    else:
        execute_specs(all_specs, execute=False, max_runs=args.max_runs)


if __name__ == "__main__":
    main()
