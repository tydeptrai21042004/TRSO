"""Offline audit of the universal fair release.

This audit does not download datasets or pretrained weights. It verifies the
router, compatibility planner, fair-protocol manifests, task metrics, and
structural baseline/TRSO preflights. Real benchmark accuracy still requires
running the generated experiments.
"""
from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import asdict
import torch

from datasets.build import available_datasets
from models.model_support import METHOD_SUPPORT
from tools.preflight_paper_baselines import run as baseline_preflight
from tools.preflight_trso import run as trso_preflight
from tools.preflight_trso_v3 import run_preflight as trso_v3_preflight
from tools.run_fair_suite import SINGLE_LABEL_DATASETS, build_suite, parser as fair_parser
from tools.verify_fairness import verify_manifest
from tools.experiment_grid import write_manifest


def _args(root: Path, task: str) -> Namespace:
    # Start from the real fair-runner parser so this audit cannot silently drift
    # when the performance protocol gains a new option.
    args = fair_parser().parse_args([])
    args.dataset = "fake"
    args.task = task
    args.data_path = str(root / "data")
    args.download = False
    args.weights = "none"
    args.pretrained = False
    args.dataset_args_json = "{}"
    args.output_root = str(root / "outputs")
    args.manifest = str(root / f"{task}.json")
    args.seeds = "0"
    args.split_seed = 2026
    args.epochs = 2
    args.batch_size = 4
    args.num_workers = 0
    args.input_size = 32
    args.backbones = "resnet50@torchvision,vit_tiny_patch16_224@timm"
    args.methods = "auto"
    args.peft_lr = 1e-3
    args.full_lr = 1e-4
    args.linear_lr = 1e-3
    args.weight_decay = 1e-4
    args.warmup_epochs = 1
    args.min_lr = 1e-6
    args.optimizer = "adamw"
    args.augmentation = "strong"
    args.peft_freeze_head = False
    args.peft_head_lr_scale = 0.5
    args.trso_budget = 0
    args.trso_auto_budget_ratio = 0.35
    args.trso_calibration_batches = 2
    args.trso_rank = 4
    args.trso_basis_trainable = True
    args.trso_max_adapters = 0
    args.trso_residual_target = 0.05
    args.ra_pretrained_checkpoint = ""
    args.device = "cpu"
    args.gpu_ids = "0"
    args.parallel_runs = 1
    args.execute = False
    args.max_runs = 0
    args.profile_efficiency = False
    args.measure_eval_latency = False
    args.allow_val_as_test = False
    return args


def run_audit() -> dict:
    datasets = available_datasets()
    hinted = set(SINGLE_LABEL_DATASETS) | {"coco", "voc2007", "celeba", "fake", "csv"}
    report = {
        "canonical_datasets": datasets,
        "dataset_count": len(datasets),
        "datasets_missing_task_resolution": sorted(set(datasets) - hinted),
        "methods": sorted(METHOD_SUPPORT),
        "unique_reported_methods": sorted(method for method in METHOD_SUPPORT if method != "adapter"),
        "tasks": ["single_label", "multilabel", "regression"],
        "baseline_preflight": baseline_preflight(),
        "trso_preflight": asdict(trso_preflight(torch.device("cpu"))),
        "trso_v3_preflight": trso_v3_preflight(),
        "task_plans": {},
    }
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        for task in report["tasks"]:
            args = _args(root / task, task)
            resolved, seeds, heads, comparisons, compatibility, _ = build_suite(args)
            manifest, _ = write_manifest([*heads, *comparisons], root / task / "manifest.json")
            fairness = verify_manifest(manifest)
            report["task_plans"][task] = {
                "resolved_task": resolved,
                "seeds": seeds,
                "runs": len(heads) + len(comparisons),
                "scheduled_pairs": sum(row["status"] == "scheduled" for row in compatibility),
                "explicit_skips": sum(row["status"] == "skipped" for row in compatibility),
                "fairness_verified": fairness["fair"],
                "fairness_errors": fairness["errors"],
                "scheduled_methods": sorted({spec.parameters["tuning_method"] for spec in [*heads, *comparisons]}),
                "prompt_explicitly_skipped": any(
                    row["method"] == "prompt" and row["status"] == "skipped"
                    for row in compatibility
                ),
            }
    report["all_ok"] = (
        not report["datasets_missing_task_resolution"]
        and report["baseline_preflight"]["all_ok"]
        and bool(report["trso_preflight"]["cnn_shape_ok"])
        and bool(report["trso_preflight"]["vit_shape_ok"])
        and bool(report["trso_preflight"]["cls_token_preserved"])
        and bool(report["trso_preflight"]["bhwc_shape_ok"])
        and report["trso_preflight"]["fused_max_error"] < 2e-5
        and report["trso_preflight"]["calibration_tangent_cosine"] > 0.999999
        and report["trso_preflight"]["recovered_kernel_cosine"] > 0.9999
        and report["trso_v3_preflight"]["all_ok"]
        and all(item["fairness_verified"] for item in report["task_plans"].values())
    )
    return report


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="test_reports/universal_release_audit.json")
    args = parser.parse_args()
    report = run_audit()
    text = json.dumps(report, indent=2)
    print(text)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if not report["all_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
