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
from tools.run_fair_suite import SINGLE_LABEL_DATASETS, build_suite
from tools.verify_fairness import verify_manifest
from tools.experiment_grid import write_manifest


def _args(root: Path, task: str) -> Namespace:
    return Namespace(
        dataset="fake", task=task, data_path=str(root / "data"), download=False,
        dataset_args_json="{}", output_root=str(root / "outputs"),
        manifest=str(root / f"{task}.json"), seeds="0", split_seed=2026,
        epochs=2, batch_size=4, num_workers=0, input_size=32,
        backbones="resnet50@torchvision,vit_tiny_patch16_224@timm",
        methods="auto", peft_lr=5e-3, full_lr=1e-4, linear_lr=1e-1,
        weight_decay=1e-4, warmup_epochs=1, min_lr=1e-6, optimizer="adamw",
        trso_budget=1000, trso_calibration_batches=2,
        ra_pretrained_checkpoint="", device="cpu", gpu_ids="0",
        parallel_runs=1, execute=False, max_runs=0,
        profile_efficiency=False, measure_eval_latency=False,
        allow_val_as_test=False,
    )


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
