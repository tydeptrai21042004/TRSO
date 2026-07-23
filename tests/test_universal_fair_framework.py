from argparse import Namespace
import json
from pathlib import Path

import torch

from engine import _multilabel_metrics, _regression_metrics
from models.model_support import (
    infer_backbone_family_name,
    static_method_compatibility,
    validate_method_task,
)
from tools.run_fair_suite import build_suite, parse_methods, resolve_task


def _suite_args(tmp_path: Path, *, task="single_label"):
    return Namespace(
        dataset="fake",
        task=task,
        data_path=str(tmp_path / "data"),
        download=False,
        dataset_args_json="{}",
        output_root=str(tmp_path / "outputs"),
        manifest=str(tmp_path / "manifest.json"),
        seeds="0,1,2",
        split_seed=2026,
        epochs=2,
        batch_size=4,
        num_workers=0,
        input_size=32,
        backbones="resnet50@torchvision,vit_tiny_patch16_224@timm",
        methods="auto",
        peft_lr=5e-3,
        full_lr=1e-4,
        linear_lr=1e-1,
        weight_decay=1e-4,
        warmup_epochs=1,
        min_lr=1e-6,
        optimizer="adamw",
        trso_budget=1000,
        trso_calibration_batches=2,
        ra_pretrained_checkpoint="",
        device="cpu",
        gpu_ids="0",
        parallel_runs=1,
        execute=False,
        max_runs=0,
        profile_efficiency=False,
        measure_eval_latency=False,
        allow_val_as_test=False,
    )


def test_auto_method_list_has_no_conv_alias_duplicate():
    methods = parse_methods("auto")
    assert "conv" in methods
    assert "adapter" not in methods
    assert len(methods) == len(set(methods))


def test_task_capability_is_explicit():
    validate_method_task("trso", "single_label")
    validate_method_task("trso", "multilabel")
    validate_method_task("trso", "regression")
    try:
        validate_method_task("prompt", "multilabel")
    except ValueError as exc:
        assert "single_label" in str(exc)
    else:
        raise AssertionError("Visual Prompting must reject multi-label tasks")


def test_static_backbone_contracts_are_not_silently_approximated():
    ok, reason, family = static_method_compatibility("conv", "resnet18", "single_label")
    assert not ok and family == "resnet" and "ResNet-50" in reason
    ok, reason, family = static_method_compatibility("adaptformer", "resnet50", "single_label")
    assert not ok and family == "resnet"
    ok, reason, family = static_method_compatibility("lora", "vit_tiny_patch16_224", "regression", source="timm")
    assert ok and family == "vit"


def test_generic_suite_schedules_all_compatible_rows_and_shared_settings(tmp_path):
    args = _suite_args(tmp_path)
    task, seeds, heads, comparisons, compatibility, _ = build_suite(args)
    assert task == "single_label"
    assert seeds == [0, 1, 2]
    assert all(row["status"] in {"scheduled", "skipped"} for row in compatibility)
    assert any(row["status"] == "skipped" for row in compatibility)
    assert not any(row["method"] == "adapter" for row in compatibility)
    peft = [spec for spec in comparisons if spec.parameters["tuning_method"] not in {"full", "linear"}]
    assert peft
    fingerprints = {
        (
            spec.parameters["fair_optimizer"], spec.parameters["fair_peft_lr"],
            spec.parameters["fair_weight_decay"], spec.parameters["fair_warmup_epochs"],
            spec.parameters["fair_min_lr"], spec.parameters["epochs"],
        )
        for spec in peft
    }
    assert len(fingerprints) == 1
    assert all(spec.parameters["head_from"].endswith("checkpoint-best.pth")
               for spec in peft if spec.parameters["tuning_method"] not in {"prompt", "residual"})


def test_multilabel_suite_skips_prompt_but_keeps_task_agnostic_methods(tmp_path):
    args = _suite_args(tmp_path, task="multilabel")
    _, _, _, comparisons, compatibility, _ = build_suite(args)
    prompt_rows = [row for row in compatibility if row["method"] == "prompt"]
    assert prompt_rows and all(row["status"] == "skipped" for row in prompt_rows)
    scheduled = {spec.parameters["tuning_method"] for spec in comparisons}
    assert "trso" in scheduled
    assert "piggyback" in scheduled
    assert "adaptformer" in scheduled


def test_multilabel_metrics_report_predictive_and_calibration_values():
    logits = torch.tensor([[3.0, -2.0, 0.2], [-1.0, 2.0, 1.0], [1.0, 1.0, -3.0]])
    targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    scalar, diagnostic = _multilabel_metrics(logits, targets)
    for key in ("map", "micro_f1", "macro_f1", "subset_accuracy", "hamming_accuracy", "ece", "brier_score"):
        assert key in scalar
    assert len(diagnostic["per_class_f1"]) == 3


def test_regression_metrics_report_quality_and_per_output_diagnostics():
    predictions = torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 5.0], [4.0, 8.0]])
    targets = torch.tensor([[1.1, 2.1], [1.9, 3.8], [3.2, 5.2], [3.8, 7.9]])
    scalar, diagnostic = _regression_metrics(predictions, targets)
    for key in ("mae", "median_absolute_error", "rmse", "r2", "pearson", "spearman"):
        assert key in scalar
    assert len(diagnostic["per_output_r2"]) == 2


def test_task_resolution_requires_explicit_csv_task(tmp_path):
    args = _suite_args(tmp_path)
    args.dataset = "csv"
    args.task = "auto"
    try:
        resolve_task(args, {})
    except ValueError as exc:
        assert "explicit" in str(exc)
    else:
        raise AssertionError("CSV task must be explicit")


def test_fairness_verifier_accepts_generated_manifest(tmp_path):
    from tools.experiment_grid import write_manifest
    from tools.verify_fairness import verify_manifest
    args = _suite_args(tmp_path)
    _, _, heads, comparisons, _, _ = build_suite(args)
    manifest, _ = write_manifest([*heads, *comparisons], tmp_path / "manifest.json")
    report = verify_manifest(manifest)
    assert report["fair"]
    assert not report["errors"]


def test_warmup_is_clamped_for_short_smoke_runs(tmp_path):
    args = _suite_args(tmp_path)
    args.epochs = 1
    args.warmup_epochs = 5
    _, _, heads, comparisons, _, _ = build_suite(args)
    assert all(spec.parameters["fair_warmup_epochs"] == 0 for spec in [*heads, *comparisons])
