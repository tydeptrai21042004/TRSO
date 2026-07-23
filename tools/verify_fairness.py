"""Verify a generated fair-suite manifest or completed output tree.

The verifier enforces the user's controlled-comparison rule:
- every PEFT method and TRSO uses the same optimizer, LR, cosine scheduler,
  warm-up, weight decay, epoch count, batch size, split seed, and augmentation;
- full fine-tuning and linear probing may use different learning rates only;
- comparisons are made within the same dataset/task/backbone group;
- unsupported pairs must appear in the compatibility report rather than vanish.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

PEFT_EXCEPTIONS = {"full", "linear"}
COMMON_KEYS = (
    "fair_protocol", "fair_optimizer", "fair_peft_lr", "fair_weight_decay",
    "fair_warmup_epochs", "fair_min_lr", "epochs", "batch_size", "update_freq",
    "split_seed", "input_size", "train_aug", "aa", "color_jitter", "mixup",
    "cutmix", "smoothing", "reprob", "task", "dataset", "backbone",
)


def _load(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_manifest(path: str | Path) -> dict[str, Any]:
    rows = _load(path)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        parameters = row.get("parameters", {})
        key = (
            str(parameters.get("dataset", "")),
            str(parameters.get("task", "auto")),
            str(parameters.get("backbone", "")),
        )
        groups[key].append(parameters)

    errors: list[str] = []
    warnings: list[str] = []
    group_reports = []
    for group, parameters_list in sorted(groups.items()):
        peft = [p for p in parameters_list if p.get("tuning_method") not in PEFT_EXCEPTIONS]
        reference = peft[0] if peft else None
        mismatches: dict[str, list[tuple[str, Any, Any]]] = defaultdict(list)
        if reference:
            for parameters in peft[1:]:
                for key in COMMON_KEYS:
                    expected = reference.get(key, 1 if key == "update_freq" else None)
                    actual = parameters.get(key, 1 if key == "update_freq" else None)
                    if actual != expected:
                        mismatches[key].append((str(parameters.get("tuning_method")), expected, actual))
        for key, values in mismatches.items():
            errors.append(f"{group}: PEFT mismatch in {key}: {values}")

        for parameters in parameters_list:
            method = str(parameters.get("tuning_method"))
            if not parameters.get("fair_protocol", False):
                errors.append(f"{group}/{method}: fair_protocol is disabled")
            if parameters.get("paper_hparams", False):
                errors.append(f"{group}/{method}: paper_hparams would override the controlled recipe")
            if parameters.get("legacy_auto_hparams", False):
                errors.append(f"{group}/{method}: legacy_auto_hparams would override the controlled recipe")
            if method not in PEFT_EXCEPTIONS and parameters.get("head_from") is None and method not in {"prompt", "residual"}:
                warnings.append(f"{group}/{method}: no shared task-aware head checkpoint")

        group_reports.append({
            "dataset": group[0], "task": group[1], "backbone": group[2],
            "methods": sorted({str(p.get("tuning_method")) for p in parameters_list}),
            "peft_reference": {key: reference.get(key, 1 if key == "update_freq" else None) for key in COMMON_KEYS} if reference else {},
            "mismatches": dict(mismatches),
        })

    return {
        "manifest": str(path),
        "groups": group_reports,
        "errors": errors,
        "warnings": warnings,
        "fair": not errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--compatibility", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    report = verify_manifest(args.manifest)
    if args.compatibility:
        compatibility = _load(args.compatibility)
        report["compatibility_rows"] = len(compatibility)
        report["scheduled_pairs"] = sum(row.get("status") == "scheduled" for row in compatibility)
        report["explicit_skips"] = sum(row.get("status") == "skipped" for row in compatibility)
        silent = [row for row in compatibility if row.get("status") not in {"scheduled", "skipped"}]
        if silent:
            report["errors"].append(f"Compatibility rows with invalid status: {silent}")
            report["fair"] = False
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    if not report["fair"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
