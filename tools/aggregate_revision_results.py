"""Aggregate experiment outputs into raw and paper-ready mean/std CSV files."""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict

import pandas as pd


def read_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def flatten_dict(prefix: str, values: Dict[str, Any]) -> Dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def infer_metadata(run_dir: str, root: str) -> Dict[str, Any]:
    rel = os.path.relpath(run_dir, root)
    parts = rel.split(os.sep)
    meta: Dict[str, Any] = {"run_dir": run_dir}
    # Grid layout: root/suite/variant/seed_<seed>_<digest>.
    if len(parts) >= 3:
        meta.update(
            {
                "experiment_suite": parts[-3],
                "experiment_name": parts[-2],
                "run_leaf": parts[-1],
            }
        )
    return meta


def _numeric_columns(df: pd.DataFrame, excluded: set[str]) -> list[str]:
    columns: list[str] = []
    for column in df.columns:
        if column in excluded:
            continue
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.notna().any():
            df[column] = converted
            columns.append(column)
    return columns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="outputs_ablation")
    parser.add_argument("--out_csv", type=str, default="experiment_summary.csv")
    args = parser.parse_args()

    run_dirs = sorted(
        set(os.path.dirname(path) for path in glob.glob(os.path.join(args.root, "**", "args.json"), recursive=True))
    )
    if not run_dirs:
        files = glob.glob(os.path.join(args.root, "**", "test_summary.json"), recursive=True)
        files += glob.glob(os.path.join(args.root, "**", "eval_summary.json"), recursive=True)
        run_dirs = sorted(set(os.path.dirname(path) for path in files))

    rows = []
    for run_dir in run_dirs:
        row = infer_metadata(run_dir, args.root)
        run_args = read_json(os.path.join(run_dir, "args.json"))
        test = read_json(os.path.join(run_dir, "test_summary.json"))
        val = read_json(os.path.join(run_dir, "eval_summary.json"))
        efficiency = read_json(os.path.join(run_dir, "efficiency_profile.json"))
        convergence = read_json(os.path.join(run_dir, "convergence_summary.json"))

        for key in (
            "experiment_suite",
            "experiment_name",
            "experiment_run_id",
            "dataset",
            "backbone",
            "tuning_method",
            "seed",
            "trso_basis_source",
            "trso_allocation",
            "trso_score_mode",
            "trso_parameter_budget",
        ):
            if key in run_args:
                row["method" if key == "tuning_method" else key] = run_args[key]
        row.update(flatten_dict("test", test))
        row.update(flatten_dict("eval", val))
        row.update(flatten_dict("eff", efficiency))
        row.update(flatten_dict("conv", convergence))
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No completed runs found under {args.root}")

    output_path = os.path.abspath(args.out_csv)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved raw summary: {output_path}")

    identifiers = {
        "run_dir",
        "run_leaf",
        "experiment_suite",
        "experiment_name",
        "experiment_run_id",
        "dataset",
        "backbone",
        "method",
        "seed",
        "trso_basis_source",
        "trso_allocation",
        "trso_score_mode",
        "trso_parameter_budget",
    }
    metric_columns = _numeric_columns(df, identifiers)
    group_columns = [
        column
        for column in ("experiment_suite", "experiment_name", "dataset", "backbone", "method")
        if column in df.columns
    ]
    if not group_columns or not metric_columns:
        print("Raw results were saved, but no numeric grouped summary was available.")
        return

    summary = df.groupby(group_columns, dropna=False)[metric_columns].agg(["mean", "std", "count"]).reset_index()
    mean_std_path = os.path.splitext(output_path)[0] + "_mean_std.csv"
    summary.to_csv(mean_std_path, index=False)
    print(f"Saved mean/std summary: {mean_std_path}")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
