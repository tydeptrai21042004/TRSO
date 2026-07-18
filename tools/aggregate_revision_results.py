# tools/aggregate_revision_results.py
"""Aggregate revision experiment outputs into CSV summaries with mean ± std."""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict

import pandas as pd


def read_json(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_dict(prefix: str, d: Dict) -> Dict:
    return {f"{prefix}_{k}": v for k, v in d.items()}


def infer_metadata(run_dir: str, root: str) -> Dict:
    rel = os.path.relpath(run_dir, root)
    parts = rel.split(os.sep)
    meta = {"run_dir": run_dir}
    # Expected: dataset/backbone/method/seed_*
    if len(parts) >= 4:
        meta.update({
            "dataset": parts[-4],
            "backbone": parts[-3],
            "method": parts[-2],
            "seed": parts[-1].replace("seed_", ""),
        })
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="outputs_revision")
    parser.add_argument("--out_csv", type=str, default="revision_summary.csv")
    args = parser.parse_args()

    run_dirs = sorted(set(os.path.dirname(p) for p in glob.glob(os.path.join(args.root, "**", "args.json"), recursive=True)))
    if not run_dirs:
        # fallback: dirs containing test_summary / convergence files
        files = glob.glob(os.path.join(args.root, "**", "test_summary.json"), recursive=True)
        files += glob.glob(os.path.join(args.root, "**", "convergence_summary.json"), recursive=True)
        run_dirs = sorted(set(os.path.dirname(p) for p in files))

    rows = []
    for run_dir in run_dirs:
        meta = infer_metadata(run_dir, args.root)
        run_args = read_json(os.path.join(run_dir, "args.json"))
        test = read_json(os.path.join(run_dir, "test_summary.json"))
        val = read_json(os.path.join(run_dir, "eval_summary.json"))
        eff = read_json(os.path.join(run_dir, "efficiency_profile.json"))
        conv = read_json(os.path.join(run_dir, "convergence_summary.json"))

        row = {}
        row.update(meta)
        # args can override inferred names if present
        for k in ("dataset", "backbone", "tuning_method", "seed"):
            if k in run_args:
                row["method" if k == "tuning_method" else k] = run_args[k]
        row.update(flatten_dict("test", test))
        row.update(flatten_dict("eval", val))
        row.update(flatten_dict("eff", eff))
        row.update(flatten_dict("conv", conv))
        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError(f"No runs found under {args.root}")

    df.to_csv(args.out_csv, index=False)
    print(f"Saved raw summary: {args.out_csv}")

    # Build paper-friendly mean/std summary.
    metric_cols = [
        c for c in df.columns
        if c not in ("run_dir", "dataset", "backbone", "method", "seed")
        and pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors="coerce"))
    ]
    for c in metric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    group_cols = [c for c in ("dataset", "backbone", "method") if c in df.columns]
    summary = df.groupby(group_cols)[metric_cols].agg(["mean", "std", "count"]).reset_index()
    out_mean = args.out_csv.replace(".csv", "_mean_std.csv")
    summary.to_csv(out_mean, index=False)
    print(f"Saved mean/std summary: {out_mean}")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
