#!/usr/bin/env python3
"""Launch reproducible TRSO parameter/layer-budget sweeps."""
from __future__ import annotations

import argparse
import itertools
import shlex
import subprocess
from pathlib import Path


def csv_values(text, cast):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="python")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--nb_classes", required=True, type=int)
    parser.add_argument("--output_root", default="experiments/trso_sweep")
    parser.add_argument("--kernel_sizes", default="3,5,7")
    parser.add_argument("--spatial_ranks", default="1,2,3")
    parser.add_argument("--channel_ratios", default="8,16,32")
    parser.add_argument("--keep_ratios", default="0.25,0.5,1.0")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--calibration_batches", type=int, default=8)
    parser.add_argument("--extra", default="", help="Additional main.py arguments.")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    kernels = csv_values(args.kernel_sizes, int)
    ranks = csv_values(args.spatial_ranks, int)
    channel_ratios = csv_values(args.channel_ratios, int)
    keep_ratios = csv_values(args.keep_ratios, float)
    seeds = csv_values(args.seeds, int)

    for kernel, rank, channel_ratio, keep_ratio, seed in itertools.product(
        kernels, ranks, channel_ratios, keep_ratios, seeds
    ):
        if rank > kernel:
            continue
        run_name = f"k{kernel}_r{rank}_cr{channel_ratio}_keep{keep_ratio:g}_seed{seed}"
        output = Path(args.output_root) / args.dataset / args.backbone / run_name
        command = [
            args.python,
            "main.py",
            "--tuning_method", "trso",
            "--dataset", args.dataset,
            "--backbone", args.backbone,
            "--data_path", args.data_path,
            "--nb_classes", str(args.nb_classes),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--seed", str(seed),
            "--trso_kernel_size", str(kernel),
            "--trso_spatial_rank", str(rank),
            "--trso_channel_ratio", str(channel_ratio),
            "--trso_keep_ratio", str(keep_ratio),
            "--trso_calibration_batches", str(args.calibration_batches),
            "--output_dir", str(output),
        ] + shlex.split(args.extra)
        print(" ".join(shlex.quote(part) for part in command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
