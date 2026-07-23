"""Deterministic universal diagnostics for TRSO-v3.

The checks target properties that must hold across datasets, task losses, and
backbone widths. They are structural/numerical checks, not benchmark claims.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from models.task_response_adapter import TaskResponseSpatialAdapter


def _calibrate(adapter: TaskResponseSpatialAdapter, x: torch.Tensor, scale: float = 1.0) -> None:
    adapter.start_calibration(reset=True)
    (scale * adapter(x).square().mean()).backward()
    adapter.accumulate_probe_gradient()
    adapter.finalize_calibration(init_scale=0.05)


def run_preflight() -> dict:
    torch.manual_seed(41)
    x = torch.randn(6, 64, 8, 8)
    v2 = TaskResponseSpatialAdapter(
        channels=64, kernel_size=3, spatial_rank=2, variant="v2",
        channel_groups=8, input_norm="rms", gate_init=1.0,
    )
    v3 = TaskResponseSpatialAdapter(
        channels=64, kernel_size=3, spatial_rank=2, variant="v3",
        channel_groups=0, input_norm="rms", gate_init=1.0,
    )
    _calibrate(v2, x)
    _calibrate(v3, x)

    scaled = TaskResponseSpatialAdapter(
        channels=64, kernel_size=3, spatial_rank=2, variant="v3",
        channel_groups=0, input_norm="rms", gate_init=1.0,
    )
    scaled.load_state_dict(TaskResponseSpatialAdapter(
        channels=64, kernel_size=3, spatial_rank=2, variant="v3",
        channel_groups=0, input_norm="rms", gate_init=1.0,
    ).state_dict(), strict=True)
    reference = TaskResponseSpatialAdapter(
        channels=64, kernel_size=3, spatial_rank=2, variant="v3",
        channel_groups=0, input_norm="rms", gate_init=1.0,
    )
    scaled.load_state_dict(reference.state_dict(), strict=True)
    _calibrate(reference, x, scale=1.0)
    _calibrate(scaled, x, scale=10000.0)
    loss_scale_basis_error = float((reference.basis_atoms - scaled.basis_atoms).abs().max())
    loss_scale_coefficient_error = float(
        (reference.response_coefficient_directions - scaled.response_coefficient_directions).abs().max()
    )

    base = torch.randn(3, 64, 7, 7)
    ratios = []
    for feature_scale in (0.01, 1.0, 100.0):
        z = base * feature_scale
        update = v3(z) - z
        ratios.append(float((update.square().mean().sqrt() / z.square().mean().sqrt()).detach()))

    token_adapter = TaskResponseSpatialAdapter(
        channels=16, kernel_size=3, spatial_rank=1, variant="v3",
        channel_groups=0, layout="bnc", grid_size=(2, 3),
        prefix_coupling=True, prefix_coupling_mode="all", gate_init=1.0,
    )
    tokens = torch.randn(4, 8, 16)  # two prefix tokens + rectangular 2x3 grid
    token_out = token_adapter(tokens)
    prefix_changes = token_out[:, :2] - tokens[:, :2]

    report = {
        "parameter_scaling": {
            "v2_parameters": v2.parameter_count_breakdown()["active_trainable"],
            "v3_parameters": v3.parameter_count_breakdown()["active_trainable"],
            "v3_auto_groups": v3.channel_groups,
        },
        "task_loss_scale_invariance": {
            "basis_max_error": loss_scale_basis_error,
            "coefficient_max_error": loss_scale_coefficient_error,
        },
        "feature_scale_invariance": {
            "update_to_feature_rms_ratios": ratios,
            "max_ratio_spread": max(ratios) - min(ratios),
        },
        "token_generality": {
            "rectangular_grid_shape_ok": tuple(token_out.shape) == tuple(tokens.shape),
            "both_prefix_tokens_changed": bool((prefix_changes.abs().sum(dim=-1) > 0).all()),
            "prefix_updates_equal": bool(torch.allclose(prefix_changes[:, 0], prefix_changes[:, 1], atol=1e-6, rtol=1e-5)),
        },
        "all_ok": bool(
            loss_scale_basis_error < 2e-5
            and loss_scale_coefficient_error < 2e-5
            and max(ratios) - min(ratios) < 2e-3
            and all(abs(value - 0.05) < 2e-3 for value in ratios)
            and tuple(token_out.shape) == tuple(tokens.shape)
            and (prefix_changes.abs().sum(dim=-1) > 0).all()
        ),
        "scope": "Universal structural/numerical diagnostics; no real-dataset accuracy is claimed.",
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="")
    args = parser.parse_args()
    report = run_preflight()
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    if not report["all_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
