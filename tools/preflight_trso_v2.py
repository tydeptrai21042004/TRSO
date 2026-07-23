"""Deterministic structural and numerical preflight for TRSO-v2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from models.task_response_adapter import TaskResponseSpatialAdapter


def calibrate(adapter, x, target, init_scale=0.2):
    adapter.start_calibration()
    F.mse_loss(adapter(x), target).backward()
    adapter.accumulate_probe_gradient()
    adapter.finalize_calibration(init_scale=init_scale)


def run_preflight() -> dict:
    torch.manual_seed(7)
    channels = 64
    x = torch.randn(16, channels, 8, 8)
    atoms = torch.randn(2, 3, 3)
    coefficients = torch.randn(channels, 2) * 0.03
    kernel = (coefficients @ atoms.flatten(1)).reshape(channels, 3, 3)
    channel_scale = torch.randn(channels) * 0.08
    target = x + F.conv2d(x, kernel.unsqueeze(1), padding=1, groups=channels)
    target = target + x * channel_scale.view(1, channels, 1, 1)

    v1 = TaskResponseSpatialAdapter(
        channels=channels, kernel_size=3, spatial_rank=2, variant="v1",
        gate_init=1.0, operator_radius=10.0, layout="bchw",
    )
    v2 = TaskResponseSpatialAdapter(
        channels=channels, kernel_size=3, spatial_rank=2, variant="v2",
        coefficient_mode="grouped", channel_groups=8, input_norm="none",
        channel_response=True, gate_init=1.0, operator_radius=10.0, layout="bchw",
    )
    calibrate(v1, x, target)
    calibrate(v2, x, target)
    baseline_mse = float(F.mse_loss(x, target).detach())
    v1_mse = float(F.mse_loss(v1(x), target).detach())
    v2_mse = float(F.mse_loss(v2(x), target).detach())
    v1_cost = v1.parameter_count_breakdown()["active_trainable"]
    v2_cost = v2.parameter_count_breakdown()["active_trainable"]

    torch.manual_seed(11)
    token_adapter = TaskResponseSpatialAdapter(
        channels=16, kernel_size=3, spatial_rank=1, variant="v2",
        coefficient_mode="grouped", channel_groups=4, input_norm="rms",
        channel_response=True, prefix_coupling=True, gate_init=1.0,
        layout="bnc", grid_size=(2, 2),
    )
    tokens = torch.randn(8, 5, 16)
    token_target = tokens.clone()
    token_target[:, 0] += tokens[:, 1:].mean(dim=1)
    calibrate(token_adapter, tokens, token_target, init_scale=0.1)
    token_output = token_adapter(tokens)
    token_loss = token_output[:, 0].square().mean()
    token_loss.backward()
    prefix_gradient = float(token_adapter.prefix_gate.grad.abs())

    report = {
        "synthetic": {
            "baseline_mse": baseline_mse,
            "v1_mse": v1_mse,
            "v2_mse": v2_mse,
            "v1_adapter_parameters": int(v1_cost),
            "v2_adapter_parameters": int(v2_cost),
            "v2_parameter_fraction_of_v1": float(v2_cost / max(1, v1_cost)),
            "v2_improves_over_v1": bool(v2_mse < v1_mse),
        },
        "transformer_prefix": {
            "class_token_changed": bool(not torch.allclose(token_output[:, 0], tokens[:, 0])),
            "prefix_gate_parameters": token_adapter.parameter_count_breakdown()["prefix_gate"],
            "prefix_gate_gradient": prefix_gradient,
        },
        "all_ok": bool(
            v2_cost <= 0.2 * v1_cost
            and v2_mse < v1_mse
            and not torch.allclose(token_output[:, 0], tokens[:, 0])
            and prefix_gradient > 0
        ),
        "scope": "Controlled structural/numerical diagnostic; not a real-dataset accuracy result.",
    }
    return report


def main():
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
