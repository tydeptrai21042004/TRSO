#!/usr/bin/env python3
"""Fast offline preflight for Scientific TRSO."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from models.task_response_adapter import TaskResponseSpatialAdapter


@dataclass
class PreflightReport:
    cnn_shape_ok: bool
    vit_shape_ok: bool
    cls_token_preserved: bool
    bhwc_shape_ok: bool
    disabled_returns_same_tensor: bool
    fused_max_error: float
    maximum_channel_kernel_l1: float
    coefficient_gradient_l1: float
    calibration_tangent_cosine: float
    recovered_kernel_cosine: float
    recovered_channel_spatial_rank: int
    configured_rank: int


def run(device: torch.device) -> PreflightReport:
    torch.manual_seed(19)
    torch.set_num_threads(1)

    adapter = TaskResponseSpatialAdapter(
        channels=8,
        kernel_size=5,
        spatial_rank=2,
        operator_radius=0.7,
        gate_init=1e-2,
    ).to(device)

    cnn = torch.randn(2, 8, 12, 12, device=device)
    vit = torch.randn(2, 144, 8, device=device)
    vit_cls = torch.randn(2, 145, 8, device=device)
    bhwc = torch.randn(2, 12, 12, 8, device=device)
    y_cnn = adapter(cnn)
    y_vit = adapter(vit)
    y_cls = adapter(vit_cls)
    y_bhwc = adapter(bhwc)

    explicit = adapter.explicit_basis_response(cnn)
    fused = adapter._depthwise(cnn, adapter.projected_kernel_bank())
    fused_error = float((explicit - fused).abs().max().item())
    maximum_l1 = float(
        adapter.projected_kernel_bank().abs().sum(dim=(1, 2)).max().item()
    )

    y_cnn.square().mean().backward()
    coefficient_gradient = float(
        sum(
            parameter.grad.abs().sum().item()
            for parameter in adapter.coefficient_vectors
        )
    )

    # Exact calibration/training tangent check.
    aligned = TaskResponseSpatialAdapter(
        channels=4,
        kernel_size=3,
        spatial_rank=2,
        operator_radius=10.0,
        gate_init=1.0,
        calibration_scale=1.0,
    ).to(device)
    x = torch.randn(3, 4, 9, 9, device=device)
    target = torch.randn_like(x)
    aligned.start_calibration()
    F.mse_loss(aligned(x), target).backward()
    probe_gradient = aligned.probe_kernel.grad.detach().clone()
    aligned.stop_calibration()
    full_kernel = torch.zeros(4, 3, 3, device=device, requires_grad=True)
    direct = x + F.conv2d(x, full_kernel.unsqueeze(1), padding=1, groups=4)
    F.mse_loss(direct, target).backward()
    tangent_cosine = float(
        F.cosine_similarity(
            probe_gradient.flatten(), full_kernel.grad.flatten(), dim=0
        ).item()
    )

    # Recover a known rank-two channel--spatial kernel bank.
    calibrator = TaskResponseSpatialAdapter(
        channels=4,
        kernel_size=5,
        spatial_rank=2,
        operator_radius=10.0,
        gate_init=1.0,
    ).to(device)
    atoms = torch.randn(2, 5, 5, device=device)
    channel_coefficients = torch.randn(4, 2, device=device)
    target_bank = (channel_coefficients @ atoms.flatten(1)).reshape(4, 5, 5)
    impulse = torch.zeros(16, 4, 13, 13, device=device)
    impulse[:, :, 6, 6] = 1.0
    desired = impulse + F.conv2d(
        impulse, target_bank.unsqueeze(1), padding=2, groups=4
    )
    calibrator.start_calibration()
    F.mse_loss(calibrator(impulse), desired).backward()
    calibrator.accumulate_probe_gradient()
    calibrator.finalize_calibration(init_scale=1.0)
    discovered = calibrator.raw_kernel_bank().detach()
    recovery_cosine = float(
        F.cosine_similarity(discovered.flatten(), target_bank.flatten(), dim=0).item()
    )
    observed_rank = int(torch.linalg.matrix_rank(discovered.flatten(1)).item())

    disabled = TaskResponseSpatialAdapter(channels=8, kernel_size=5, spatial_rank=2).to(device)
    disabled.set_active_rank(0)
    disabled_tokens = torch.randn(2, 145, 8, device=device)
    disabled_output = disabled(disabled_tokens)

    return PreflightReport(
        cnn_shape_ok=tuple(y_cnn.shape) == tuple(cnn.shape),
        vit_shape_ok=tuple(y_vit.shape) == tuple(vit.shape),
        cls_token_preserved=bool(torch.equal(y_cls[:, :1], vit_cls[:, :1])),
        bhwc_shape_ok=tuple(y_bhwc.shape) == tuple(bhwc.shape),
        disabled_returns_same_tensor=disabled_output is disabled_tokens,
        fused_max_error=fused_error,
        maximum_channel_kernel_l1=maximum_l1,
        coefficient_gradient_l1=coefficient_gradient,
        calibration_tangent_cosine=tangent_cosine,
        recovered_kernel_cosine=recovery_cosine,
        recovered_channel_spatial_rank=observed_rank,
        configured_rank=2,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json", default="")
    args = parser.parse_args()
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    report = run(device)
    payload = asdict(report)
    print(json.dumps(payload, indent=2))

    assert report.cnn_shape_ok and report.vit_shape_ok
    assert report.cls_token_preserved and report.bhwc_shape_ok
    assert report.disabled_returns_same_tensor
    assert report.fused_max_error < 2e-5
    assert report.maximum_channel_kernel_l1 <= 0.70001
    assert report.coefficient_gradient_l1 > 0.0
    assert report.calibration_tangent_cosine > 0.999999
    assert report.recovered_kernel_cosine > 0.9999
    assert report.recovered_channel_spatial_rank <= report.configured_rank

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("[ALL OK] Scientific TRSO preflight passed")


if __name__ == "__main__":
    main()
