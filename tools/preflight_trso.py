#!/usr/bin/env python3
"""Fast TRSO preflight and synthetic research checks.

This script does not download datasets or pretrained weights. It verifies:
- CNN, ViT-token, class-token, and channels-last support;
- task-response calibration and SVD basis discovery;
- exact basis fusion;
- stability projection;
- gradient flow at the recommended non-zero gate;
- a synthetic comparison against an axial-cross operator family;
- a small CPU latency comparison between one fused 2-D operator and six
  separate axial branches.
"""
from __future__ import annotations

import argparse
import json
import time
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
    fused_max_error: float
    projected_kernel_l1: float
    internal_gradient_l1: float
    calibration_cosine: float
    rank2_relative_error: float
    axial_cross_relative_error: float
    rank2_output_mse: float
    axial_cross_output_mse: float
    fused_latency_ms: float
    six_axial_latency_ms: float
    latency_ratio: float


def _bench(fn, warmup=5, repeats=30):
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) * 1000.0 / repeats


def _axial_cross_projection(kernel: torch.Tensor) -> torch.Tensor:
    k = kernel.shape[0]
    center = k // 2
    projected = torch.zeros_like(kernel)
    projected[center, :] = kernel[center, :]
    projected[:, center] = kernel[:, center]
    projected[center, center] = kernel[center, center]
    return projected


def run(device: torch.device) -> PreflightReport:
    torch.manual_seed(19)
    torch.set_num_threads(1)

    adapter = TaskResponseSpatialAdapter(
        channels=8,
        kernel_size=5,
        spatial_rank=2,
        channel_ratio=4,
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

    z = torch.randn(2, adapter.hidden_channels, 12, 12, device=device)
    explicit = adapter.explicit_basis_response(z)
    fused = adapter._shared_depthwise(z, adapter.projected_fused_kernel())
    fused_error = float((explicit - fused).abs().max().item())

    loss = y_cnn.square().mean()
    loss.backward()
    internal_grad = float(adapter.down.weight.grad.abs().sum().item() + adapter.coefficients.grad.abs().sum().item())

    # Synthetic task-response recovery from an impulse experiment.
    target = torch.tensor(
        [
            [0.00, 0.00, 0.12, 0.00, -0.05],
            [0.00, 0.18, 0.00, -0.12, 0.00],
            [0.08, 0.00, 0.32, 0.00, 0.08],
            [0.00, -0.12, 0.00, 0.18, 0.00],
            [-0.05, 0.00, 0.12, 0.00, 0.00],
        ],
        device=device,
    )
    calibrator = TaskResponseSpatialAdapter(
        channels=1,
        kernel_size=5,
        spatial_rank=2,
        channel_ratio=1,
        operator_radius=10.0,
        gate_init=1e-2,
    ).to(device)
    impulse = torch.zeros(16, 1, 13, 13, device=device)
    impulse[:, :, 6, 6] = 1.0
    desired = impulse + F.conv2d(impulse, target.view(1, 1, 5, 5), padding=2)
    calibrator.start_calibration()
    F.mse_loss(calibrator(impulse), desired).backward()
    calibrator.accumulate_probe_gradient()
    calibrator.finalize_calibration(init_scale=1.0)
    discovered = calibrator.raw_fused_kernel().detach()
    cosine = float(F.cosine_similarity(discovered.flatten(), target.flatten(), dim=0).item())

    # Approximation comparison: best rank-2 2-D operator vs axial-cross support.
    u, s, vh = torch.linalg.svd(target, full_matrices=False)
    rank2 = (u[:, :2] * s[:2]) @ vh[:2]
    axial = _axial_cross_projection(target)
    target_norm = target.norm().clamp_min(1e-12)
    rank2_err = float(((target - rank2).norm() / target_norm).item())
    axial_err = float(((target - axial).norm() / target_norm).item())

    signal = torch.randn(32, 1, 32, 32, device=device)
    target_y = F.conv2d(signal, target.view(1, 1, 5, 5), padding=2)
    rank2_y = F.conv2d(signal, rank2.view(1, 1, 5, 5), padding=2)
    axial_y = F.conv2d(signal, axial.view(1, 1, 5, 5), padding=2)
    rank2_mse = float(F.mse_loss(rank2_y, target_y).item())
    axial_mse = float(F.mse_loss(axial_y, target_y).item())

    # Structural latency check: one fused 5x5 depthwise convolution vs six
    # independent 1-D axial branch launches. This is a microbenchmark, not a GPU claim.
    feature = torch.randn(8, 64, 28, 28, device=device)
    fused_weight = torch.randn(64, 1, 5, 5, device=device)
    h_weights = [torch.randn(64, 1, 5, 1, device=device) for _ in range(3)]
    w_weights = [torch.randn(64, 1, 1, 5, device=device) for _ in range(3)]

    def fused_call():
        return F.conv2d(feature, fused_weight, padding=2, groups=64)

    def axial_call():
        out = 0.0
        for weight in h_weights:
            out = out + F.conv2d(feature, weight, padding=(2, 0), groups=64)
        for weight in w_weights:
            out = out + F.conv2d(feature, weight, padding=(0, 2), groups=64)
        return out

    with torch.no_grad():
        fused_ms = _bench(fused_call)
        axial_ms = _bench(axial_call)

    return PreflightReport(
        cnn_shape_ok=tuple(y_cnn.shape) == tuple(cnn.shape),
        vit_shape_ok=tuple(y_vit.shape) == tuple(vit.shape),
        cls_token_preserved=bool(torch.equal(y_cls[:, :1], vit_cls[:, :1])),
        bhwc_shape_ok=tuple(y_bhwc.shape) == tuple(bhwc.shape),
        fused_max_error=fused_error,
        projected_kernel_l1=float(adapter.projected_fused_kernel().abs().sum().item()),
        internal_gradient_l1=internal_grad,
        calibration_cosine=cosine,
        rank2_relative_error=rank2_err,
        axial_cross_relative_error=axial_err,
        rank2_output_mse=rank2_mse,
        axial_cross_output_mse=axial_mse,
        fused_latency_ms=fused_ms,
        six_axial_latency_ms=axial_ms,
        latency_ratio=axial_ms / max(fused_ms, 1e-12),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json", default="")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    report = run(device)
    payload = asdict(report)
    print(json.dumps(payload, indent=2))

    assert report.cnn_shape_ok and report.vit_shape_ok and report.cls_token_preserved and report.bhwc_shape_ok
    assert report.fused_max_error < 2e-5
    assert report.projected_kernel_l1 <= 0.70001
    assert report.internal_gradient_l1 > 0.0
    assert report.calibration_cosine > 0.95
    assert report.rank2_relative_error < report.axial_cross_relative_error
    assert report.rank2_output_mse < report.axial_cross_output_mse

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("[ALL OK] TRSO preflight passed")


if __name__ == "__main__":
    main()
