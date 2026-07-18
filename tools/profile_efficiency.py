# tools/profile_efficiency.py
"""Utility functions for FLOPs, parameter, latency, and memory profiling."""

from __future__ import annotations

import json
import os
import time
from contextlib import nullcontext
from typing import Dict, Optional

import torch

try:
    from fvcore.nn import FlopCountAnalysis
except Exception:  # optional dependency
    FlopCountAnalysis = None


def count_params(model: torch.nn.Module) -> Dict[str, int | float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "trainable_param_ratio": float(trainable / max(total, 1)),
    }


def _amp_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda" and torch.cuda.is_available():
        return torch.amp.autocast(device_type="cuda")
    return nullcontext()


@torch.no_grad()
def measure_flops(model: torch.nn.Module, device: torch.device, input_size: int = 224) -> Optional[int]:
    if FlopCountAnalysis is None:
        return None
    was_training = model.training
    model.eval()
    x = torch.randn(1, 3, input_size, input_size, device=device)
    try:
        flops = int(FlopCountAnalysis(model, x).total())
    except Exception as e:
        print(f"[Warn] FLOPs computation failed: {e}")
        flops = None
    model.train(was_training)
    return flops


@torch.no_grad()
def measure_latency(
    model: torch.nn.Module,
    device: torch.device,
    input_size: int = 224,
    batch_size: int = 32,
    warmup: int = 20,
    iters: int = 100,
    use_amp: bool = False,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    x = torch.randn(batch_size, 3, input_size, input_size, device=device)

    for _ in range(max(0, warmup)):
        with _amp_context(device, use_amp):
            _ = model(x)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)

    start = time.time()
    for _ in range(max(1, iters)):
        with _amp_context(device, use_amp):
            _ = model(x)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elapsed = time.time() - start

    ms_per_batch = elapsed * 1000.0 / max(1, iters)
    ms_per_image = ms_per_batch / batch_size
    model.train(was_training)
    return {
        "latency_ms_per_batch": float(ms_per_batch),
        "latency_ms_per_image": float(ms_per_image),
        "fps": float(1000.0 / max(ms_per_image, 1e-12)),
    }


@torch.no_grad()
def measure_peak_inference_memory(
    model: torch.nn.Module,
    device: torch.device,
    input_size: int = 224,
    batch_size: int = 32,
    use_amp: bool = False,
) -> Optional[float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    was_training = model.training
    model.eval()
    x = torch.randn(batch_size, 3, input_size, input_size, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    with _amp_context(device, use_amp):
        _ = model(x)
    torch.cuda.synchronize(device)
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    model.train(was_training)
    return float(peak_mb)


def profile_model(
    model: torch.nn.Module,
    device: torch.device,
    input_size: int = 224,
    batch_size: int = 32,
    use_amp: bool = False,
    warmup: int = 20,
    iters: int = 100,
) -> Dict:
    profile = count_params(model)
    flops = measure_flops(model, device, input_size=input_size)
    profile["flops"] = None if flops is None else int(flops)
    profile["flops_g"] = None if flops is None else float(flops / 1e9)
    profile.update(
        measure_latency(
            model,
            device,
            input_size=input_size,
            batch_size=batch_size,
            warmup=warmup,
            iters=iters,
            use_amp=use_amp,
        )
    )
    profile["peak_inference_memory_mb"] = measure_peak_inference_memory(
        model,
        device,
        input_size=input_size,
        batch_size=batch_size,
        use_amp=use_amp,
    )
    profile["profile_batch_size"] = int(batch_size)
    profile["input_size"] = int(input_size)
    return profile


def save_profile(profile: Dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, default=str)
