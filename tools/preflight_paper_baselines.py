#!/usr/bin/env python3
"""Offline structural/gradient preflight for strict paper baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torchvision.models import resnet18, resnet50
from torchvision.models.vision_transformer import VisionTransformer

from main import set_trainability_policy
from models.tuning_modules.bam_adapter import BAMResNet50
from models.tuning_modules.conv_adapter import apply_conv_adapter_resnet50, set_conv_adapter_trainability
from models.tuning_modules.lora_transformer import apply_lora_transformer
from models.tuning_modules.prompter import VisualPromptingClassifier
from models.tuning_modules.residual_adapter import ResidualAdapterResNet26, set_residual_adapter_trainability
from models.tuning_modules.side_tuning import SideTuningClassifier
from models.tuning_modules.ssf import apply_ssf, set_ssf_trainability


def step_ok(model, x):
    model.eval()
    model.zero_grad(set_to_none=True)
    y = model(x)
    y.float().square().mean().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    return list(y.shape), bool(grads and any(g is not None for g in grads))


def run():
    torch.manual_seed(7)
    x = torch.randn(2, 3, 32, 32)
    report = {}

    prompt = VisualPromptingClassifier(
        resnet18(weights=None, num_classes=1000),
        5,
        prompt_size=2,
        image_size=32,
        prompt_type="padding",
    )
    set_trainability_policy(prompt, SimpleNamespace(tuning_method="prompt", weight_decay=1e-4))
    report["visual_prompt"] = step_ok(prompt, x)

    conv = resnet50(weights=None, num_classes=5)
    apply_conv_adapter_resnet50(conv, mode="conv_parallel")
    set_conv_adapter_trainability(conv)
    report["conv_adapter"] = step_ok(conv, x)

    bam = BAMResNet50(resnet50(weights=None), num_classes=5)
    for p in bam.parameters(): p.requires_grad_(True)
    report["bam"] = step_ok(bam, x)

    residual = ResidualAdapterResNet26(5, mode="parallel")
    set_residual_adapter_trainability(residual)
    report["residual_adapter"] = step_ok(residual, x)

    ssf = VisionTransformer(image_size=32, patch_size=8, num_layers=2, num_heads=2, hidden_dim=32, mlp_dim=64, num_classes=5)
    apply_ssf(ssf)
    set_ssf_trainability(ssf)
    report["ssf"] = step_ok(ssf, x)

    lora = VisionTransformer(image_size=32, patch_size=8, num_layers=2, num_heads=2, hidden_dim=32, mlp_dim=64, num_classes=5)
    apply_lora_transformer(lora, rank=2, alpha=4)
    set_trainability_policy(lora, SimpleNamespace(tuning_method="lora"))
    report["lora"] = step_ok(lora, x)

    bitfit = VisionTransformer(image_size=32, patch_size=8, num_layers=1, num_heads=2, hidden_dim=32, mlp_dim=64, num_classes=5)
    set_trainability_policy(bitfit, SimpleNamespace(tuning_method="bitfit", bitfit_train_head=True, weight_decay=1e-4))
    report["bitfit"] = step_ok(bitfit, x)

    side = SideTuningClassifier(
        resnet18(weights=None),
        num_classes=5,
        side_arch="lightweight",
        side_width=16,
        side_depth=4,
    )
    set_trainability_policy(side, SimpleNamespace(tuning_method="sidetune"))
    report["side_tuning"] = step_ok(side, x)

    failures = [name for name, (_shape, grad_ok) in report.items() if not grad_ok]
    return {"baselines": {name: {"output_shape": shape, "gradient_ok": grad} for name, (shape, grad) in report.items()}, "all_ok": not failures, "failures": failures}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="")
    args = parser.parse_args()
    report = run()
    print(json.dumps(report, indent=2))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["all_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
