#!/usr/bin/env bash
set -euo pipefail

# Capability-aware controlled comparison runner.
# Override any argument after the defaults, for example:
#   scripts/run_universal_fair.sh --dataset cifar100 --data_path ./data --download True
python -m tools.run_fair_suite \
  --dataset dtd \
  --task auto \
  --data_path ./data \
  --download True \
  --backbones resnet50@torchvision,vit_tiny_patch16_224@timm \
  --methods auto \
  --seeds 0,1,2 \
  --epochs 50 \
  --input_size 0 \
  --peft_lr 1e-3 \
  --full_lr 1e-4 \
  --linear_lr 1e-3 \
  --warmup_epochs 5 \
  --augmentation strong \
  --peft_freeze_head False \
  --peft_head_lr_scale 0.5 \
  --trso_calibration_batches 16 \
  --trso_auto_budget_ratio 0.35 \
  --trso_rank 4 \
  --trso_basis_trainable True \
  --trso_residual_target 0.05 \
  --weight_decay 1e-4 \
  --manifest experiments/universal_fair_manifest.json \
  --output_root outputs_universal_fair \
  --execute \
  "$@"
