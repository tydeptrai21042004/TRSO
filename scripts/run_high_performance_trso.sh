#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "$MODE" == "smoke" ]]; then
  OUT="${2:-outputs_performance_smoke}"
  rm -rf "$OUT"
  python main.py \
    --dataset fake --task single_label --nb_classes 10 \
    --fake_train_size 64 --fake_val_size 32 --fake_test_size 32 \
    --backbone resnet18 --model_source torchvision \
    --pretrained False --weights none --input_size 32 \
    --tuning_method trso --batch_size 8 --epochs 2 --num_workers 0 \
    --device cpu --use_amp False --profile_efficiency False \
    --measure_eval_latency False --fair_protocol True \
    --fair_peft_lr 1e-3 --fair_linear_lr 1e-3 \
    --fair_warmup_epochs 1 --clip_grad 1.0 --peft_freeze_head False \
    --peft_head_lr_scale 0.5 --trso_calibration_batches 2 \
    --trso_head_warmup_steps 2 --trso_gate_search_batches 2 \
    --trso_auto_budget_ratio 0.35 --output_dir "$OUT" \
    --save_ckpt True --save_history True --final_test True \
    --evaluate_before_training True --auto_resume False
  echo "Smoke run completed: $OUT"
  exit 0
fi

if [[ "$MODE" == "dtd" ]]; then
  DATA_PATH="${2:-./data}"
  OUTPUT_ROOT="${3:-outputs_performance_dtd}"
  python -m tools.run_fair_suite \
    --dataset dtd --task auto --data_path "$DATA_PATH" --download True \
    --backbones resnet50@torchvision,vit_tiny_patch16_224@timm \
    --methods linear,trso,full,lora,bitfit,ssf,adaptformer,conv \
    --seeds 0,1,2 --epochs 50 --batch_size 64 --input_size 0 \
    --peft_lr 1e-3 --full_lr 1e-4 --linear_lr 1e-3 \
    --weight_decay 1e-4 --warmup_epochs 5 --augmentation strong \
    --peft_freeze_head False --peft_head_lr_scale 0.5 \
    --trso_budget 0 --trso_auto_budget_ratio 0.35 \
    --trso_calibration_batches 16 --trso_rank 4 \
    --trso_basis_trainable True --trso_residual_target 0.05 \
    --output_root "$OUTPUT_ROOT" \
    --manifest "$OUTPUT_ROOT/manifest.json" --execute
  exit 0
fi

echo "Usage: $0 [smoke [output_dir] | dtd [data_path] [output_root]]" >&2
exit 2
