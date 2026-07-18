#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
python main.py \
  --tuning_method trso \
  --dataset fake \
  --backbone resnet18 \
  --weights none \
  --pretrained False \
  --nb_classes 4 \
  --input_size 64 \
  --fake_train_size 16 \
  --fake_val_size 8 \
  --fake_test_size 8 \
  --epochs 1 \
  --batch_size 4 \
  --num_workers 0 \
  --device cpu \
  --use_amp False \
  --profile_efficiency False \
  --save_ckpt False \
  --final_test False \
  --trso_calibration_batches 2 \
  --trso_max_adapters 2 \
  --output_dir test_reports/fake_smoke
