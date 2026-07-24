# TRSO-v3 Performance-Corrected Release

This release corrects the training and allocation choices that previously made
TRSO unnecessarily weak or slow. It does not claim a benchmark gain without a
completed real-dataset run; it provides a stronger, reproducible protocol for
running that benchmark.

## What changed

### 1. Shared classifier is trainable by default

The best linear-probe checkpoint is still reused for every compatible PEFT
method, but the classifier is now an initialization rather than a frozen
bottleneck.

```text
--peft_freeze_head False
--peft_head_lr_scale 0.5
```

The adapter uses the shared PEFT learning rate and the classifier uses half of
that rate. Full fine-tuning and linear probing retain a multiplier of 1.0.

### 2. Safer learning rates

```text
PEFT AdamW LR:         1e-3
Linear-probe AdamW LR: 1e-3
Full fine-tuning LR:   1e-4
Warm-up:               5 epochs
Gradient clipping:     1.0
```

Biases, normalization parameters, scalar gates, and shifts are excluded from
weight decay.

### 3. Calibration preserves cross-layer response strength

The former V3 calibration independently normalized each layer gradient. That
removed useful magnitude differences before layer selection. The new
`global_rms` mode computes one normalization factor over all candidate probe
gradients in a batch:

```text
--trso_calibration_grad_norm global_rms
```

This keeps calibration invariant to overall loss units while preserving the
relative strength of different layers.

### 4. Automatic budgeting is actually sparse

The former automatic budget was based on total backbone parameters and often
exceeded the entire TRSO candidate capacity. The corrected rule uses a fraction
of candidate-adapter capacity:

```text
--trso_parameter_budget 0
--trso_auto_budget_ratio 0.35
--trso_auto_sparse True
```

When `--trso_max_adapters 0`, V3 also resolves a sparse maximum from the number
of candidate layers. An explicit budget or maximum still overrides the
automatic behavior.

### 5. Stronger but still compact proposal

```text
--trso_spatial_rank 4
--trso_basis_trainable True
--trso_coefficient_mode grouped
--trso_channel_groups 0
--trso_grouping_mode response
```

The calibrated rank-constrained basis remains the scientific core, but selected
basis atoms can now move during optimization instead of remaining completely
fixed.

### 6. Global residual perturbation budget

The 5% target is now interpreted as a network-level target:

```text
--trso_residual_target 0.05
--trso_residual_budget_mode global
```

For `L` selected adapters, each adapter receives `0.05 / sqrt(L)`. This avoids
injecting a full 5% residual independently at every block.

### 7. Safer gate search

```text
--trso_gate_search_values 0.0,0.05,0.1,0.25,0.5,1.0
--trso_gate_search_batches 16
```

The identity candidate `0.0` prevents calibration from forcing a harmful
initial perturbation.

### 8. Strong common augmentation for single-label classification

The fair runner uses the same augmentation for all methods:

```text
RandAugment
Color jitter 0.2
Mixup 0.2
Label smoothing 0.1
Random erasing 0.1
```

Multi-label and regression tasks automatically retain the basic protocol.

## Recommended DTD command

```bash
python -m tools.run_fair_suite \
  --dataset dtd \
  --task auto \
  --data_path ./data \
  --download True \
  --backbones resnet50@torchvision,vit_tiny_patch16_224@timm \
  --methods linear,trso,full,lora,bitfit,ssf,adaptformer,conv \
  --seeds 0,1,2 \
  --epochs 50 \
  --batch_size 64 \
  --input_size 0 \
  --peft_lr 1e-3 \
  --full_lr 1e-4 \
  --linear_lr 1e-3 \
  --weight_decay 1e-4 \
  --warmup_epochs 5 \
  --augmentation strong \
  --peft_freeze_head False \
  --peft_head_lr_scale 0.5 \
  --trso_budget 0 \
  --trso_auto_budget_ratio 0.35 \
  --trso_calibration_batches 16 \
  --trso_rank 4 \
  --trso_basis_trainable True \
  --trso_residual_target 0.05 \
  --output_root outputs_performance \
  --manifest experiments/performance_manifest.json \
  --execute
```

## Fast validation command

Use this before a full three-seed run:

```bash
bash scripts/run_high_performance_trso.sh smoke
```

Then run:

```bash
bash scripts/run_high_performance_trso.sh dtd
```

## Validation completed for this release

- Original repository tests: passed.
- New performance-regression tests: passed.
- End-to-end CPU FakeData run: calibration, sparse allocation, gate search,
  optimizer groups, checkpoint selection, reload, and final test all completed.

Real DTD/Flowers accuracy still requires execution with the real downloaded
dataset and pretrained weights. FakeData is a software smoke test, not evidence
of benchmark quality.
