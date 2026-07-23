> **Legacy dataset-specific example.** The active release is dataset/task/backbone generic; use `tools.run_fair_suite` and `TRSO_V3_METHOD.md` for new experiments.

# Fair all-baseline DTD experiment

This suite replaces the earlier six-epoch smoke test. It is designed to compare
adaptation mechanisms rather than different optimization recipes.

## Controlled protocol

Every PEFT method and TRSO uses the same settings:

| Setting | Value |
|---|---:|
| Dataset | Complete DTD, official partition 1 |
| Image size | 224 x 224 |
| Seeds | 0, 1, 2 |
| Epochs | 20 |
| Optimizer | AdamW |
| Learning rate | 5e-3 |
| Scheduler | Cosine decay |
| Warm-up | 5 epochs |
| Minimum learning rate | 1e-6 |
| Weight decay | 1e-4 |
| Training augmentation | RandomResizedCrop + horizontal flip |
| Validation selection | Best validation top-1 |
| Test protocol | Official DTD test split, evaluated once after selection |

Full fine-tuning uses `1e-4`, and linear probing uses `1e-1`, as the only
permitted learning-rate exceptions. The optimizer family, scheduler, warm-up,
input resolution, augmentation, split, checkpoint selection and seed list are
otherwise controlled.

## Task-aware shared head

A linear head is trained once per backbone and seed. Its best checkpoint is
loaded by every compatible PEFT method before adapter construction or TRSO
calibration. This avoids calibrating TRSO from a random classifier and prevents
head optimization from dominating comparisons among small adapters. The shared
head preparation cost is preserved as a separate suite and must be reported.

Visual Prompting is the architectural exception because it retains the original
pretrained source classifier. Its source-to-target map is estimated only from
the DTD training split using frequency counts and a one-to-one maximum-weight
assignment. Validation and test labels are never used to build the map.

## Comparison groups

### CNN: common ResNet-50

- full fine-tuning;
- linear probing;
- visual prompting;
- Conv-Adapter;
- BAM;
- Side-Tuning;
- Piggyback;
- TRSO.

### Transformer: common ViT-Tiny

- full fine-tuning;
- linear probing;
- visual prompting;
- SSF;
- LoRA;
- BitFit;
- AdaptFormer;
- TRSO.

### Residual Adapter: separate ResNet-26 group

Residual Adapter uses its dedicated ResNet-26 architecture and released shared
checkpoint. It is not mixed into the ResNet-50 table. Supply
`--ra_pretrained_checkpoint` to add series and parallel runs.

## TRSO corrections relevant to the previous low result

- The task head is loaded before response calibration.
- ViT adapters are inserted before each Transformer block, so even the final
  adapter can influence the class token through self-attention.
- DTD uses the pretrained native 224 resolution instead of 96.
- Calibration uses 16 batches and a noise-adjusted score.
- The common PEFT learning rate is 5e-3 instead of 3e-4.
- Training uses 20 epochs and five warm-up epochs instead of six short epochs.
- Parameter budgets are large enough to test the method but should still be
  accompanied by budget-matched ablations.

These corrections remove known protocol disadvantages. They do not guarantee a
particular accuracy; final results still depend on the method, backbone,
pretrained checkpoint and hyperparameter robustness.

## Run

Generate the manifest without training:

```bash
python tools/run_fair_dtd_suite.py
```

Execute all default runs:

```bash
python tools/run_fair_dtd_suite.py \
  --execute \
  --data_path ./data \
  --output_root outputs_fair_dtd \
  --seeds 0,1,2
```

Include the dedicated Residual Adapter group:

```bash
python tools/run_fair_dtd_suite.py \
  --execute \
  --ra_pretrained_checkpoint /path/to/resnet26_shared.pth
```

Aggregate results:

```bash
python tools/aggregate_revision_results.py \
  --root outputs_fair_dtd \
  --out_csv outputs_fair_dtd/all_runs.csv
```

The aggregator writes:

- raw per-run results;
- a complete numeric mean/std/count table;
- a compact paper-facing table with accuracy, macro metrics, calibration,
  parameter counts, FLOPs, latency, memory and convergence.

## Reported metrics

### Predictive

- top-1 and top-5 accuracy;
- loss;
- macro precision, recall and F1;
- weighted F1;
- balanced accuracy;
- per-class precision, recall, F1, accuracy and support;
- confusion matrix.

### Calibration

- expected calibration error;
- Brier score;
- mean confidence.

### Efficiency and convergence

- trainable and total parameters;
- trainable-parameter ratio;
- FLOPs;
- inference latency and throughput;
- peak inference memory;
- best validation accuracy and epoch;
- total and mean epoch time;
- epochs to 95% of the best validation result;
- Piggyback training mask-score memory and deployed one-bit mask storage.
