# Universal fair experiment framework

This release is not tied to DTD. The active experiment router supports every
canonical dataset in `datasets/build.py`, all three task types, and every
backbone/method pairing that the strict implementation can represent without
silently changing the baseline.

## What "works on all datasets/tasks/backbones" means

The repository guarantees the following:

1. **Datasets** — every dataset listed by `python main.py --list_backbones` under
   `[datasets]` can be passed to the generic runner. Generic `imagefolder` and
   `csv` inputs allow additional datasets without changing source code.
2. **Tasks** — single-label, multi-label, and regression use separate losses,
   primary validation metrics, final-test metrics, and aggregation columns.
3. **Backbones** — torchvision and timm classifiers are resolved dynamically.
   `--input_size 0` reads the pretrained model's native spatial resolution
   before constructing dataset transforms.
4. **Baselines** — every requested method/backbone/task combination receives a
   row in the compatibility report. Unsupported combinations are marked
   `skipped` with a reason; they are never silently omitted or replaced by a
   different architecture.

This does **not** mean that every paper baseline is mathematically defined for
every backbone. For example, Conv-Adapter and BAM require ResNet-50, AdaptFormer
requires ViT, and Residual Adapter requires its dedicated ResNet-26 checkpoint.
The framework preserves these scientific boundaries.

## Fair controlled protocol

For a given dataset, task, backbone, and seed, all PEFT methods and TRSO share:

- optimizer;
- learning rate;
- cosine scheduler;
- warm-up;
- minimum learning rate;
- weight decay;
- epoch count;
- batch size and accumulation frequency;
- augmentation;
- train/validation/test split and split seed;
- input resolution;
- checkpoint-selection rule.

Only full fine-tuning and linear probing may use different learning rates. A
best linear-probe head is trained once per backbone/seed and loaded into every
compatible PEFT method before adapter construction and TRSO calibration.
Visual Prompting is the deliberate exception because its method definition
retains the frozen source classifier and learns a source-to-target label map.

Every run writes `resolved_protocol.json`. The suite manifest can be checked by:

```bash
python -m tools.verify_fairness \
  --manifest experiments/fair_manifest.json \
  --compatibility experiments/fair_manifest_compatibility.json
```

## Generic comparison runner

```bash
python -m tools.run_fair_suite \
  --dataset dtd \
  --task auto \
  --data_path ./data \
  --download True \
  --backbones resnet50@torchvision,vit_tiny_patch16_224@timm \
  --methods auto \
  --seeds 0,1,2 \
  --epochs 20 \
  --input_size 0 \
  --execute
```

`--methods auto` requests every registered unique method. The legacy name
`adapter` is treated only as an alias of `conv` and is not reported as a second
baseline.

### Multi-label example

```bash
python -m tools.run_fair_suite \
  --dataset voc2007 \
  --task multilabel \
  --data_path ./VOCdevkit \
  --backbones resnet50@torchvision,swin_t@torchvision \
  --seeds 0,1,2 \
  --execute
```

Visual Prompting will appear as an explicit compatibility skip because its
frozen source-logit label mapping is a single-label formulation. Other
feature/weight adapters remain task-head agnostic.

### Regression CSV example

```bash
python -m tools.run_fair_suite \
  --dataset csv \
  --task regression \
  --data_path ./my_regression_data \
  --dataset_args_json '{"train_csv":"train.csv","val_csv":"val.csv","test_csv":"test.csv","label_column":"target"}' \
  --backbones resnet50@torchvision,vit_tiny_patch16_224@timm \
  --execute
```

For `csv`, `--task` must be explicit.

## Fair TRSO ablation

The ablation runner is also dataset/task/backbone agnostic. It first trains a
linear task head per seed and supplies that exact head to every ablation:

```bash
python -m tools.run_ablation_suite \
  --dataset dtd \
  --task auto \
  --data_path ./data \
  --download True \
  --backbone resnet50 \
  --model_source torchvision \
  --input_size 0 \
  --seeds 0,1,2 \
  --epochs 20 \
  --execute
```

The suite covers response/random/DCT bases, exact/greedy/uniform allocation,
noise-aware and normalized scores, rank, kernel support, calibration size, and
parameter budget.

## Metrics

### Single-label

- top-1 and top-5 accuracy;
- macro precision, recall and F1;
- weighted F1 and balanced accuracy;
- expected calibration error, Brier score and mean confidence;
- per-class metrics and confusion matrix.

### Multi-label

- mean average precision;
- micro and macro precision/recall/F1;
- weighted F1;
- subset and Hamming accuracy;
- label-cardinality error;
- calibration error, Brier score and confidence;
- per-label AP, precision, recall, F1 and support.

### Regression

- MAE, median absolute error and RMSE;
- R-squared;
- Pearson and Spearman correlation;
- per-output MAE, RMSE, R-squared and correlations.

### Efficiency and convergence

- trainable/total parameters and trainable ratio;
- FLOPs, latency, throughput and GPU memory when profiling is enabled;
- Piggyback training mask memory and one-bit deployment storage;
- best validation metric/epoch, total time, mean epoch time and time-to-target.

Aggregate all tasks with:

```bash
python -m tools.aggregate_revision_results \
  --root outputs_fair \
  --out_csv outputs_fair/all_results.csv
```

## Result expectations

The framework removes known causes of artificially low TRSO results: random-head
calibration, non-native resolution, too-short schedules, inconsistent learning
rates, noisy Transformer response scoring, and post-final-block ViT insertion.
It cannot honestly guarantee that TRSO will outperform every baseline on every
dataset. The output is designed to reveal the actual result with a controlled
protocol rather than force a preferred conclusion.
