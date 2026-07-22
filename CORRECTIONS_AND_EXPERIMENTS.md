# Corrections, tests and experiment workflow

## 1. Correctness fixes

### Checkpoint resumption

The model checkpoint is loaded before optimizer creation. TRSO rank buffers are
then synchronized with parameter trainability, after which the optimizer and
its state are constructed/restored. This prevents inactive ranks from remaining
in the optimizer after resume.

Training-state checkpoints now include and restore:

- best validation metric;
- best epoch;
- full history;
- primary metric and maximize/minimize rule;
- global update count.

Final testing requires a valid strictly loaded best checkpoint; it no longer
silently evaluates the final epoch after a restoration failure.

### Gradient accumulation

The number of updates uses ceiling division. A final partial accumulation window
is normalized by its actual number of microbatches and receives an optimizer
step instead of being discarded.

### Distributed TRSO calibration

When classifier-head warm-up is enabled under distributed training, head
gradients are averaged across ranks before each update so all workers calibrate
the same model state.

## 2. Baseline corrections

- **Conv-Adapter:** reduced grouped spatial adapter, intermediate bottleneck
  insertion, pointwise projection and learned output-channel scale.
- **Residual Adapter:** corrected stem behavior, series BN/1x1/BN ordering,
  option-A zero-channel shortcut, dedicated ResNet-26 and strict shared-filter
  checkpoint coverage.
- **Visual Prompting:** padding, fixed-patch and random-patch prompt families;
  frozen pretrained classifier and explicit fixed output mapping.
- **SSF:** merge-to-backbone export with output-equivalence verification.
- **LoRA:** stricter rank/dropout validation and corrected bias policy.
- **BitFit:** explicit bias scopes and head policy.
- **Side-Tuning:** lightweight side network is the primary path; full-copy side
  remains only as a named capacity ablation.

## 3. TRSO ablation suite

Generate a manifest without starting training:

```bash
python -m tools.run_ablation_suite \
  --dataset dtd \
  --data_path ./data \
  --backbone resnet50 \
  --weights DEFAULT \
  --seeds 0,1,2,3,4 \
  --parameter_budget 50000
```

Execute the generated runs by adding `--execute`.

The suite changes one scientific component at a time:

| Group | Variants |
|---|---|
| Basis | task response, random orthonormal, DCT |
| Allocation | exact dynamic program, greedy, uniform |
| Layer value | raw energy, energy/parameter, energy/channel, noise-adjusted |
| Rank | 1, 2, 4 |
| Spatial support | 3, 5, 7 |
| Calibration data | 1, 4, 8, 16 batches |
| Head preparation | zero or 25 warm-up steps |
| Budget | half, reference, double, unconstrained candidates |

## 4. Hyperparameter experiments

One-factor-at-a-time is the default because every run has one interpretable
change from a reference setting:

```bash
python -m tools.run_hparam_sweep \
  --method conv \
  --dataset flowers102 \
  --data_path ./data \
  --weights DEFAULT \
  --seeds 0,1,2 \
  --epochs 100 \
  --execute
```

Supported sweep methods are:

```text
full, linear, trso, conv, prompt, ssf, lora, bam, residual, bitfit, sidetune
```

Use `--strategy grid` for a Cartesian grid. The default backbone is selected per
method domain; an explicit `--backbone` overrides it. Residual Adapter transfer
sweeps should provide `--ra_pretrained_checkpoint`.

## 5. Offline smoke runs

To avoid pretrained-weight downloads while validating the pipeline:

```bash
python -m tools.run_ablation_suite \
  --dataset fake --weights none --device cpu --epochs 1 --seeds 0 --max_runs 2 --execute
```

These synthetic runs verify software operation only and must not be used as
scientific accuracy evidence.

## 6. Aggregation

After completing a suite:

```bash
python -m tools.aggregate_revision_results \
  --root outputs_ablation \
  --out_csv experiments/ablation_results.csv
```

The tool produces:

- a raw per-run CSV;
- a grouped CSV with mean, standard deviation and seed count for every numeric
  metric;
- grouping by suite, ablation/sweep variant, dataset, backbone and method.

## 7. Fair-comparison checklist

For paper tables, hold constant across methods:

- pretrained checkpoint;
- train/validation/test split;
- preprocessing and augmentation;
- number of seeds;
- model-selection metric;
- hyperparameter-search budget;
- total training epochs or update count.

Report separately:

- trainable parameters;
- total parameters;
- calibration time;
- training time;
- peak memory;
- inference latency;
- search cost.
