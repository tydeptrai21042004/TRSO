# Experiment suites

## Generate plans without training

```bash
./scripts/run_ablation_suite.sh \
  --dataset fake --backbone resnet18 --weights none --device cpu --epochs 1 --seeds 0 --max_runs 3

./scripts/run_hparam_sweep.sh \
  --method trso --dataset fake --backbone resnet18 --weights none --device cpu --epochs 1 --seeds 0 --max_runs 5
```

Add `--execute` to run the generated commands. Every run receives:

- deterministic suite, variant and run identifiers;
- a unique output path;
- JSON and CSV manifests;
- automatic completed-run skipping.

Use at least three seeds, preferably five, for reported tables.

## TRSO ablations

The suite isolates basis source, allocation algorithm, response-score
normalization, rank, kernel support, calibration sample count, classifier-head
warm-up and parameter budget. Every row changes one component from the proposed
reference configuration.

## Hyperparameter sweeps

The tool covers:

```text
full, linear, trso, conv, prompt, ssf, lora, bam, residual, bitfit, sidetune
```

It defaults to one-factor-at-a-time sweeps. `--strategy grid` enables a full
factorial grid. Method-compatible default backbones are selected automatically,
but can be overridden using `--backbone`.

## Aggregate results

```bash
python -m tools.aggregate_revision_results \
  --root outputs_ablation \
  --out_csv experiments/ablation_results.csv
```

This writes raw per-run results and grouped mean/std/count tables by experiment
suite, variant, dataset, backbone and method.
