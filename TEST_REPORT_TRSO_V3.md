# TRSO-v3 verification report

## Automated suite

```text
118 passed
```

The suite covers baseline fidelity, checkpoint resumption, accumulation,
classification metrics, multi-label metrics, regression metrics, fair protocol
planning, V1/V2/V3 behavior, named backbones, generic CNN fallback, generic
Transformer fallback, rectangular token grids, and multiple prefix tokens.

## Universal numerical preflight

The release preflight reports:

| Check | Result |
|---|---:|
| V3 parameters in the 64-channel rank-2 diagnostic | 25 |
| Automatically selected channel groups | 8 |
| Basis error after positive loss rescaling | 7.75e-7 |
| Coefficient error after positive loss rescaling | 7.45e-9 |
| Target update/feature RMS ratio | approximately 0.05 |
| Maximum ratio spread across feature scales | 3.73e-6 |
| Rectangular grid path | passed |
| Multiple prefix tokens updated | passed |

The preflight is stored in `test_reports/trso_v3_preflight.json`.

## End-to-end task execution

Tiny deterministic smoke datasets were used only to exercise complete program
paths. Each run completed:

```text
build dataset/model
-> prepare task head
-> calibrate V3
-> allocate ranks
-> train
-> select best validation checkpoint
-> strictly restore checkpoint
-> final test
-> write task-specific metrics
```

Verified task paths:

- single-label classification: accuracy and macro-F1 artifacts;
- multi-label classification: mAP, micro-F1 and macro-F1 artifacts;
- multi-output regression: MAE, RMSE and R-squared artifacts.

The smoke numbers are not benchmark claims. Their purpose is to verify that all
losses, selection metrics, checkpoints, and result schemas execute successfully.
See `test_reports/trso_v3_all_tasks_smoke.json` and the three task logs.

## Fairness and compatibility audit

The universal audit covers 37 canonical dataset routes and three task types.
Representative manifests produced:

| Task | Scheduled compatible pairs | Explicit incompatible skips | Fairness errors |
|---|---:|---:|---:|
| Single-label | 16 | 10 | 0 |
| Multi-label | 14 | 12 | 0 |
| Regression | 14 | 12 | 0 |

Explicit skips are intentional. They preserve the scientific architecture/task
boundaries of paper baselines rather than silently replacing them with a
non-equivalent implementation.

## Scope of the guarantee

The release verifies method invariants, supported layouts, task execution,
protocol equality, and failure behavior. It does not guarantee that one
hyperparameter configuration wins on every dataset or backbone. Controlled
multi-seed real-data experiments remain necessary for performance claims.
