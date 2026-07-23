# Verification report — universal fair release

## Automated tests

```text
110 passed
```

The suite covers baseline modules, paper-oriented architecture fidelity,
checkpoint resume, partial gradient accumulation, experiment grids, task-aware
metrics, fair-manifest generation, compatibility skips, input/task routing,
AdaptFormer, Piggyback, universal task routing, and TRSO-v1/v2 calibration/allocation controls.

## Structural baseline preflight

`test_reports/paper_baselines_preflight.json` reports successful output and
trainable-gradient checks for:

- Visual Prompting;
- Conv-Adapter;
- BAM;
- Residual Adapter;
- SSF;
- LoRA;
- BitFit;
- AdaptFormer;
- Piggyback;
- Side-Tuning.

## TRSO-v2 checks

The V2-specific suite verifies grouped task-locked coefficients, low parameter
cost, synthetic response approximation, bounded stronger gating, direct class-token
coupling, and SNR-per-parameter allocation. See `TEST_REPORT_TRSO_V2.md`.

## TRSO preflight

`test_reports/trso_preflight_universal.json` verifies:

- BCHW, BNC and BHWC layouts;
- V1 class-token preservation and V2 direct class-token coupling;
- fused and explicit operator equivalence;
- coefficient gradients;
- exact calibration/training tangent alignment;
- known low-rank kernel recovery;
- rank bound and operator-radius projection.

## Universal release audit

`test_reports/universal_release_audit.json` verifies offline that:

- all 37 canonical dataset entries have a task-resolution path;
- single-label, multi-label and regression manifests are generated;
- all scheduled PEFT rows share the controlled recipe;
- unsupported method/backbone/task pairs are explicit skips;
- Visual Prompting is explicitly excluded from multi-label/regression rather
  than silently producing an invalid result;
- the structural baseline and TRSO preflights pass.

## Executed task smoke tests

The development audit executed one-epoch synthetic train/validation/test runs
for single-label, multi-label and regression paths using frozen-head and TRSO
configurations. Each path produced a strict best checkpoint and task-specific
final-test JSON. These runs validate plumbing only; they are not benchmark
accuracy claims.

## Executable manifest and fairness dry runs

The release entry points were executed in planning mode for all three tasks:

| Task | Executable runs | Scheduled method/backbone pairs | Explicit skips | Fairness errors |
|---|---:|---:|---:|---:|
| Single-label, 3 seeds | 48 | 16 | 10 | 0 |
| Multi-label, 1 seed | 14 | 14 | 12 | 0 |
| Regression, 1 seed | 14 | 14 | 12 | 0 |

The universal TRSO ablation planner generated 69 runs for three seeds, and the
AdaptFormer hyperparameter planner generated 10 valid configurations in its
smoke manifest. These were planner/CLI checks, not accuracy claims.
