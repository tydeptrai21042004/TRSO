# TRSO-v3 changelog

TRSO-v3 generalizes the low-parameter V2 proposal beyond one dataset, loss,
feature scale, token layout, or named backbone.

## Performance-corrected update

- Replaced independent per-layer calibration normalization with one shared
  `global_rms` factor, preserving cross-layer response magnitude.
- Replaced backbone-size budgeting with sparse candidate-capacity budgeting.
- Changed the default proposal to rank four with trainable response bases.
- Interpreted the residual target as one global network perturbation budget and
  divided it by the square root of the selected adapter count.
- Added identity and small-value gate-search candidates.
- Reused the best linear head as initialization and jointly adapted it at a
  lower learning-rate multiplier instead of freezing it by default.
- Added optimizer groups with no decay for bias, normalization, gate, scale,
  and shift parameters; the scheduler now preserves group LR multipliers.
- Corrected AdamW defaults to `1e-3` for linear probing and PEFT.
- Added five targeted performance regression tests and an offline full fair-run
  execution check.

## Method changes

- Made V3 the default proposal while retaining V1 and V2 as ablations.
- Added per-batch calibration-gradient normalization for task-loss scale
  invariance.
- Added response-derived balanced channel grouping and automatic width-aware
  group counts.
- Added per-sample residual RMS control with a configurable target update ratio.
- Added stable response-energy and stability-per-cost allocation scores.
- Added automatic calibration-batch resolution by loader size and task type.
- Added automatic candidate-capacity parameter budgets with minimum and maximum
  clipping plus a sparse adapter-count cap.
- Generalized prefix coupling to first/all/mean modes and multiple prefix tokens.
- Added rectangular token-grid support through backbone metadata.
- Added conservative generic insertion fallbacks for unknown pre-norm token
  blocks and Conv2d-based CNNs.

## Framework changes

- Updated fair-suite defaults to V3 automatic settings.
- Added a universal V3 search runner and shell entry point.
- Expanded ablations to compare V1, V2, V3, gradient normalization, residual
  normalization, grouping, prefix coupling, score, rank, kernel, calibration,
  and budget choices.
- Expanded aggregation with TRSO stability, grouping, calibration-gradient, and
  selected-layer diagnostics.
- Added complete end-to-end smoke execution for single-label, multi-label, and
  regression tasks.

## Verification

- 123 automated tests pass in the performance-corrected release.
- Universal release audit passes.
- Fairness verification reports no protocol mismatches in representative
  single-label, multi-label, and regression manifests.
- V3 preflight verifies loss-scale invariance, feature-scale invariance, compact
  parameter scaling, rectangular token grids, and multiple prefix tokens.
