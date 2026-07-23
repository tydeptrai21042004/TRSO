# Modification summary: TRSO-v3 Universal Adaptive

## Objective

Generalize the proposal beyond the low DTD/ResNet-18/ViT-Tiny case while
retaining a very small adapter state and preserving V1/V2 for scientific
ablation.

## Core corrections

- Normalized each calibration batch response so task-loss units cannot dominate
  the learned basis.
- Replaced channel-index grouping with response-derived balanced grouping.
- Added width-aware automatic group counts.
- Added feature-scale-invariant residual RMS control.
- Added stable response-energy-per-parameter allocation.
- Added automatic task/data-aware calibration size.
- Added model-relative automatic parameter budgets.
- Generalized token handling to rectangular patch grids and multiple prefix
  tokens.
- Added named-contract-first generic CNN and Transformer insertion fallbacks.
- Recorded all resolved choices and TRSO diagnostics in result artifacts.

## Framework corrections

- V3 is the default in direct runs, fair suites, Kaggle, and search runners.
- V1 and V2 remain available and are included in ablation/search manifests.
- Single-label, multi-label, and regression paths complete calibration, training,
  strict best-checkpoint restoration, final testing, and metric export.
- Paper baselines retain strict compatibility boundaries and explicit skip rows.

## Verification

- 118 automated tests pass.
- Universal audit and fairness verification pass.
- V3 numerical preflight passes loss-scale, feature-scale, parameter-count,
  rectangular-token, and multi-prefix checks.
- End-to-end smoke runs pass all three task families.
