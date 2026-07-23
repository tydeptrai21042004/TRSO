# Modification summary — universal fair release

## Core framework

- Replaced the DTD-only comparison assumption with a dataset/task/backbone-aware
  fair-suite planner.
- Added explicit compatibility planning and automatic fairness verification.
- Added native input-size resolution before dataset transforms.
- Added resolved dataset and training protocol JSON files to every run.

## Methods

- Added paper-oriented AdaptFormer and Piggyback implementations and tests.
- Retained strict paper-domain contracts for all existing baselines.
- Kept `adapter` as a CLI alias only, preventing duplicate Conv-Adapter rows.
- Corrected shared-head loading before TRSO calibration and pre-block ViT TRSO
  insertion so patch responses can influence the class token.

## Tasks and metrics

- Single-label: top-1/top-5, loss, macro/weighted metrics, balanced accuracy,
  ECE, Brier score, confidence, per-class metrics, confusion matrix.
- Multi-label: mAP, micro/macro precision/recall/F1, weighted F1, subset and
  Hamming accuracy, label-cardinality error, calibration and per-class metrics.
- Regression: MAE, median AE, RMSE, R², Pearson, Spearman and per-output metrics.
- Efficiency and convergence fields are aggregated with mean/std/seed count.

## Experiment entry points

- `python -m tools.run_fair_suite`
- `python -m tools.run_ablation_suite`
- `python -m tools.run_hparam_sweep`
- `python -m tools.verify_fairness`
- `kaggle/TRSO_Universal_Fair_OneCell.ipynb`

## Verification

- 106 automated tests pass.
- All paper-named baseline structural preflights pass.
- TRSO BCHW/BNC/BHWC and calibration/recovery preflights pass.
- Universal audit covers 37 canonical dataset routes and all three task types.
