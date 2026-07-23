# TRSO-v2 change log

## Proposal method

- Added `--trso_variant v2`; preserved V1 for ablation.
- Added grouped task-locked response coefficients.
- Added a calibrated low-parameter channel-response tangent.
- Added RMS-normalized spatial response.
- Added direct ViT class-token coupling.
- Added bounded stronger gate initialization and training-only gate search.
- Added `snr`, `snr_per_param`, and normalized energy-per-parameter allocation scores.
- Reduced the default fair-suite adapter budget to 512 parameters.

## Fair optimization

- Added a common task-aware linear-head starting point.
- Added `--peft_freeze_head` so compatible PEFT methods can share and freeze the same classifier.
- Added `--evaluate_before_training` to expose whether an adapter damages the linear starting point.
- Kept full fine-tuning and linear probing learning rates independently configurable.

## Reporting

- Added separate head and adapter trainable-parameter counts.
- Added V2 architecture metadata to aggregate results.
- Preserved task-specific predictive, calibration, efficiency and convergence metrics.

## Experiments

- Updated the universal fair runner to use the V2 proposal.
- Expanded the TRSO ablation runner with V1/V2 and component controls.
- Added `tools/run_trso_v2_search.py` for a focused three-seed V2 search.
- Updated the universal Kaggle runner for the V2 release.

## Verification

- Added V2 parameter-efficiency, synthetic-fit, gate-gradient, prefix-coupling and score tests.
- Final automated result: 110 tests passed.
