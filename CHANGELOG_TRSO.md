# TRSO Replacement Changelog

## Removed

- former HCC/DT1D proposal implementation;
- shifted symmetric axial-kernel construction;
- manual dilation bank and axis-scale router;
- former proposal-specific tests, preflight scripts, sweeps, cells, and plots;
- obsolete proposal command-line arguments.

## Added

- `models/task_response_adapter.py`;
- task-gradient calibration with a zero virtual 2-D probe operator;
- truncated-SVD spatial basis discovery;
- response-energy layer ranking;
- parameter-budget-aware adapter selection;
- single fused 2-D depthwise operator;
- L1 operator-radius projection;
- CNN, ViT-token, and Swin/BHWC layout support;
- class/prefix token preservation;
- strict calibration JSON export/reload;
- offline deterministic FakeData workflow;
- TRSO preflight and budget-sweep tools;
- expanded unit, integration, regression, dataset, and smoke tests;
- frozen-BatchNorm protection during PEFT training;
- CPU-safe AMP, pin-memory, and smoke-test behavior.

## Corrected

- zero-initialized proposal gate was replaced by a nonzero default gate so internal adapter parameters receive gradients immediately;
- static routing and redundant multi-call filtering were removed;
- adapter accounting now distinguishes structural capacity from active trainable parameters;
- configuration loading now fails clearly when calibrated module names or tensor shapes do not match the target model;
- deprecated `timm` import paths were updated.
