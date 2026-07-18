# TRSO Project Verification Report

**Verification date:** 2026-07-18  
**Scope:** implementation correctness, architecture integration, calibration/configuration flow, offline smoke training, and synthetic operator checks.

## Release status

The former shifted axial/Hartley proposal has been removed and replaced by **TRSO — Task-Response Spatial Operator Adaptation**. The release supports CNN feature maps, Vision Transformer patch tokens, and Swin/channels-last feature maps through one adapter implementation.

## Commands executed

```bash
python -m compileall -q .
python -m pytest -q
bash scripts/run_preflight_trso.sh
bash scripts/run_smoke_trso_fake.sh
```

A second offline training run loaded the saved `trso_calibration.json` with strict module-name and tensor-shape validation before training.

## Automated test result

```text
32 passed, 10 subtests passed in 4.43s
```

The focused preflight suite additionally reported:

```text
20 passed in 2.81s
```

### Covered cases

- exact residual identity when the gate is zero;
- nonzero-gate first-step gradient flow;
- documented zero-gate gradient starvation behavior;
- CNN `B x C x H x W` input/output shape;
- ViT `B x N x C` patch-token input/output shape;
- class/prefix token preservation;
- explicit non-square token-grid handling;
- invalid grid detection;
- Swin-style `B x H x W x C` input/output shape;
- fused operator equivalence to an explicit basis sum;
- L1 operator-radius projection;
- autograd-based recovery of a synthetic task operator;
- rank-two truncated-SVD optimality check;
- response-score layer selection;
- parameter-budget-aware layer selection;
- calibration JSON save/load round trip;
- strict saved-configuration reload;
- adapter parameter accounting;
- ResNet integration;
- torchvision ViT integration;
- selected-adapter training update;
- deterministic offline FakeData train/validation/test splits;
- baseline-module regression tests.

## Synthetic operator and fusion checks

Latest `tools/preflight_trso.py` result:

| Check | Result |
|---|---:|
| CNN shape | pass |
| ViT shape | pass |
| Class token preserved | pass |
| BHWC/Swin shape | pass |
| Fused-versus-explicit maximum error | `1.49e-08` |
| Internal adapter gradient L1 | `1.43e-07` |
| Calibration cosine with target direction | `0.96835` |
| Rank-2 relative approximation error | `0.24958` |
| Axial-cross relative approximation error | `0.63752` |
| Rank-2 output MSE | `0.01359` |
| Axial-cross output MSE | `0.09249` |

For this controlled target, the task-response rank-two operator had substantially lower reconstruction and output error than the broader axial-cross comparison family.

## CPU microbenchmark

For the small synthetic preflight tensor:

| Implementation | Latency |
|---|---:|
| One fused TRSO depthwise operator | `0.372 ms` |
| Six separate axial calls | `9.966 ms` |
| Observed ratio | `26.81x` |

This is a CPU microbenchmark of operator execution, not a GPU deployment claim. Actual latency depends on hardware, tensor sizes, memory layout, compiler, and insertion count.

## Offline end-to-end smoke training

`bash scripts/run_smoke_trso_fake.sh` completed all of the following without network access:

1. built a torchvision ResNet-18;
2. inserted eight candidate TRSO modules;
3. performed two task-response calibration batches;
4. selected two layers;
5. saved `test_reports/fake_smoke/trso_calibration.json`;
6. constructed the optimizer with only selected adapter/head parameters;
7. completed one training epoch;
8. completed validation.

The run used 16 random FakeData training samples and 8 validation samples. Its accuracy is deliberately not interpreted as model performance.

A separate reload run loaded a saved calibration file using `strict=True`, selected the same stored adapter configuration, and completed training and validation.

## What these tests establish

They establish that the replacement method is internally consistent, differentiable, budget-selectable, serializable, and executable on both convolutional and Transformer-style feature layouts.

They do **not** establish that TRSO outperforms published PEFT methods on real benchmarks. A research claim still requires multi-seed, matched-budget experiments on real datasets and hardware, including at least one CNN and one Transformer backbone.
