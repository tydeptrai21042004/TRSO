# Scientific TRSO Verification Report

**Verification date:** 2026-07-21  
**Scope:** mathematical mechanism, implementation correctness, exact allocation, serialization, backbone integration, offline training, and controlled transfer behavior.

## Commands executed

```bash
python -m pytest -q
bash scripts/run_preflight_trso.sh
bash scripts/run_smoke_trso_fake.sh
bash scripts/run_scientific_trso.sh
```

## Regression result

```text
36 passed, 10 subtests passed in 5.62 s
```

The suite covers CNN, ViT, Swin/BHWC, class-token preservation, non-square grids, exact fusion, per-channel stability projection, calibration alignment, SVD optimality, rank preservation, exact budget optimization, JSON reload, ResNet and ViT insertion, baseline regressions, and FakeData execution.

## Fast preflight

| Check | Result |
|---|---:|
| CNN shape | pass |
| ViT shape | pass |
| Class token preserved | pass |
| BHWC shape | pass |
| Disabled adapter returns the original tensor | pass |
| Explicit-vs-fused maximum error | `1.12e-08` |
| Maximum channel-kernel L1 norm | `0.07112` with radius `0.7` |
| Coefficient gradient L1 | `3.53e-03` |
| Calibration/training tangent cosine | `1.000000` |
| Known rank-2 kernel recovery cosine | `1.000000` |
| Recovered flattened kernel-bank rank | `2` |

## Scientific mechanism experiments

| Test | Result |
|---|---:|
| Aligned tangent cosine, 24 trials, mean | `0.99999998` |
| Aligned tangent cosine, minimum | `0.99999982` |
| Legacy surrogate cosine, mean | `0.03898` |
| Legacy surrogate cosine, median | `-0.11429` |
| Optimal rank-3 SVD relative error | `0.20109` |
| Best of 1,000 random rank-3 subspaces | `0.76911` |
| Random/SVD error ratio | `3.8246x` |
| Channel-specific rank-2 output MSE | `1.97e-12` |
| Shared-kernel output MSE | `57.26109` |
| Exact budget value | `18` |
| Greedy budget value | `12` |
| Maximum observed factorized rank | `2` for configured rank `2` |

## Five-seed controlled transfer experiment

The source classifier was frozen. Both TRSO and the random-subspace baseline used the same task-gradient calibration data, the same rank, the same initialization norm, and the same `13` trainable parameters. The full-kernel comparison used `151` trainable parameters.

| Quantity | Mean |
|---|---:|
| Source accuracy | `100.00%` |
| Corrupted target accuracy | `47.17%` |
| TRSO initial target accuracy | `84.77%` |
| Random-subspace initial accuracy | `57.43%` |
| TRSO after one epoch | `99.43%` |
| Random subspace after one epoch | `88.93%` |
| Full kernel after one epoch | `99.73%` |

Initial standard deviations:

- TRSO: `4.77` percentage points;
- random subspace: `5.54` percentage points.

## Offline ResNet-18 smoke run

The run:

1. inserted eight candidate adapters;
2. calibrated on two mini-batches;
3. selected two layers;
4. saved the basis and active rank configuration;
5. trained one epoch;
6. completed validation.

Selected adapter trainable costs were `129` and `257` parameters. The complete run had `2,438` trainable parameters including the classifier head.

FakeData accuracy is not interpreted as scientific performance.

A second end-to-end run imposed an exact adapter budget of `130` parameters. The allocator selected two rank-one 64-channel adapters, each costing `65` parameters, and the optimizer reported exactly `130` adapter-like trainable parameters.

## What is established

The tests establish that:

- the calibration gradient is the exact tangent of the trained spatial operator;
- SVD gives the correct best rank-constrained response projection;
- channel-specific mixtures are materially more expressive than one shared kernel;
- the rank constraint survives optimization;
- exact global allocation can beat greedy selection;
- the method executes in the complete repository pipeline.

## What is not established

The tests do not establish superiority on real visual-transfer benchmarks. Publication-level evidence still requires matched-budget, multi-seed experiments against official implementations on real datasets and backbones.
