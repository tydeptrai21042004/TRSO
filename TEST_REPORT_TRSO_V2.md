# TRSO-v2 verification report

## Automated suite

```text
110 passed
```

The suite covers the existing universal framework, corrected baselines, task-specific evaluation, checkpoint restoration, fair-suite planning, and the new V2 operator.

## New V2 checks

1. V2 uses no more than 20% of V1 adapter parameters in the controlled rank-two test.
2. V2 provides lower calibrated synthetic approximation error than V1 in that controlled task.
3. One-parameter class-token coupling changes the ViT prefix token and receives gradients.
4. Gate initialization at one avoids the gradient suppression caused by the V1 gate initialized at `0.01`.
5. SNR-per-parameter allocation scores are finite.

## Controlled numerical diagnostic

For a synthetic `C=64`, rank-two response:

| Variant | Adapter cost | Initial response MSE |
|---|---:|---:|
| No adapter | 0 | approximately 0.01498 |
| TRSO-v1 | 129 | approximately 0.01015 |
| TRSO-v2 | 25 | approximately 0.00896 |

This diagnostic verifies operator behavior only. It is not a real-dataset accuracy result.

## End-to-end smoke test

A CNN fake-data run completed:

- task-aware setup;
- V2 calibration;
- exact allocation;
- optional gate search;
- training;
- strict best-validation checkpoint restoration;
- final test evaluation.

The example gate search selected `1.0` from `0.25, 0.5, 1.0, 1.5` using training calibration batches only.

## Transformer status

Structural Transformer tests pass, including nonzero class-token coupling gradients. A local real timm-model smoke run was not executed because `timm` was unavailable in the local container. The Kaggle runner installs `timm` before execution.

## Accuracy limitation

The previous DTD result was produced by the old V1, six-epoch, 96-pixel protocol. A full three-seed DTD rerun of V2 is still required. No real-data accuracy improvement is claimed without that rerun.
