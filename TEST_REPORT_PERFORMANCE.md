# TRSO-v3 Performance Release Test Report

## Validation status

| Check | Result |
|---|---|
| Complete pytest suite | **123 passed** |
| New performance regression tests | **5 passed** as part of the suite |
| Direct TRSO FakeData smoke | **Passed** |
| Full fair-runner linear → TRSO flow | **Passed** offline with `--weights none` |
| Universal release audit | **Passed**, `all_ok=true` |
| Kaggle one-cell notebook synchronization | **Completed** |

## Behaviors verified

The automated tests and execution smokes verify that:

- global-RMS calibration preserves relative cross-layer response magnitude and
  remains invariant to an overall loss-scale change;
- automatic budgeting uses candidate-adapter capacity and resolves a genuinely
  sparse maximum adapter count;
- the global residual target is divided by `sqrt(number_of_selected_adapters)`;
- adapter and classifier parameters use separate learning-rate multipliers;
- bias, normalization, gates, scales, and shifts are excluded from weight decay;
- cosine scheduling preserves per-parameter-group learning-rate multipliers;
- the fair runner trains a shared linear head, reloads its best checkpoint for
  TRSO, calibrates TRSO, trains both adapter and head, and restores the best TRSO
  checkpoint for final testing.

## Direct smoke details

The CPU smoke used FakeData, ResNet-18, two epochs, and no downloaded weights.
It resolved:

- candidate capacity: `1376`;
- automatic budget: `482` (`35%`);
- sparse cap: `5` adapters from `8` candidates;
- global residual target: `0.05`;
- per-adapter target: `0.05 / sqrt(5) = 0.0223607`;
- trainable state: `455` TRSO parameters plus `5130` classifier parameters.

## Scientific limitation

FakeData validates execution and invariants, not benchmark accuracy. Real DTD,
Flowers102, VTAB, or other transfer results require pretrained weights and full
multi-seed training. This release intentionally does not report synthetic smoke
accuracy as evidence of method quality.

## Reproduction

```bash
# Fast offline integration check
scripts/run_high_performance_trso.sh smoke outputs_performance_smoke

# Full DTD protocol
scripts/run_high_performance_trso.sh dtd ./data outputs_performance_dtd

# Automated tests
python -m pytest -q

# Offline structural/protocol audit
python -m tools.audit_universal_release --output test_reports/universal_release_audit_performance.json
```
