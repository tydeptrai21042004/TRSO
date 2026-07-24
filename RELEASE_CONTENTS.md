# Release contents

Start here:

1. `PERFORMANCE_RELEASE.md` — corrected optimization, sparse allocation, global calibration and residual controls.
2. `TEST_REPORT_PERFORMANCE.md` — automated, smoke, and audit validation for the corrected release.
3. `TRSO_V3_METHOD.md` — universal adaptive proposal equations, automatic resolution rules and scope.
4. `CHANGELOG_TRSO_V3.md` — exact V3 and performance-release changes.
5. `GENERALIZATION_SCOPE.md` — supported generality and honest scientific boundaries.
6. `UNIVERSAL_RELEASE_CHECKLIST.md` — framework tickets and scientific limits.
7. `UNIVERSAL_FAIR_FRAMEWORK.md` — datasets, tasks, protocol and commands.
8. `SUPPORT_MATRIX.md` — method/backbone/task compatibility.
9. `BASELINE_FIDELITY.md` — paper-fidelity boundaries for every baseline.
10. `kaggle/TRSO_Universal_Fair_OneCell.ipynb` — configurable corrected Kaggle runner.

Primary commands:

```bash
# Verify the complete release
./scripts/verify_universal_release.sh

# Fast corrected TRSO integration smoke
./scripts/run_high_performance_trso.sh smoke outputs_performance_smoke

# Full corrected DTD protocol
./scripts/run_high_performance_trso.sh dtd ./data outputs_performance_dtd

# Generate and execute a fair all-compatible-baseline suite
python -m tools.run_fair_suite --help

# Run the focused universal adaptive TRSO-v3 search
python -m tools.run_trso_v3_search --help

# Verify that comparable PEFT methods share the same protocol
python -m tools.verify_fairness --help

# Generate V1/V2/V3 and component ablations
python -m tools.run_ablation_suite --help

# Generate baseline/TRSO hyperparameter manifests
python -m tools.run_hparam_sweep --help
```

The fair runner always writes compatibility JSON/CSV files. Read them before
interpreting a result table: every requested method/backbone/task pair appears,
including scientifically invalid combinations that were explicitly skipped.
