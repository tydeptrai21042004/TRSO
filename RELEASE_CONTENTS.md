# Release contents

Start here:

1. `TRSO_V3_METHOD.md` — universal adaptive proposal equations, automatic resolution rules and scope.
2. `TEST_REPORT_TRSO_V3.md` — V3 invariance, layout, task-path and fairness verification.
3. `CHANGELOG_TRSO_V3.md` — exact changes from V2 to V3.
4. `GENERALIZATION_SCOPE.md` — supported generality and honest scientific boundaries.
5. `UNIVERSAL_RELEASE_CHECKLIST.md` — framework tickets and scientific limits.
6. `UNIVERSAL_FAIR_FRAMEWORK.md` — datasets, tasks, protocol and commands.
7. `SUPPORT_MATRIX.md` — method/backbone/task compatibility.
8. `BASELINE_FIDELITY.md` — paper-fidelity boundaries for every baseline.
9. `kaggle/TRSO_Universal_Fair_OneCell.ipynb` — configurable Kaggle runner.

Primary commands:

```bash
# Verify the complete release
./scripts/verify_universal_release.sh

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
