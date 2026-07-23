# Release contents

Start here:

1. `UNIVERSAL_RELEASE_CHECKLIST.md` — tickets addressed and scientific limits.
2. `UNIVERSAL_FAIR_FRAMEWORK.md` — datasets, tasks, protocol and commands.
3. `SUPPORT_MATRIX.md` — method/backbone/task compatibility.
4. `TEST_REPORT.md` — automated and structural verification.
5. `kaggle/TRSO_Universal_Fair_OneCell.ipynb` — configurable Kaggle runner.

Primary commands:

```bash
# Verify the release
./scripts/verify_universal_release.sh

# Generate and execute a fair all-compatible-baseline suite
python -m tools.run_fair_suite --help

# Verify that comparable PEFT methods share the same protocol
python -m tools.verify_fairness --help

# Generate task-aware TRSO ablations
python -m tools.run_ablation_suite --help

# Generate baseline/TRSO hyperparameter manifests
python -m tools.run_hparam_sweep --help
```

The fair runner always writes a compatibility JSON/CSV. Read that file before
interpreting a result table: it shows every requested method/backbone/task pair,
including scientifically invalid combinations that were explicitly skipped.
