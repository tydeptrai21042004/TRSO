# Verification report

## Automated result

```text
85 passed
```

Run with:

```bash
./scripts/test_all.sh
```

## New regression coverage

### Training and resumption

- incomplete gradient-accumulation windows still perform an optimizer step;
- exact accumulation windows preserve the expected step count;
- frozen BatchNorm running statistics remain unchanged during PEFT/linear-probe training;
- TRSO active rank is restored before optimizer construction;
- the post-load hook restores rank-specific `requires_grad` state;
- strict resume rejects architecture-mismatched checkpoints;
- optimizer state, epoch, best metric, best epoch and history are restored.

### Baseline fidelity

- Conv-Adapter released equation and ResNet bottleneck insertion location;
- padding, fixed-patch and random-patch visual prompts;
- Residual Adapter option-A shortcut, series order and shared-filter coverage;
- SSF merged/unmerged numerical equivalence;
- LoRA nonzero merge/unmerge equivalence and unsupported MHA dropout guard;
- BitFit bias-scope and task-head policy;
- lightweight Side-Tuning default and frozen base model.

### TRSO scientific controls

- response, random and DCT basis construction;
- exact, greedy and uniform allocation under a common budget;
- energy, energy-per-parameter, energy-per-channel and noise-adjusted scores;
- deterministic experiment-grid and manifest generation.

## Baseline-only release test

```bash
./scripts/test_all_baselines.sh
```

This script now uses pytest consistently and no longer reports zero integration
tests through an incompatible unittest invocation.
