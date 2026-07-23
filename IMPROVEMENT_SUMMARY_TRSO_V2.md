# Proposal improvement summary

## Problem observed in the old DTD result

The old experiment produced approximately 22.38% CNN and 5.73% Transformer
accuracy for TRSO. The ablation means differed by less than one test image.
The method entered training with a very small effective update, calibrated
through an untrained classifier, used a full channel-by-rank coefficient matrix,
and had no direct final class-token path.

## New default proposal

`TRSO-v2` now uses:

- a best task-aware linear head before calibration;
- optional frozen shared head for strict adapter-only comparison;
- task-derived SVD spatial atoms;
- fixed task-response channel directions;
- only group-by-rank trainable amplitudes;
- a complementary grouped channel-response branch;
- RMS-normalized spatial filtering;
- one-scalar class-token coupling for ViTs;
- stronger bounded residual gates;
- training-only global gate search;
- SNR-per-parameter exact layer/rank allocation;
- a 512-parameter default adapter budget.

## Parameter example

At rank two and 64 channels:

```text
V1: 64*2 + 1 = 129 adapter parameters per CNN layer
V2: 8*2 + 8 + 1 = 25 adapter parameters per CNN layer
```

V2 uses approximately 19.4% of the V1 cost in the controlled test.

## Evidence included in the release

- 110 automated tests pass.
- Every baseline structural preflight passes.
- The universal dataset/task/backbone audit passes.
- The V2 controlled diagnostic improves response MSE from about 0.01015 to
  0.00896 while reducing adapter cost from 129 to 25.
- A CNN fake-data end-to-end run completed calibration, allocation, gate search,
  training, strict checkpoint restoration and final evaluation.

## Honest limitation

The complete DTD three-seed experiment has not been rerun here. The release is a
corrected and stronger implementation, not a fabricated accuracy claim. Use the
included Kaggle notebook to obtain the real comparison.
