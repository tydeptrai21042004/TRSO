# Verification report

## Automated tests

```text
58 passed
```

The suite covers:

- exact visual-prompt trainability and fixed output mapping;
- Conv-Adapter topology and ResNet-50 insertion;
- BAM equation, two-dilation spatial branch, and stage placement;
- dedicated ResNet-26 residual-adapter architecture and series/parallel modes;
- SSF insertion inside ViT operations;
- LoRA zero-update identity and merge/unmerge;
- BitFit trainability;
- copy-initialized Side-Tuning;
- dataset/task protocols and TRSO mathematical/integration tests.

## Paper-baseline preflight

`test_reports/paper_baselines_preflight.json` reports successful forward and
backward passes for visual prompting, Conv-Adapter, BAM, residual adapters, SSF,
LoRA, BitFit, and Side-Tuning.

## TRSO preflight

`test_reports/trso_preflight.json` verifies CNN, ViT and BHWC shapes, class-token
preservation, fused-kernel agreement, gradient flow, calibration tangent
alignment, and configured spatial rank.

## Training smoke

`test_reports/prompt_training_smoke/` contains a complete one-epoch FakeData
training run using strict visual prompting. The run trains 720 prompt parameters
and leaves the original ResNet classifier frozen.
