# Generalization scope and scientific boundaries

## What is generalized

TRSO-v3 is dataset-name independent. It operates on the task loss and feature
layout supplied by the active experiment, with scale-normalized calibration and
scale-controlled residual updates. It supports the repository's three task
families:

- single-label classification;
- multi-label classification;
- image regression, including multiple outputs.

It supports recognized CNN, ViT and Swin insertion contracts and conservative
generic fallbacks for Conv2d CNNs and pre-normalized token blocks. Token support
includes rectangular grids and multiple prefix tokens when grid metadata is
available.

Automatic calibration size, response grouping, and model-relative budget
selection prevent the default proposal from being hard-coded for DTD,
ResNet-18, ViT-Tiny, 47 classes, or one loss scale.

## What is not claimed

No method can guarantee high accuracy for every dataset, task, architecture,
distribution shift, or training budget. V3 improves generality and removes known
structural disadvantages; it does not fabricate a universal accuracy guarantee.

Paper baselines are also not universal by definition. Conv-Adapter, BAM,
AdaptFormer, Residual Adapter, and other methods retain their strict architecture
contracts. The fair runner records incompatible pairs as explicit skips. It does
not distort those methods merely to fill every cell of a result table.

## Source of truth for an experiment

For each experiment, inspect:

```text
resolved_protocol.json
dataset_protocol.json
trso_calibration.json
*_compatibility.json
fairness_verification.json
```

These artifacts state the resolved task, input size, optimizer/scheduler,
calibration size, parameter budget, insertion points, selected ranks, metrics,
and any compatibility skips.
