# Universal release checklist

This release addresses the project tickets at the framework level. “Universal”
means every registered dataset route and task type can use the common training
pipeline, while every method/backbone pairing is checked against its scientific
architecture contract. It does **not** mean that a paper-specific baseline is
silently rewritten to run on an incompatible backbone.

## Tickets closed

- AdaptFormer added as a Transformer baseline.
- Piggyback added as a CNN baseline.
- Thirteen unique reported methods; the legacy `adapter` name is only an alias
  of Conv-Adapter and is not duplicated in result tables.
- One controlled AdamW/cosine recipe for every PEFT method and TRSO.
- Full fine-tuning and linear probing may use separate learning rates.
- A best task-aware linear head is reused before PEFT construction and TRSO
  calibration when the method definition permits it.
- Native pretrained input size can be resolved with `--input_size 0`.
- Single-label, multi-label, and regression losses, selection rules, and final
  metrics are supported.
- Every requested method/backbone/task pair appears in the compatibility report
  as either `scheduled` or `skipped` with a reason.
- ResNet-50-only, ViT-only, and dedicated ResNet-26 methods are never silently
  approximated on another architecture.
- TRSO ablations use one shared head per seed and the same training recipe.
- Hyperparameter manifests support every reported method.
- Result aggregation includes predictive, calibration, efficiency, convergence,
  per-class, per-label, and regression metrics as appropriate.
- Kaggle one-cell execution supports configurable datasets, tasks, backbones,
  methods, seeds, and one or multiple GPUs.
- Automatic fairness verification detects protocol mismatches before results are
  interpreted.
- TRSO-v3 removes task-loss-unit dependence through per-batch response normalization.
- Response-derived channel grouping replaces fixed channel-index assumptions.
- Residual RMS control makes initial adapter strength comparable across feature scales.
- Automatic calibration and model-relative budget resolution replace DTD/backbone-specific defaults.
- Rectangular token grids and multiple prefix tokens are supported.
- Conservative generic CNN and Transformer insertion fallbacks are tested and recorded.

## Verified scope

- 37 canonical dataset routes have an automatic or explicit task-resolution path.
- Tasks: single-label classification, multi-label classification, image regression.
- Representative named and generic CNN/Transformer layouts pass structural TRSO preflights.
- Rectangular grids, multiple prefix tokens, task-loss scaling and feature scaling pass numerical invariance tests.
- Complete single-label, multi-label and regression smoke runs pass checkpoint restoration and final metrics.
- The complete 118-test automated suite passes.
- Dry-run manifests and fairness checks pass for all three task types.

## Scientific boundary

Some baselines are not defined for every backbone:

- Conv-Adapter and BAM use their strict ResNet-50 path.
- AdaptFormer uses ViT blocks.
- SSF, LoRA, and BitFit require supported Transformer internals.
- Residual Adapter uses the dedicated ResNet-26 and released shared checkpoint.
- Visual Prompting is a single-label method in this framework because its
  one-to-one source-label mapping is not defined for multi-label/regression.

The compatibility report is the source of truth for a requested experiment.
