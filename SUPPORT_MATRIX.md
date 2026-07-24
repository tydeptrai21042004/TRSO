# Strict method/backbone/task support matrix

Unsupported combinations stop before training or are marked as explicit skips by
`tools.run_fair_suite`. A method is never silently rewritten into a different
baseline.

| Method | Supported backbone domain | Supported tasks | Trainable state |
|---|---|---|---|
| Full fine-tuning | Any supported classifier/regressor | Single-label, multi-label, regression | All parameters |
| Linear probe | Any model with a replaceable task head | Single-label, multi-label, regression | Task head only |
| Visual Prompting | Pretrained image classifier with enough source classes | Single-label only | Prompt only; original classifier frozen |
| Conv-Adapter | torchvision ResNet-50 bottlenecks | Single-label, multi-label, regression task-head extension | Released reduced grouped adapter + task head |
| BAM | ResNet-50 bottlenecks | Single-label, multi-label, regression task-head extension | Backbone + BAM + head |
| Residual Adapter | Dedicated ResNet-26 | Single-label, multi-label, regression task-head extension | Adapter 1x1, task BN and task head |
| SSF | Supported ViT, Swin and ConvNeXt operations | Single-label, multi-label, regression task-head extension | SSF affine parameters + task head |
| LoRA | Transformer with identifiable Q/V or packed QKV projections | Single-label, multi-label, regression task-head extension | Q/V LoRA + task head |
| BitFit | Transformer | Single-label, multi-label, regression task-head extension | Selected biases + task head |
| AdaptFormer | Recognized ViT blocks | Single-label, multi-label, regression task-head extension | Parallel FFN bottleneck + task head |
| Piggyback | CNN/ResNet convolutional weights | Single-label, multi-label, regression task-head extension | Real mask scores + task head |
| Side-Tuning | ResNet | Single-label, multi-label, regression task-head extension | Lightweight side path, alpha and task head |
| TRSO-v3 | Named CNN/ViT/Swin contracts plus conservative generic Conv2d and pre-norm token-block fallbacks; BCHW/BHWC/BNC, rectangular grids and multiple prefixes | Single-label, multi-label, regression | Response-group amplitudes, bounded gates, optional prefix couplings and task head |

“Task-head extension” means the paper’s adaptation operator is preserved while
the downstream task head/loss is changed. These rows must not be described as
an exact reproduction of a paper’s original non-classification benchmark when
the paper evaluated only classification.

## Required strict choices

- `prompt`: use training-only frequency/Hungarian mapping or provide fixed
  `--prompt_output_indices`; it is intentionally unavailable for multi-label and
  regression tasks.
- `conv`: use `--backbone resnet50` and
  `--conv_adapter_mode conv_parallel` for the released Design-2/v4 path.
- `bam`: use `--backbone resnet50`.
- `residual`: use a separate `resnet26_adapter@auto` group and provide
  `--ra_pretrained_checkpoint`.
- `ssf`: use a supported ViT, Swin or ConvNeXt implementation.
- `lora`: use a Transformer with identifiable Q/V or packed QKV projections;
  packed torchvision MHA requires zero LoRA dropout.
- `adaptformer`: use a recognized ViT; Swin/window blocks are rejected rather
  than approximated.
- `piggyback`: paper defaults are mask score `1e-2`, threshold `5e-3`, frozen
  biases/BN and Adam-family optimization.
- `sidetune`: use `--sidetune_arch lightweight` for the primary baseline.
- `trso`: V3 defaults to native resolution (`--input_size 0`), a task-aware shared head, global-RMS calibration, sparse candidate-capacity allocation, a global residual budget, rank-4 trainable response bases, jointly adapted task head, named insertion contracts with recorded generic fallbacks, and pre-block token insertion.

`adapter` is a legacy alias of `conv` and is not a separate baseline row.
`lora_conv` is deliberately rejected by the strict factory.
