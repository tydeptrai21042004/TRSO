# Strict method/backbone support matrix

Unsupported architecture pairings stop before training. A method is not
silently rewritten into a different adapter when its required architecture is
unavailable.

| Method | Supported backbone domain | Trainable state |
|---|---|---|
| Full fine-tuning | Any supported classifier | All parameters |
| Linear probe | Any model with a replaceable task head | Task head only |
| Visual prompting | Pretrained image classifier with enough original output classes | Prompt only; original classifier frozen |
| Conv-Adapter | torchvision ResNet-50 bottlenecks | Released reduced grouped adapter + task head |
| BAM | ResNet-50 bottlenecks | Backbone + BAM + head |
| Residual Adapter | Dedicated ResNet-26 | Adapter 1x1, task BN and task head |
| SSF | Supported torchvision ViT, Swin and ConvNeXt operations | SSF affine parameters + task head |
| LoRA | ViT/Swin/Transformer with identifiable Q/V or packed QKV projections | Q/V LoRA + task head |
| BitFit | Transformer | Selected biases and optionally task head |
| Side-Tuning | ResNet | Lightweight side path, alpha and task head; base frozen |
| TRSO | Recognized CNN, ViT and Swin feature layouts | Selected TRSO ranks/gates + task head |

## Required strict choices

- `prompt`: set `--prompt_output_indices` to the fixed output-label mapping used
  in the experiment. Mapping estimation must use training data only.
- `conv`: use `--backbone resnet50` and
  `--conv_adapter_mode conv_parallel` for the released Design-2/v4 path. Other
  modes are retained as explicitly named ablations.
- `bam`: use `--backbone resnet50`.
- `residual`: provide `--ra_pretrained_checkpoint` for transfer experiments.
- `ssf`: use a supported `vit_*`, `swin_*` or `convnext_*` model.
- `lora`: use a Transformer with identifiable Q/V or QKV projections. Packed
  torchvision MHA requires `--lora_dropout 0`.
- `sidetune`: use `--sidetune_arch lightweight` for the primary baseline.

`lora_conv` is deliberately rejected by the strict factory.
