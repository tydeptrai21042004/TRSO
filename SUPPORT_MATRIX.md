# Strict method/backbone support matrix

Unsupported pairings stop before training. Methods are not rewritten for an
architecture family outside the reproduced configuration.

| Method | Supported backbone | Training rule |
|---|---|---|
| Full fine-tuning | Any supported classifier | All parameters |
| Linear probe | Any model with a replaceable task head | Task head only |
| Visual prompt | Pretrained image classifier with sufficient original outputs | Prompt only; original classifier frozen |
| Conv-Adapter | ResNet-50 Bottleneck | Conv-Adapter + task head; backbone frozen |
| BAM | ResNet-50 Bottleneck | End-to-end backbone + BAM + head |
| Residual Adapter | Dedicated ResNet-26 | Adapter 1x1, task BN, task head |
| SSF | torchvision ViT, Swin, ConvNeXt | SSF affine parameters + task head |
| LoRA | ViT, Swin, Transformer | Q/V LoRA + task head |
| BitFit | Transformer | Biases + task head |
| Side-Tuning | ResNet | Copied side + alpha + task head; base frozen |
| TRSO | Recognized CNN, ViT, Swin | Selected TRSO parameters + task head |

## Required strict choices

- `prompt`: keep the pretrained output head. The default fixed mapping is
  `0,1,...,C-1`; use `--prompt_output_indices` for another fixed mapping.
- `conv`: use `--backbone resnet50` and choose one of
  `conv_parallel`, `conv_sequential`, `residual_parallel`, or
  `residual_sequential`.
- `bam`: use `--backbone resnet50`.
- `residual`: supply `--ra_pretrained_checkpoint` for transfer experiments.
  `--weights none` is accepted only for architecture/tests.
- `ssf`: use a torchvision `vit_*`, `swin_*`, or `convnext_*` model.
- `lora`: use a Transformer with identifiable Q/V or packed QKV projections.

`lora_conv` is deliberately rejected by the strict factory.
