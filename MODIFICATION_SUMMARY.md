# Paper-fidelity baseline correction summary

The active experiment factory was changed from generic hook approximations to
strict, architecture-specific baseline reproductions.

## Replaced implementations

- Visual prompting: frozen original classifier, fixed label mapping, prompt-only optimization.
- Conv-Adapter: depthwise/pointwise module, channel scaling, four published ResNet-50 insertion schemes.
- BAM: complete channel/spatial gates with two dilated convolutions; end-to-end ResNet-50.
- Residual Adapters: dedicated ResNet-26 with task BN and original series/parallel 1x1 mappings.
- SSF: affine parameters after internal ViT/Swin/ConvNeXt operations.
- LoRA: Q/V low-rank updates with loralib initialization and merge/unmerge.
- BitFit: Transformer biases and task head only.
- Side-Tuning: copied side network rather than random initialization.

## Removed from strict results

- LoRA-Conv.
- post-block SSF approximation.
- frozen-backbone BAM PEFT approximation.
- arbitrary torchvision residual-adapter wrappers.
- random shallow side network.
- pixel prompt plus trainable task head.

Unsupported method/backbone combinations now fail before training.
