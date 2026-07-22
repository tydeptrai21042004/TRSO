# Strict baseline reproduction policy

This repository separates **paper reproductions** from the TRSO proposal. A
method is exposed under a paper name only when the active model factory follows
the publication's computational graph, insertion locations, initialization,
and trainability rule for the supported backbone.

## Reproduced baselines

| CLI method | Reproduced method | Strict supported configuration |
|---|---|---|
| `prompt` | Exploring Visual Prompts for Adapting Large-Scale Models | Frozen pretrained image classifier, padding prompt, fixed output-label mapping, no new task head |
| `conv` / `adapter` | Conv-Adapter | ResNet-50 Bottleneck, depthwise + pointwise adapter, learnable channel scaling, four published insertion schemes |
| `bam` | Bottleneck Attention Module | End-to-end ResNet-50 with BAM after layer1/layer2/layer3; two dilated spatial convolutions |
| `residual` | Residual Adapters | Dedicated reduced-resolution ResNet-26, task-specific BN, series or parallel 1x1 adapters |
| `ssf` | Scaling & Shifting Your Features | torchvision ViT, Swin, or ConvNeXt with SSF after the internal operations specified by the paper |
| `lora` | LoRA | Transformer Q/V projections, alpha/r scaling, Kaiming A, zero B, merge/unmerge |
| `bitfit` | BitFit | Transformer biases plus downstream task head |
| `sidetune` | Side-Tuning | ResNet frozen base plus a copied, trainable side network and learned mixture coefficient |

`full` and `linear` remain standard controls.

## Explicitly excluded

`lora_conv` is not exposed as an original LoRA baseline. The old convolutional
low-rank module remains only as historical source code and is unreachable from
the strict experiment factory.

The former hook-based approximations for BAM, SSF, Conv-Adapter, residual
adapters, and Side-Tuning have been removed from the active path.

## Reproduction boundary

The code reproduces the **method definition** for the configurations above.
Reproducing a paper's reported number additionally requires the exact paper
checkpoint, dataset split, preprocessing, optimizer grid, and random seeds.
Where a method requires a nonstandard pretrained model, the factory requires an
explicit checkpoint rather than silently substituting another backbone.

## Primary references

- Bahng et al., *Exploring Visual Prompts for Adapting Large-Scale Models*, ECCV 2022, arXiv:2203.17274.
- Chen et al., *Conv-Adapter: Exploring Parameter Efficient Transfer Learning for ConvNets*, arXiv:2208.07463 / CVPRW 2024.
- Park et al., *BAM: Bottleneck Attention Module*, BMVC 2018, arXiv:1807.06514.
- Rebuffi et al., *Learning Multiple Visual Domains with Residual Adapters*, NeurIPS 2017, arXiv:1705.08045.
- Rebuffi et al., *Efficient Parametrization of Multi-domain Deep Neural Networks*, CVPR 2018, arXiv:1803.10082.
- Lian et al., *Scaling & Shifting Your Features: A New Baseline for Efficient Model Tuning*, NeurIPS 2022, arXiv:2210.08823.
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022, arXiv:2106.09685.
- Zaken et al., *BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models*, ACL 2022, arXiv:2106.10199.
- Zhang et al., *Side-Tuning: A Baseline for Network Adaptation via Additive Side Networks*, ECCV 2020, arXiv:1912.13503.
