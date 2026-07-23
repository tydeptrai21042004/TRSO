# Baseline fidelity and reproduction boundary

This repository separates three claims that are often incorrectly combined:

1. **Method-graph fidelity** — the active module follows the released/published computation graph.
2. **Protocol fidelity** — the data split, preprocessing, optimizer, schedule, checkpoint and label mapping match a paper.
3. **Numerical reproduction** — the reported paper result is reproduced within an explicitly stated tolerance.

The corrected code targets method-graph fidelity. Numerical reproduction still
requires the original assets and protocol for each paper.

## Corrected active baselines

| CLI method | Active implementation | Enforced boundary |
|---|---|---|
| `full` | Full fine-tuning | All model parameters and the task head are trainable |
| `linear` | Linear probe | Backbone frozen; task head trainable |
| `prompt` | Visual prompting | Frozen pretrained classifier; padding, fixed-patch or random-patch prompt; fixed output-label map; no replacement task classifier |
| `conv` / `adapter` | Conv-Adapter Design-2/v4 path | ResNet-50 bottleneck; reduced grouped spatial adapter from the bottleneck intermediate activation; pointwise projection and channel scale |
| `bam` | Bottleneck Attention Module | ResNet-50 stage bottlenecks; end-to-end BAM training, not falsely labeled frozen-backbone PEFT |
| `residual` | Residual Adapters | Dedicated reduced-resolution ResNet-26; official option-A shortcut; task BN; parallel or series 1x1 adapters; strict shared-checkpoint coverage |
| `ssf` | Scaling and Shifting Features | SSF inside supported ViT/Swin/ConvNeXt operations; merge/export function with numerical-equivalence test |
| `lora` | Transformer LoRA | Q/V low-rank updates, alpha/r scaling, zero-update initialization and merge/unmerge; unsupported packed-MHA dropout is rejected |
| `bitfit` | BitFit-style bias tuning | Explicit `all`, `transformer`, or `attention` bias scope and explicit task-head policy |
| `adaptformer` | AdaptFormer | Frozen ViT; parallel ReLU bottleneck on the FFN branch; Kaiming down projection and zero up projection |
| `piggyback` | Piggyback | Frozen pretrained CNN weights multiplied by learned binary masks; paper threshold/init; straight-through mask gradients |
| `sidetune` | Side-Tuning | Frozen ResNet base plus a lightweight trainable side network and learned mixture; full-copy side is retained only as a labeled high-capacity ablation |
| `trso` | Proposed method | Task-response, random or DCT basis controls; exact, greedy or uniform allocation controls |

## Important protocol notes

### Visual prompting

The code keeps the original classifier frozen. For a strict experiment, provide
a training-derived and then fixed output-label mapping through
`--prompt_output_indices`. The default identity map is only a deterministic
fallback and must not be presented as the paper's mapping unless that is truly
the selected protocol.

### Residual adapters

Transfer experiments require `--ra_pretrained_checkpoint`. The loader validates
that every expected shared convolution filter is present. `--weights none` is
accepted only for architecture tests and synthetic smoke runs.

### SSF

`merge_ssf_` folds trained SSF affine parameters into supported base operations.
The test suite requires merged and unmerged logits to agree numerically.

### AdaptFormer

The implementation patches only recognized ViT blocks. The original attention
and FFN parameters remain frozen, and the adapter up-projection is initialized
to zero so insertion initially preserves the pretrained function. Swin/window
blocks are rejected because silently reusing the ViT forward equation would not
be a faithful reproduction.

### Piggyback

Convolution weights and biases remain frozen. Real mask scores initialize to
`1e-2` and are thresholded at `5e-3`, so all masks initially equal one. The
backward pass uses the released straight-through rule. Batch-normalization
statistics remain frozen. Reports distinguish full-precision mask scores during
training from one-bit masks at deployment.

### Side-Tuning

`--sidetune_arch lightweight` is the default paper-oriented path.
`--sidetune_arch copy` is intentionally labeled as a high-capacity ablation and
must be reported with its trainable-parameter and compute cost.

## Excluded misleading names

`lora_conv` is not exposed as original LoRA. Historical convolutional low-rank
source code is not part of strict baseline experiments.

## Reproduction requirement

A baseline should be called a **numerical paper reproduction** only after the
following are recorded and matched:

- original pretrained checkpoint and checksum;
- exact train/validation/test split;
- image normalization, resizing and augmentation;
- optimizer, scheduler, epochs and warm-up;
- classifier and label-map initialization;
- random seeds and number of runs;
- parameter-count definition;
- validation model-selection rule.
