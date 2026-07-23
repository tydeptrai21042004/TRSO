# AdaptFormer, Piggyback and fair-evaluation release

## Added baselines

- `adaptformer`: parallel FFN bottleneck for recognized ViT blocks, with frozen
  pretrained parameters and identity-safe zero up-projection.
- `piggyback`: learned binary masks over frozen CNN weights, paper threshold and
  initialization, straight-through gradients, and separate training/deployment
  storage reporting.

## Fair training protocol

- one optimizer family, PEFT learning rate, cosine scheduler, warm-up, weight
  decay, augmentation, resolution, epochs and validation rule for every PEFT
  method and TRSO;
- only full fine-tuning and linear probing use their explicitly allowed learning
  rates;
- one task-aware shared head per backbone and seed is loaded before adapter
  training and TRSO calibration;
- complete DTD official split, three seeds and native 224 resolution;
- frequency/Hungarian training-only label mapping for Visual Prompting.

## TRSO corrections

- task head loaded before response calibration;
- noise-adjusted allocation available and used by the fair suite;
- ViT TRSO uses pre-block insertion, allowing the final block's patch response
  to affect the class token;
- checkpoint/trainability synchronization and partial accumulation-window fixes
  retained from the previous corrected release.

## Reporting

- top-1/top-5, loss, macro precision/recall/F1, weighted F1 and balanced accuracy;
- ECE, Brier score and confidence;
- per-class metrics and confusion matrix;
- trainable/total parameters, FLOPs, latency, throughput, memory and convergence;
- Piggyback one-bit deployment storage;
- raw, complete mean/std/count and compact paper-facing CSVs.

## Verification

- `96 passed` in the complete pytest suite;
- all ten paper-named baseline preflight forward/backward checks pass;
- 54-run default DTD manifest verified, including eight CNN and eight
  Transformer configurations over three seeds plus shared-head preparation.
