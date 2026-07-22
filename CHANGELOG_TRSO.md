# Scientific TRSO Changelog

## 2026-07-21: scientific reformulation

### Removed from the proposal core

- random `1x1` down/up bottleneck;
- GELU between calibration and the learned spatial operator;
- surrogate direct-feature calibration;
- one spatial kernel shared identically by every channel;
- greedy response-per-parameter layer selection;
- trainable full-matrix atoms that could violate the stated rank constraint.

### Added

- full channel-specific probe kernel bank;
- calibration and training through the same depthwise operator family;
- SVD of the flattened `C x k^2` task-response matrix;
- channel-specific coefficients over shared task-derived spatial atoms;
- exact factorized rank preservation, including trainable-basis mode;
- exact sparse dynamic programming for joint layer-and-rank allocation;
- per-channel `L1` kernel projection;
- distributed aggregation of calibration statistics before SVD;
- immediate return for disabled token adapters;
- controlled scientific experiment suite;
- five-seed synthetic transfer validation.

### Verified

- `36 passed, 10 subtests passed`;
- calibration/training tangent cosine above `0.9999998` in all 24 trials;
- exact recovery of a known rank-two channel--spatial kernel bank;
- exact rank preservation during optimization;
- exact budget allocation beating a constructed greedy counterexample;
- same-budget task-derived basis outperforming a random basis in controlled transfer.

## 2026-07-22 — Domain-safe framework extension

- Added a method/backbone compatibility registry. Unsupported pairings now fail
  before calibration or optimizer construction.
- Preserved CNN-only baselines as CNN-only and Transformer-only baselines as
  Transformer-only.
- Added original-domain Transformer LoRA on attention Q/V projections while
  retaining `lora_conv` as a separate CNN method.
- Replaced the duplicated residual-adapter modes with distinct paper-style
  series and parallel 1x1 adapters for BasicBlock ResNets only.
- Fixed torchvision ViT head replacement and added optional timm, CIFAR Hub,
  and CLIP/OpenCLIP backbone resolution.
- Expanded the active dataset router to 37 canonical entries, including CUB,
  NABirds, Stanford Dogs, official-style VTAB lists, ImageFolder, CSV, COCO,
  Pascal VOC, CelebA, Country211, Places365 and iNaturalist.
- Added single-label, multi-label and regression losses, metrics and checkpoint
  selection to the shared trainer.
- Added a tested generic metadata-list few-shot route and replaced broken dormant
  dataset modules with active aliases or explicit fail-fast notices.
- Corrected CelebA landmark regression to normalize coordinates and avoid
  classification crops/flips that invalidate spatial targets.
- Removed silent validation-as-test fallback by default and added deterministic,
  disjoint split handling.
- Added optional dependency and support/protocol documentation.
