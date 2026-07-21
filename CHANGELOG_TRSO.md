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
