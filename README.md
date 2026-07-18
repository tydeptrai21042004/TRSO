# TRSO-Adapter

**Task-Response Spatial Operator Adaptation for parameter-efficient visual fine-tuning**

TRSO replaces the former hand-designed axial/Hartley adapter with a task-derived two-dimensional spatial operator. A short calibration pass measures the downstream-loss response to a zero virtual depthwise kernel at every candidate layer. The leading singular directions of that response define the spatial basis, and layers are selected by captured singular energy under a user-defined budget.

The same adapter works with:

- CNN feature maps in `B x C x H x W` format;
- Vision Transformer patch tokens in `B x N x C` format;
- Swin-style channels-last maps in `B x H x W x C` format.

Class or prefix tokens are preserved and are never spatially filtered.

## Main differences from the removed proposal

| Removed design | TRSO design |
|---|---|
| Fixed shifted symmetric axial kernels | Task-loss-derived full 2-D spatial directions |
| Manually selected dilation bank | No dilation bank |
| Static axis-scale router | No router |
| Same insertion policy for all layers | Response-score layer selection |
| Restricted even axial operator family | Rank-constrained 2-D operator family |
| Multiple height/width convolution launches | One fused depthwise convolution |
| Theory added after architecture design | SVD basis follows from a rank-constrained local objective |

The previous `models/hcc_adapter.py`, HCC/DT1D tests, preflight scripts, and experiment cells have been removed from this project.

## Method summary

For candidate layer `l`, calibration exposes a virtual shared depthwise kernel `K_l`, initialized to zero. For calibration samples, TRSO accumulates

```text
G_l = average gradient of downstream loss with respect to K_l at K_l = 0.
```

The rank-constrained local problem

```text
minimize  <G_l, K_l> + (lambda / 2) ||K_l||_F^2
subject to rank(K_l) <= r
```

has solution proportional to the rank-`r` truncated SVD of `-G_l`. TRSO retains those rank-one spatial atoms, scores each layer by captured squared singular energy, and enables the highest-scoring layers under `--trso_keep_ratio` and `--trso_max_adapters`.

During normal training, selected adapters use

```text
feature -> 1x1 down projection -> one fused depthwise 2-D operator
        -> GELU -> 1x1 up projection -> gated residual update
```

The fused kernel is projected to a prescribed L1 radius with `--trso_operator_radius`.

## Project structure

```text
TRSO-Adapter/
├── main.py
├── engine.py
├── models/
│   ├── task_response_adapter.py
│   ├── backbones/
│   └── tuning_modules/
├── datasets/
├── tools/
│   ├── preflight_trso.py
│   ├── run_trso_budget_sweep.py
│   ├── profile_efficiency.py
│   └── aggregate_revision_results.py
├── scripts/
│   ├── run_preflight_trso.sh
│   ├── run_smoke_trso_fake.sh
│   └── test_all_baselines.sh
├── tests/
│   ├── test_trso_adapter.py
│   ├── test_trso_integration.py
│   ├── test_baseline_modules.py
│   ├── test_baseline_integration.py
│   └── test_fake_dataset_smoke.py
├── requirements.txt
├── CHANGELOG_TRSO.md
└── TEST_REPORT.md
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate             # Linux/macOS
# .venv\Scripts\activate              # Windows
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build separately when GPU training is required.

## Fast verification

```bash
bash scripts/run_preflight_trso.sh
```

Or run the complete test suite:

```bash
python -m pytest -q
```

The preflight covers CNN tensors, Transformer tokens, class-token preservation, Swin-style layout, gradient-based calibration, SVD recovery, fused-kernel equivalence, stability projection, gradient flow, synthetic approximation quality, and a CPU kernel-launch microbenchmark.

## CNN example

```bash
python main.py \
  --tuning_method trso \
  --backbone resnet50 \
  --weights DEFAULT \
  --dataset dtd \
  --data_path ./data \
  --nb_classes 47 \
  --epochs 100 \
  --batch_size 32 \
  --trso_kernel_size 5 \
  --trso_spatial_rank 2 \
  --trso_channel_ratio 16 \
  --trso_calibration True \
  --trso_calibration_batches 8 \
  --trso_keep_ratio 0.5 \
  --trso_max_adapters 8 \
  --trso_parameter_budget 500000 \
  --output_dir ./experiments/dtd_resnet50_trso
```

## Vision Transformer example

Torchvision ViT encoder blocks produce `B x N x C` tokens. TRSO automatically detects the patch grid when the patch count is square and preserves the class token.

```bash
python main.py \
  --tuning_method trso \
  --backbone vit_b_16 \
  --weights DEFAULT \
  --dataset flowers102 \
  --data_path ./data \
  --nb_classes 102 \
  --epochs 50 \
  --batch_size 16 \
  --trso_kernel_size 5 \
  --trso_spatial_rank 2 \
  --trso_channel_ratio 32 \
  --trso_calibration_batches 8 \
  --trso_keep_ratio 0.5 \
  --output_dir ./experiments/flowers102_vitb16_trso
```

## Swin Transformer example

Torchvision Swin blocks use channels-last spatial tensors and are supported by the same hook path.

```bash
python main.py \
  --tuning_method trso \
  --backbone swin_t \
  --weights DEFAULT \
  --dataset eurosat \
  --data_path ./data \
  --nb_classes 10 \
  --epochs 50 \
  --batch_size 32 \
  --trso_spatial_rank 2 \
  --trso_keep_ratio 0.5 \
  --output_dir ./experiments/eurosat_swint_trso
```

## Calibration configuration reuse

A calibrated run saves `trso_calibration.json` inside the output directory. Reuse it without recalibrating:

```bash
python main.py \
  --tuning_method trso \
  --backbone resnet50 \
  --trso_config ./experiments/dtd_resnet50_trso/trso_calibration.json \
  ...
```

The JSON contains selected layers, task-response scores, singular values, basis atoms, and initial coefficients.

## TRSO arguments

| Argument | Meaning | Default |
|---|---|---:|
| `--trso_kernel_size` | Odd spatial support | `5` |
| `--trso_spatial_rank` | Number of SVD atoms per selected layer | `2` |
| `--trso_channel_ratio` | Channel bottleneck ratio | `16` |
| `--trso_operator_radius` | Maximum fused-kernel L1 norm | `1.0` |
| `--trso_gate_init` | Residual gate initialization | `0.01` |
| `--trso_basis_init_scale` | Initial total basis coefficient scale | `0.05` |
| `--trso_basis_trainable` | Fine-tune discovered atoms themselves | `False` |
| `--trso_calibration` | Run task-response calibration | `True` |
| `--trso_calibration_batches` | Calibration mini-batches | `8` |
| `--trso_head_warmup_steps` | Optional classifier warm-up steps | `0` |
| `--trso_keep_ratio` | Fraction of candidate layers retained | `1.0` |
| `--trso_max_adapters` | Hard layer cap; `0` means no additional cap | `0` |
| `--trso_parameter_budget` | Adapter-only trainable-parameter budget; `0` disables | `0` |
| `--trso_config` | Load prior calibration JSON | empty |
| `--trso_save_config` | Explicit calibration JSON output | empty |

## Fair evaluation recommendations

For a paper-quality comparison:

1. match adapter-only parameter budgets across methods;
2. report classifier parameters separately;
3. use at least three seeds and report mean plus standard deviation;
4. select checkpoints only on validation data;
5. report latency, throughput, training memory, and per-task storage;
6. test at least one CNN and one Transformer;
7. compare calibrated layer selection with random and uniform placement;
8. compare task-derived atoms with random, DCT, Fourier, and axial-only bases.

## Limitations

The included tests establish implementation correctness and synthetic operator recovery. They do **not** establish benchmark superiority. Real accuracy claims require full multi-seed experiments under matched parameter and compute budgets.
