# Scientific TRSO

> **Strict baseline fidelity:** paper-named baselines now use only their
> published architecture domain and active computational graph. Unsupported
> pairings fail before training. See `BASELINE_FIDELITY.md`,
> `SUPPORT_MATRIX.md`, and `PAPER_REPRODUCTION.md`.


**Task-Response Spatial Operator Adaptation for parameter-efficient visual fine-tuning**

Scientific TRSO is intentionally based on three components only:

1. **aligned task-response measurement** using the same depthwise operator in calibration and training;
2. **exact rank-constrained local projection** by truncated SVD;
3. **exact layer-and-rank allocation** under a global trainable-parameter budget.

The method does not use a random channel bottleneck, GELU inside the proposal operator, a hand-designed dilation bank, a router, frequency branches, or a collection of auxiliary losses.

It supports:

- CNN tensors in `B x C x H x W` format;
- Vision Transformer tokens in `B x N x C` format;
- Swin-style tensors in `B x H x W x C` format.

Spatial filtering preserves prefix tokens exactly. Automatic token-grid inference supports zero, one or two prefix tokens on square grids; rectangular grids must be supplied explicitly by backbone metadata.

## Framework support extension

The repository now uses one shared, task-aware training pipeline for:

- single-label classification;
- multi-label classification;
- image regression.

It exposes 37 active dataset routes, 80 filtered torchvision
image-classification builders in the current environment, optional timm and
Torch Hub backbones, and OpenAI CLIP/OpenCLIP visual encoders.

Baseline support is intentionally **domain-safe**. CNN baselines remain CNN
baselines, Transformer baselines remain Transformer baselines, and unsupported
method/backbone pairs stop before training rather than being silently adapted.

```bash
python main.py --list_compatibility
python main.py --list_backbones
```

See [SUPPORT_MATRIX.md](SUPPORT_MATRIX.md) for the method contract,
[BASELINE_FIDELITY.md](BASELINE_FIDELITY.md) for paper-versus-implementation
labels, and [DATASETS.md](DATASETS.md) for dataset layouts and split protocols.

## 1. Scientific formulation

At candidate layer `l`, let

```text
X_l in R^(B x C_l x H_l x W_l)
```

and introduce a full channel-specific zero probe kernel bank

```text
P_l in R^(C_l x k x k),  P_l = 0.
```

Calibration uses

```text
X_l' = X_l + alpha * DepthwiseConv(X_l; P_l).
```

For `M` calibration mini-batches, the measured response is

```text
G_l = (1/M) sum_b d L_b / d P_l evaluated at P_l = 0.
```

Flatten the spatial dimensions:

```text
G_l^flat in R^(C_l x k^2).
```

For a selected rank `r`, consider

```text
minimize  <G_l^flat, W_l^flat> + (lambda/2) ||W_l^flat||_F^2
subject to rank(W_l^flat) <= r.
```

If

```text
-G_l^flat = U Sigma V^T,
```

then the exact solution direction is the truncated SVD

```text
W_l^flat proportional to U_r Sigma_r V_r^T.
```

This follows directly from completing the square and the Eckart--Young--Mirsky theorem.

## 2. Trainable operator

TRSO parameterizes the depthwise kernel bank as

```text
W_l^flat = A_l B_l,
A_l in R^(C_l x r),
B_l in R^(r x k^2).
```

The rows of `B_l` are the task-derived spatial atoms. The rows of `A_l` give each feature channel its own mixture of those atoms.

The normal forward path is

```text
Y_l = X_l + g_l * DepthwiseConv(X_l; W_l).
```

Therefore:

- calibration and training use the same operator family;
- `rank(W_l^flat) <= r` is preserved exactly;
- fixed-basis trainable cost is only `C_l * r + 1` parameters;
- the channel kernels are not forced to be identical.

Each channel kernel is projected independently to satisfy

```text
||W_l,c||_1 <= rho.
```

## 3. Exact layer-and-rank allocation

For layer `l`, the predicted value of rank `r` is

```text
V_l(r) = sum_(i=1)^r sigma_(l,i)^2.
```

With a fixed task-derived basis, its trainable cost is

```text
C_l(r) = C_l * r + 1,  r > 0,
C_l(0) = 0.
```

TRSO solves

```text
maximize   sum_l V_l(r_l)
subject to sum_l C_l(r_l) <= budget,
           r_l in {0, ..., r_max}.
```

The implementation uses sparse dynamic programming rather than a response-per-cost greedy rule. This permits different layers to receive different ranks.

## 4. Why the previous bottleneck formulation was removed

The earlier code measured the response of

```text
X + DepthwiseConv(X; K)
```

but trained

```text
X + g * Up(GELU(DepthwiseConv(Down(X); K))).
```

The SVD direction was therefore optimal only for a surrogate perturbation, not for the trained adapter. In 24 controlled random trials, the old calibration/training gradient cosine had:

- mean: `0.03898`;
- median: `-0.11429`.

The aligned formulation achieved:

- mean cosine: `0.99999998`;
- minimum cosine: `0.99999982`.

## 5. Verification

Run the fast implementation checks:

```bash
bash scripts/run_preflight_trso.sh
```

Run all controlled scientific experiments:

```bash
bash scripts/run_scientific_trso.sh
```

Run the complete regression suite:

```bash
python -m pytest -q
```

Latest local regression and integration result is recorded in
[TEST_REPORT.md](TEST_REPORT.md).

### Controlled scientific results

The offline experiment suite uses no downloaded data or pretrained weights.

| Hypothesis | Controlled result |
|---|---:|
| Aligned calibration equals the trainable tangent | mean cosine `0.99999998` |
| Legacy surrogate equals the trainable tangent | mean cosine `0.03898` |
| SVD is better than 1,000 random rank-3 spatial subspaces | relative error `0.2011` vs `0.7691` |
| Channel-specific rank-2 bank vs one shared kernel | output MSE `1.97e-12` vs `57.2611` |
| Exact budget allocation vs greedy counterexample | value `18` vs `12` |
| Rank during optimization with configured rank 2 | maximum observed rank `2` |

### Synthetic transfer test

A frozen source classifier was evaluated after a known rank-two spatial domain shift. Results are means over five seeds.

| Method | Trainable operator parameters | Initial target accuracy | After one epoch |
|---|---:|---:|---:|
| No adapter | `0` | `47.17%` | — |
| Random rank-2 spatial subspace | `13` | `57.43%` | `88.93%` |
| **Task-response rank-2 TRSO** | **13** | **84.77%** | **99.43%** |
| Full channel-specific `5x5` kernel | `151` | — | `99.73%` |

This is a controlled mechanism test, not evidence of superiority on VTAB, FGVC, ImageNet, or other real benchmarks.

## 6. Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional COCO, CLIP and experiment-logging integrations
pip install -r requirements-optional.txt
```

Install the appropriate CUDA-enabled PyTorch build separately for GPU experiments.

## 7. CNN example

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
  --trso_spatial_rank 3 \
  --trso_calibration_batches 16 \
  --trso_parameter_budget 5000 \
  --output_dir ./experiments/dtd_resnet50_trso
```

## 8. Transformer example

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
  --trso_spatial_rank 3 \
  --trso_parameter_budget 10000 \
  --output_dir ./experiments/flowers102_vitb16_trso
```

## 9. Arguments

| Argument | Scientific role | Default |
|---|---|---:|
| `--trso_kernel_size` | Spatial support `k` | `5` |
| `--trso_spatial_rank` | Maximum channel--spatial rank | `2` |
| `--trso_operator_radius` | Per-channel kernel `L1` radius | `1.0` |
| `--trso_gate_init` | Nonzero residual scale for first-step gradients | `0.01` |
| `--trso_basis_init_scale` | Frobenius norm of the initial SVD update | `0.05` |
| `--trso_basis_trainable` | Refine spatial atoms while preserving factorized rank | `False` |
| `--trso_calibration_batches` | Mini-batches used to estimate task response | `8` |
| `--trso_basis_source` | Proposed response basis or controlled `random`/`dct` basis | `response` |
| `--trso_allocation` | `exact`, `greedy`, or `uniform` rank allocation | `exact` |
| `--trso_score_mode` | Layer value: energy, per-parameter, per-channel, or noise-adjusted | `energy` |
| `--trso_noise_beta` | Penalty coefficient for noise-adjusted response value | `0.0` |
| `--trso_head_warmup_steps` | Optional synchronized shared-head preparation | `0` |
| `--trso_keep_ratio` | Maximum retained layer fraction | `1.0` |
| `--trso_max_adapters` | Maximum number of adapted layers | `0` |
| `--trso_parameter_budget` | Exact global adapter budget | `0` |
| `--trso_config` | Load a prior calibration allocation | empty |
| `--trso_save_config` | Save calibration and rank allocation | empty |
| `--trso_channel_ratio` | Legacy accepted argument; not used | `16` |

## 10. Required real-data evidence

Before making a publication claim, run:

- at least three seeds;
- strictly matched trainable-parameter budgets;
- task-derived vs random and fixed-basis ablations;
- fixed-rank vs exact adaptive-rank allocation;
- CNN and Transformer backbones;
- calibration-cost, memory, throughput, and storage measurements;
- a common classifier-head preparation protocol across methods.

## Limitations

The repository now verifies the mathematics and implementation of the proposal. It does not yet prove that TRSO outperforms strong PEFT baselines on real visual-transfer benchmarks. The synthetic transfer result establishes only that the proposed response subspace can recover a controlled spatial shift more efficiently than a random subspace.

## 11. Additional task examples

### Pascal VOC 2007 multi-label classification

```bash
python main.py \
  --tuning_method full \
  --backbone resnet50 \
  --weights DEFAULT \
  --dataset voc2007 \
  --task multilabel \
  --data_path ./data \
  --epochs 50 \
  --output_dir ./experiments/voc2007_resnet50_full
```

### CelebA landmark regression

```bash
python main.py \
  --tuning_method linear \
  --backbone vit_b_16 \
  --weights DEFAULT \
  --dataset celeba \
  --celeba_task landmarks \
  --task regression \
  --data_path ./data \
  --regression_loss smooth_l1 \
  --output_dir ./experiments/celeba_landmarks_vit_linear
```

### Original Transformer LoRA

```bash
python main.py \
  --tuning_method lora \
  --backbone vit_b_16 \
  --weights DEFAULT \
  --dataset flowers102 \
  --data_path ./data \
  --lora_r 8 \
  --lora_alpha 16 \
  --output_dir ./experiments/flowers102_vit_lora
```

`lora` is restricted to Transformer Q/V attention projections. The former
`lora_conv` control is excluded from strict paper-reproduction runs because it
is not the original LoRA method.

## 12. Corrected release workflow

The current release includes corrected baseline graphs, strict checkpoint
resumption, partial gradient-accumulation handling, **85 automated tests**,
method-compatible hyperparameter sweeps and a controlled TRSO ablation suite.

```bash
# All tests
./scripts/test_all.sh

# Baseline fidelity tests only
./scripts/test_all_baselines.sh

# Plan an ablation suite; add --execute to run
python -m tools.run_ablation_suite \
  --dataset fake --weights none --device cpu --epochs 1 --seeds 0 --max_runs 3

# Plan a baseline hyperparameter sweep
python -m tools.run_hparam_sweep \
  --method lora --dataset fake --weights none --device cpu --epochs 1 --seeds 0 --max_runs 3

# Aggregate completed runs
python -m tools.aggregate_revision_results \
  --root outputs_ablation --out_csv experiments/ablation_results.csv
```

See `CORRECTIONS_AND_EXPERIMENTS.md` for the full correction log, ablation
rationale, sweep spaces and fair-comparison checklist.

