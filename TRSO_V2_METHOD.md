# TRSO-v2: Low-Parameter Task-Locked Response Adaptation

## Purpose

TRSO-v2 addresses the failure mode observed in the original short DTD run:

- the task head and adapters were under-trained;
- the residual perturbation started at approximately `0.05 x 0.01 = 0.0005`;
- the rank budget activated nearly every candidate layer;
- the full per-channel coefficient matrix consumed most adapter parameters;
- post-block ViT spatial updates could fail to influence the class token.

The original formulation remains available as `--trso_variant v1`. The default CLI proposal is now `v2`.

TRSO-v2 is not an unrelated engineering stack. It retains the scientific core:

1. measure task gradients in the same operator family used during adaptation;
2. obtain shared spatial atoms by truncated SVD;
3. allocate rank by an exact budgeted dynamic program.

V2 changes the **parameterization and optimization geometry** so that a small budget has a visible, task-aligned effect.

## 1. Task-aware starting point

For each backbone and seed, the fair runner first trains a linear task head and selects its best validation checkpoint. Compatible PEFT methods can then start from the same head:

```text
pretrained backbone -> best linear head -> adapter calibration -> adapter training
```

With `--peft_freeze_head True`, the comparison isolates the adapter because the same task-aware classifier is frozen for every compatible PEFT method.

This avoids measuring TRSO responses through a random classifier and prevents the classifier from dominating the trainable-parameter count.

## 2. Spatial response basis

At layer `l`, calibration introduces a zero depthwise probe

```text
P_l in R^(C_l x k x k)
```

and measures

```text
G_l = E_batch[dL/dP_l] at P_l = 0.
```

After flattening `G_l` to `C_l x k^2`, truncated SVD gives

```text
-G_l = U_l Sigma_l V_l^T.
```

The first `r_l` rows of `V_l^T` are fixed task-derived spatial atoms. The corresponding channel directions from `U_l Sigma_l` are stored as non-trainable response directions.

## 3. Grouped task-locked coefficients

V1 learns a complete coefficient matrix

```text
A_l in R^(C_l x r_l),
```

with cost `C_l r_l`.

V2 locks the calibrated channel directions and learns only one amplitude per channel group and rank component:

```text
A_l = D_l .* Gamma_l[group(c), :],
```

where:

- `D_l in R^(C_l x r_l)` is the fixed task-response direction;
- `Gamma_l in R^(G_l x r_l)` is trainable;
- `G_l` is a small number of channel groups, eight by default.

The resulting kernel bank remains

```text
W_l^flat = A_l B_l,
```

and still satisfies

```text
rank(W_l^flat) <= r_l.
```

### Parameter cost

For a fixed spatial basis, V2 uses approximately

```text
G_l r_l + G_l + 1
```

trainable parameters per CNN layer:

- `G_l r_l` grouped spatial amplitudes;
- `G_l` channel-response gains;
- one residual gate.

For ViT layers with class-token coupling, add one prefix gate:

```text
G_l r_l + G_l + 2.
```

With `G_l = 8` and `r_l = 2`:

```text
CNN layer:  8*2 + 8 + 1 = 25 parameters
ViT layer:  8*2 + 8 + 2 = 26 parameters
```

By comparison, V1 costs `C_l r_l + 1`. At `C_l = 64` and `r_l = 2`, that is 129 parameters.

## 4. Complementary channel-response tangent

A spatial depthwise perturbation alone may be too weak when the pretrained representation already contains useful task information. V2 therefore calibrates a second, multiplicative tangent:

```text
Delta_channel(X_l) = X_l .* s_l,
```

where the full calibration probe `s_l in R^(C_l)` is reduced after calibration to a fixed response direction plus `G_l` trainable group gains.

The V2 update is

```text
Y_l = X_l + g_l [
    DWConv(Norm(X_l); W_l)
    + X_l .* s_l
].
```

This adds channel reweighting without introducing a dense channel bottleneck.

## 5. RMS-normalized spatial response

V2 optionally uses per-sample, per-channel spatial RMS normalization:

```text
Norm(X_l) = X_l / sqrt(mean_hw(X_l^2) + epsilon).
```

This reduces the tendency of high-energy layers to dominate calibration and makes one shared gate scale more meaningful across stages.

The multiplicative channel response is applied to the unnormalized feature, preserving its semantic scale.

## 6. Direct class-token coupling

For ViT tokens, patch responses are reshaped into a spatial grid. V2 additionally pools the patch delta and updates the first prefix token:

```text
cls_l' = cls_l + q_l * mean_patches(Delta_l).
```

Only one scalar `q_l` is learned per active layer. This guarantees that the final candidate layer can influence the classifier even when no later attention block remains to transfer patch information into the class token.

V1 preserves prefix tokens exactly; V2 intentionally adds this low-parameter coupling.

## 7. Strong but bounded initialization

V1 used a small basis scale and a gate initialized near `0.01`, resulting in a negligible initial perturbation.

V2 defaults to:

```text
spatial/channel response directions: calibrated from the task
gate initialization: 1.0
gate clamp: [-2, 2]
```

An optional training-only gate search evaluates a small set of global trust-region scales on calibration batches:

```text
--trso_gate_search True
--trso_gate_search_values 0.25,0.5,1.0,1.5
```

No validation or test examples are used by this search.

## 8. Noise- and cost-aware allocation

V2 supports the score

```text
score_l(r) = captured_energy_l(r)
             / response_noise_l(r)
             / parameter_cost_l(r).
```

Use:

```text
--trso_score_mode snr_per_param
```

This favors stable response components that provide more task signal per trainable parameter. Exact dynamic programming then selects layer ranks under the global budget.

The default V2 fair-suite budget is 512 adapter parameters rather than the old 12,000/2,000-style budget. Actual active cost is recorded layer by layer.

## 9. Recommended controlled protocol

For a small image dataset:

```text
native pretrained input resolution
best linear head prepared per backbone and seed
same frozen head for compatible PEFT methods
AdamW for every PEFT method
same PEFT learning rate and cosine schedule
20-50 epochs depending on dataset size
5 warm-up epochs
three or five seeds
best-validation checkpoint -> one final test evaluation
```

Suggested V2 command:

```bash
python -m tools.run_fair_suite \
  --dataset dtd --task auto --data_path ./data --download True \
  --backbones resnet50@torchvision,vit_tiny_patch16_224@timm \
  --methods auto --seeds 0,1,2 --epochs 30 \
  --input_size 0 --peft_lr 5e-3 --full_lr 1e-4 --linear_lr 1e-1 \
  --warmup_epochs 5 --trso_budget 512 \
  --trso_calibration_batches 16 --peft_freeze_head True --execute
```

Focused V2 search:

```bash
python -m tools.run_trso_v2_search \
  --dataset dtd --data_path ./data --download True \
  --backbone resnet50 --model_source torchvision \
  --seeds 0,1,2 --epochs 30 --peft_lr 5e-3 --execute
```

## 10. Required ablations

The release includes these scientifically interpretable controls:

- V1 versus V2;
- response, random and DCT spatial bases;
- full, locked and grouped coefficients;
- channel-response branch on/off;
- RMS normalization on/off;
- class-token coupling on/off;
- group count `1, 4, 8, 16`;
- exact, greedy and uniform allocation;
- energy versus SNR-per-parameter scoring;
- rank, kernel, calibration-batch and budget sweeps.

## 11. Verified evidence and current limitation

Verified locally:

- all automated tests pass;
- V2 uses substantially fewer adapter parameters than V1;
- class-token coupling receives and propagates classification gradients;
- V2 avoids the old `0.01` gate suppression;
- a controlled synthetic task showed lower initial approximation error than V1 while using fewer parameters;
- a CNN end-to-end fake-data smoke run completed calibration, allocation, training, strict best-checkpoint restoration and final evaluation.

The complete DTD three-seed benchmark has **not** been rerun in this environment. Therefore, this release removes known failure modes and supplies a stronger hypothesis, but does not guarantee a particular real-data accuracy.
