# TRSO-v3 Universal Adaptive Method

## Purpose

TRSO-v3 is the default proposal method in this release. It is designed to keep
TRSO parameter-efficient while removing assumptions tied to one dataset, loss
scale, feature magnitude, image shape, token layout, or backbone family.

The original methods remain available for controlled ablation:

```text
--trso_variant v1   original full channel-by-rank TRSO
--trso_variant v2   grouped task-locked low-parameter TRSO
--trso_variant v3   universal adaptive TRSO (default)
```

TRSO-v3 does not promise that one fixed configuration is optimal on every
benchmark. Instead, it makes calibration and residual strength comparable
across tasks and architectures, resolves safe defaults from the active model
and data loader, and records the resolved choices for reproducibility.

## 1. Universal task-response calibration

At candidate layer \(\ell\), let the calibration probe parameters be collected
in \(p_\ell\). A mini-batch produces task loss \(L_b\) and raw probe gradient

\[
 g_{\ell,b}=\nabla_{p_\ell}L_b.
\]

Loss units differ substantially between cross-entropy, binary
cross-entropy, MAE, MSE, and multi-output regression. V3 therefore normalizes
each batch response before accumulation. With the default RMS rule,

\[
 \widehat g_{\ell,b}
 =
 \frac{g_{\ell,b}}
 {\sqrt{\operatorname{mean}(g_{\ell,b}^{2})}+\varepsilon}.
\]

The calibrated response is

\[
 \overline g_\ell=\frac{1}{M}\sum_{b=1}^{M}\widehat g_{\ell,b}.
\]

Consequently, multiplying a task loss by any positive constant does not change
the recovered response direction apart from numerical precision. The release
preflight verifies this property directly.

Supported calibration normalization modes are:

```text
none | unit | rms | auto
```

For V3, `auto` resolves to `rms`.

## 2. Spatial and channel response components

For CNN features \(X\in\mathbb{R}^{B\times C\times H\times W}\), V3 uses two
calibrated tangent families:

\[
 \Delta_{\mathrm{sp}}(X)
 =
 \operatorname{DWConv}(\operatorname{RMSNorm}(X);W),
\]

\[
 \Delta_{\mathrm{ch}}(X)
 = X\odot s.
\]

The depthwise kernel bank remains rank constrained after flattening the spatial
kernel dimensions:

\[
 W^{\mathrm{flat}}=AB,
 \qquad
 \operatorname{rank}(W^{\mathrm{flat}})\le r.
\]

The spatial atoms and channel response directions are derived from calibration.
Only small grouped amplitudes, a bounded gate, and optional prefix-coupling
scalars are trained.

## 3. Response-derived channel grouping

Fixed contiguous channel groups assume that neighboring channel indices have a
meaningful relationship. That assumption is not general across CNNs,
Transformers, learned channel permutations, or custom backbones.

V3 forms a response signature for each channel and computes a deterministic
low-dimensional ordering using SVD/PCA. Channels are assigned to balanced
quantile groups according to this response ordering. Thus channels are grouped
by measured task behavior rather than index position.

The automatic group count is width-aware, approximately square-root in channel
width, rounded to a power of two, and capped to retain a small parameter count.
It can be overridden explicitly.

```text
--trso_channel_groups 0
--trso_grouping_mode auto
```

For V3, these settings select response grouping with an automatically resolved
group count.

## 4. Scale-controlled residual adaptation

A fixed gate can be too weak for one layer and destabilizing for another because
feature RMS varies by architecture, depth, task, and normalization convention.
V3 normalizes the combined adapter residual per sample:

\[
 \Delta_\ell=\Delta_{\mathrm{sp},\ell}+\Delta_{\mathrm{ch},\ell},
\]

\[
 \widetilde\Delta_\ell
 =
 \Delta_\ell
 \frac{	au\,\operatorname{RMS}(X_\ell)}
 {\operatorname{RMS}(\Delta_\ell)+\varepsilon},
\]

where \(\tau\) is the target update-to-feature RMS ratio. The default is

```text
--trso_residual_target 0.05
```

The scaling factor is bounded by `--trso_residual_scale_limit`. This preserves
the calibrated direction while making initial update magnitude comparable
across feature scales. The preflight verifies the same approximately 0.05 ratio
when input scale changes by four orders of magnitude.

## 5. General token layouts and prefix coupling

V3 supports:

- `B x C x H x W` CNN tensors;
- `B x H x W x C` window/Swin tensors;
- `B x N x C` token tensors;
- square and rectangular patch grids;
- zero, one, or multiple prefix tokens when grid metadata is known.

For token models, the spatial response is reshaped over the patch grid. Prefix
coupling can update the first prefix, every prefix, or their mean-coupled
representation:

```text
--trso_prefix_coupling_mode first | all | mean | auto
```

V3 `auto` resolves to `all`, with energy-preserving normalization across prefix
tokens. This avoids the previous special case in which only one class token was
assumed.

## 6. Automatic calibration size and parameter budget

When the user specifies zero, V3 resolves calibration and budget from the actual
experiment:

```text
--trso_calibration_batches 0
--trso_parameter_budget 0
```

Calibration batches depend on data-loader length and task type. Multi-label and
regression receive more calibration evidence because their gradient estimates
are often noisier.

The automatic adapter budget is

\[
 B=\operatorname{clip}
 \left(
 \rho P_{\mathrm{model}},
 B_{\min},
 B_{\max}
 \right),
\]

then clipped to the actual candidate-layer capacity. Defaults are:

```text
rho  = 5e-5
min  = 128
max  = 1024
```

This keeps the method low-parameter across small and large backbones instead of
using a dataset-specific fixed budget.

## 7. Stable utility per parameter

V3 records response energy and stability across calibration batches. Its default
allocation score is stable response energy per trainable parameter:

\[
 U_\ell(r)
 =
 \frac{E_\ell(r)\,S_\ell(r)}{C_\ell(r)},
\]

where \(E_\ell(r)\) is captured singular energy, \(S_\ell(r)\) is empirical
response stability, and \(C_\ell(r)\) is trainable cost. Exact sparse dynamic
programming allocates layer ranks under the resolved global budget.

Available V3-oriented score modes include:

```text
stable_energy
stable_energy_per_param
stability
stability_per_param
```

## 8. Generic backbone insertion with safe precedence

Named architecture contracts remain preferred because they provide the most
scientifically interpretable insertion points. If no named contract matches,
V3 uses conservative fallbacks:

1. generic pre-normalization token blocks with identifiable attention/block
   structure;
2. uniformly sampled Conv2d outputs for generic CNNs.

The number of generic candidates is capped. Every resolved insertion point is
written to the calibration/configuration artifacts. Paper-specific baselines
remain restricted to their original architecture domains; only TRSO uses the
generic fallback.

## 9. Parameter count

For rank \(r\), \(G\) response groups, channel response enabled, and prefix
coupling enabled, the typical V3 layer cost is approximately

\[
 Gr+G+1+I_{\mathrm{prefix}}.
\]

With \(G=8\) and \(r=2\), this is about 25 parameters for a CNN layer and 26 for
a token layer. The exact count is recorded per run.

## 10. Recommended universal command

```bash
python main.py \
  --dataset dtd --task auto --data_path ./data --download True \
  --backbone resnet50 --model_source torchvision --pretrained True \
  --tuning_method trso --trso_variant v3 \
  --trso_parameter_budget 0 --trso_calibration_batches 0 \
  --trso_channel_groups 0 --trso_grouping_mode auto \
  --trso_calibration_grad_norm auto --trso_residual_norm auto \
  --trso_score_mode stable_energy_per_param \
  --trso_prefix_coupling_mode auto --input_size 0
```

The same settings can be used for single-label, multi-label, and regression
routes. Dataset/task/backbone-specific values are resolved and saved rather
than hard-coded in the method.

## 11. Evidence and limits

Verified in this release:

- loss-scale invariant calibration;
- feature-scale controlled residual magnitude;
- response-derived balanced grouping;
- rectangular grids and multiple prefix tokens;
- named and generic CNN/Transformer insertion paths;
- full single-label, multi-label, and regression execution;
- strict checkpoint restoration and task-specific metrics;
- fair-suite protocol verification.

These are structural, numerical, and execution guarantees. They are not a
claim that TRSO-v3 achieves the highest accuracy on every dataset. Real
performance must be established with controlled multi-seed experiments on the
chosen datasets and compatible backbones.
