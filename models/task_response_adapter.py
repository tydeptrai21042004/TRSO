"""Scientific Task-Response Spatial Operator (TRSO).

The adapter is deliberately small and mathematically aligned with calibration.
For a feature tensor X with C channels, TRSO learns a channel-specific depthwise
kernel bank W in a low-dimensional spatial subspace:

    Y = X + g * DWConv(X; W),
    W_flat = A B,

where A has shape C x r and B has shape r x k^2.  Hence the matrix obtained by
flattening every channel kernel has rank at most r.  During calibration, a full
zero kernel bank is exposed.  The task-loss gradient G with respect to that bank
is projected by truncated SVD.  By Eckart--Young--Mirsky, this is the exact
solution of the local rank-constrained proximal problem

    min_rank(W_flat)<=r <G, W> + lambda/2 ||W||_F^2.

The same depthwise operator is used in calibration and training; there is no
random bottleneck or nonlinear surrogate between the response measurement and
the learned operator.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SpatialLayout:
    kind: str
    grid_size: Tuple[int, int]
    prefix_tokens: int = 0


def _as_pair(value: Optional[Sequence[int] | int]) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    if isinstance(value, int):
        return (int(value), int(value))
    if len(value) != 2:
        raise ValueError(f"grid_size must contain two integers, got {value!r}")
    return (int(value[0]), int(value[1]))


def _infer_token_grid(n_tokens: int, requested: Optional[Tuple[int, int]]) -> Tuple[Tuple[int, int], int]:
    if requested is not None:
        h, w = requested
        if h <= 0 or w <= 0:
            raise ValueError(f"grid_size must be positive, got {requested!r}")
        patches = h * w
        prefix = n_tokens - patches
        if prefix < 0:
            raise ValueError(
                f"Token count {n_tokens} is smaller than requested grid {h}x{w}={patches}."
            )
        return (h, w), prefix

    side = int(math.isqrt(n_tokens))
    if side * side == n_tokens:
        return (side, side), 0
    for prefix in (1, 2):
        if n_tokens > prefix:
            patch_tokens = n_tokens - prefix
            side = int(math.isqrt(patch_tokens))
            if side * side == patch_tokens:
                return (side, side), prefix
    raise ValueError(
        "Cannot infer a square patch grid from token count "
        f"{n_tokens}. Pass grid_size=(height, width) explicitly."
    )


def tensor_to_bchw(
    x: torch.Tensor,
    channels: int,
    layout: str = "auto",
    grid_size: Optional[Sequence[int] | int] = None,
) -> Tuple[torch.Tensor, SpatialLayout, Optional[torch.Tensor]]:
    """Convert BCHW, BHWC, or BNC tensors to BCHW.

    Prefix tokens are excluded from spatial filtering and restored unchanged.
    """
    if x.ndim not in (3, 4):
        raise ValueError(f"TRSO expects a 3-D or 4-D tensor, got shape {tuple(x.shape)}")

    requested = _as_pair(grid_size)
    kind = layout.lower()
    if kind not in {"auto", "bchw", "bhwc", "bnc"}:
        raise ValueError(f"Unsupported layout {layout!r}; use auto, bchw, bhwc, or bnc.")

    if x.ndim == 4:
        if kind == "bchw" or (kind == "auto" and x.shape[1] == channels):
            return x, SpatialLayout("bchw", (x.shape[2], x.shape[3])), None
        if kind == "bhwc" or (kind == "auto" and x.shape[-1] == channels):
            return (
                x.permute(0, 3, 1, 2).contiguous(),
                SpatialLayout("bhwc", (x.shape[1], x.shape[2])),
                None,
            )
        raise ValueError(
            f"Cannot determine 4-D layout for shape {tuple(x.shape)} and channels={channels}."
        )

    if kind not in {"auto", "bnc"}:
        raise ValueError(f"A 3-D tensor requires layout='bnc' or 'auto', got {layout!r}.")
    if x.shape[-1] != channels:
        raise ValueError(
            f"Expected token embedding dimension {channels}, got shape {tuple(x.shape)}."
        )
    (h, w), prefix_count = _infer_token_grid(x.shape[1], requested)
    prefix = x[:, :prefix_count] if prefix_count else None
    patches = x[:, prefix_count:]
    spatial = patches.transpose(1, 2).reshape(x.shape[0], channels, h, w).contiguous()
    return spatial, SpatialLayout("bnc", (h, w), prefix_count), prefix


def bchw_to_tensor(x: torch.Tensor, meta: SpatialLayout, prefix: Optional[torch.Tensor]) -> torch.Tensor:
    if meta.kind == "bchw":
        return x
    if meta.kind == "bhwc":
        return x.permute(0, 2, 3, 1).contiguous()
    if meta.kind == "bnc":
        patches = x.flatten(2).transpose(1, 2).contiguous()
        return torch.cat((prefix, patches), dim=1) if prefix is not None else patches
    raise RuntimeError(f"Unknown stored layout {meta.kind!r}")


def _dct_fallback_basis(kernel_size: int, rank: int) -> torch.Tensor:
    """Deterministic orthonormal 2-D DCT atoms used only without calibration."""
    coords = torch.arange(kernel_size, dtype=torch.float32)
    vectors: List[torch.Tensor] = []
    for frequency in range(kernel_size):
        v = torch.cos(math.pi * (2.0 * coords + 1.0) * frequency / (2.0 * kernel_size))
        v = v / v.norm().clamp_min(1e-8)
        vectors.append(v)
    atoms: List[torch.Tensor] = []
    # Low frequencies first, ordered by total frequency.
    for total in range(2 * kernel_size - 1):
        for p in range(kernel_size):
            q = total - p
            if 0 <= q < kernel_size:
                atoms.append(torch.outer(vectors[p], vectors[q]))
                if len(atoms) == rank:
                    return torch.stack(atoms, dim=0)
    raise RuntimeError("Unable to construct enough fallback atoms.")


def _auto_group_count(channels: int) -> int:
    """Resolve a small, architecture-scale-aware number of channel groups.

    A fixed group count is unnecessarily rigid: eight groups is reasonable for
    a 64-channel ResNet block but too coarse for a 768-dimensional Transformer
    and excessive for a tiny mobile block.  The square-root rule grows slowly,
    is clipped to a small PEFT-friendly range, and uses a power of two so groups
    remain balanced for common backbone widths.
    """
    channels = max(1, int(channels))
    target = max(1.0, math.sqrt(float(channels)))
    power = int(round(math.log2(target))) if target > 1.0 else 0
    return max(1, min(channels, 16, 2 ** power))


class TaskResponseSpatialAdapter(nn.Module):
    """Task-derived low-rank channel--spatial operator.

    The rank constraint is on the matrix ``W_flat in R^{C x k^2}``, whose rows
    are the flattened depthwise kernels of individual channels.  Thus rank ``r``
    means all channel kernels are mixtures of ``r`` shared spatial atoms.

    ``channel_ratio`` is accepted only for compatibility with earlier command
    lines; the scientific TRSO formulation has no channel bottleneck.
    """

    def __init__(
        self,
        channels: Optional[int] = None,
        *,
        C: Optional[int] = None,
        kernel_size: int = 5,
        spatial_rank: int = 2,
        channel_ratio: int = 16,
        operator_radius: float = 1.0,
        gate_init: float = 1e-2,
        basis_init_scale: float = 5e-2,
        calibration_scale: float = 1.0,
        layout: str = "auto",
        grid_size: Optional[Sequence[int] | int] = None,
        basis_trainable: bool = False,
        enabled: bool = True,
        variant: str = "v1",
        coefficient_mode: str = "auto",
        channel_groups: int = 8,
        grouping_mode: str = "auto",
        input_norm: str = "auto",
        calibration_grad_norm: str = "auto",
        residual_norm: str = "auto",
        residual_target: float = 5e-2,
        residual_scale_limit: float = 1024.0,
        channel_response: Optional[bool] = None,
        channel_init_scale: float = 1e-2,
        prefix_coupling: Optional[bool] = None,
        prefix_coupling_mode: str = "auto",
        prefix_gate_init: float = 1.0,
        gate_limit: float = 2.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if channels is None:
            channels = C
        if channels is None:
            raise ValueError("channels (or legacy alias C) must be provided.")
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        max_rank = min(int(channels), int(kernel_size) * int(kernel_size))
        if spatial_rank <= 0 or spatial_rank > max_rank:
            raise ValueError(
                "spatial_rank must satisfy 1 <= rank <= min(channels, kernel_size^2); "
                f"got rank={spatial_rank}, channels={channels}, k={kernel_size}"
            )
        if operator_radius <= 0:
            raise ValueError(f"operator_radius must be positive, got {operator_radius}")
        if calibration_scale <= 0:
            raise ValueError(f"calibration_scale must be positive, got {calibration_scale}")

        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.spatial_rank = int(spatial_rank)
        self.channel_ratio = int(channel_ratio)  # legacy metadata; not used by the operator.
        self.operator_radius = float(operator_radius)
        self.calibration_scale = float(calibration_scale)
        self.layout = str(layout)
        self.grid_size = _as_pair(grid_size)
        self.basis_trainable = bool(basis_trainable)
        self.eps = float(eps)
        self.basis_init_scale = float(basis_init_scale)
        self.variant = str(variant).lower()
        if self.variant not in {"v1", "v2", "v3"}:
            raise ValueError("variant must be 'v1', 'v2', or 'v3'")
        if coefficient_mode == "auto":
            coefficient_mode = "full" if self.variant == "v1" else "grouped"
        self.coefficient_mode = str(coefficient_mode).lower()
        if self.coefficient_mode not in {"full", "locked", "grouped"}:
            raise ValueError("coefficient_mode must be full, locked, grouped, or auto")
        if channel_groups < 0:
            raise ValueError("channel_groups must be non-negative; zero selects an automatic value")
        self.channel_groups = (
            _auto_group_count(self.channels)
            if int(channel_groups) == 0
            else min(int(channel_groups), self.channels)
        )
        if grouping_mode == "auto":
            grouping_mode = "response" if self.variant == "v3" else "contiguous"
        self.grouping_mode = str(grouping_mode).lower()
        if self.grouping_mode not in {"contiguous", "response"}:
            raise ValueError("grouping_mode must be contiguous, response, or auto")
        if input_norm == "auto":
            input_norm = "none" if self.variant == "v1" else "rms"
        self.input_norm = str(input_norm).lower()
        if self.input_norm not in {"none", "rms"}:
            raise ValueError("input_norm must be none, rms, or auto")
        if calibration_grad_norm == "auto":
            # V3 uses one shared normalization factor across every candidate
            # adapter.  Unlike independent per-layer RMS normalization, this
            # preserves the relative task-response magnitude between layers,
            # which is essential for meaningful layer allocation.
            calibration_grad_norm = "global_rms" if self.variant == "v3" else "none"
        self.calibration_grad_norm = str(calibration_grad_norm).lower()
        if self.calibration_grad_norm not in {"none", "unit", "rms", "global_rms"}:
            raise ValueError(
                "calibration_grad_norm must be none, unit, rms, global_rms, or auto"
            )
        if residual_norm == "auto":
            residual_norm = "rms" if self.variant == "v3" else "none"
        self.residual_norm = str(residual_norm).lower()
        if self.residual_norm not in {"none", "rms"}:
            raise ValueError("residual_norm must be none, rms, or auto")
        self.residual_target = float(residual_target)
        self.residual_scale_limit = float(residual_scale_limit)
        if self.residual_target <= 0:
            raise ValueError("residual_target must be positive")
        if self.residual_scale_limit <= 0:
            raise ValueError("residual_scale_limit must be positive")
        self.channel_response = bool(self.variant == "v2" if channel_response is None else channel_response)
        if channel_response is None and self.variant == "v3":
            self.channel_response = True
        self.channel_init_scale = float(channel_init_scale)
        self.prefix_coupling = bool(
            (self.variant in {"v2", "v3"} and self.layout == "bnc")
            if prefix_coupling is None else prefix_coupling
        )
        if prefix_coupling_mode == "auto":
            prefix_coupling_mode = "all" if self.variant == "v3" else "first"
        self.prefix_coupling_mode = str(prefix_coupling_mode).lower()
        if self.prefix_coupling_mode not in {"first", "all", "mean"}:
            raise ValueError("prefix_coupling_mode must be first, all, mean, or auto")
        self.gate_limit = float(gate_limit)
        if self.gate_limit <= 0:
            raise ValueError("gate_limit must be positive")

        fallback = _dct_fallback_basis(self.kernel_size, self.spatial_rank)
        if self.basis_trainable:
            self.basis_atom_parameters = nn.ParameterList(
                [nn.Parameter(atom.clone()) for atom in fallback]
            )
            self.register_buffer("_basis_atoms", torch.empty(0))
        else:
            self.basis_atom_parameters = nn.ParameterList()
            self.register_buffer("_basis_atoms", fallback)

        initial = float(basis_init_scale) / math.sqrt(max(1, self.channels * self.spatial_rank))
        self.coefficient_vectors = nn.ParameterList(
            [nn.Parameter(torch.full((self.channels,), initial)) for _ in range(self.spatial_rank)]
        )
        # V2 freezes the calibrated channel directions and learns only a handful
        # of layer/group amplitudes. This is analogous to vector scaling over a
        # fixed low-rank basis, but the basis is task-derived rather than random.
        self.register_buffer(
            "response_coefficient_directions",
            torch.full((self.channels, self.spatial_rank), initial),
        )
        self.component_amplitudes = nn.Parameter(torch.ones(self.spatial_rank))
        self.group_amplitudes = nn.Parameter(torch.ones(self.channel_groups, self.spatial_rank))
        group_index = torch.div(
            torch.arange(self.channels) * self.channel_groups,
            self.channels,
            rounding_mode="floor",
        ).clamp_max(self.channel_groups - 1)
        self.register_buffer("channel_group_index", group_index.long())

        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.prefix_gate = nn.Parameter(torch.tensor(float(prefix_gate_init)))
        self.channel_group_gains = nn.Parameter(
            torch.full((self.channel_groups,), float(channel_init_scale))
        )
        self.register_buffer("channel_response_direction", torch.zeros(self.channels))

        # A full channel-specific spatial probe and a multiplicative channel probe
        # measure two complementary tangent directions of the exact V2 operator.
        self.probe_kernel = nn.Parameter(
            torch.zeros(self.channels, self.kernel_size, self.kernel_size),
            requires_grad=False,
        )
        self.channel_probe = nn.Parameter(torch.zeros(self.channels), requires_grad=False)
        self.register_buffer("gradient_sum", torch.zeros_like(self.probe_kernel))
        self.register_buffer("gradient_square_sum", torch.zeros_like(self.probe_kernel))
        self.register_buffer("channel_gradient_sum", torch.zeros_like(self.channel_probe))
        self.register_buffer("channel_gradient_square_sum", torch.zeros_like(self.channel_probe))
        self.register_buffer("gradient_samples", torch.tensor(0, dtype=torch.long))
        self.register_buffer("singular_values", torch.zeros(self.spatial_rank))
        self.register_buffer("component_noise", torch.zeros(self.spatial_rank))
        self.register_buffer("response_score", torch.tensor(0.0))
        self.register_buffer("response_noise", torch.tensor(0.0))
        self.register_buffer("channel_response_score", torch.tensor(0.0))
        self.register_buffer("channel_response_noise", torch.tensor(0.0))
        # Scale-invariant calibration diagnostics are non-persistent so older V1/V2
        # checkpoints remain strict-load compatible. They are recomputed whenever
        # calibration is run.
        self.register_buffer("response_stability", torch.tensor(0.0), persistent=False)
        self.register_buffer("channel_response_stability", torch.tensor(0.0), persistent=False)
        self.register_buffer("component_stability", torch.zeros(self.spatial_rank), persistent=False)
        self.register_buffer("raw_gradient_norm_sum", torch.tensor(0.0), persistent=False)
        self.register_buffer("raw_gradient_norm_square_sum", torch.tensor(0.0), persistent=False)
        self.register_buffer("input_energy_sum", torch.tensor(0.0))
        self.register_buffer("input_energy_samples", torch.tensor(0, dtype=torch.long))
        self.register_buffer("enabled_flag", torch.tensor(bool(enabled), dtype=torch.bool))
        self.register_buffer("active_rank_buffer", torch.tensor(self.spatial_rank, dtype=torch.long))
        self.register_buffer("calibration_flag", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("calibrated_flag", torch.tensor(False, dtype=torch.bool))

        self.is_trso_adapter = True
        self.set_enabled(enabled)
        self.register_load_state_dict_post_hook(
            lambda module, incompatible_keys: module.sync_trainability_from_state()
        )

    @property
    def enabled(self) -> bool:
        return bool(self.enabled_flag.item())

    @property
    def calibrating(self) -> bool:
        return bool(self.calibration_flag.item())

    @property
    def calibrated(self) -> bool:
        return bool(self.calibrated_flag.item())

    @property
    def active_rank(self) -> int:
        return int(self.active_rank_buffer.item())

    @property
    def basis_atoms(self) -> torch.Tensor:
        if self.basis_trainable:
            return torch.stack(list(self.basis_atom_parameters), dim=0)
        return self._basis_atoms

    @property
    def coefficients(self) -> torch.Tensor:
        """Return the effective C x R coefficient matrix."""
        if self.coefficient_mode == "full":
            return torch.stack(list(self.coefficient_vectors), dim=1)
        base = self.response_coefficient_directions
        if self.coefficient_mode == "locked":
            return base * self.component_amplitudes.view(1, -1)
        gains = self.group_amplitudes[self.channel_group_index]
        return base * gains

    def _copy_basis(self, atoms: torch.Tensor) -> None:
        if tuple(atoms.shape) != (self.spatial_rank, self.kernel_size, self.kernel_size):
            raise ValueError(f"Unexpected basis shape {tuple(atoms.shape)}")
        if self.basis_trainable:
            for parameter, atom in zip(self.basis_atom_parameters, atoms):
                parameter.data.copy_(atom)
        else:
            self._basis_atoms.copy_(atoms)

    def _copy_coefficients(self, coefficients: torch.Tensor) -> None:
        if tuple(coefficients.shape) != (self.channels, self.spatial_rank):
            raise ValueError(f"Unexpected coefficient shape {tuple(coefficients.shape)}")
        self.response_coefficient_directions.copy_(coefficients)
        for index, parameter in enumerate(self.coefficient_vectors):
            parameter.data.copy_(coefficients[:, index])
        self.component_amplitudes.data.fill_(1.0)
        self.group_amplitudes.data.fill_(1.0)

    @torch.no_grad()
    def _update_response_groups(
        self,
        coefficients: torch.Tensor,
        channel_response: Optional[torch.Tensor] = None,
    ) -> None:
        """Build deterministic balanced channel groups from calibration response.

        Contiguous channel IDs have no semantic meaning across most CNN and
        Transformer backbones. V3 instead clusters channels by a one-dimensional
        principal response score, then assigns balanced quantile groups. This
        retains a tiny number of trainable amplitudes while grouping channels
        that react similarly to the downstream task.
        """
        if self.grouping_mode != "response" or self.channel_groups <= 1:
            return
        features = coefficients.detach()
        if channel_response is not None and channel_response.numel() == self.channels:
            features = torch.cat((features, channel_response.detach().view(-1, 1)), dim=1)
        if features.numel() == 0:
            return
        features = features - features.mean(dim=0, keepdim=True)
        column_rms = features.square().mean(dim=0, keepdim=True).sqrt().clamp_min(self.eps)
        features = features / column_rms
        if features.shape[1] == 1:
            scores = features[:, 0]
        else:
            try:
                u, s, _ = torch.linalg.svd(features, full_matrices=False)
                scores = u[:, 0] * s[0]
            except RuntimeError:
                # Deterministic fallback for rare low-precision SVD failures.
                weights = torch.linspace(1.0, 2.0, features.shape[1], device=features.device, dtype=features.dtype)
                scores = features @ weights
        order = torch.argsort(scores, stable=True)
        group_index = torch.empty(self.channels, dtype=torch.long, device=order.device)
        positions = torch.arange(self.channels, device=order.device)
        groups = torch.div(
            positions * self.channel_groups,
            self.channels,
            rounding_mode="floor",
        ).clamp_max(self.channel_groups - 1)
        group_index[order] = groups
        self.channel_group_index.copy_(group_index.to(self.channel_group_index.device))

    def set_active_rank(self, rank: int) -> None:
        rank = int(rank)
        if not (0 <= rank <= self.spatial_rank):
            raise ValueError(f"active rank must be in [0, {self.spatial_rank}], got {rank}")
        self.active_rank_buffer.fill_(rank)
        self.enabled_flag.fill_(rank > 0)
        enabled = rank > 0
        for index, parameter in enumerate(self.coefficient_vectors):
            parameter.requires_grad_(enabled and self.coefficient_mode == "full" and index < rank)
        self.component_amplitudes.requires_grad_(enabled and self.coefficient_mode == "locked")
        self.group_amplitudes.requires_grad_(enabled and self.coefficient_mode == "grouped")
        if self.basis_trainable:
            for index, parameter in enumerate(self.basis_atom_parameters):
                parameter.requires_grad_(enabled and index < rank)
        self.gate.requires_grad_(enabled)
        self.channel_group_gains.requires_grad_(enabled and self.channel_response)
        self.prefix_gate.requires_grad_(enabled and self.prefix_coupling)

    def set_enabled(self, enabled: bool) -> None:
        if enabled:
            rank = max(1, self.active_rank)
            self.set_active_rank(rank)
        else:
            # Preserve the allocated rank so re-enabling restores it.
            saved_rank = self.active_rank
            self.enabled_flag.fill_(False)
            for parameter in self.coefficient_vectors:
                parameter.requires_grad_(False)
            for parameter in self.basis_atom_parameters:
                parameter.requires_grad_(False)
            self.component_amplitudes.requires_grad_(False)
            self.group_amplitudes.requires_grad_(False)
            self.channel_group_gains.requires_grad_(False)
            self.prefix_gate.requires_grad_(False)
            self.gate.requires_grad_(False)
            self.active_rank_buffer.fill_(saved_rank)

    def sync_trainability_from_state(self) -> None:
        """Restore ``requires_grad`` from persistent rank/enabled buffers."""
        rank = self.active_rank
        enabled = self.enabled and rank > 0
        for index, parameter in enumerate(self.coefficient_vectors):
            parameter.requires_grad_(enabled and self.coefficient_mode == "full" and index < rank)
        self.component_amplitudes.requires_grad_(enabled and self.coefficient_mode == "locked")
        self.group_amplitudes.requires_grad_(enabled and self.coefficient_mode == "grouped")
        for index, parameter in enumerate(self.basis_atom_parameters):
            parameter.requires_grad_(enabled and index < rank)
        self.gate.requires_grad_(enabled)
        self.channel_group_gains.requires_grad_(enabled and self.channel_response)
        self.prefix_gate.requires_grad_(enabled and self.prefix_coupling)
        self.probe_kernel.requires_grad_(self.calibrating)
        self.channel_probe.requires_grad_(self.calibrating and self.channel_response)

    def start_calibration(self, reset: bool = True) -> None:
        if reset:
            self.gradient_sum.zero_()
            self.gradient_square_sum.zero_()
            self.channel_gradient_sum.zero_()
            self.channel_gradient_square_sum.zero_()
            self.gradient_samples.zero_()
            self.response_stability.zero_()
            self.channel_response_stability.zero_()
            self.component_stability.zero_()
            self.raw_gradient_norm_sum.zero_()
            self.raw_gradient_norm_square_sum.zero_()
            self.input_energy_sum.zero_()
            self.input_energy_samples.zero_()
            self.probe_kernel.data.zero_()
            self.channel_probe.data.zero_()
        self.calibration_flag.fill_(True)
        self.probe_kernel.requires_grad_(True)
        self.channel_probe.requires_grad_(self.channel_response)

    def stop_calibration(self) -> None:
        self.calibration_flag.fill_(False)
        self.probe_kernel.requires_grad_(False)
        self.channel_probe.requires_grad_(False)
        self.probe_kernel.grad = None
        self.channel_probe.grad = None

    @torch.no_grad()
    def accumulate_probe_gradient(
        self,
        *,
        global_squared_norm: Optional[torch.Tensor] = None,
        global_element_count: Optional[int] = None,
    ) -> None:
        grad = self.probe_kernel.grad
        channel_grad = self.channel_probe.grad
        detached = grad.detach() if grad is not None else None
        detached_channel = channel_grad.detach() if channel_grad is not None else None

        # Classification, multi-label, and regression losses can differ by large
        # arbitrary scale factors (number of labels, target units, reduction
        # conventions). V3 normalizes the combined tangent gradient per batch so
        # calibration measures a stable direction rather than the loss units.
        if detached is not None or detached_channel is not None:
            squared_norm = torch.zeros((), device=self.gradient_sum.device, dtype=self.gradient_sum.dtype)
            element_count = 0
            if detached is not None:
                squared_norm = squared_norm + detached.square().sum()
                element_count += detached.numel()
            if detached_channel is not None:
                squared_norm = squared_norm + detached_channel.square().sum()
                element_count += detached_channel.numel()
            raw_norm = squared_norm.sqrt()
            self.raw_gradient_norm_sum.add_(raw_norm)
            self.raw_gradient_norm_square_sum.add_(squared_norm)
            if self.calibration_grad_norm == "unit":
                multiplier = raw_norm.clamp_min(self.eps).reciprocal()
            elif self.calibration_grad_norm == "rms":
                multiplier = math.sqrt(max(1, element_count)) / raw_norm.clamp_min(self.eps)
            elif self.calibration_grad_norm == "global_rms":
                # One multiplier is shared by every layer.  This removes the
                # arbitrary task-loss scale while retaining cross-layer
                # response magnitudes.  The fallback keeps direct unit tests
                # and external one-adapter use backward compatible.
                if global_squared_norm is None:
                    reference_norm = raw_norm
                    reference_count = element_count
                else:
                    reference_norm = global_squared_norm.to(
                        device=raw_norm.device, dtype=raw_norm.dtype
                    ).clamp_min(0.0).sqrt()
                    reference_count = int(global_element_count or element_count)
                multiplier = math.sqrt(max(1, reference_count)) / reference_norm.clamp_min(self.eps)
            else:
                multiplier = torch.ones_like(raw_norm)
            if detached is not None:
                detached = detached * multiplier
            if detached_channel is not None:
                detached_channel = detached_channel * multiplier

        if detached is not None:
            self.gradient_sum.add_(detached)
            self.gradient_square_sum.add_(detached.square())
            self.probe_kernel.grad = None
        if detached_channel is not None:
            self.channel_gradient_sum.add_(detached_channel)
            self.channel_gradient_square_sum.add_(detached_channel.square())
            self.channel_probe.grad = None
        if grad is not None or channel_grad is not None:
            self.gradient_samples.add_(1)

    @torch.no_grad()
    def finalize_calibration(
        self,
        init_scale: float = 5e-2,
        basis_source: str = "response",
        random_seed: int = 0,
    ) -> float:
        """Finalize response statistics and initialize the requested basis.

        ``response`` uses truncated SVD. ``random`` and ``dct`` project the same
        measured response onto controlled orthonormal bases, enabling a clean
        basis ablation without changing the operator or parameter count.
        """
        samples = int(self.gradient_samples.item())
        if samples <= 0:
            raise RuntimeError("No probe gradients were accumulated for this TRSO adapter.")
        basis_source = str(basis_source).lower()
        if basis_source not in {"response", "random", "dct"}:
            raise ValueError("basis_source must be response, random, or dct")

        response = self.gradient_sum / float(samples)
        descending = -response.reshape(self.channels, self.kernel_size * self.kernel_size)
        max_available = min(self.spatial_rank, self.channels, self.kernel_size * self.kernel_size)

        if basis_source == "response":
            u, s, vh = torch.linalg.svd(descending, full_matrices=False)
            atoms_flat = vh[:max_available]
            coefficients_raw = u[:, :max_available] * s[:max_available]
            component_strength = s[:max_available]
        elif basis_source == "dct":
            atoms_flat = _dct_fallback_basis(
                self.kernel_size, self.spatial_rank
            ).to(response).flatten(1)[:max_available]
            coefficients_raw = descending @ atoms_flat.transpose(0, 1)
            component_strength = coefficients_raw.square().sum(dim=0).sqrt()
        else:
            generator = torch.Generator(device="cpu").manual_seed(
                int(random_seed) + 104729 * self.channels + 1009 * self.kernel_size
            )
            random_matrix = torch.randn(
                self.kernel_size * self.kernel_size, max_available, generator=generator
            ).to(device=response.device, dtype=response.dtype)
            atoms_flat = torch.linalg.qr(random_matrix, mode="reduced").Q.transpose(0, 1)
            coefficients_raw = descending @ atoms_flat.transpose(0, 1)
            component_strength = coefficients_raw.square().sum(dim=0).sqrt()

        # Order controlled bases by captured response energy so rank-r is nested.
        order = torch.argsort(component_strength, descending=True)
        atoms_flat = atoms_flat[order]
        coefficients_raw = coefficients_raw[:, order]
        component_strength = component_strength[order]

        atoms = _dct_fallback_basis(self.kernel_size, self.spatial_rank).to(response)
        coefficients = torch.zeros(
            self.channels, self.spatial_rank, dtype=response.dtype, device=response.device
        )
        available = min(max_available, atoms_flat.shape[0])
        if available > 0:
            atoms[:available].copy_(atoms_flat[:available].reshape(available, self.kernel_size, self.kernel_size))
            coefficients[:, :available].copy_(coefficients_raw[:, :available])
            reconstructed = coefficients[:, :available] @ atoms_flat[:available]
            scale = float(init_scale) / reconstructed.norm().clamp_min(self.eps)
            coefficients[:, :available].mul_(scale)

        self._copy_basis(atoms)
        self._copy_coefficients(coefficients)
        self.singular_values.zero_()
        self.singular_values[:available].copy_(component_strength[:available])
        score = component_strength[:available].square().sum()
        self.response_score.copy_(score)

        mean_square = self.gradient_square_sum / float(samples)
        variance = (mean_square - response.square()).clamp_min(0.0)
        self.response_noise.copy_(variance.sum())
        spatial_signal = response.square().sum()
        spatial_second_moment = mean_square.sum().clamp_min(self.eps)
        self.response_stability.copy_((spatial_signal / spatial_second_moment).clamp(0.0, 1.0))
        self.component_noise.zero_()
        self.component_stability.zero_()
        variance_flat = variance.reshape(self.channels, -1)
        for index in range(available):
            direction = coefficients_raw[:, index:index + 1] * atoms_flat[index:index + 1]
            direction = direction / direction.norm().clamp_min(self.eps)
            projected_noise = (variance_flat * direction.square()).sum()
            self.component_noise[index].copy_(projected_noise)
            projected_signal = component_strength[index].square()
            self.component_stability[index].copy_(
                (projected_signal / (projected_signal + projected_noise).clamp_min(self.eps)).clamp(0.0, 1.0)
            )

        if self.channel_response:
            channel_response = -(self.channel_gradient_sum / float(samples))
            channel_mean_square = self.channel_gradient_square_sum / float(samples)
            channel_variance = (channel_mean_square - channel_response.square()).clamp_min(0.0)
            rms = channel_response.square().mean().sqrt().clamp_min(self.eps)
            self.channel_response_direction.copy_(channel_response / rms)
            self.channel_response_score.copy_(channel_response.square().sum())
            self.channel_response_noise.copy_(channel_variance.sum())
            channel_second_moment = channel_mean_square.sum().clamp_min(self.eps)
            self.channel_response_stability.copy_(
                (channel_response.square().sum() / channel_second_moment).clamp(0.0, 1.0)
            )
            self.channel_group_gains.data.fill_(self.channel_init_scale)
        else:
            self.channel_response_direction.zero_()
            self.channel_response_score.zero_()
            self.channel_response_noise.zero_()
            self.channel_response_stability.zero_()
        self._update_response_groups(
            coefficients_raw[:, :available],
            self.channel_response_direction if self.channel_response else None,
        )
        self.calibrated_flag.fill_(True)
        self.set_active_rank(self.spatial_rank)
        self.stop_calibration()
        return float(score.item())

    def rank_value(
        self,
        rank: int,
        score_mode: str = "energy",
        noise_beta: float = 0.0,
    ) -> float:
        rank = max(0, min(int(rank), self.spatial_rank))
        if rank == 0:
            return 0.0
        component_energy = self.singular_values[:rank].square()
        energy = float(component_energy.sum().item() + self.channel_response_score.item())
        stable_energy = float(
            (component_energy * self.component_stability[:rank]).sum().item()
            + self.channel_response_score.item() * self.channel_response_stability.item()
        )
        component_noise = float(
            self.component_noise[:rank].sum().item() + self.channel_response_noise.item()
        )
        mode = str(score_mode).lower()
        cost = max(1, self.parameter_cost_for_rank(rank))
        activation = max(self.eps, float(self.input_energy_sum.item()) / max(1, int(self.input_energy_samples.item())))
        if mode == "energy":
            return energy
        if mode == "energy_per_param":
            return energy / cost
        if mode == "energy_per_channel":
            return energy / max(1, self.channels)
        if mode == "noise_adjusted":
            return max(0.0, energy - float(noise_beta) * component_noise)
        if mode == "snr":
            return energy / max(self.eps, component_noise)
        if mode == "snr_per_param":
            return energy / max(self.eps, component_noise) / cost
        if mode == "normalized_energy_per_param":
            return energy / activation / cost
        if mode == "normalized_stable_energy_per_param":
            return stable_energy / activation / cost
        if mode == "stable_energy":
            return stable_energy
        if mode == "stable_energy_per_param":
            return stable_energy / cost
        if mode == "stability":
            return stable_energy / max(self.eps, energy)
        if mode == "stability_per_param":
            return stable_energy / max(self.eps, energy) / cost
        raise ValueError(
            "score_mode must be energy, energy_per_param, energy_per_channel, "
            "noise_adjusted, snr, snr_per_param, normalized_energy_per_param, "
            "stable_energy, stable_energy_per_param, normalized_stable_energy_per_param, "
            "stability, or stability_per_param"
        )

    def parameter_cost_for_rank(self, rank: int) -> int:
        rank = int(rank)
        if rank <= 0:
            return 0
        if self.coefficient_mode == "full":
            coefficient_cost = self.channels * rank
        elif self.coefficient_mode == "locked":
            coefficient_cost = rank
        else:
            coefficient_cost = self.channel_groups * rank
        basis_cost = self.kernel_size * self.kernel_size * rank if self.basis_trainable else 0
        channel_cost = self.channel_groups if self.channel_response else 0
        prefix_cost = 1 if self.prefix_coupling else 0
        return int(coefficient_cost + basis_cost + channel_cost + prefix_cost + 1)

    def raw_kernel_bank(self, rank: Optional[int] = None) -> torch.Tensor:
        used_rank = self.active_rank if rank is None else int(rank)
        if used_rank <= 0:
            return torch.zeros(
                self.channels,
                self.kernel_size,
                self.kernel_size,
                dtype=self.gate.dtype,
                device=self.gate.device,
            )
        coefficients = self.coefficients[:, :used_rank]
        atoms = self.basis_atoms[:used_rank].flatten(1)
        return (coefficients @ atoms).reshape(
            self.channels, self.kernel_size, self.kernel_size
        )

    # Backward-compatible name; the scientific version returns a channel bank.
    def raw_fused_kernel(self) -> torch.Tensor:
        return self.raw_kernel_bank()

    def projected_kernel_bank(self) -> torch.Tensor:
        kernels = self.raw_kernel_bank()
        l1 = kernels.abs().sum(dim=(1, 2), keepdim=True)
        scale = torch.clamp(self.operator_radius / (l1 + self.eps), max=1.0)
        return kernels * scale

    def projected_fused_kernel(self) -> torch.Tensor:
        return self.projected_kernel_bank()

    def _normalized_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_norm == "none":
            return x
        rms = x.square().mean(dim=(2, 3), keepdim=True).add(self.eps).sqrt()
        return x / rms

    def _channel_modulation(self, x: torch.Tensor, calibrating: bool = False) -> torch.Tensor:
        if not self.channel_response:
            return torch.zeros_like(x)
        if calibrating:
            scale = self.channel_probe.view(1, self.channels, 1, 1)
        else:
            gains = self.channel_group_gains[self.channel_group_index]
            scale = (self.channel_response_direction * gains).view(1, self.channels, 1, 1)
        return x * scale

    def _effective_gate(self) -> torch.Tensor:
        return self.gate.clamp(min=-self.gate_limit, max=self.gate_limit)

    def _normalize_residual(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        if self.residual_norm == "none":
            return delta
        # Per-example scaling makes the residual magnitude independent of stage
        # width, spatial resolution, activation units, and task-loss scale while
        # preserving the calibrated response direction. The detached scale avoids
        # introducing a normalization Jacobian into the tiny adapter optimizer.
        reduce_dims = tuple(range(1, delta.ndim))
        x_rms = x.detach().square().mean(dim=reduce_dims, keepdim=True).add(self.eps).sqrt()
        delta_rms = delta.detach().square().mean(dim=reduce_dims, keepdim=True).add(self.eps).sqrt()
        scale = (self.residual_target * x_rms / delta_rms).clamp(max=self.residual_scale_limit)
        return delta * scale

    def _depthwise(self, x: torch.Tensor, kernel_bank: torch.Tensor) -> torch.Tensor:
        if tuple(kernel_bank.shape) != (self.channels, self.kernel_size, self.kernel_size):
            raise ValueError(
                "Kernel bank shape mismatch: expected "
                f"{(self.channels, self.kernel_size, self.kernel_size)}, got {tuple(kernel_bank.shape)}"
            )
        weight = kernel_bank.unsqueeze(1)
        return F.conv2d(
            x,
            weight,
            padding=self.kernel_size // 2,
            groups=self.channels,
        )

    # Backward-compatible internal alias.
    def _shared_depthwise(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        if kernel.ndim == 2:
            kernel = kernel.unsqueeze(0).expand(self.channels, -1, -1)
        return self._depthwise(x, kernel)

    def explicit_basis_response(self, x: torch.Tensor) -> torch.Tensor:
        """Reference sum used to verify exact fusion of spatial atoms."""
        raw = self.raw_kernel_bank()
        l1 = raw.abs().sum(dim=(1, 2), keepdim=True)
        scale = torch.clamp(self.operator_radius / (l1 + self.eps), max=1.0)
        out = torch.zeros_like(x)
        for index in range(self.active_rank):
            atom_bank = self.basis_atoms[index].unsqueeze(0).expand(self.channels, -1, -1)
            response = self._depthwise(x, atom_bank)
            out = out + response * self.coefficient_vectors[index].view(1, self.channels, 1, 1)
        return out * scale.view(1, self.channels, 1, 1)

    def _forward_spatial(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        normalized = self._normalized_input(x)
        if self.calibrating:
            self.input_energy_sum.add_(x.detach().square().mean())
            self.input_energy_samples.add_(1)
            delta = self._depthwise(normalized, self.probe_kernel)
            delta = delta + self._channel_modulation(x, calibrating=True)
            return x + self.calibration_scale * delta, delta
        if not self.enabled or self.active_rank <= 0:
            return x, torch.zeros_like(x)
        delta = self._depthwise(normalized, self.projected_kernel_bank())
        delta = delta + self._channel_modulation(x, calibrating=False)
        delta = self._normalize_residual(x, delta)
        return x + self._effective_gate() * delta, delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.calibrating and (not self.enabled or self.active_rank <= 0):
            return x
        spatial, meta, prefix = tensor_to_bchw(x, self.channels, self.layout, self.grid_size)
        out, delta = self._forward_spatial(spatial)
        if meta.kind == "bnc" and prefix is not None and self.prefix_coupling:
            pooled = delta.mean(dim=(2, 3))
            prefix = prefix.clone()
            coupling = (self.prefix_gate * self._effective_gate()) if not self.calibrating else torch.ones_like(self.prefix_gate)
            if self.prefix_coupling_mode == "first" or prefix.shape[1] == 1:
                prefix[:, 0] = prefix[:, 0] + coupling * pooled
            elif self.prefix_coupling_mode == "mean":
                prefix = prefix + coupling * pooled.unsqueeze(1) / float(prefix.shape[1])
            else:  # all: energy-preserving coupling for class/distillation/register tokens.
                prefix = prefix + coupling * pooled.unsqueeze(1) / math.sqrt(float(prefix.shape[1]))
        return bchw_to_tensor(out, meta, prefix)

    def parameter_count_breakdown(self) -> Dict[str, int]:
        active = self.parameter_cost_for_rank(self.active_rank) if self.enabled else 0
        maximum = self.parameter_cost_for_rank(self.spatial_rank)
        if self.coefficient_mode == "full":
            coefficient_count = self.channels * self.active_rank
        elif self.coefficient_mode == "locked":
            coefficient_count = self.active_rank
        else:
            coefficient_count = self.channel_groups * self.active_rank
        return {
            "coefficient_mode": self.coefficient_mode,
            "coefficients": int(coefficient_count if self.enabled else 0),
            "basis": int(
                self.kernel_size * self.kernel_size * self.active_rank
                if self.enabled and self.basis_trainable else 0
            ),
            "channel_response": int(self.channel_groups if self.enabled and self.channel_response else 0),
            "gate": int(1 if self.enabled else 0),
            "prefix_gate": int(1 if self.enabled and self.prefix_coupling else 0),
            "adapter_capacity": int(maximum),
            "active_trainable": int(active),
            "total_trainable": int(active),
            "maximum_trainable": int(maximum),
        }

    def export_state(self) -> Dict[str, object]:
        return {
            "channels": self.channels,
            "kernel_size": self.kernel_size,
            "spatial_rank": self.spatial_rank,
            "active_rank": self.active_rank,
            "operator_radius": self.operator_radius,
            "variant": self.variant,
            "coefficient_mode": self.coefficient_mode,
            "channel_groups": self.channel_groups,
            "grouping_mode": self.grouping_mode,
            "channel_group_index": self.channel_group_index.detach().cpu().tolist(),
            "input_norm": self.input_norm,
            "calibration_grad_norm": self.calibration_grad_norm,
            "residual_norm": self.residual_norm,
            "residual_target": self.residual_target,
            "channel_response": self.channel_response,
            "prefix_coupling": self.prefix_coupling,
            "prefix_coupling_mode": self.prefix_coupling_mode,
            "enabled": self.enabled,
            "calibrated": self.calibrated,
            "response_score": float(self.response_score.item()),
            "response_noise": float(self.response_noise.item()),
            "channel_response_score": float(self.channel_response_score.item()),
            "channel_response_noise": float(self.channel_response_noise.item()),
            "response_stability": float(self.response_stability.item()),
            "channel_response_stability": float(self.channel_response_stability.item()),
            "component_stability": self.component_stability.detach().cpu().tolist(),
            "mean_raw_gradient_norm": float(
                self.raw_gradient_norm_sum.item() / max(1, int(self.gradient_samples.item()))
            ),
            "singular_values": self.singular_values.detach().cpu().tolist(),
            "component_noise": self.component_noise.detach().cpu().tolist(),
            "basis_atoms": self.basis_atoms.detach().cpu().tolist(),
            "coefficients": self.coefficients.detach().cpu().tolist(),
            "response_coefficient_directions": self.response_coefficient_directions.detach().cpu().tolist(),
            "component_amplitudes": self.component_amplitudes.detach().cpu().tolist(),
            "group_amplitudes": self.group_amplitudes.detach().cpu().tolist(),
            "channel_response_direction": self.channel_response_direction.detach().cpu().tolist(),
            "channel_group_gains": self.channel_group_gains.detach().cpu().tolist(),
            "gate": float(self.gate.detach().cpu().item()),
            "prefix_gate": float(self.prefix_gate.detach().cpu().item()),
        }

    @torch.no_grad()
    def load_exported_state(self, state: Mapping[str, object]) -> None:
        basis = torch.as_tensor(
            state["basis_atoms"], dtype=self.basis_atoms.dtype, device=self.basis_atoms.device
        )
        coefficients = torch.as_tensor(
            state["coefficients"],
            dtype=self.coefficients.dtype,
            device=self.coefficients.device,
        )
        self._copy_basis(basis)
        self._copy_coefficients(coefficients)
        if "response_coefficient_directions" in state:
            self.response_coefficient_directions.copy_(torch.as_tensor(
                state["response_coefficient_directions"], dtype=self.response_coefficient_directions.dtype,
                device=self.response_coefficient_directions.device))
        for key, target in (("component_amplitudes", self.component_amplitudes),
                            ("group_amplitudes", self.group_amplitudes),
                            ("channel_group_gains", self.channel_group_gains)):
            if key in state:
                target.copy_(torch.as_tensor(state[key], dtype=target.dtype, device=target.device))
        if "channel_response_direction" in state:
            self.channel_response_direction.copy_(torch.as_tensor(
                state["channel_response_direction"], dtype=self.channel_response_direction.dtype,
                device=self.channel_response_direction.device))
        if "channel_group_index" in state:
            restored_groups = torch.as_tensor(
                state["channel_group_index"], dtype=self.channel_group_index.dtype,
                device=self.channel_group_index.device,
            )
            if restored_groups.numel() == self.channel_group_index.numel():
                self.channel_group_index.copy_(restored_groups.reshape_as(self.channel_group_index))
        self.response_score.fill_(float(state.get("response_score", 0.0)))
        self.response_noise.fill_(float(state.get("response_noise", 0.0)))
        singular = torch.as_tensor(
            state.get("singular_values", []),
            dtype=self.singular_values.dtype,
            device=self.singular_values.device,
        )
        self.singular_values.zero_()
        self.singular_values[: min(singular.numel(), self.singular_values.numel())].copy_(
            singular[: self.singular_values.numel()]
        )
        if "gate" in state:
            self.gate.copy_(torch.as_tensor(state["gate"], dtype=self.gate.dtype, device=self.gate.device))
        if "prefix_gate" in state:
            self.prefix_gate.copy_(torch.as_tensor(state["prefix_gate"], dtype=self.prefix_gate.dtype, device=self.prefix_gate.device))
        self.channel_response_score.fill_(float(state.get("channel_response_score", 0.0)))
        self.channel_response_noise.fill_(float(state.get("channel_response_noise", 0.0)))
        self.response_stability.fill_(float(state.get("response_stability", 0.0)))
        self.channel_response_stability.fill_(float(state.get("channel_response_stability", 0.0)))
        if "component_stability" in state:
            restored = torch.as_tensor(
                state["component_stability"], dtype=self.component_stability.dtype,
                device=self.component_stability.device,
            )
            self.component_stability.zero_()
            self.component_stability[: min(restored.numel(), self.component_stability.numel())].copy_(
                restored[: self.component_stability.numel()]
            )
        self.calibrated_flag.fill_(bool(state.get("calibrated", True)))
        active_rank = int(state.get("active_rank", self.spatial_rank))
        if bool(state.get("enabled", True)) and active_rank > 0:
            self.set_active_rank(active_rank)
        else:
            self.set_active_rank(0)


# Concise public alias.
TRSOAdapter = TaskResponseSpatialAdapter


def iter_trso_adapters(model: nn.Module) -> Iterable[Tuple[str, TaskResponseSpatialAdapter]]:
    for name, module in model.named_modules():
        if isinstance(module, TaskResponseSpatialAdapter):
            yield name, module


def sync_trso_trainability(model: nn.Module) -> List[str]:
    """Synchronize every TRSO adapter after checkpoint/config loading."""
    selected: List[str] = []
    for name, adapter in iter_trso_adapters(model):
        adapter.sync_trainability_from_state()
        if adapter.enabled and adapter.active_rank > 0:
            selected.append(name)
    return selected


def _prune_dominated_states(states):
    """Remove states that cost more without producing a larger value."""
    grouped = {}
    for (cost, count), payload in states.items():
        grouped.setdefault(count, []).append((cost, payload))
    pruned = {}
    for count, rows in grouped.items():
        rows.sort(key=lambda row: row[0])
        best_value = -float("inf")
        for cost, payload in rows:
            value = payload[0]
            if value > best_value + 1e-12:
                pruned[(cost, count)] = payload
                best_value = value
    return pruned


def select_trso_layers(
    model: nn.Module,
    *,
    max_adapters: Optional[int] = None,
    keep_ratio: float = 1.0,
    parameter_budget: Optional[int] = None,
    allocation: str = "exact",
    score_mode: str = "energy",
    noise_beta: float = 0.0,
) -> List[str]:
    """Allocate TRSO ranks under controlled exact/greedy/uniform rules.

    ``exact`` solves the multiple-choice knapsack for the selected score
    surrogate. ``greedy`` selects the best incremental value/cost step.
    ``uniform`` assigns a common rank to the highest-scoring layers.
    """
    adapters = list(iter_trso_adapters(model))
    if not adapters:
        return []
    if not (0 < keep_ratio <= 1.0):
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
    allocation = str(allocation).lower()
    if allocation not in {"exact", "greedy", "uniform"}:
        raise ValueError("allocation must be exact, greedy, or uniform")
    by_ratio = max(1, int(math.ceil(len(adapters) * keep_ratio)))
    keep = by_ratio if max_adapters is None or max_adapters <= 0 else min(by_ratio, int(max_adapters))

    def value(adapter: TaskResponseSpatialAdapter, rank: int) -> float:
        return adapter.rank_value(rank, score_mode=score_mode, noise_beta=noise_beta)

    ranked = sorted(adapters, key=lambda row: value(row[1], row[1].spatial_rank), reverse=True)
    budget = int(parameter_budget or 0)
    if budget <= 0:
        chosen_names = {name for name, _ in ranked[:keep]}
        for name, adapter in adapters:
            adapter.set_active_rank(adapter.spatial_rank if name in chosen_names else 0)
        return [name for name, _ in ranked[:keep]]

    cheapest = min(adapter.parameter_cost_for_rank(1) for _, adapter in adapters)
    if budget < cheapest:
        raise ValueError(
            f"TRSO parameter budget {budget} is smaller than the cheapest rank-one candidate ({cheapest})."
        )

    if allocation == "uniform":
        best_ranks = [0] * len(adapters)
        name_to_index = {name: index for index, (name, _) in enumerate(adapters)}
        # Highest common rank that permits at least one selected layer; add
        # layers in score order until the next one exceeds budget/keep.
        for common_rank in range(max(a.spatial_rank for _, a in adapters), 0, -1):
            ranks = [0] * len(adapters)
            used = 0
            count = 0
            for name, adapter in ranked:
                rank = min(common_rank, adapter.spatial_rank)
                cost = adapter.parameter_cost_for_rank(rank)
                if count < keep and used + cost <= budget:
                    ranks[name_to_index[name]] = rank
                    used += cost
                    count += 1
            if count > 0:
                best_ranks = ranks
                break
    elif allocation == "greedy":
        best_ranks = [0] * len(adapters)
        used_cost = 0
        active_count = 0
        while True:
            candidates = []
            for index, (_, adapter) in enumerate(adapters):
                current = best_ranks[index]
                if current >= adapter.spatial_rank:
                    continue
                if current == 0 and active_count >= keep:
                    continue
                next_rank = current + 1
                incremental_cost = adapter.parameter_cost_for_rank(next_rank) - adapter.parameter_cost_for_rank(current)
                if used_cost + incremental_cost > budget:
                    continue
                gain = value(adapter, next_rank) - value(adapter, current)
                ratio = gain / max(1, incremental_cost)
                candidates.append((ratio, gain, -incremental_cost, index, next_rank))
            if not candidates:
                break
            _, _, neg_cost, index, next_rank = max(candidates)
            old_rank = best_ranks[index]
            incremental_cost = -neg_cost
            best_ranks[index] = next_rank
            used_cost += incremental_cost
            if old_rank == 0:
                active_count += 1
    else:
        # Exact sparse dynamic program. State payload: value, rank tuple.
        states = {(0, 0): (0.0, tuple())}
        for _, adapter in adapters:
            updated = {}
            for (used_cost, used_count), (total_value, choices) in states.items():
                for rank in range(adapter.spatial_rank + 1):
                    new_count = used_count + int(rank > 0)
                    if new_count > keep:
                        continue
                    new_cost = used_cost + adapter.parameter_cost_for_rank(rank)
                    if new_cost > budget:
                        continue
                    candidate = (total_value + value(adapter, rank), choices + (rank,))
                    key = (new_cost, new_count)
                    current = updated.get(key)
                    if current is None or candidate[0] > current[0] + 1e-12:
                        updated[key] = candidate
            states = _prune_dominated_states(updated)
        if not states:
            raise RuntimeError("No feasible TRSO allocation was found.")
        _, (_, best_ranks) = max(states.items(), key=lambda item: (item[1][0], -item[0][0]))
        best_ranks = list(best_ranks)

    for (_, adapter), rank in zip(adapters, best_ranks):
        adapter.set_active_rank(rank)
    selected_rows = [
        (name, value(adapter, rank))
        for (name, adapter), rank in zip(adapters, best_ranks)
        if rank > 0
    ]
    selected_rows.sort(key=lambda row: row[1], reverse=True)
    return [name for name, _ in selected_rows]


def save_trso_config(model: nn.Module, path: str | Path, selected_layers: Optional[Sequence[str]] = None) -> None:
    payload = {
        "method": "Scientific-TRSO",
        "selected_layers": list(
            selected_layers
            or [name for name, adapter in iter_trso_adapters(model) if adapter.enabled]
        ),
        "adapters": {name: adapter.export_state() for name, adapter in iter_trso_adapters(model)},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_trso_config(model: nn.Module, path: str | Path, strict: bool = True) -> List[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    states: MutableMapping[str, Mapping[str, object]] = payload.get("adapters", {})
    modules = dict(iter_trso_adapters(model))
    missing = []
    for name, state in states.items():
        if name not in modules:
            missing.append(name)
            continue
        modules[name].load_exported_state(state)
    if strict and missing:
        raise KeyError(f"TRSO config contains unknown adapter layers: {missing}")
    selected = [name for name, adapter in modules.items() if adapter.enabled]
    return sorted(selected)
