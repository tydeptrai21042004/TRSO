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
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

        # A full channel-specific probe bank measures the exact tangent of the
        # same depthwise operator family later used for training.
        self.probe_kernel = nn.Parameter(
            torch.zeros(self.channels, self.kernel_size, self.kernel_size),
            requires_grad=False,
        )
        self.register_buffer("gradient_sum", torch.zeros_like(self.probe_kernel))
        self.register_buffer("gradient_square_sum", torch.zeros_like(self.probe_kernel))
        self.register_buffer("gradient_samples", torch.tensor(0, dtype=torch.long))
        self.register_buffer("singular_values", torch.zeros(self.spatial_rank))
        self.register_buffer("response_score", torch.tensor(0.0))
        self.register_buffer("response_noise", torch.tensor(0.0))
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
        """Return the C x R coefficient matrix."""
        return torch.stack(list(self.coefficient_vectors), dim=1)

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
        for index, parameter in enumerate(self.coefficient_vectors):
            parameter.data.copy_(coefficients[:, index])

    def set_active_rank(self, rank: int) -> None:
        rank = int(rank)
        if not (0 <= rank <= self.spatial_rank):
            raise ValueError(f"active rank must be in [0, {self.spatial_rank}], got {rank}")
        self.active_rank_buffer.fill_(rank)
        self.enabled_flag.fill_(rank > 0)
        for index, parameter in enumerate(self.coefficient_vectors):
            parameter.requires_grad_(rank > 0 and index < rank)
        if self.basis_trainable:
            for index, parameter in enumerate(self.basis_atom_parameters):
                parameter.requires_grad_(rank > 0 and index < rank)
        self.gate.requires_grad_(rank > 0)

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
            self.gate.requires_grad_(False)
            self.active_rank_buffer.fill_(saved_rank)

    def sync_trainability_from_state(self) -> None:
        """Restore ``requires_grad`` from persistent rank/enabled buffers.

        PyTorch state dictionaries do not serialize ``requires_grad``. This
        method is therefore called automatically after state loading and may
        also be called explicitly before optimizer construction.
        """
        rank = self.active_rank
        enabled = self.enabled and rank > 0
        for index, parameter in enumerate(self.coefficient_vectors):
            parameter.requires_grad_(enabled and index < rank)
        for index, parameter in enumerate(self.basis_atom_parameters):
            parameter.requires_grad_(enabled and index < rank)
        self.gate.requires_grad_(enabled)
        self.probe_kernel.requires_grad_(self.calibrating)

    def start_calibration(self, reset: bool = True) -> None:
        if reset:
            self.gradient_sum.zero_()
            self.gradient_square_sum.zero_()
            self.gradient_samples.zero_()
            self.probe_kernel.data.zero_()
        self.calibration_flag.fill_(True)
        self.probe_kernel.requires_grad_(True)

    def stop_calibration(self) -> None:
        self.calibration_flag.fill_(False)
        self.probe_kernel.requires_grad_(False)
        self.probe_kernel.grad = None

    @torch.no_grad()
    def accumulate_probe_gradient(self) -> None:
        grad = self.probe_kernel.grad
        if grad is None:
            return
        detached = grad.detach()
        self.gradient_sum.add_(detached)
        self.gradient_square_sum.add_(detached.square())
        self.gradient_samples.add_(1)
        self.probe_kernel.grad = None

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
        energy = float(self.singular_values[:rank].square().sum().item())
        mode = str(score_mode).lower()
        if mode == "energy":
            return energy
        if mode == "energy_per_param":
            return energy / max(1, self.parameter_cost_for_rank(rank))
        if mode == "energy_per_channel":
            return energy / max(1, self.channels)
        if mode == "noise_adjusted":
            noise_share = float(self.response_noise.item()) * rank / max(1, self.spatial_rank)
            return max(0.0, energy - float(noise_beta) * noise_share)
        raise ValueError(
            "score_mode must be energy, energy_per_param, energy_per_channel, or noise_adjusted"
        )

    def parameter_cost_for_rank(self, rank: int) -> int:
        rank = int(rank)
        if rank <= 0:
            return 0
        coefficient_cost = self.channels * rank
        basis_cost = self.kernel_size * self.kernel_size * rank if self.basis_trainable else 0
        return int(coefficient_cost + basis_cost + 1)  # one residual gate

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

    def _forward_spatial(self, x: torch.Tensor) -> torch.Tensor:
        if self.calibrating:
            delta = self._depthwise(x, self.probe_kernel)
            return x + self.calibration_scale * delta
        if not self.enabled or self.active_rank <= 0:
            return x
        delta = self._depthwise(x, self.projected_kernel_bank())
        return x + self.gate * delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Avoid token reshapes and memory copies for unselected adapters.
        if not self.calibrating and (not self.enabled or self.active_rank <= 0):
            return x
        spatial, meta, prefix = tensor_to_bchw(x, self.channels, self.layout, self.grid_size)
        out = self._forward_spatial(spatial)
        return bchw_to_tensor(out, meta, prefix)

    def parameter_count_breakdown(self) -> Dict[str, int]:
        active = self.parameter_cost_for_rank(self.active_rank) if self.enabled else 0
        maximum = self.parameter_cost_for_rank(self.spatial_rank)
        return {
            "channel_down": 0,
            "channel_up": 0,
            "coefficients": int(self.channels * self.active_rank if self.enabled else 0),
            "basis": int(
                self.kernel_size * self.kernel_size * self.active_rank
                if self.enabled and self.basis_trainable
                else 0
            ),
            "gate": int(1 if self.enabled else 0),
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
            "enabled": self.enabled,
            "calibrated": self.calibrated,
            "response_score": float(self.response_score.item()),
            "response_noise": float(self.response_noise.item()),
            "singular_values": self.singular_values.detach().cpu().tolist(),
            "basis_atoms": self.basis_atoms.detach().cpu().tolist(),
            "coefficients": self.coefficients.detach().cpu().tolist(),
            "gate": float(self.gate.detach().cpu().item()),
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
