"""Task-Response Spatial Operator (TRSO) adapter.

TRSO replaces the previous hand-designed shifted axial adapter with a task-derived
2-D spatial operator.  During a short calibration pass each adapter exposes a
zero-initialized virtual depthwise kernel.  The downstream-loss gradient with
respect to that kernel is accumulated and factorized by SVD.  The leading
rank-one spatial directions become the adapter basis, while layers are selected
from their captured singular energy under a global adapter budget.

The same module accepts CNN feature maps (B,C,H,W), channels-last maps
(B,H,W,C), and Transformer patch tokens (B,N,C).  At normal inference all
selected basis atoms are merged into a single depthwise convolution kernel.
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

    if n_tokens > 1:
        patch_tokens = n_tokens - 1
        side = int(math.isqrt(patch_tokens))
        if side * side == patch_tokens:
            return (side, side), 1

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

    Returns the spatial tensor, layout metadata, and optional prefix tokens.
    Prefix tokens are never spatially filtered and are restored unchanged.
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
            return x.permute(0, 3, 1, 2).contiguous(), SpatialLayout("bhwc", (x.shape[1], x.shape[2])), None
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


class TaskResponseSpatialAdapter(nn.Module):
    """Task-derived, rank-constrained spatial PEFT adapter.

    Parameters
    ----------
    channels:
        Feature dimension of the target CNN block or Transformer token block.
    kernel_size:
        Odd support size of the discovered 2-D operator.
    spatial_rank:
        Number of SVD-derived rank-one spatial atoms retained per selected layer.
    channel_ratio:
        Bottleneck ratio for the trainable 1x1 channel projections.
    operator_radius:
        Maximum l1 norm of the fused spatial kernel.
    gate_init:
        Small non-zero residual scale.  Unlike a zero scalar gate, this allows all
        internal adapter parameters to receive gradients from the first step.
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
        if spatial_rank <= 0 or spatial_rank > kernel_size:
            raise ValueError(
                f"spatial_rank must satisfy 1 <= rank <= kernel_size; got rank={spatial_rank}, k={kernel_size}"
            )
        if channel_ratio <= 0:
            raise ValueError(f"channel_ratio must be positive, got {channel_ratio}")
        if operator_radius <= 0:
            raise ValueError(f"operator_radius must be positive, got {operator_radius}")
        if calibration_scale <= 0:
            raise ValueError(f"calibration_scale must be positive, got {calibration_scale}")

        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.spatial_rank = int(spatial_rank)
        self.channel_ratio = int(channel_ratio)
        self.operator_radius = float(operator_radius)
        self.calibration_scale = float(calibration_scale)
        self.layout = str(layout)
        self.grid_size = _as_pair(grid_size)
        self.basis_trainable = bool(basis_trainable)
        self.eps = float(eps)

        hidden = max(1, self.channels // self.channel_ratio)
        self.hidden_channels = hidden
        self.down = nn.Conv2d(self.channels, hidden, kernel_size=1, bias=False)
        self.up = nn.Conv2d(hidden, self.channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.normal_(self.up.weight, mean=0.0, std=1e-3)

        # The basis is populated by finalize_calibration().  A deterministic
        # orthonormal fallback keeps the module runnable before calibration, but
        # production runs should use the calibration stage enabled by default.
        fallback = self._fallback_basis(self.kernel_size, self.spatial_rank)
        if self.basis_trainable:
            self.basis_atoms = nn.Parameter(fallback)
        else:
            self.register_buffer("basis_atoms", fallback)
        self.coefficients = nn.Parameter(torch.full((self.spatial_rank,), float(basis_init_scale) / self.spatial_rank))
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

        # The virtual kernel is used only during task-response calibration.
        self.probe_kernel = nn.Parameter(torch.zeros(self.kernel_size, self.kernel_size), requires_grad=False)
        self.register_buffer("gradient_sum", torch.zeros(self.kernel_size, self.kernel_size))
        self.register_buffer("gradient_samples", torch.tensor(0, dtype=torch.long))
        self.register_buffer("singular_values", torch.zeros(self.spatial_rank))
        self.register_buffer("response_score", torch.tensor(0.0))
        self.register_buffer("enabled_flag", torch.tensor(bool(enabled), dtype=torch.bool))
        self.register_buffer("calibration_flag", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("calibrated_flag", torch.tensor(False, dtype=torch.bool))

        self.is_trso_adapter = True

    @staticmethod
    def _fallback_basis(kernel_size: int, rank: int) -> torch.Tensor:
        # Deterministic separable orthonormal atoms. They are only a safe fallback;
        # calibration replaces them with task-derived directions.
        coords = torch.linspace(-1.0, 1.0, kernel_size)
        vectors: List[torch.Tensor] = []
        for degree in range(rank):
            v = coords.pow(degree)
            for prev in vectors:
                v = v - torch.dot(v, prev) * prev
            v = v / v.norm().clamp_min(1e-8)
            vectors.append(v)
        atoms = [torch.outer(vectors[i], vectors[(i + 1) % len(vectors)]) for i in range(rank)]
        return torch.stack(atoms, dim=0)

    @property
    def enabled(self) -> bool:
        return bool(self.enabled_flag.item())

    @property
    def calibrating(self) -> bool:
        return bool(self.calibration_flag.item())

    @property
    def calibrated(self) -> bool:
        return bool(self.calibrated_flag.item())

    def set_enabled(self, enabled: bool) -> None:
        self.enabled_flag.fill_(bool(enabled))
        for p in (self.down.weight, self.up.weight, self.coefficients, self.gate):
            p.requires_grad_(bool(enabled))
        if isinstance(self.basis_atoms, nn.Parameter):
            self.basis_atoms.requires_grad_(bool(enabled))

    def start_calibration(self, reset: bool = True) -> None:
        if reset:
            self.gradient_sum.zero_()
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
        self.gradient_sum.add_(grad.detach())
        self.gradient_samples.add_(1)
        self.probe_kernel.grad = None

    @torch.no_grad()
    def finalize_calibration(self, init_scale: float = 5e-2) -> float:
        if int(self.gradient_samples.item()) <= 0:
            raise RuntimeError("No probe gradients were accumulated for this TRSO adapter.")
        response = self.gradient_sum / self.gradient_samples.to(self.gradient_sum.dtype)
        # The descending task-response direction is -gradient.
        u, s, vh = torch.linalg.svd(-response, full_matrices=False)
        r = min(self.spatial_rank, s.numel())
        atoms = torch.stack([torch.outer(u[:, i], vh[i, :]) for i in range(r)], dim=0)
        if r < self.spatial_rank:
            pad = self._fallback_basis(self.kernel_size, self.spatial_rank - r).to(atoms)
            atoms = torch.cat((atoms, pad), dim=0)
        if isinstance(self.basis_atoms, nn.Parameter):
            self.basis_atoms.data.copy_(atoms)
        else:
            self.basis_atoms.copy_(atoms)

        captured = s[:r]
        self.singular_values.zero_()
        self.singular_values[:r].copy_(captured)
        score = captured.square().sum()
        self.response_score.copy_(score)
        weights = captured / captured.sum().clamp_min(self.eps)
        self.coefficients.data.zero_()
        self.coefficients.data[:r].copy_(weights * float(init_scale))
        self.calibrated_flag.fill_(True)
        self.stop_calibration()
        return float(score.item())

    def raw_fused_kernel(self) -> torch.Tensor:
        return torch.einsum("r,rhw->hw", self.coefficients, self.basis_atoms)

    def projected_fused_kernel(self) -> torch.Tensor:
        kernel = self.raw_fused_kernel()
        l1 = kernel.abs().sum()
        scale = torch.clamp(self.operator_radius / (l1 + self.eps), max=1.0)
        return kernel * scale

    def explicit_basis_response(self, x: torch.Tensor) -> torch.Tensor:
        """Reference implementation used by tests to verify fusion exactly."""
        out = torch.zeros_like(x)
        for coefficient, atom in zip(self.coefficients, self.basis_atoms):
            weight = atom.view(1, 1, self.kernel_size, self.kernel_size).expand(x.shape[1], 1, -1, -1)
            out = out + coefficient * F.conv2d(x, weight, padding=self.kernel_size // 2, groups=x.shape[1])
        raw = self.raw_fused_kernel()
        scale = torch.clamp(self.operator_radius / (raw.abs().sum() + self.eps), max=1.0)
        return out * scale

    def _shared_depthwise(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        weight = kernel.view(1, 1, self.kernel_size, self.kernel_size).expand(x.shape[1], 1, -1, -1)
        return F.conv2d(x, weight, padding=self.kernel_size // 2, groups=x.shape[1])

    def _forward_spatial(self, x: torch.Tensor) -> torch.Tensor:
        if self.calibrating:
            # The zero probe leaves the forward path unchanged while exposing a
            # meaningful directional derivative for every candidate layer.
            delta = self._shared_depthwise(x, self.probe_kernel)
            return x + self.calibration_scale * delta
        if not self.enabled:
            return x
        z = self.down(x)
        z = self._shared_depthwise(z, self.projected_fused_kernel())
        delta = self.up(F.gelu(z))
        return x + self.gate * delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial, meta, prefix = tensor_to_bchw(x, self.channels, self.layout, self.grid_size)
        out = self._forward_spatial(spatial)
        return bchw_to_tensor(out, meta, prefix)

    def parameter_count_breakdown(self) -> Dict[str, int]:
        basis = self.basis_atoms.numel() if isinstance(self.basis_atoms, nn.Parameter) else 0
        capacity = (
            self.down.weight.numel()
            + self.up.weight.numel()
            + self.coefficients.numel()
            + basis
            + self.gate.numel()
        )
        return {
            "channel_down": int(self.down.weight.numel()),
            "channel_up": int(self.up.weight.numel()),
            "coefficients": int(self.coefficients.numel()),
            "basis": int(basis),
            "gate": int(self.gate.numel()),
            "adapter_capacity": int(capacity),
            "active_trainable": int(sum(
                p.numel() for p in self.parameters()
                if p.requires_grad and p is not self.probe_kernel
            )),
            # Backward-compatible key used by tests and reporting. This is the
            # structural adapter cost, independent of temporary calibration state.
            "total_trainable": int(capacity),
        }

    def export_state(self) -> Dict[str, object]:
        return {
            "channels": self.channels,
            "kernel_size": self.kernel_size,
            "spatial_rank": self.spatial_rank,
            "operator_radius": self.operator_radius,
            "enabled": self.enabled,
            "calibrated": self.calibrated,
            "response_score": float(self.response_score.item()),
            "singular_values": self.singular_values.detach().cpu().tolist(),
            "basis_atoms": self.basis_atoms.detach().cpu().tolist(),
            "coefficients": self.coefficients.detach().cpu().tolist(),
        }

    @torch.no_grad()
    def load_exported_state(self, state: Mapping[str, object]) -> None:
        basis = torch.as_tensor(state["basis_atoms"], dtype=self.basis_atoms.dtype, device=self.basis_atoms.device)
        coeff = torch.as_tensor(state["coefficients"], dtype=self.coefficients.dtype, device=self.coefficients.device)
        if tuple(basis.shape) != tuple(self.basis_atoms.shape):
            raise ValueError(f"Basis shape mismatch: expected {tuple(self.basis_atoms.shape)}, got {tuple(basis.shape)}")
        if tuple(coeff.shape) != tuple(self.coefficients.shape):
            raise ValueError(f"Coefficient shape mismatch: expected {tuple(self.coefficients.shape)}, got {tuple(coeff.shape)}")
        if isinstance(self.basis_atoms, nn.Parameter):
            self.basis_atoms.data.copy_(basis)
        else:
            self.basis_atoms.copy_(basis)
        self.coefficients.data.copy_(coeff)
        self.response_score.fill_(float(state.get("response_score", 0.0)))
        singular = torch.as_tensor(state.get("singular_values", []), dtype=self.singular_values.dtype, device=self.singular_values.device)
        self.singular_values.zero_()
        self.singular_values[: min(singular.numel(), self.singular_values.numel())].copy_(singular[: self.singular_values.numel()])
        self.calibrated_flag.fill_(bool(state.get("calibrated", True)))
        self.set_enabled(bool(state.get("enabled", True)))


# Concise public alias.
TRSOAdapter = TaskResponseSpatialAdapter


def iter_trso_adapters(model: nn.Module) -> Iterable[Tuple[str, TaskResponseSpatialAdapter]]:
    for name, module in model.named_modules():
        if isinstance(module, TaskResponseSpatialAdapter):
            yield name, module


def select_trso_layers(
    model: nn.Module,
    *,
    max_adapters: Optional[int] = None,
    keep_ratio: float = 1.0,
    parameter_budget: Optional[int] = None,
) -> List[str]:
    """Select task-responsive layers under count and/or trainable-parameter budgets.

    Without a parameter budget, layers are ranked by captured singular energy.
    With a budget, greedy benefit density ``response_score / trainable_cost`` is
    used, which accounts for the different channel widths of CNN/Transformer
    blocks. The exact selected parameter total is reported by the saved config.
    """
    adapters = list(iter_trso_adapters(model))
    if not adapters:
        return []
    if not (0 < keep_ratio <= 1.0):
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
    by_ratio = max(1, int(math.ceil(len(adapters) * keep_ratio)))
    keep = by_ratio if max_adapters is None or max_adapters <= 0 else min(by_ratio, int(max_adapters))

    rows = []
    for name, adapter in adapters:
        cost = max(1, int(adapter.parameter_count_breakdown()["total_trainable"]))
        score = float(adapter.response_score.item())
        density = score / cost
        rows.append((name, adapter, score, cost, density))

    if parameter_budget is not None and parameter_budget > 0:
        minimum = min(row[3] for row in rows)
        if parameter_budget < minimum:
            raise ValueError(
                f"TRSO parameter budget {parameter_budget} is smaller than the cheapest "
                f"candidate adapter ({minimum} parameters)."
            )
        ranked_rows = sorted(rows, key=lambda row: (row[4], row[2]), reverse=True)
        chosen = []
        used = 0
        for row in ranked_rows:
            if len(chosen) >= keep:
                break
            if used + row[3] <= parameter_budget:
                chosen.append(row)
                used += row[3]
    else:
        ranked_rows = sorted(rows, key=lambda row: row[2], reverse=True)
        chosen = ranked_rows[:keep]

    selected = {row[0] for row in chosen}
    for name, adapter in adapters:
        adapter.set_enabled(name in selected)
    return [row[0] for row in chosen]


def save_trso_config(model: nn.Module, path: str | Path, selected_layers: Optional[Sequence[str]] = None) -> None:
    payload = {
        "method": "TRSO",
        "selected_layers": list(selected_layers or [name for name, adapter in iter_trso_adapters(model) if adapter.enabled]),
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
    selected = set(payload.get("selected_layers", []))
    if selected:
        for name, adapter in modules.items():
            adapter.set_enabled(name in selected)
    return sorted(selected)
