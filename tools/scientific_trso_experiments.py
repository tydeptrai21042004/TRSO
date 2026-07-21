#!/usr/bin/env python3
"""Controlled scientific validation for the aligned TRSO proposal.

The experiments are intentionally synthetic and mechanism-focused. They test
claims that can be isolated without downloading datasets or pretrained weights:

1. calibration is the exact tangent of the trainable operator family;
2. truncated SVD is the optimal rank-r channel--spatial response projection;
3. channel-specific mixtures are necessary when channels need different filters;
4. exact budget allocation can outperform response-per-cost greedy selection;
5. the factorized parameterization preserves its rank during optimization;
6. task-derived bases improve immediate and one-epoch transfer performance over
   random bases under the same 13-parameter budget.

These checks validate the mechanism, not state-of-the-art benchmark accuracy.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.task_response_adapter import TaskResponseSpatialAdapter, select_trso_layers


@dataclass
class ScientificReport:
    aligned_tangent_cosine_mean: float
    aligned_tangent_cosine_min: float
    legacy_surrogate_cosine_mean: float
    legacy_surrogate_cosine_median: float
    svd_optimal_relative_error: float
    best_random_subspace_relative_error: float
    random_to_svd_error_ratio: float
    channel_specific_output_mse: float
    shared_kernel_output_mse: float
    shared_to_specific_mse_ratio: float
    exact_budget_value: float
    greedy_budget_value: float
    exact_budget_selected: List[str]
    maximum_factorized_rank_during_training: int
    configured_factorized_rank: int
    transfer_source_accuracy_mean: float
    transfer_corrupted_accuracy_mean: float
    transfer_trso_initial_accuracy_mean: float
    transfer_random_initial_accuracy_mean: float
    transfer_trso_one_epoch_accuracy_mean: float
    transfer_random_one_epoch_accuracy_mean: float
    transfer_full_kernel_one_epoch_accuracy_mean: float
    transfer_trso_initial_accuracy_std: float
    transfer_random_initial_accuracy_std: float
    trso_trainable_parameters: int
    random_trainable_parameters: int
    full_kernel_trainable_parameters: int


def _median(values: List[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    midpoint = n // 2
    if n % 2:
        return ordered[midpoint]
    return 0.5 * (ordered[midpoint - 1] + ordered[midpoint])


def _alignment_experiment(trials: int = 24) -> Dict[str, float]:
    aligned: List[float] = []
    legacy: List[float] = []
    for seed in range(trials):
        torch.manual_seed(1000 + seed)
        channels, hidden, kernel = 6, 3, 3
        x = torch.randn(4, channels, 9, 9)
        target = torch.randn_like(x)

        adapter = TaskResponseSpatialAdapter(
            channels=channels,
            kernel_size=kernel,
            spatial_rank=2,
            operator_radius=10.0,
            gate_init=1.0,
            calibration_scale=1.0,
        )
        adapter.start_calibration()
        F.mse_loss(adapter(x), target).backward()
        probe_gradient = adapter.probe_kernel.grad.detach().clone()
        adapter.stop_calibration()

        full_kernel = torch.zeros(channels, kernel, kernel, requires_grad=True)
        direct = x + F.conv2d(
            x, full_kernel.unsqueeze(1), padding=kernel // 2, groups=channels
        )
        F.mse_loss(direct, target).backward()
        direct_gradient = full_kernel.grad.detach()
        aligned.append(
            float(
                F.cosine_similarity(
                    probe_gradient.flatten(), direct_gradient.flatten(), dim=0
                ).item()
            )
        )

        # Legacy mismatch: calibration perturbs X directly with one shared
        # kernel, whereas training inserts that kernel after random down/up
        # projections and GELU.
        calibration_kernel = torch.zeros(kernel, kernel, requires_grad=True)
        shared_weight = calibration_kernel.view(1, 1, kernel, kernel).expand(
            channels, 1, -1, -1
        )
        calibration_output = x + F.conv2d(
            x, shared_weight, padding=kernel // 2, groups=channels
        )
        F.mse_loss(calibration_output, target).backward()
        legacy_probe = calibration_kernel.grad.detach().clone()

        down_weight = torch.randn(hidden, channels, 1, 1) / math.sqrt(channels)
        up_weight = torch.randn(channels, hidden, 1, 1) * 1e-3
        training_kernel = torch.zeros(kernel, kernel, requires_grad=True)
        z = F.conv2d(x, down_weight)
        spatial_weight = training_kernel.view(1, 1, kernel, kernel).expand(
            hidden, 1, -1, -1
        )
        z = F.conv2d(z, spatial_weight, padding=kernel // 2, groups=hidden)
        delta = F.conv2d(F.gelu(z), up_weight)
        legacy_output = x + 1e-2 * delta
        F.mse_loss(legacy_output, target).backward()
        legacy_actual = training_kernel.grad.detach()
        legacy.append(
            float(
                F.cosine_similarity(
                    legacy_probe.flatten(), legacy_actual.flatten(), dim=0, eps=1e-12
                ).item()
            )
        )

    return {
        "aligned_mean": mean(aligned),
        "aligned_min": min(aligned),
        "legacy_mean": mean(legacy),
        "legacy_median": _median(legacy),
    }


def _svd_experiment(random_trials: int = 1000) -> Dict[str, float]:
    torch.manual_seed(203)
    channels, kernel, rank = 10, 5, 3
    # A full response with decaying spectrum.
    left, _ = torch.linalg.qr(torch.randn(channels, channels))
    right, _ = torch.linalg.qr(torch.randn(kernel * kernel, kernel * kernel))
    singular = torch.tensor([8.0, 5.0, 3.0, 1.5, 1.0, 0.7, 0.5, 0.3, 0.2, 0.1])
    response = (left * singular) @ right[:channels]

    u, s, vh = torch.linalg.svd(response, full_matrices=False)
    optimal = (u[:, :rank] * s[:rank]) @ vh[:rank]
    denominator = response.norm().clamp_min(1e-12)
    optimal_error = float(((response - optimal).norm() / denominator).item())

    best_random = float("inf")
    for _ in range(random_trials):
        q, _ = torch.linalg.qr(torch.randn(kernel * kernel, rank))
        # For a fixed random spatial subspace Q, response @ Q gives the optimal
        # channel coefficients in least squares.
        reconstruction = (response @ q) @ q.T
        error = float(((response - reconstruction).norm() / denominator).item())
        best_random = min(best_random, error)

    return {
        "svd_error": optimal_error,
        "random_error": best_random,
        "ratio": best_random / max(optimal_error, 1e-12),
    }


def _channel_specific_experiment() -> Dict[str, float]:
    torch.manual_seed(307)
    channels, kernel, rank = 8, 5, 2
    atoms = torch.randn(rank, kernel, kernel)
    coefficients = torch.randn(channels, rank)
    target_bank = (coefficients @ atoms.flatten(1)).reshape(channels, kernel, kernel)

    response = target_bank.flatten(1)
    u, s, vh = torch.linalg.svd(response, full_matrices=False)
    low_rank_bank = ((u[:, :rank] * s[:rank]) @ vh[:rank]).reshape_as(target_bank)
    shared = target_bank.mean(dim=0, keepdim=True).expand_as(target_bank)

    x = torch.randn(64, channels, 24, 24)
    target_y = F.conv2d(x, target_bank.unsqueeze(1), padding=kernel // 2, groups=channels)
    specific_y = F.conv2d(x, low_rank_bank.unsqueeze(1), padding=kernel // 2, groups=channels)
    shared_y = F.conv2d(x, shared.unsqueeze(1), padding=kernel // 2, groups=channels)
    specific_mse = float(F.mse_loss(specific_y, target_y).item())
    shared_mse = float(F.mse_loss(shared_y, target_y).item())
    return {
        "specific_mse": specific_mse,
        "shared_mse": shared_mse,
        "ratio": shared_mse / max(specific_mse, 1e-12),
    }


def _set_spectrum(adapter: TaskResponseSpatialAdapter, values: List[float]) -> None:
    tensor = torch.tensor(values, dtype=adapter.singular_values.dtype)
    adapter.singular_values.zero_()
    adapter.singular_values[: tensor.numel()].copy_(tensor.sqrt())
    adapter.response_score.copy_(tensor.sum())


def _budget_experiment() -> Dict[str, object]:
    model = nn.ModuleDict(
        {
            "large_value": TaskResponseSpatialAdapter(
                channels=5, kernel_size=3, spatial_rank=1
            ),
            "small_a": TaskResponseSpatialAdapter(
                channels=4, kernel_size=3, spatial_rank=1
            ),
            "small_b": TaskResponseSpatialAdapter(
                channels=4, kernel_size=3, spatial_rank=1
            ),
        }
    )
    values = {"large_value": 12.0, "small_a": 9.0, "small_b": 9.0}
    for name, value in values.items():
        _set_spectrum(model[name], [value])

    budget = 10
    selected = select_trso_layers(
        model, max_adapters=2, keep_ratio=1.0, parameter_budget=budget
    )
    exact_value = sum(values[name] for name in selected)

    rows = []
    for name, adapter in model.items():
        cost = adapter.parameter_cost_for_rank(1)
        rows.append((values[name] / cost, name, cost, values[name]))
    used = 0
    greedy_value = 0.0
    for _, _, cost, value in sorted(rows, reverse=True):
        if used + cost <= budget:
            used += cost
            greedy_value += value
    return {
        "exact_value": exact_value,
        "greedy_value": greedy_value,
        "selected": selected,
    }


def _rank_preservation_experiment() -> Dict[str, int]:
    torch.manual_seed(401)
    configured_rank = 2
    adapter = TaskResponseSpatialAdapter(
        channels=7,
        kernel_size=3,
        spatial_rank=configured_rank,
        basis_trainable=True,
        operator_radius=10.0,
        gate_init=1.0,
    )
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.05)
    maximum_rank = 0
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        x = torch.randn(4, 7, 10, 10)
        target = torch.randn_like(x)
        loss = F.mse_loss(adapter(x), target)
        loss.backward()
        optimizer.step()
        observed = int(torch.linalg.matrix_rank(adapter.raw_kernel_bank().flatten(1)))
        maximum_rank = max(maximum_rank, observed)
    return {"maximum_rank": maximum_rank, "configured_rank": configured_rank}


def _make_transfer_data(seed: int):
    torch.manual_seed(seed)
    channels, height, width, classes = 6, 12, 12, 4
    yy, xx = torch.meshgrid(
        torch.arange(height), torch.arange(width), indexing="ij"
    )
    centers = [(4, 4), (4, 6), (6, 4), (6, 6)]
    patterns = torch.stack(
        [
            torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 1.2**2))
            for cy, cx in centers
        ]
    ).float()
    channel_mix = torch.tensor(
        [
            [1.0, 0.8, 0.6, 0.4, 0.2, 0.1],
            [0.2, 1.0, 0.8, 0.6, 0.4, 0.2],
            [0.4, 0.2, 1.0, 0.8, 0.6, 0.4],
            [0.6, 0.4, 0.2, 1.0, 0.8, 0.6],
        ]
    )
    prototypes = channel_mix[:, :, None, None] * patterns[:, None]
    prototypes = (
        prototypes
        / prototypes.flatten(1).norm(dim=1)[:, None, None, None]
        * math.sqrt(channels * height * width)
    )

    def sample(count: int):
        labels = torch.randint(0, classes, (count,))
        images = prototypes[labels] + 1.4 * torch.randn(
            count, channels, height, width
        )
        return images, labels

    source_train = sample(1200)
    source_test = sample(600)
    target_train = sample(700)
    target_test = sample(600)
    return source_train, source_test, target_train, target_test


def _accuracy(head: nn.Module, x: torch.Tensor, y: torch.Tensor, adapter=None) -> float:
    with torch.no_grad():
        if adapter is not None:
            x = adapter(x)
        predictions = head(x.flatten(1)).argmax(dim=1)
        return float((predictions == y).float().mean().item())


def _collect_response(
    adapter: TaskResponseSpatialAdapter,
    head: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 64,
) -> torch.Tensor:
    adapter.start_calibration(reset=True)
    for start in range(0, len(images), batch_size):
        adapter.zero_grad(set_to_none=True)
        batch = images[start : start + batch_size]
        target = labels[start : start + batch_size]
        loss = F.cross_entropy(head(adapter(batch).flatten(1)), target)
        loss.backward()
        adapter.accumulate_probe_gradient()
    return -(adapter.gradient_sum / float(adapter.gradient_samples.item())).flatten(1)


def _train_adapter_one_epoch(
    adapter: nn.Module,
    head: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    learning_rate: float = 0.03,
) -> None:
    parameters = [
        parameter
        for parameter in adapter.parameters()
        if parameter.requires_grad
        and not (
            hasattr(adapter, "probe_kernel") and parameter is adapter.probe_kernel
        )
    ]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
    order = torch.randperm(len(images))
    for start in range(0, len(order), 64):
        indices = order[start : start + 64]
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(adapter(images[indices]).flatten(1)), labels[indices])
        loss.backward()
        optimizer.step()


class _FullKernelAdapter(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.kernel = nn.Parameter(torch.zeros(channels, kernel_size, kernel_size))
        self.gate = nn.Parameter(torch.tensor(1.0))
        self.channels = channels
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = F.conv2d(
            x,
            self.kernel.unsqueeze(1),
            padding=self.kernel_size // 2,
            groups=self.channels,
        )
        return x + self.gate * delta


def _transfer_trial(seed: int) -> Dict[str, float]:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    channels, kernel, rank, classes = 6, 5, 2, 4
    (source_x, source_y), (source_test_x, source_test_y), (
        target_x,
        target_y,
    ), (target_test_x, target_test_y) = _make_transfer_data(seed)

    head = nn.Linear(channels * 12 * 12, classes)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.03, weight_decay=1e-4)
    for _ in range(40):
        order = torch.randperm(len(source_x))
        for start in range(0, len(order), 128):
            indices = order[start : start + 128]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(source_x[indices].flatten(1)), source_y[indices])
            loss.backward()
            optimizer.step()
    for parameter in head.parameters():
        parameter.requires_grad_(False)

    center = kernel // 2
    atom_h = torch.zeros(kernel, kernel)
    atom_h[center, kernel - 1] = 1.0
    atom_h[center, center] = -1.0
    atom_v = torch.zeros(kernel, kernel)
    atom_v[kernel - 1, center] = 1.0
    atom_v[center, center] = -1.0
    corruption_atoms = torch.stack((atom_h, atom_v))
    corruption_coefficients = torch.tensor(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.0, 1.0],
            [0.7, 0.3],
            [0.3, 0.7],
        ]
    )
    corruption = (
        corruption_coefficients @ corruption_atoms.flatten(1)
    ).reshape(channels, kernel, kernel)

    def corrupt(x: torch.Tensor) -> torch.Tensor:
        return x + F.conv2d(
            x, corruption.unsqueeze(1), padding=center, groups=channels
        )

    target_x = corrupt(target_x)
    target_test_x = corrupt(target_test_x)

    trso = TaskResponseSpatialAdapter(
        channels=channels,
        kernel_size=kernel,
        spatial_rank=rank,
        operator_radius=2.5,
        gate_init=1.0,
        basis_init_scale=0.25,
    )
    _collect_response(trso, head, target_x, target_y)
    trso.finalize_calibration(init_scale=0.25)

    random_adapter = TaskResponseSpatialAdapter(
        channels=channels,
        kernel_size=kernel,
        spatial_rank=rank,
        operator_radius=2.5,
        gate_init=1.0,
        basis_init_scale=0.25,
    )
    descending_response = _collect_response(
        random_adapter, head, target_x, target_y
    )
    random_q, _ = torch.linalg.qr(torch.randn(kernel * kernel, rank))
    random_atoms = random_q.T
    random_coefficients = descending_response @ random_atoms.T
    random_reconstruction = random_coefficients @ random_atoms
    random_coefficients = random_coefficients * (
        0.25 / random_reconstruction.norm().clamp_min(1e-8)
    )
    with torch.no_grad():
        random_adapter._copy_basis(random_atoms.reshape(rank, kernel, kernel))
        random_adapter._copy_coefficients(random_coefficients)
        random_adapter.calibrated_flag.fill_(True)
        random_adapter.stop_calibration()

    full = _FullKernelAdapter(channels, kernel)

    result = {
        "source": _accuracy(head, source_test_x, source_test_y),
        "corrupted": _accuracy(head, target_test_x, target_test_y),
        "trso_initial": _accuracy(head, target_test_x, target_test_y, trso),
        "random_initial": _accuracy(
            head, target_test_x, target_test_y, random_adapter
        ),
    }
    _train_adapter_one_epoch(trso, head, target_x, target_y)
    _train_adapter_one_epoch(random_adapter, head, target_x, target_y)
    _train_adapter_one_epoch(full, head, target_x, target_y)
    result.update(
        {
            "trso_one_epoch": _accuracy(
                head, target_test_x, target_test_y, trso
            ),
            "random_one_epoch": _accuracy(
                head, target_test_x, target_test_y, random_adapter
            ),
            "full_one_epoch": _accuracy(
                head, target_test_x, target_test_y, full
            ),
        }
    )
    return result


def run(transfer_seeds: int = 5) -> ScientificReport:
    torch.set_num_threads(1)
    alignment = _alignment_experiment()
    svd = _svd_experiment()
    channel = _channel_specific_experiment()
    budget = _budget_experiment()
    rank = _rank_preservation_experiment()
    transfer = [_transfer_trial(700 + seed) for seed in range(transfer_seeds)]

    def metric(name: str) -> List[float]:
        return [row[name] for row in transfer]

    trso_example = TaskResponseSpatialAdapter(
        channels=6, kernel_size=5, spatial_rank=2, basis_trainable=False
    )
    trso_parameters = trso_example.parameter_cost_for_rank(2)
    full_parameters = 6 * 5 * 5 + 1

    return ScientificReport(
        aligned_tangent_cosine_mean=alignment["aligned_mean"],
        aligned_tangent_cosine_min=alignment["aligned_min"],
        legacy_surrogate_cosine_mean=alignment["legacy_mean"],
        legacy_surrogate_cosine_median=alignment["legacy_median"],
        svd_optimal_relative_error=svd["svd_error"],
        best_random_subspace_relative_error=svd["random_error"],
        random_to_svd_error_ratio=svd["ratio"],
        channel_specific_output_mse=channel["specific_mse"],
        shared_kernel_output_mse=channel["shared_mse"],
        shared_to_specific_mse_ratio=channel["ratio"],
        exact_budget_value=float(budget["exact_value"]),
        greedy_budget_value=float(budget["greedy_value"]),
        exact_budget_selected=list(budget["selected"]),
        maximum_factorized_rank_during_training=rank["maximum_rank"],
        configured_factorized_rank=rank["configured_rank"],
        transfer_source_accuracy_mean=mean(metric("source")),
        transfer_corrupted_accuracy_mean=mean(metric("corrupted")),
        transfer_trso_initial_accuracy_mean=mean(metric("trso_initial")),
        transfer_random_initial_accuracy_mean=mean(metric("random_initial")),
        transfer_trso_one_epoch_accuracy_mean=mean(metric("trso_one_epoch")),
        transfer_random_one_epoch_accuracy_mean=mean(metric("random_one_epoch")),
        transfer_full_kernel_one_epoch_accuracy_mean=mean(metric("full_one_epoch")),
        transfer_trso_initial_accuracy_std=pstdev(metric("trso_initial")),
        transfer_random_initial_accuracy_std=pstdev(metric("random_initial")),
        trso_trainable_parameters=trso_parameters,
        random_trainable_parameters=trso_parameters,
        full_kernel_trainable_parameters=full_parameters,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transfer_seeds", type=int, default=5)
    parser.add_argument("--json", default="")
    args = parser.parse_args()
    report = run(transfer_seeds=args.transfer_seeds)
    payload = asdict(report)
    print(json.dumps(payload, indent=2))

    assert report.aligned_tangent_cosine_min > 0.999999
    assert report.aligned_tangent_cosine_mean > 0.999999
    assert report.random_to_svd_error_ratio > 1.05
    assert report.shared_kernel_output_mse > report.channel_specific_output_mse
    assert report.exact_budget_value > report.greedy_budget_value
    assert (
        report.maximum_factorized_rank_during_training
        <= report.configured_factorized_rank
    )
    assert (
        report.transfer_trso_initial_accuracy_mean
        > report.transfer_random_initial_accuracy_mean
    )
    assert (
        report.transfer_trso_one_epoch_accuracy_mean
        >= report.transfer_random_one_epoch_accuracy_mean
    )

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("[ALL OK] Scientific TRSO experiments passed")


if __name__ == "__main__":
    main()
