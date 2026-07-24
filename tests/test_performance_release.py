from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from engine import train_one_epoch
from main import (
    apply_trso_residual_budget,
    build_optimizer_parameter_groups,
    resolve_trso_budget_and_sparsity,
)
from models.task_response_adapter import TaskResponseSpatialAdapter


def _adapter(channels: int = 8) -> TaskResponseSpatialAdapter:
    return TaskResponseSpatialAdapter(
        channels=channels,
        kernel_size=3,
        spatial_rank=2,
        variant="v3",
        channel_groups=2,
        channel_response=False,
        calibration_grad_norm="global_rms",
        residual_norm="rms",
    )


def test_global_rms_calibration_preserves_cross_layer_magnitude_and_loss_scale():
    first, second = _adapter(), _adapter()
    first.start_calibration(reset=True)
    second.start_calibration(reset=True)
    first.probe_kernel.grad = torch.ones_like(first.probe_kernel)
    second.probe_kernel.grad = 2.0 * torch.ones_like(second.probe_kernel)

    squared = first.probe_kernel.grad.square().sum() + second.probe_kernel.grad.square().sum()
    count = first.probe_kernel.numel() + second.probe_kernel.numel()
    first.accumulate_probe_gradient(global_squared_norm=squared, global_element_count=count)
    second.accumulate_probe_gradient(global_squared_norm=squared, global_element_count=count)

    ratio = second.gradient_sum.abs().mean() / first.gradient_sum.abs().mean()
    assert float(ratio) == pytest.approx(2.0, rel=1e-6)

    scaled_first, scaled_second = _adapter(), _adapter()
    scaled_first.start_calibration(reset=True)
    scaled_second.start_calibration(reset=True)
    scaled_first.probe_kernel.grad = 100.0 * torch.ones_like(scaled_first.probe_kernel)
    scaled_second.probe_kernel.grad = 200.0 * torch.ones_like(scaled_second.probe_kernel)
    scaled_squared = (
        scaled_first.probe_kernel.grad.square().sum()
        + scaled_second.probe_kernel.grad.square().sum()
    )
    scaled_first.accumulate_probe_gradient(
        global_squared_norm=scaled_squared, global_element_count=count
    )
    scaled_second.accumulate_probe_gradient(
        global_squared_norm=scaled_squared, global_element_count=count
    )
    assert torch.allclose(first.gradient_sum, scaled_first.gradient_sum, atol=1e-6, rtol=1e-6)
    assert torch.allclose(second.gradient_sum, scaled_second.gradient_sum, atol=1e-6, rtol=1e-6)


def test_capacity_budget_and_auto_sparse_limit_do_not_select_every_candidate():
    model = nn.Module()
    model.adapters = nn.ModuleList([
        TaskResponseSpatialAdapter(
            channels=16,
            kernel_size=5,
            spatial_rank=4,
            variant="v3",
            channel_groups=4,
            basis_trainable=True,
        )
        for _ in range(16)
    ])
    adapters = [(f"adapters.{index}", adapter) for index, adapter in enumerate(model.adapters)]
    args = SimpleNamespace(
        trso_parameter_budget=0,
        trso_max_adapters=0,
        trso_variant="v3",
        trso_auto_sparse=True,
        trso_auto_budget_ratio=0.35,
        trso_auto_budget_min=1,
        trso_auto_budget_max=4096,
    )
    budget, max_adapters = resolve_trso_budget_and_sparsity(model, adapters, args)
    total_capacity = sum(a.parameter_cost_for_rank(a.spatial_rank) for _, a in adapters)
    assert 0 < budget < total_capacity
    assert 1 < max_adapters < len(adapters)


def test_global_residual_budget_scales_with_selected_layer_count():
    adapters = [(f"a{index}", _adapter()) for index in range(6)]
    args = SimpleNamespace(trso_residual_target=0.04, trso_residual_budget_mode="global")
    selected = [name for name, _ in adapters[:4]]
    per_adapter = apply_trso_residual_budget(adapters, selected, args)
    assert per_adapter == pytest.approx(0.02)
    for name, adapter in adapters:
        if name in selected:
            assert adapter.residual_target == pytest.approx(0.02)


class _GroupedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.pet_adapter = nn.Module()
        self.pet_adapter.coefficients = nn.Parameter(torch.ones(4, 2))
        self.pet_adapter.gate = nn.Parameter(torch.tensor(1.0))
        self.head = nn.Linear(4, 3)


def test_optimizer_groups_use_reduced_head_lr_and_no_decay_for_scalars():
    model = _GroupedModel()
    args = SimpleNamespace(
        tuning_method="trso",
        peft_head_lr_scale=0.5,
        no_decay_bias_norm=True,
        weight_decay=1e-4,
        weight_decay_adapter=1e-4,
        lr=1e-3,
    )
    groups = build_optimizer_parameter_groups(model, args)
    head_groups = [group for group in groups if group["group_name"].startswith("head_")]
    assert head_groups and all(group["lr_scale"] == pytest.approx(0.5) for group in head_groups)
    gate_id = id(model.pet_adapter.gate)
    gate_group = next(group for group in groups if any(id(p) == gate_id for p in group["params"]))
    assert gate_group["weight_decay"] == 0.0


def test_training_scheduler_respects_parameter_group_lr_scale():
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD([
        {"params": [model.weight], "lr": 0.1, "lr_scale": 1.0},
        {"params": [model.bias], "lr": 0.05, "lr_scale": 0.5},
    ])
    loader = DataLoader(
        TensorDataset(torch.randn(4, 4), torch.tensor([0, 1, 0, 1])),
        batch_size=4,
    )
    train_one_epoch(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        data_loader=loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        epoch=0,
        loss_scaler=None,
        start_steps=0,
        lr_schedule_values=[0.02],
        wd_schedule_values=None,
        num_training_steps_per_epoch=1,
        update_freq=1,
        use_amp=False,
        task_type="single_label",
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.02)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(0.01)
