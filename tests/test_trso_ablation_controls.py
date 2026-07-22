from __future__ import annotations

import copy

import torch
import torch.nn as nn

from models.task_response_adapter import TaskResponseSpatialAdapter, select_trso_layers


class AdapterBank(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = TaskResponseSpatialAdapter(channels=4, kernel_size=3, spatial_rank=2)
        self.b = TaskResponseSpatialAdapter(channels=8, kernel_size=3, spatial_rank=2)
        self.c = TaskResponseSpatialAdapter(channels=12, kernel_size=3, spatial_rank=2)


def _set_values(model):
    model.a.singular_values.copy_(torch.tensor([4.0, 1.0]))
    model.b.singular_values.copy_(torch.tensor([3.0, 2.0]))
    model.c.singular_values.copy_(torch.tensor([2.5, 2.4]))


def test_allocation_modes_obey_budget_and_keep_limit():
    for mode in ("exact", "greedy", "uniform"):
        model = AdapterBank()
        _set_values(model)
        selected = select_trso_layers(
            model,
            parameter_budget=20,
            max_adapters=2,
            allocation=mode,
            score_mode="energy",
        )
        active = [adapter for adapter in (model.a, model.b, model.c) if adapter.enabled]
        assert len(selected) <= 2
        assert sum(adapter.parameter_cost_for_rank(adapter.active_rank) for adapter in active) <= 20


def test_response_random_and_dct_basis_are_controlled_and_distinct():
    torch.manual_seed(0)
    base = TaskResponseSpatialAdapter(channels=6, kernel_size=3, spatial_rank=2)
    base.gradient_samples.fill_(3)
    base.gradient_sum.copy_(torch.randn_like(base.gradient_sum))
    base.gradient_square_sum.copy_(base.gradient_sum.square() / 3 + 0.1)

    response = copy.deepcopy(base)
    random = copy.deepcopy(base)
    dct = copy.deepcopy(base)
    response.finalize_calibration(basis_source="response", random_seed=1)
    random.finalize_calibration(basis_source="random", random_seed=1)
    dct.finalize_calibration(basis_source="dct", random_seed=1)

    random_atoms = random.basis_atoms.flatten(1)
    assert torch.allclose(random_atoms @ random_atoms.T, torch.eye(2), atol=1e-5)
    assert not torch.allclose(response.basis_atoms, random.basis_atoms)
    assert not torch.allclose(response.basis_atoms, dct.basis_atoms)


def test_score_modes_are_finite_and_noise_penalty_is_monotone():
    adapter = TaskResponseSpatialAdapter(channels=8, kernel_size=3, spatial_rank=2)
    adapter.singular_values.copy_(torch.tensor([3.0, 2.0]))
    adapter.response_noise.fill_(5.0)
    assert adapter.rank_value(2, "energy") > 0
    assert adapter.rank_value(2, "energy_per_param") > 0
    assert adapter.rank_value(2, "energy_per_channel") > 0
    low = adapter.rank_value(2, "noise_adjusted", noise_beta=0.1)
    high = adapter.rank_value(2, "noise_adjusted", noise_beta=1.0)
    assert high <= low
