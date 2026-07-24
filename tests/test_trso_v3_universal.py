from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from main import _attach_hook_adapters
from models.task_response_adapter import TaskResponseSpatialAdapter
from tools.run_fair_suite import method_variant


def _calibrate_once(adapter: TaskResponseSpatialAdapter, x: torch.Tensor, loss_scale: float = 1.0) -> None:
    adapter.start_calibration(reset=True)
    y = adapter(x)
    loss = loss_scale * y.square().mean()
    loss.backward()
    adapter.accumulate_probe_gradient()
    adapter.finalize_calibration(init_scale=0.05, basis_source="response")


def test_v3_automatic_groups_scale_with_width_but_remain_small():
    tiny = TaskResponseSpatialAdapter(channels=8, kernel_size=3, spatial_rank=1, variant="v3", channel_groups=0)
    medium = TaskResponseSpatialAdapter(channels=64, kernel_size=3, spatial_rank=1, variant="v3", channel_groups=0)
    wide = TaskResponseSpatialAdapter(channels=768, kernel_size=3, spatial_rank=1, variant="v3", channel_groups=0)
    assert 1 <= tiny.channel_groups <= 8
    assert medium.channel_groups == 8
    assert wide.channel_groups == 16
    assert wide.parameter_cost_for_rank(1) < 40


def test_v3_calibration_is_invariant_to_positive_task_loss_scaling():
    torch.manual_seed(7)
    x = torch.randn(4, 8, 5, 5)
    a = TaskResponseSpatialAdapter(
        channels=8, kernel_size=3, spatial_rank=2, variant="v3",
        channel_groups=4, calibration_grad_norm="rms", input_norm="rms",
    )
    b = TaskResponseSpatialAdapter(
        channels=8, kernel_size=3, spatial_rank=2, variant="v3",
        channel_groups=4, calibration_grad_norm="rms", input_norm="rms",
    )
    b.load_state_dict(a.state_dict(), strict=True)
    _calibrate_once(a, x, loss_scale=1.0)
    _calibrate_once(b, x, loss_scale=1000.0)
    assert torch.allclose(a.basis_atoms, b.basis_atoms, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        a.response_coefficient_directions,
        b.response_coefficient_directions,
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.equal(a.channel_group_index, b.channel_group_index)


def test_v3_response_grouping_is_data_driven_and_balanced():
    adapter = TaskResponseSpatialAdapter(
        channels=8, kernel_size=3, spatial_rank=1, variant="v3",
        channel_groups=2, grouping_mode="response", channel_response=False,
        calibration_grad_norm="none",
    )
    adapter.start_calibration(reset=True)
    pattern = torch.zeros_like(adapter.gradient_sum)
    pattern[0::2, 1, 1] = 1.0
    pattern[1::2, 1, 1] = -1.0
    adapter.gradient_sum.copy_(pattern)
    adapter.gradient_square_sum.copy_(pattern.square())
    adapter.gradient_samples.fill_(1)
    adapter.finalize_calibration(init_scale=0.05)
    contiguous = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    assert not torch.equal(adapter.channel_group_index.cpu(), contiguous)
    counts = torch.bincount(adapter.channel_group_index.cpu(), minlength=2)
    assert counts.tolist() == [4, 4]
    assert adapter.channel_group_index[0] == adapter.channel_group_index[2]
    assert adapter.channel_group_index[1] == adapter.channel_group_index[3]
    assert adapter.channel_group_index[0] != adapter.channel_group_index[1]


def test_v3_residual_target_is_feature_scale_invariant():
    torch.manual_seed(3)
    adapter = TaskResponseSpatialAdapter(
        channels=8, kernel_size=3, spatial_rank=1, variant="v3",
        channel_groups=4, input_norm="rms", residual_norm="rms",
        residual_target=0.05, channel_response=False, gate_init=1.0,
    )
    adapter.set_active_rank(1)
    x = torch.randn(3, 8, 7, 7)
    ratios = []
    for scale in (0.01, 1.0, 100.0):
        z = x * scale
        update = adapter(z) - z
        ratios.append(float((update.square().mean().sqrt() / z.square().mean().sqrt()).detach()))
    for ratio in ratios:
        assert abs(ratio - 0.05) < 2e-3
    assert max(ratios) - min(ratios) < 2e-3


def test_v3_rectangular_grid_and_multiple_prefix_tokens_are_coupled():
    torch.manual_seed(11)
    adapter = TaskResponseSpatialAdapter(
        channels=6, kernel_size=3, spatial_rank=1, variant="v3",
        channel_groups=2, layout="bnc", grid_size=(2, 3),
        prefix_coupling=True, prefix_coupling_mode="all",
        residual_norm="rms", residual_target=0.05, gate_init=1.0,
    )
    adapter.set_active_rank(1)
    x = torch.randn(2, 8, 6)  # two prefix tokens + 2x3 patches
    y = adapter(x)
    assert y.shape == x.shape
    first_change = y[:, 0] - x[:, 0]
    second_change = y[:, 1] - x[:, 1]
    assert first_change.abs().sum() > 0
    assert second_change.abs().sum() > 0
    assert torch.allclose(first_change, second_change, atol=1e-6, rtol=1e-5)


def test_generic_cnn_fallback_attaches_when_no_named_block_contract_exists():
    class PlainCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
                nn.Conv2d(8, 12, 3, padding=1), nn.ReLU(),
            )

        def forward(self, x):
            return self.features(x)

    model = PlainCNN()
    args = SimpleNamespace(
        tuning_method="trso", trso_generic_fallback=True,
        trso_generic_max_candidates=8, adapt_scale=1.0,
    )

    def make_adapter(ch, layout="auto", grid_size=None):
        return TaskResponseSpatialAdapter(
            channels=ch, kernel_size=3, spatial_rank=1, variant="v3",
            channel_groups=0, layout=layout, grid_size=grid_size,
        )

    count = _attach_hook_adapters(model, args, make_adapter)
    assert count == 2
    y = model(torch.randn(2, 3, 8, 8))
    assert y.shape == (2, 12, 8, 8)


def test_fair_suite_uses_v3_universal_defaults():
    args = SimpleNamespace(
        trso_budget=0,
        trso_calibration_batches=0,
    )
    row = method_variant("trso", "cnn", 0, None, args)
    assert row["trso_variant"] == "v3"
    assert row["trso_channel_groups"] == 0
    assert row["trso_grouping_mode"] == "response"
    assert row["trso_calibration_grad_norm"] == "global_rms"
    assert row["trso_residual_norm"] == "rms"
    assert row["trso_score_mode"] == "normalized_stable_energy_per_param"
    assert row["trso_spatial_rank"] == 4
    assert row["trso_basis_trainable"] is True
    assert row["trso_residual_budget_mode"] == "global"


def test_generic_transformer_fallback_handles_square_tokens_with_prefix():
    class TokenBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = nn.Identity()

        def forward(self, x):
            return x + 0.1 * self.norm1(x)

    class PlainTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([TokenBlock(8), TokenBlock(8)])

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return x

    model = PlainTransformer()
    args = SimpleNamespace(
        tuning_method="trso", trso_generic_fallback=True,
        trso_generic_max_candidates=8, adapt_scale=1.0,
    )

    def make_adapter(ch, layout="auto", grid_size=None):
        return TaskResponseSpatialAdapter(
            channels=ch, kernel_size=3, spatial_rank=1, variant="v3",
            channel_groups=0, layout=layout, grid_size=grid_size,
        )

    count = _attach_hook_adapters(model, args, make_adapter)
    assert count == 2
    tokens = torch.randn(2, 5, 8)  # one prefix + 2x2 patches
    output = model(tokens)
    assert output.shape == tokens.shape
