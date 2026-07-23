import torch
import torch.nn.functional as F

from models.task_response_adapter import TaskResponseSpatialAdapter


def _calibrate(adapter, x, target, init_scale=0.2):
    adapter.start_calibration()
    F.mse_loss(adapter(x), target).backward()
    adapter.accumulate_probe_gradient()
    adapter.finalize_calibration(init_scale=init_scale)


def test_v2_grouped_dual_response_uses_far_fewer_parameters_and_better_initial_fit():
    torch.manual_seed(7)
    channels = 64
    x = torch.randn(16, channels, 8, 8)
    atoms = torch.randn(2, 3, 3)
    coefficients = torch.randn(channels, 2) * 0.03
    kernel = (coefficients @ atoms.flatten(1)).reshape(channels, 3, 3)
    channel_scale = torch.randn(channels) * 0.08
    target = (
        x
        + F.conv2d(x, kernel.unsqueeze(1), padding=1, groups=channels)
        + x * channel_scale.view(1, channels, 1, 1)
    )

    v1 = TaskResponseSpatialAdapter(
        channels=channels, kernel_size=3, spatial_rank=2, variant="v1",
        gate_init=1.0, operator_radius=10.0, layout="bchw",
    )
    v2 = TaskResponseSpatialAdapter(
        channels=channels, kernel_size=3, spatial_rank=2, variant="v2",
        coefficient_mode="grouped", channel_groups=8, input_norm="none",
        channel_response=True, gate_init=1.0, operator_radius=10.0, layout="bchw",
    )
    _calibrate(v1, x, target)
    _calibrate(v2, x, target)

    assert v2.parameter_count_breakdown()["active_trainable"] <= 0.2 * v1.parameter_count_breakdown()["active_trainable"]
    assert F.mse_loss(v2(x), target) < F.mse_loss(v1(x), target)


def test_v2_prefix_coupling_updates_class_token_with_only_one_extra_parameter():
    torch.manual_seed(11)
    adapter = TaskResponseSpatialAdapter(
        channels=16, kernel_size=3, spatial_rank=1, variant="v2",
        coefficient_mode="grouped", channel_groups=4, input_norm="rms",
        channel_response=True, prefix_coupling=True, gate_init=1.0,
        layout="bnc", grid_size=(2, 2),
    )
    tokens = torch.randn(8, 5, 16)
    target = tokens.clone()
    target[:, 0] = target[:, 0] + tokens[:, 1:].mean(dim=1)
    _calibrate(adapter, tokens, target, init_scale=0.1)
    output = adapter(tokens)

    assert not torch.allclose(output[:, 0], tokens[:, 0])
    assert adapter.parameter_count_breakdown()["prefix_gate"] == 1


def test_v2_gate_one_avoids_original_gradient_suppression():
    torch.manual_seed(19)
    x = torch.randn(4, 32, 8, 8)
    weak = TaskResponseSpatialAdapter(
        channels=32, kernel_size=3, spatial_rank=2, variant="v1", gate_init=1e-2,
    )
    strong = TaskResponseSpatialAdapter(
        channels=32, kernel_size=3, spatial_rank=2, variant="v2", gate_init=1.0,
        coefficient_mode="full", input_norm="none", channel_response=False,
    )
    strong._copy_coefficients(weak.coefficients.detach().clone())
    strong._copy_basis(weak.basis_atoms.detach().clone())
    weak(x).square().mean().backward()
    strong(x).square().mean().backward()
    weak_grad = sum(float(p.grad.abs().sum()) for p in weak.coefficient_vectors)
    strong_grad = sum(float(p.grad.abs().sum()) for p in strong.coefficient_vectors)
    assert strong_grad > 50.0 * weak_grad


def test_v2_snr_per_parameter_is_finite_after_calibration():
    torch.manual_seed(23)
    adapter = TaskResponseSpatialAdapter(
        channels=16, kernel_size=3, spatial_rank=2, variant="v2",
        coefficient_mode="grouped", channel_groups=4, gate_init=1.0,
    )
    x = torch.randn(8, 16, 6, 6)
    target = torch.randn_like(x)
    _calibrate(adapter, x, target)
    score = adapter.rank_value(2, score_mode="snr_per_param")
    assert score >= 0.0
    assert torch.isfinite(torch.tensor(score))
