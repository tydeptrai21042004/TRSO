"""Focused mathematical and implementation tests for TRSO."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.task_response_adapter import (
    TaskResponseSpatialAdapter,
    iter_trso_adapters,
    load_trso_config,
    save_trso_config,
    select_trso_layers,
)


class TRSOAdapterTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def make(self, **kwargs):
        defaults = dict(
            channels=8,
            kernel_size=5,
            spatial_rank=2,
            channel_ratio=4,
            operator_radius=0.8,
            gate_init=1e-2,
        )
        defaults.update(kwargs)
        return TaskResponseSpatialAdapter(**defaults)

    def test_cnn_shape_and_exact_identity_at_zero_gate(self):
        x = torch.randn(2, 8, 11, 13)
        adapter = self.make(gate_init=0.0)
        y = adapter(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertTrue(torch.equal(y, x))

    def test_transformer_tokens_without_class_token(self):
        x = torch.randn(2, 16, 8)
        adapter = self.make()
        y = adapter(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertFalse(torch.equal(y, x))

    def test_transformer_class_token_is_preserved_exactly(self):
        x = torch.randn(2, 17, 8)
        adapter = self.make()
        y = adapter(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertTrue(torch.equal(y[:, :1], x[:, :1]))
        self.assertFalse(torch.equal(y[:, 1:], x[:, 1:]))

    def test_non_square_token_grid_can_be_explicit(self):
        x = torch.randn(2, 13, 8)  # one prefix token + 3x4 patch grid
        adapter = self.make(grid_size=(3, 4))
        y = adapter(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertTrue(torch.equal(y[:, :1], x[:, :1]))

    def test_invalid_token_grid_raises_clear_error(self):
        adapter = self.make()
        with self.assertRaisesRegex(ValueError, "Cannot infer a square patch grid"):
            adapter(torch.randn(2, 14, 8))

    def test_channels_last_swin_layout(self):
        x = torch.randn(2, 6, 7, 8)
        adapter = self.make(layout="bhwc")
        y = adapter(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))

    def test_projected_kernel_respects_l1_radius(self):
        adapter = self.make(operator_radius=0.35)
        with torch.no_grad():
            adapter.coefficients.fill_(10.0)
        kernel = adapter.projected_fused_kernel()
        self.assertLessEqual(float(kernel.abs().sum().detach()), 0.350001)

    def test_fused_kernel_matches_explicit_basis_sum(self):
        adapter = self.make(operator_radius=10.0)
        z = torch.randn(2, adapter.hidden_channels, 9, 9)
        explicit = adapter.explicit_basis_response(z)
        fused = adapter._shared_depthwise(z, adapter.projected_fused_kernel())
        self.assertTrue(torch.allclose(explicit, fused, atol=2e-6, rtol=2e-5))

    def test_nonzero_gate_allows_internal_gradients_on_first_step(self):
        adapter = self.make(gate_init=1e-2)
        x = torch.randn(2, 8, 9, 9)
        loss = adapter(x).square().mean()
        loss.backward()
        self.assertGreater(float(adapter.down.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(adapter.up.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(adapter.coefficients.grad.abs().sum()), 0.0)
        self.assertGreater(float(adapter.gate.grad.abs().sum()), 0.0)

    def test_zero_gate_exposes_expected_gradient_starvation(self):
        adapter = self.make(gate_init=0.0)
        x = torch.randn(2, 8, 9, 9)
        adapter(x).square().mean().backward()
        self.assertEqual(float(adapter.down.weight.grad.abs().sum()), 0.0)
        self.assertEqual(float(adapter.coefficients.grad.abs().sum()), 0.0)
        self.assertGreaterEqual(float(adapter.gate.grad.abs().sum()), 0.0)

    def test_autograd_task_response_recovers_synthetic_operator_direction(self):
        adapter = TaskResponseSpatialAdapter(
            channels=1,
            kernel_size=5,
            spatial_rank=2,
            channel_ratio=1,
            gate_init=1e-2,
        )
        target = torch.tensor(
            [
                [0.0, 0.0, 0.10, 0.0, 0.0],
                [0.0, 0.20, 0.0, -0.10, 0.0],
                [0.10, 0.0, 0.35, 0.0, 0.10],
                [0.0, -0.10, 0.0, 0.20, 0.0],
                [0.0, 0.0, 0.10, 0.0, 0.0],
            ]
        )
        impulse = torch.zeros(8, 1, 11, 11)
        impulse[:, :, 5, 5] = 1.0
        weight = target.view(1, 1, 5, 5)
        desired = impulse + F.conv2d(impulse, weight, padding=2)

        adapter.start_calibration()
        loss = F.mse_loss(adapter(impulse), desired)
        loss.backward()
        adapter.accumulate_probe_gradient()
        score = adapter.finalize_calibration(init_scale=1.0)
        discovered = adapter.raw_fused_kernel().detach().flatten()
        cosine = F.cosine_similarity(discovered, target.flatten(), dim=0)
        self.assertGreater(score, 0.0)
        self.assertGreater(float(cosine), 0.97)

    def test_direct_svd_finalization_matches_best_rank_two_approximation(self):
        adapter = self.make()
        target = torch.randn(5, 2) @ torch.randn(2, 5)
        with torch.no_grad():
            adapter.gradient_sum.copy_(-target)
            adapter.gradient_samples.fill_(1)
        adapter.finalize_calibration(init_scale=1.0)
        u, s, vh = torch.linalg.svd(target, full_matrices=False)
        best = (u[:, :2] * s[:2]) @ vh[:2]
        reconstructed = adapter.raw_fused_kernel()
        rel_error = (reconstructed / reconstructed.norm() - best / best.norm()).norm()
        self.assertLess(float(rel_error.detach()), 1e-5)

    def test_layer_selection_uses_response_score(self):
        model = nn.Sequential(self.make(), self.make(), self.make())
        scores = [0.2, 3.0, 1.0]
        for (_, adapter), score in zip(iter_trso_adapters(model), scores):
            adapter.response_score.fill_(score)
        selected = select_trso_layers(model, max_adapters=2, keep_ratio=1.0)
        self.assertEqual(selected, ["1", "2"])
        enabled = {name: adapter.enabled for name, adapter in iter_trso_adapters(model)}
        self.assertEqual(enabled, {"0": False, "1": True, "2": True})

    def test_parameter_budget_uses_response_per_cost(self):
        model = nn.ModuleDict({
            "small": TaskResponseSpatialAdapter(channels=8, channel_ratio=4),
            "large": TaskResponseSpatialAdapter(channels=64, channel_ratio=4),
        })
        model["small"].response_score.fill_(2.0)
        model["large"].response_score.fill_(3.0)
        small_cost = model["small"].parameter_count_breakdown()["total_trainable"]
        selected = select_trso_layers(
            model,
            keep_ratio=1.0,
            max_adapters=2,
            parameter_budget=small_cost,
        )
        self.assertEqual(selected, ["small"])
        self.assertTrue(model["small"].enabled)
        self.assertFalse(model["large"].enabled)

    def test_json_config_roundtrip(self):
        model_a = nn.Sequential(self.make(), self.make())
        for index, (_, adapter) in enumerate(iter_trso_adapters(model_a)):
            with torch.no_grad():
                adapter.basis_atoms.copy_(torch.randn_like(adapter.basis_atoms))
                adapter.coefficients.copy_(torch.randn_like(adapter.coefficients))
                adapter.response_score.fill_(index + 0.5)
                adapter.calibrated_flag.fill_(True)
        model_a[1].set_enabled(False)

        model_b = nn.Sequential(self.make(), self.make())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trso.json"
            save_trso_config(model_a, path, selected_layers=["0"])
            payload = json.loads(path.read_text())
            self.assertEqual(payload["method"], "TRSO")
            selected = load_trso_config(model_b, path)
        self.assertEqual(selected, ["0"])
        self.assertTrue(torch.allclose(model_a[0].basis_atoms, model_b[0].basis_atoms))
        self.assertTrue(torch.allclose(model_a[0].coefficients, model_b[0].coefficients))
        self.assertTrue(model_b[0].enabled)
        self.assertFalse(model_b[1].enabled)

    def test_parameter_breakdown_excludes_fixed_basis(self):
        adapter = self.make(basis_trainable=False)
        breakdown = adapter.parameter_count_breakdown()
        self.assertEqual(breakdown["basis"], 0)
        self.assertEqual(breakdown["total_trainable"], sum(
            p.numel() for name, p in adapter.named_parameters()
            if p.requires_grad and name != "probe_kernel"
        ))


if __name__ == "__main__":
    unittest.main()
