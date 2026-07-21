"""Scientific and implementation tests for aligned TRSO."""
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
            operator_radius=0.8,
            gate_init=1e-2,
        )
        defaults.update(kwargs)
        return TaskResponseSpatialAdapter(**defaults)

    @staticmethod
    def set_spectrum(adapter: TaskResponseSpatialAdapter, component_values):
        values = torch.as_tensor(component_values, dtype=adapter.singular_values.dtype)
        adapter.singular_values.zero_()
        adapter.singular_values[: values.numel()].copy_(values.sqrt())
        adapter.response_score.copy_(values.sum())

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
        self.assertTrue(torch.equal(y[:, :1], x[:, :1]))
        self.assertFalse(torch.equal(y[:, 1:], x[:, 1:]))

    def test_non_square_token_grid_can_be_explicit(self):
        x = torch.randn(2, 13, 8)
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

    def test_disabled_adapter_returns_input_without_layout_conversion(self):
        x = torch.randn(2, 17, 8)
        adapter = self.make()
        adapter.set_active_rank(0)
        y = adapter(x)
        self.assertIs(y, x)

    def test_each_channel_kernel_respects_l1_radius(self):
        adapter = self.make(operator_radius=0.35)
        with torch.no_grad():
            for vector in adapter.coefficient_vectors:
                vector.fill_(10.0)
        kernel_bank = adapter.projected_kernel_bank()
        per_channel_l1 = kernel_bank.abs().sum(dim=(1, 2))
        self.assertLessEqual(float(per_channel_l1.max().detach()), 0.350001)

    def test_fused_kernel_bank_matches_explicit_basis_sum(self):
        adapter = self.make(operator_radius=10.0)
        x = torch.randn(2, adapter.channels, 9, 9)
        explicit = adapter.explicit_basis_response(x)
        fused = adapter._depthwise(x, adapter.projected_kernel_bank())
        self.assertTrue(torch.allclose(explicit, fused, atol=2e-6, rtol=2e-5))

    def test_nonzero_gate_allows_coefficients_to_learn_on_first_step(self):
        adapter = self.make(gate_init=1e-2)
        x = torch.randn(2, 8, 9, 9)
        adapter(x).square().mean().backward()
        coefficient_grad = sum(
            float(parameter.grad.abs().sum()) for parameter in adapter.coefficient_vectors
        )
        self.assertGreater(coefficient_grad, 0.0)
        self.assertGreater(float(adapter.gate.grad.abs().sum()), 0.0)

    def test_zero_gate_has_expected_coefficient_gradient_starvation(self):
        adapter = self.make(gate_init=0.0)
        x = torch.randn(2, 8, 9, 9)
        adapter(x).square().mean().backward()
        coefficient_grad = sum(
            float(parameter.grad.abs().sum()) for parameter in adapter.coefficient_vectors
        )
        self.assertEqual(coefficient_grad, 0.0)
        self.assertGreaterEqual(float(adapter.gate.grad.abs().sum()), 0.0)

    def test_calibration_tangent_matches_actual_full_kernel_path(self):
        adapter = TaskResponseSpatialAdapter(
            channels=4,
            kernel_size=3,
            spatial_rank=2,
            operator_radius=10.0,
            gate_init=1.0,
            calibration_scale=1.0,
        )
        x = torch.randn(3, 4, 8, 8)
        target = torch.randn_like(x)

        adapter.start_calibration()
        F.mse_loss(adapter(x), target).backward()
        probe_gradient = adapter.probe_kernel.grad.detach().clone()
        adapter.stop_calibration()

        full_kernel = torch.zeros(4, 3, 3, requires_grad=True)
        direct = x + F.conv2d(x, full_kernel.unsqueeze(1), padding=1, groups=4)
        F.mse_loss(direct, target).backward()
        direct_gradient = full_kernel.grad.detach()

        cosine = F.cosine_similarity(probe_gradient.flatten(), direct_gradient.flatten(), dim=0)
        relative_error = (probe_gradient - direct_gradient).norm() / direct_gradient.norm()
        self.assertGreater(float(cosine), 0.999999)
        self.assertLess(float(relative_error), 1e-7)

    def test_autograd_response_recovers_rank_two_channel_kernel_bank(self):
        adapter = TaskResponseSpatialAdapter(
            channels=4,
            kernel_size=5,
            spatial_rank=2,
            operator_radius=10.0,
            gate_init=1.0,
        )
        atoms = torch.randn(2, 5, 5)
        channel_coefficients = torch.randn(4, 2)
        target_bank = (channel_coefficients @ atoms.flatten(1)).reshape(4, 5, 5)

        impulse = torch.zeros(8, 4, 11, 11)
        impulse[:, :, 5, 5] = 1.0
        desired = impulse + F.conv2d(
            impulse, target_bank.unsqueeze(1), padding=2, groups=4
        )

        adapter.start_calibration()
        F.mse_loss(adapter(impulse), desired).backward()
        adapter.accumulate_probe_gradient()
        score = adapter.finalize_calibration(init_scale=1.0)

        discovered = adapter.raw_kernel_bank().detach()
        cosine = F.cosine_similarity(discovered.flatten(), target_bank.flatten(), dim=0)
        flattened_rank = torch.linalg.matrix_rank(discovered.flatten(1))
        self.assertGreater(score, 0.0)
        self.assertGreater(float(cosine), 0.9999)
        self.assertLessEqual(int(flattened_rank), 2)

    def test_svd_finalization_is_best_rank_two_channel_spatial_approximation(self):
        adapter = self.make(channels=6, kernel_size=3, spatial_rank=2, operator_radius=10.0)
        target_flat = torch.randn(6, 3) @ torch.randn(3, 9)
        target_bank = target_flat.reshape(6, 3, 3)
        with torch.no_grad():
            adapter.gradient_sum.copy_(-target_bank)
            adapter.gradient_square_sum.copy_(target_bank.square())
            adapter.gradient_samples.fill_(1)
        adapter.finalize_calibration(init_scale=1.0)

        u, s, vh = torch.linalg.svd(target_flat, full_matrices=False)
        best = (u[:, :2] * s[:2]) @ vh[:2]
        reconstructed = adapter.raw_kernel_bank().flatten(1)
        rel_direction_error = (
            reconstructed / reconstructed.norm() - best / best.norm()
        ).norm()
        self.assertLess(float(rel_direction_error.detach()), 1e-5)

        optimal_error = (target_flat - best).norm()
        discovered_scaled = reconstructed * (
            torch.sum(reconstructed * target_flat) / reconstructed.square().sum()
        )
        discovered_error = (target_flat - discovered_scaled).norm()
        self.assertTrue(torch.allclose(discovered_error, optimal_error, atol=1e-5, rtol=1e-5))

    def test_trainable_factorization_preserves_channel_spatial_rank(self):
        adapter = self.make(
            channels=7,
            kernel_size=3,
            spatial_rank=2,
            basis_trainable=True,
            operator_radius=10.0,
            gate_init=1.0,
        )
        optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
        for _ in range(4):
            optimizer.zero_grad(set_to_none=True)
            x = torch.randn(2, 7, 8, 8)
            loss = adapter(x).square().mean()
            loss.backward()
            optimizer.step()
            rank = int(torch.linalg.matrix_rank(adapter.raw_kernel_bank().flatten(1)))
            self.assertLessEqual(rank, 2)

    def test_layer_selection_uses_captured_singular_energy(self):
        model = nn.Sequential(self.make(), self.make(), self.make())
        component_values = [[0.1, 0.1], [2.0, 1.0], [0.7, 0.3]]
        for (_, adapter), values in zip(iter_trso_adapters(model), component_values):
            self.set_spectrum(adapter, values)
        selected = select_trso_layers(model, max_adapters=2, keep_ratio=1.0)
        self.assertEqual(selected, ["1", "2"])
        self.assertEqual([model[i].active_rank for i in range(3)], [0, 2, 2])

    def test_exact_budget_selection_beats_greedy_density_counterexample(self):
        # Costs are C*r + 1: 6, 5, and 5. Under budget 10, selecting the two
        # smaller layers yields value 18, while density-greedy chooses value 12.
        model = nn.ModuleDict({
            "large_value": TaskResponseSpatialAdapter(channels=5, kernel_size=3, spatial_rank=1),
            "small_a": TaskResponseSpatialAdapter(channels=4, kernel_size=3, spatial_rank=1),
            "small_b": TaskResponseSpatialAdapter(channels=4, kernel_size=3, spatial_rank=1),
        })
        self.set_spectrum(model["large_value"], [12.0])
        self.set_spectrum(model["small_a"], [9.0])
        self.set_spectrum(model["small_b"], [9.0])

        selected = select_trso_layers(
            model,
            keep_ratio=1.0,
            max_adapters=2,
            parameter_budget=10,
        )
        self.assertEqual(set(selected), {"small_a", "small_b"})
        self.assertEqual(model["large_value"].active_rank, 0)

    def test_budget_allocator_can_choose_layer_specific_rank(self):
        model = nn.ModuleDict({
            "a": TaskResponseSpatialAdapter(channels=4, kernel_size=3, spatial_rank=2),
            "b": TaskResponseSpatialAdapter(channels=4, kernel_size=3, spatial_rank=2),
        })
        self.set_spectrum(model["a"], [8.0, 1.0])
        self.set_spectrum(model["b"], [7.0, 7.0])
        # Rank two costs 9; two rank-one adapters cost 10. Budget 9 therefore
        # tests whether the allocator chooses b at rank two (value 14).
        selected = select_trso_layers(model, parameter_budget=9, max_adapters=2)
        self.assertEqual(selected, ["b"])
        self.assertEqual(model["a"].active_rank, 0)
        self.assertEqual(model["b"].active_rank, 2)

    def test_json_config_roundtrip(self):
        model_a = nn.Sequential(self.make(), self.make())
        for index, (_, adapter) in enumerate(iter_trso_adapters(model_a)):
            with torch.no_grad():
                adapter._copy_basis(torch.randn_like(adapter.basis_atoms))
                adapter._copy_coefficients(torch.randn_like(adapter.coefficients))
                self.set_spectrum(adapter, [index + 0.5, 0.25])
                adapter.calibrated_flag.fill_(True)
        model_a[0].set_active_rank(1)
        model_a[1].set_active_rank(0)

        model_b = nn.Sequential(self.make(), self.make())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trso.json"
            save_trso_config(model_a, path, selected_layers=["0"])
            payload = json.loads(path.read_text())
            self.assertEqual(payload["method"], "Scientific-TRSO")
            selected = load_trso_config(model_b, path)
        self.assertEqual(selected, ["0"])
        self.assertTrue(torch.allclose(model_a[0].basis_atoms, model_b[0].basis_atoms))
        self.assertTrue(torch.allclose(model_a[0].coefficients, model_b[0].coefficients))
        self.assertEqual(model_b[0].active_rank, 1)
        self.assertEqual(model_b[1].active_rank, 0)

    def test_parameter_breakdown_matches_active_factorization(self):
        adapter = self.make(channels=8, spatial_rank=2, basis_trainable=False)
        adapter.set_active_rank(1)
        breakdown = adapter.parameter_count_breakdown()
        self.assertEqual(breakdown["coefficients"], 8)
        self.assertEqual(breakdown["basis"], 0)
        self.assertEqual(breakdown["total_trainable"], 9)
        actual = sum(p.numel() for p in adapter.parameters() if p.requires_grad and p is not adapter.probe_kernel)
        self.assertEqual(actual, 9)


if __name__ == "__main__":
    unittest.main()
