"""TRSO integration tests for CNNs, Transformers, calibration, and hooks."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from main import _attach_hook_adapters, calibrate_trso_model
from models.task_response_adapter import TaskResponseSpatialAdapter, iter_trso_adapters


class TinyTRSOClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, padding=1)
        self.adapter_a = TaskResponseSpatialAdapter(channels=8, channel_ratio=4, kernel_size=5, spatial_rank=2)
        self.block = nn.Conv2d(8, 8, 3, padding=1)
        self.adapter_b = TaskResponseSpatialAdapter(channels=8, channel_ratio=4, kernel_size=5, spatial_rank=2)
        self.head = nn.Linear(8, 3)

    def forward(self, x):
        x = torch.relu(self.stem(x))
        x = self.adapter_a(x)
        x = torch.relu(self.block(x))
        x = self.adapter_b(x)
        return self.head(x.mean(dim=(2, 3)))


class TRSOIntegrationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)

    @staticmethod
    def args(tmpdir):
        return SimpleNamespace(
            trso_config="",
            trso_calibration=True,
            trso_calibration_batches=3,
            trso_head_warmup_steps=1,
            trso_basis_init_scale=0.05,
            trso_max_adapters=1,
            trso_parameter_budget=0,
            trso_keep_ratio=1.0,
            trso_save_config=str(Path(tmpdir) / "calibration.json"),
            output_dir=str(tmpdir),
            lr=1e-3,
            weight_decay=0.0,
        )

    def test_end_to_end_calibration_selects_and_saves_one_layer(self):
        model = TinyTRSOClassifier()
        images = torch.randn(12, 3, 12, 12)
        labels = torch.randint(0, 3, (12,))
        loader = DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=False)
        with tempfile.TemporaryDirectory() as tmp:
            selected = calibrate_trso_model(model, loader, torch.device("cpu"), self.args(tmp))
            self.assertEqual(len(selected), 1)
            self.assertTrue((Path(tmp) / "calibration.json").exists())
        adapters = list(iter_trso_adapters(model))
        self.assertTrue(all(adapter.calibrated for _, adapter in adapters))
        self.assertEqual(sum(adapter.enabled for _, adapter in adapters), 1)
        self.assertTrue(all(not adapter.probe_kernel.requires_grad for _, adapter in adapters))

    def test_resnet_hook_insertion_and_forward(self):
        from torchvision.models import resnet18

        model = resnet18(weights=None, num_classes=4)
        args = SimpleNamespace(tuning_method="trso", adapt_scale=1.0)
        _attach_hook_adapters(
            model,
            args,
            lambda ch, layout="auto", grid_size=None: TaskResponseSpatialAdapter(
                channels=ch, channel_ratio=16, layout=layout, grid_size=grid_size
            ),
        )
        model.eval()
        adapters = list(iter_trso_adapters(model))
        self.assertEqual(len(adapters), 8)
        y = model(torch.randn(2, 3, 64, 64))
        self.assertEqual(tuple(y.shape), (2, 4))
        self.assertTrue(all(adapter.layout == "bchw" for _, adapter in adapters))

    def test_torchvision_vit_hook_insertion_and_forward(self):
        from torchvision.models.vision_transformer import VisionTransformer

        model = VisionTransformer(
            image_size=32,
            patch_size=8,
            num_layers=2,
            num_heads=2,
            hidden_dim=32,
            mlp_dim=64,
            num_classes=4,
        )
        args = SimpleNamespace(tuning_method="trso", adapt_scale=1.0)
        _attach_hook_adapters(
            model,
            args,
            lambda ch, layout="auto", grid_size=None: TaskResponseSpatialAdapter(
                channels=ch, channel_ratio=8, layout=layout, grid_size=grid_size
            ),
        )
        model.eval()
        adapters = list(iter_trso_adapters(model))
        self.assertEqual(len(adapters), 2)
        y = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(y.shape), (2, 4))
        self.assertTrue(all(adapter.layout == "bnc" for _, adapter in adapters))

    def test_single_training_step_updates_selected_adapter(self):
        model = TinyTRSOClassifier()
        model.adapter_b.set_enabled(False)
        before = model.adapter_a.coefficients.detach().clone()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-2, weight_decay=0.0
        )
        x = torch.randn(4, 3, 12, 12)
        labels = torch.randint(0, 3, (4,))
        optimizer.zero_grad(set_to_none=True)
        loss = nn.CrossEntropyLoss()(model(x), labels)
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.adapter_a.coefficients.detach()))
        self.assertTrue(all(parameter.grad is None for parameter in model.adapter_b.coefficient_vectors))


if __name__ == "__main__":
    unittest.main()
