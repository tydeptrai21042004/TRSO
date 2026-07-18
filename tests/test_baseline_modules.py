# tests/test_baseline_modules.py
"""Unit tests for individual PEFT baseline modules.

These tests check implementation-level faithfulness properties used in the
paper revision: shape preservation, identity-safe initialization where expected,
frozen base behavior for LoRA, no recursive wrapping, and side-tuning feature
compatibility with torchvision ResNets.
"""
from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from models.task_response_adapter import TaskResponseSpatialAdapter
from models.tuning_modules.bam_adapter import BAMAdapter
from models.tuning_modules.lora_conv import LoRAConv2d, apply_lora_conv2d
from models.tuning_modules.residual_adapter import ParallelResidualAdapter
from models.tuning_modules.ssf import SSF
from models.tuning_modules.side_tuning import SideTuningClassifier
from models.tuning_modules.prompter import PadPrompter


class BaselineModuleFaithfulnessTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_trso_identity_safe_and_shape_preserving(self):
        x = torch.randn(2, 8, 7, 7)
        adapter = TaskResponseSpatialAdapter(
            channels=8,
            kernel_size=5,
            spatial_rank=2,
            channel_ratio=4,
            gate_init=0.0,
        )
        y = adapter(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertTrue(torch.allclose(y, x, atol=1e-7), "TRSO must be identity-safe at gate_init=0.")

    def test_bam_identity_safe_shape_and_attention(self):
        x = torch.randn(2, 16, 8, 8)
        bam = BAMAdapter(channels=16, reduction=4, dilation=2, gate_init=0.0, use_bn=True)
        bam.eval()
        y = bam(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertTrue(torch.allclose(y, x, atol=1e-6), "BAM-Tuning must be identity-safe at gate_init=0.")
        att = bam.attention(x)
        self.assertEqual(tuple(att.shape), (2, 16, 8, 8))
        self.assertTrue(torch.all(att >= 0) and torch.all(att <= 1), "BAM attention must be sigmoid-bounded.")

    def test_ssf_identity_safe(self):
        x = torch.randn(2, 10, 4, 4)
        ssf = SSF(10, init_scale=1.0, init_shift=0.0)
        y = ssf(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertTrue(torch.allclose(y, x, atol=1e-6), "SSF should be identity at scale=1, shift=0.")

    def test_lora_conv_identity_safe_and_no_recursive_wrapping(self):
        base = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(8, 8, 1, bias=False),
        )
        x = torch.randn(2, 3, 8, 8)
        with torch.no_grad():
            y_before = base(x)
        apply_lora_conv2d(base, r=2, alpha=4.0, target="all")
        wrapped = [m for m in base.modules() if isinstance(m, LoRAConv2d)]
        self.assertEqual(len(wrapped), 2, "Only original Conv2d layers should be wrapped once.")
        nested = any(isinstance(m.base, LoRAConv2d) for m in wrapped)
        self.assertFalse(nested, "LoRAConv2d should not recursively wrap LoRAConv2d.")
        with torch.no_grad():
            y_after = base(x)
        self.assertTrue(torch.allclose(y_before, y_after, atol=1e-6), "LoRA init should preserve frozen base output.")
        for m in wrapped:
            self.assertFalse(m.base.weight.requires_grad)
            if m.lora_down is not None:
                self.assertTrue(m.lora_down.weight.requires_grad)
                self.assertTrue(m.lora_up.weight.requires_grad)

    def test_residual_adapter_downsample_block_runs(self):
        # Channel-changing block simulates ResNet downsample stages.
        block = nn.Sequential(
            nn.Conv2d(8, 16, 1, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        adapter = ParallelResidualAdapter(block, channels=16, reduction=4, gate_init=0.0)
        adapter.eval()
        x = torch.randn(2, 8, 16, 16)
        y = adapter(x)
        self.assertEqual(tuple(y.shape), (2, 16, 8, 8))

    def test_side_tuning_works_with_torchvision_resnet_features(self):
        from torchvision.models import resnet18

        base = resnet18(weights=None)
        model = SideTuningClassifier(base_model=base, num_classes=5, side_width=8, side_depth=2, use_checkpoint=False)
        x = torch.randn(2, 3, 64, 64)
        y = model(x)
        self.assertEqual(tuple(y.shape), (2, 5))
        self.assertTrue(any(p.requires_grad for p in model.side_net.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.base.parameters()))

    def test_prompt_pad_prompter_shape(self):
        prompter = PadPrompter(prompt_size=2, image_size=16)
        x = torch.zeros(3, 3, 16, 16)
        y = prompter(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertTrue(any(p.requires_grad for p in prompter.parameters()))


if __name__ == "__main__":
    unittest.main()
