"""Paper-fidelity unit tests for reproduced baselines."""
from __future__ import annotations

import copy
import unittest

import torch
import torch.nn as nn

from models.tuning_modules.bam_adapter import BAM, SpatialGate
from models.tuning_modules.conv_adapter import ConvAdapter, ConvAdapterBottleneck
from models.tuning_modules.lora_transformer import LoRALinear
from models.tuning_modules.prompter import PadPrompter, VisualPromptingClassifier
from models.tuning_modules.residual_adapter import ConvTask, ResidualAdapterResNet26, SeriesAdapter
from models.tuning_modules.side_tuning import SideTuningClassifier
from models.tuning_modules.ssf import SSFPost, apply_ssf


class BaselineModuleFaithfulnessTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_visual_prompt_has_frozen_original_classifier_and_fixed_mapping(self):
        from torchvision.models import resnet18
        backbone = resnet18(weights=None, num_classes=1000)
        original_fc = backbone.fc
        model = VisualPromptingClassifier(backbone, num_classes=3, prompt_size=2, image_size=32, output_indices=[4, 2, 7])
        self.assertIs(model.backbone.fc, original_fc)
        self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.tuning_module.parameters()))
        out = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(out.shape), (2, 3))

    def test_pad_prompter_changes_only_border(self):
        prompt = PadPrompter(prompt_size=2, image_size=16)
        x = torch.zeros(1, 3, 16, 16)
        y = prompt(x)
        self.assertTrue(torch.equal(y[:, :, 2:-2, 2:-2], x[:, :, 2:-2, 2:-2]))
        self.assertFalse(torch.equal(y, x))

    def test_bam_exact_equation_and_two_dilated_convolutions(self):
        bam = BAM(gate_channel=16, reduction_ratio=4, dilation_conv_num=2, dilation_val=2)
        dilated = [m for m in bam.spatial_att.modules() if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3)]
        self.assertEqual(len(dilated), 2)
        self.assertTrue(all(m.dilation == (2, 2) for m in dilated))
        x = torch.randn(2, 16, 8, 8)
        bam.eval()
        with torch.no_grad():
            expected = (1 + torch.sigmoid(bam.channel_att(x) * bam.spatial_att(x))) * x
            actual = bam(x)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_conv_adapter_is_depthwise_pointwise_with_channel_scale(self):
        adapter = ConvAdapter(16, 32, kernel_size=3)
        self.assertEqual(adapter.depthwise.groups, 16)
        self.assertEqual(adapter.depthwise.in_channels, 16)
        self.assertEqual(adapter.depthwise.out_channels, 16)
        self.assertEqual(adapter.pointwise.kernel_size, (1, 1))
        self.assertEqual(tuple(adapter.scale.shape), (1, 32, 1, 1))
        self.assertEqual(tuple(adapter(torch.randn(2, 16, 8, 8)).shape), (2, 32, 8, 8))

    def test_residual_adapter_resnet26_architecture_and_modes(self):
        parallel = ResidualAdapterResNet26(5, mode="parallel")
        series = ResidualAdapterResNet26(5, mode="series")
        self.assertEqual([len(parallel.layer1), len(parallel.layer2), len(parallel.layer3)], [4, 4, 4])
        self.assertIsInstance(parallel.layer1[0].conv1, ConvTask)
        self.assertIsInstance(parallel.layer1[0].conv1.adapter, nn.Conv2d)
        self.assertIsInstance(series.layer1[0].conv1.adapter, SeriesAdapter)
        x = torch.randn(2, 3, 64, 64)
        self.assertEqual(tuple(parallel(x).shape), (2, 5))
        self.assertEqual(tuple(series(x).shape), (2, 5))

    def test_ssf_is_inserted_after_internal_vit_operations(self):
        from torchvision.models.vision_transformer import VisionTransformer
        model = VisionTransformer(image_size=32, patch_size=8, num_layers=2, num_heads=2, hidden_dim=32, mlp_dim=64, num_classes=3)
        records = apply_ssf(model, init_std=0.02)
        self.assertIn("conv_proj", records)
        self.assertIn("encoder.layers.encoder_layer_0.mlp.0", records)
        self.assertIsInstance(model.conv_proj, SSFPost)
        self.assertTrue(hasattr(model.encoder.layers[0].self_attention, "qkv_scale"))
        self.assertEqual(tuple(model(torch.randn(2, 3, 32, 32)).shape), (2, 3))

    def test_lora_merge_and_unmerge_preserve_outputs_at_zero_init(self):
        base = nn.Linear(8, 6)
        wrapper = LoRALinear(base, rank=2, alpha=4, merge_weights=True)
        x = torch.randn(3, 8)
        wrapper.train()
        train_out = wrapper(x)
        wrapper.eval()
        self.assertTrue(wrapper.merged)
        eval_out = wrapper(x)
        self.assertTrue(torch.allclose(train_out, eval_out, atol=1e-7))
        wrapper.train()
        self.assertFalse(wrapper.merged)

    def test_side_tuning_side_is_copied_from_base(self):
        from torchvision.models import resnet18
        base = resnet18(weights=None)
        original = copy.deepcopy(base.state_dict())
        model = SideTuningClassifier(base_model=base, num_classes=4)
        for key, value in model.side.state_dict().items():
            if key.startswith("fc."):
                continue
            self.assertTrue(torch.equal(value, original[key]))
        self.assertTrue(all(not p.requires_grad for p in model.base.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.side.parameters()))
        self.assertEqual(tuple(model(torch.randn(2, 3, 64, 64)).shape), (2, 4))


if __name__ == "__main__":
    unittest.main()
