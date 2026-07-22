from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from main import set_trainability_policy
from models.tuning_modules.conv_adapter import ConvAdapter, ConvAdapterBottleneck
from models.tuning_modules.lora_transformer import LoRALinear, LoRAMultiheadAttention
from models.tuning_modules.prompter import FixedPatchPrompter, PadPrompter, RandomPatchPrompter
from models.tuning_modules.residual_adapter import ConvTask, ResidualAdapterBasicBlock, ResidualAdapterResNet26
from models.tuning_modules.side_tuning import ConvSideNetwork, SideTuningClassifier
from models.tuning_modules.ssf import SSFPost, apply_ssf, merge_ssf_


def test_conv_adapter_matches_released_design2_equation():
    torch.manual_seed(0)
    adapter = ConvAdapter(16, 12, width=4, groups=4, kernel_size=3, act_layer=nn.ReLU)
    x = torch.randn(2, 16, 9, 9)
    expected = adapter.conv2(adapter.act(adapter.conv1(x))) * adapter.se
    assert adapter.conv1.groups == 4
    assert adapter.conv1.out_channels == 4
    assert torch.allclose(adapter(x), expected, atol=1e-7)


def test_conv_parallel_wrapper_uses_official_insertion_location():
    from torchvision.models import resnet50

    block = copy.deepcopy(resnet50(weights=None).layer2[0]).eval()
    wrapper = ConvAdapterBottleneck(block, mode="conv_parallel", reduction=8).eval()
    x = torch.randn(1, block.conv1.in_channels, 16, 16)
    with torch.no_grad():
        z = block.relu(block.bn1(block.conv1(x)))
        adapt = wrapper.adapter(z)
        z = block.relu(block.bn2(block.conv2(z))) + adapt
        z = block.bn3(block.conv3(z))
        identity = block.downsample(x)
        expected = block.relu(z + identity)
        actual = wrapper(x)
    assert wrapper.is_official_conv_adapter_path
    assert torch.allclose(actual, expected, atol=1e-6)


def test_visual_prompt_families_preserve_nonprompt_pixels():
    x = torch.zeros(2, 3, 16, 16)
    pad = PadPrompter(2, 16)
    assert torch.equal(pad(x)[:, :, 2:-2, 2:-2], x[:, :, 2:-2, 2:-2])

    fixed = FixedPatchPrompter(3, 16)
    fixed_out = fixed(x)
    assert torch.count_nonzero(fixed_out[:, :, :-3, :]).item() == 0
    assert torch.count_nonzero(fixed_out[:, :, :, :-3]).item() == 0

    torch.manual_seed(3)
    random_prompt = RandomPatchPrompter(3, 16)
    random_out = random_prompt(x)
    top, left = random_prompt.last_location.tolist()
    mask = torch.zeros_like(random_out)
    mask[:, :, top : top + 3, left : left + 3] = 1
    assert torch.count_nonzero(random_out * (1 - mask)).item() == 0


def test_residual_adapter_option_a_shortcut_appends_zero_channels():
    block = ResidualAdapterBasicBlock(8, 16, stride=2, mode="parallel")
    x = torch.randn(2, 8, 12, 12)
    shortcut = block._shortcut(x)
    pooled = torch.nn.functional.avg_pool2d(x, 2, 2)
    assert torch.equal(shortcut[:, :8], pooled)
    assert torch.count_nonzero(shortcut[:, 8:]).item() == 0


def test_residual_series_order_is_series_then_final_bn():
    module = ConvTask(4, 4, mode="series").eval()
    x = torch.randn(2, 4, 8, 8)
    with torch.no_grad():
        expected = module.bn(module.adapter(module.conv(x)))
        actual = module(x)
    assert torch.allclose(actual, expected, atol=1e-7)


def test_residual_checkpoint_requires_all_shared_filters():
    model = ResidualAdapterResNet26(3, mode="parallel")
    incomplete = {"pre_layers_conv.conv.weight": model.pre_layers_conv.conv.weight.detach().clone()}
    with pytest.raises(RuntimeError, match="shared filters"):
        model.load_shared_state_dict(incomplete, require_shared_coverage=True)
    model.load_shared_state_dict(model.state_dict(), require_shared_coverage=True)
    assert model.last_checkpoint_coverage.complete


def test_ssf_merge_removes_wrappers_and_preserves_nontrivial_output():
    from torchvision.models.vision_transformer import VisionTransformer

    torch.manual_seed(5)
    model = VisionTransformer(32, 8, 2, 2, 32, 64, num_classes=3).eval()
    apply_ssf(model, init_std=0.0)
    for module in model.modules():
        if isinstance(module, SSFPost):
            module.ssf_scale.data.uniform_(0.8, 1.2)
            module.ssf_shift.data.uniform_(-0.1, 0.1)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        before = model(x)
    records = merge_ssf_(model)
    with torch.no_grad():
        after = model(x)
    assert records
    assert not any(isinstance(module, SSFPost) for module in model.modules())
    assert torch.allclose(before, after, atol=2e-5, rtol=2e-5)


def test_lora_nonzero_merge_unmerge_equivalence_and_mha_dropout_guard():
    base = nn.Linear(7, 5)
    wrapper = LoRALinear(base, rank=2, alpha=4, merge_weights=True)
    wrapper.lora_b.data.normal_()
    x = torch.randn(3, 7)
    wrapper.train()
    train_output = wrapper(x)
    wrapper.eval()
    assert torch.allclose(train_output, wrapper(x), atol=1e-6)
    wrapper.train()
    assert torch.allclose(train_output, wrapper(x), atol=1e-6)

    mha = nn.MultiheadAttention(8, 2, batch_first=True)
    with pytest.raises(ValueError, match="dropout=0"):
        LoRAMultiheadAttention(mha, rank=2, dropout=0.1)


def test_bitfit_explicit_scope_and_head_policy():
    from torchvision.models.vision_transformer import VisionTransformer

    model = VisionTransformer(32, 8, 1, 2, 32, 64, num_classes=3)
    args = SimpleNamespace(
        tuning_method="bitfit",
        bitfit_train_head=True,
        bitfit_bias_scope="attention",
        weight_decay=0.1,
    )
    set_trainability_policy(model, args)
    names = {name for name, p in model.named_parameters() if p.requires_grad}
    assert "heads.head.weight" in names and "heads.head.bias" in names
    assert any("self_attention" in name and name.endswith("bias") for name in names)
    assert all(name.startswith("heads.") or "self_attention" in name for name in names)


def test_default_main_side_network_is_lightweight_and_trainable():
    from torchvision.models import resnet18

    base = resnet18(weights=None)
    model = SideTuningClassifier(
        base_model=base,
        num_classes=4,
        side_width=16,
        side_depth=4,
    )
    assert isinstance(model.side, ConvSideNetwork)
    assert sum(p.numel() for p in model.side.parameters()) < sum(p.numel() for p in model.base.parameters())
    assert all(not p.requires_grad for p in model.base.parameters())
    assert all(p.requires_grad for p in model.side.parameters())
    assert model(torch.randn(2, 3, 64, 64)).shape == (2, 4)
