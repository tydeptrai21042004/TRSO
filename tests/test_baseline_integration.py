"""Factory and trainability tests for strict paper reproductions."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from main import build_model_for_experiment, canonicalize_args, get_args_parser, set_trainability_policy
from models.tuning_modules.bam_adapter import BAM, BAMResNet50
from models.tuning_modules.conv_adapter import ConvAdapter, ConvAdapterBottleneck
from models.tuning_modules.prompter import VisualPromptingClassifier
from models.tuning_modules.residual_adapter import ResidualAdapterResNet26
from models.tuning_modules.side_tuning import SideTuningClassifier


def make_args(method: str, backbone: str, size: int = 64):
    parser = get_args_parser()
    args = parser.parse_args([
        "--backbone", backbone,
        "--weights", "none",
        "--pretrained", "False",
        "--tuning_method", method,
        "--nb_classes", "5",
        "--input_size", str(size),
        "--device", "cpu",
        "--use_amp", "False",
        "--profile_efficiency", "False",
        "--save_ckpt", "False",
        "--final_test", "False",
        "--batch_size", "1",
        "--num_workers", "0",
        "--prompt_size", "2",
        "--trso_calibration", "False",
    ])
    return canonicalize_args(args)


def build(method, backbone, size=64):
    args = make_args(method, backbone, size)
    model, ids = build_model_for_experiment(args)
    model = set_trainability_policy(model, args, ids)
    return args, model


def trainable(model):
    return {name for name, p in model.named_parameters() if p.requires_grad}


def test_full_and_linear_contracts():
    _, full = build("full", "resnet18")
    assert "conv1.weight" in trainable(full) and "fc.weight" in trainable(full)
    _, linear = build("linear", "resnet18")
    assert trainable(linear) == {"fc.weight", "fc.bias"}


def test_visual_prompt_factory_keeps_original_head_frozen():
    _, model = build("prompt", "resnet18")
    assert isinstance(model, VisualPromptingClassifier)
    names = trainable(model)
    assert names and all(name.startswith("tuning_module.") for name in names)
    assert model.backbone.fc.out_features == 1000
    model.eval()
    assert model(torch.randn(1, 3, 64, 64)).shape == (1, 5)


def test_conv_adapter_resnet50_factory_and_trainability():
    _, model = build("conv", "resnet50")
    names = trainable(model)
    assert any(isinstance(module, ConvAdapterBottleneck) for module in model.modules())
    assert any(isinstance(module, ConvAdapter) for module in model.modules())
    assert any("adapter." in name for name in names)
    assert "fc.weight" in names
    assert "conv1.weight" not in names
    model.eval()
    assert model(torch.randn(1, 3, 64, 64)).shape == (1, 5)


def test_bam_is_end_to_end_resnet50_with_three_transition_modules():
    _, model = build("bam", "resnet50")
    assert isinstance(model, BAMResNet50)
    assert sum(isinstance(module, BAM) for module in model.modules()) == 3
    names = trainable(model)
    assert "conv1.weight" in names and "fc.weight" in names
    model.eval()
    assert model(torch.randn(1, 3, 64, 64)).shape == (1, 5)


def test_residual_adapter_factory_uses_dedicated_resnet26():
    _, model = build("residual", "resnet18")
    assert isinstance(model, ResidualAdapterResNet26)
    names = trainable(model)
    assert "conv1.weight" not in names
    assert "fc.weight" in names
    assert any("adapter" in name for name in names)
    assert any("bn" in name for name in names)
    model.eval()
    assert model(torch.randn(1, 3, 64, 64)).shape == (1, 5)


def test_side_tuning_trains_copied_side_not_base():
    _, model = build("sidetune", "resnet18")
    assert isinstance(model, SideTuningClassifier)
    names = trainable(model)
    assert any(name.startswith("side.") for name in names)
    assert any(name.startswith("head.") for name in names)
    assert "alpha_logit" in names
    assert not any(name.startswith("base.") for name in names)


def test_unsupported_paper_domain_pairs_fail_explicitly():
    with pytest.raises(ValueError, match="ResNet-50"):
        build("conv", "resnet18")
    with pytest.raises(ValueError, match="ResNet-50"):
        build("bam", "resnet18")
    args = make_args("lora_conv", "resnet18")
    with pytest.raises(ValueError, match="Unknown tuning method"):
        build_model_for_experiment(args)
