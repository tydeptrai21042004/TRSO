from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torchvision.models.vision_transformer import VisionTransformer
from torch.utils.data import DataLoader, TensorDataset

from engine import _single_label_detailed_metrics
from main import (
    _attach_hook_adapters,
    calibrate_prompt_frequency_mapping,
    canonicalize_args,
    get_args_parser,
    load_matching_head,
    set_trainability_policy,
)
from models.model_support import validate_method_backbone
from models.task_response_adapter import TaskResponseSpatialAdapter
from models.tuning_modules.adaptformer import (
    AdaptFormerAdapter,
    apply_adaptformer,
)
from models.tuning_modules.prompter import VisualPromptingClassifier
from tools.run_fair_dtd_suite import (
    CNN_METHODS,
    TRANSFORMER_METHODS,
    build_comparison_specs,
    build_head_specs,
    parser as fair_suite_parser,
)
from models.tuning_modules.piggyback import (
    PiggybackConv2d,
    apply_piggyback,
    binary_mask,
    piggyback_storage,
)


class TinyCNN(nn.Module):
    def __init__(self, classes: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, classes)

    def forward(self, x):
        return self.fc(self.pool(torch.relu(self.conv(x))).flatten(1))


def test_adaptformer_is_identity_safe_and_trains_only_adapter_and_head():
    torch.manual_seed(3)
    model = VisionTransformer(
        image_size=32,
        patch_size=8,
        num_layers=2,
        num_heads=2,
        hidden_dim=32,
        mlp_dim=64,
        num_classes=5,
        dropout=0.0,
        attention_dropout=0.0,
    ).eval()
    reference = copy.deepcopy(model).eval()
    records = apply_adaptformer(model, bottleneck=4, dropout=0.0, scale=0.1)
    assert len(records) == 2
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(reference(x), model(x), atol=1e-7, rtol=1e-6)

    set_trainability_policy(model, SimpleNamespace(tuning_method="adaptformer"))
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable
    assert all("adaptformer_adapter" in name or name.startswith("heads.") for name in trainable)

    loss = model(x).square().mean()
    loss.backward()
    assert any(
        p.grad is not None
        for name, p in model.named_parameters()
        if "adaptformer_adapter" in name and name.endswith("up_proj.weight")
    )


def test_adaptformer_official_initialization_contract():
    adapter = AdaptFormerAdapter(dim=16, bottleneck=4, scale=0.1)
    assert torch.count_nonzero(adapter.up_proj.weight) == 0
    assert torch.count_nonzero(adapter.up_proj.bias) == 0
    assert torch.count_nonzero(adapter.down_proj.weight) > 0


def test_piggyback_ones_init_preserves_pretrained_function_and_mask_gradients():
    torch.manual_seed(4)
    model = TinyCNN().eval()
    reference = copy.deepcopy(model).eval()
    records = apply_piggyback(model, threshold=5e-3, mask_init="ones", mask_linear=False)
    assert records == ["conv"]
    assert isinstance(model.conv, PiggybackConv2d)
    assert torch.allclose(model.conv.mask_scores, torch.full_like(model.conv.mask_scores, 1e-2))
    x = torch.randn(3, 3, 16, 16)
    with torch.no_grad():
        assert torch.allclose(reference(x), model(x), atol=1e-7, rtol=1e-6)

    set_trainability_policy(
        model,
        SimpleNamespace(tuning_method="piggyback", piggyback_train_head=True),
    )
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert "conv.mask_scores" in trainable
    assert "fc.weight" in trainable and "fc.bias" in trainable
    assert not model.conv.weight.requires_grad

    model.train()
    model(x).sum().backward()
    assert model.conv.mask_scores.grad is not None
    storage = piggyback_storage(model)
    assert storage.masked_weights == model.conv.weight.numel()
    assert storage.deployed_mask_bits == storage.masked_weights


def test_binary_mask_ste_threshold_and_gradient():
    scores = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
    mask = binary_mask(scores, threshold=0.0)
    assert mask.tolist() == [0.0, 1.0, 1.0]
    mask.sum().backward()
    assert scores.grad.tolist() == [1.0, 1.0, 1.0]


def _fair_args(method: str):
    parser = get_args_parser()
    args = parser.parse_args([
        "--tuning_method", method,
        "--fair_protocol", "True",
        "--fair_optimizer", "adamw",
        "--fair_peft_lr", "0.005",
        "--fair_full_lr", "0.0001",
        "--fair_linear_lr", "0.1",
        "--fair_weight_decay", "0.0001",
        "--fair_warmup_epochs", "5",
    ])
    return canonicalize_args(args)


def test_fair_protocol_is_identical_for_all_peft_methods():
    peft = [
        "prompt", "conv", "trso", "bam", "residual", "ssf", "lora",
        "bitfit", "adaptformer", "piggyback", "sidetune",
    ]
    rows = [_fair_args(method) for method in peft]
    assert {row.optimizer for row in rows} == {"adamw"}
    assert {row.lr for row in rows} == {0.005}
    assert {row.weight_decay for row in rows} == {0.0001}
    assert {row.weight_decay_adapter for row in rows} == {0.0001}
    assert {row.warmup_epochs for row in rows} == {5}
    assert {row.min_lr for row in rows} == {1e-6}
    assert all(not row.paper_hparams and not row.legacy_auto_hparams for row in rows)

    assert _fair_args("full").lr == 0.0001
    assert _fair_args("linear").lr == 0.1


def test_detailed_metrics_include_macro_calibration_and_confusion():
    logits = torch.tensor([
        [8.0, 0.0, 0.0],
        [0.0, 8.0, 0.0],
        [0.0, 0.0, 8.0],
        [0.0, 7.0, 0.0],
    ])
    targets = torch.tensor([0, 1, 2, 1])
    scalar, diagnostic = _single_label_detailed_metrics(logits, targets)
    assert scalar["macro_f1"] == pytest.approx(100.0)
    assert scalar["balanced_accuracy"] == pytest.approx(100.0)
    assert scalar["num_samples"] == 4
    assert scalar["num_classes"] == 3
    assert diagnostic["confusion_matrix"] == [[1, 0, 0], [0, 2, 0], [0, 0, 1]]
    assert len(diagnostic["per_class_f1"]) == 3
    assert 0.0 <= scalar["ece"] <= 100.0


def test_shared_head_checkpoint_loads_exactly(tmp_path):
    source = TinyCNN(classes=3)
    target = TinyCNN(classes=3)
    with torch.no_grad():
        source.fc.weight.fill_(0.25)
        source.fc.bias.fill_(-0.5)
    checkpoint = tmp_path / "head.pth"
    torch.save({"model": source.state_dict()}, checkpoint)
    loaded = load_matching_head(target, str(checkpoint), strict=True)
    assert loaded == 2
    assert torch.equal(target.fc.weight, source.fc.weight)
    assert torch.equal(target.fc.bias, source.fc.bias)
    assert not torch.equal(target.conv.weight, source.conv.weight)




class MappingBackbone(nn.Module):
    def __init__(self, source_classes: int = 5):
        super().__init__()
        self.source_classes = source_classes
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        indices = x[:, 0, 0, 0].long()
        logits = x.new_full((x.shape[0], self.source_classes), -10.0)
        logits.scatter_(1, indices[:, None], 10.0)
        return logits + self.anchor * 0.0


def test_visual_prompt_frequency_mapping_uses_training_data_and_unique_assignment():
    target = torch.tensor([0, 0, 1, 1, 2, 2])
    source = torch.tensor([4, 4, 2, 2, 3, 3])
    images = torch.zeros(6, 3, 4, 4)
    images[:, 0, 0, 0] = source.float()
    loader = DataLoader(TensorDataset(images, target), batch_size=2, shuffle=False)
    model = VisualPromptingClassifier(
        MappingBackbone(),
        num_classes=3,
        prompt_size=1,
        image_size=4,
        output_indices=[0, 1, 2],
        prompt_type="fixed_patch",
    )
    report = calibrate_prompt_frequency_mapping(
        model, loader, torch.device("cpu"), num_classes=3
    )
    assert model.output_indices.tolist() == [4, 2, 3]
    assert report["mapping_precision"] == pytest.approx(1.0)

def test_trso_vit_uses_pre_block_insertion_so_final_adapter_reaches_cls_token():
    torch.manual_seed(17)
    model = VisionTransformer(
        image_size=32,
        patch_size=8,
        num_layers=2,
        num_heads=2,
        hidden_dim=32,
        mlp_dim=64,
        num_classes=3,
        dropout=0.0,
        attention_dropout=0.0,
    )

    def make_adapter(ch, layout="auto", grid_size=None):
        return TaskResponseSpatialAdapter(
            channels=ch,
            kernel_size=3,
            spatial_rank=1,
            layout=layout,
            grid_size=grid_size,
        )

    with torch.no_grad():
        model.heads.head.weight.normal_(mean=0.0, std=0.02)
        model.heads.head.bias.zero_()
    count = _attach_hook_adapters(
        model,
        SimpleNamespace(tuning_method="trso", adapt_scale=1.0),
        make_adapter,
    )
    assert count == 2
    final_adapter = model.encoder.layers[-1].pet_adapter
    final_adapter.start_calibration(reset=True)
    model(torch.randn(2, 3, 32, 32)).sum().backward()
    assert final_adapter.probe_kernel.grad is not None
    assert final_adapter.probe_kernel.grad.abs().sum().item() > 0


def test_fair_dtd_manifest_includes_every_common_backbone_baseline(tmp_path):
    args = fair_suite_parser().parse_args([
        "--output_root", str(tmp_path / "outputs"),
        "--manifest", str(tmp_path / "manifest.json"),
        "--seeds", "0,1,2",
    ])
    heads, paths = build_head_specs(args, [0, 1, 2])
    comparison = build_comparison_specs(args, [0, 1, 2], paths)
    assert len(heads) == 6
    assert len(comparison) == 3 * (len(CNN_METHODS) + len(TRANSFORMER_METHODS))
    cnn = {spec.parameters["tuning_method"] for spec in comparison if spec.suite == "fair_dtd_cnn"}
    transformer = {
        spec.parameters["tuning_method"]
        for spec in comparison
        if spec.suite == "fair_dtd_transformer"
    }
    assert cnn == set(CNN_METHODS)
    assert transformer == set(TRANSFORMER_METHODS)
    peft = [
        spec.parameters for spec in comparison
        if spec.parameters["tuning_method"] not in {"full", "linear"}
    ]
    assert {row["fair_optimizer"] for row in peft} == {"adamw"}
    assert {row["fair_peft_lr"] for row in peft} == {5e-3}
    assert {row["fair_warmup_epochs"] for row in peft} == {5}
    assert {row["fair_weight_decay"] for row in peft} == {1e-4}

def test_new_method_compatibility_contracts():
    validate_method_backbone("adaptformer", "vit")
    validate_method_backbone("piggyback", "resnet")
    with pytest.raises(ValueError):
        validate_method_backbone("adaptformer", "resnet")
    with pytest.raises(ValueError):
        validate_method_backbone("piggyback", "vit")
