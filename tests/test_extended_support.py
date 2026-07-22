from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
from PIL import Image

from datasets.build import (
    TASK_MULTILABEL,
    TASK_REGRESSION,
    TASK_SINGLE_LABEL,
    build_dataset_split,
)
from engine import evaluate
from main import _replace_classifier_head, build_model_for_experiment, canonicalize_args, get_args_parser, set_trainability_policy
from models.model_support import detect_backbone_family, validate_method_backbone
from models.tuning_modules.lora_transformer import (
    LoRAMultiheadAttention,
    LoRAQKVLinear,
    apply_lora_transformer,
)


def _dataset_args(tmp_path=None, task="auto", classes=5):
    return SimpleNamespace(
        dataset="fake",
        task=task,
        task_type=None,
        data_path=str(tmp_path or "./data"),
        input_size=32,
        crop_ratio=0.875,
        train_interpolation="bicubic",
        train_aug="resize",
        imagenet_norm=True,
        imagenet_default_mean_and_std=True,
        color_jitter=0.0,
        aa="none",
        reprob=0.0,
        nb_classes=classes,
        seed=7,
        fake_train_size=9,
        fake_val_size=5,
        fake_test_size=4,
        val_ratio=0.1,
    )


def test_baseline_domain_contracts_are_explicit():
    validate_method_backbone("conv", "resnet")
    validate_method_backbone("lora", "vit")
    validate_method_backbone("bitfit", "swin")
    validate_method_backbone("trso", "cnn")
    validate_method_backbone("trso", "vit")

    for method, family in (("conv", "vit"), ("bam", "swin"), ("lora", "resnet"), ("bitfit", "cnn")):
        try:
            validate_method_backbone(method, family)
        except ValueError as exc:
            assert "not supported" in str(exc)
        else:
            raise AssertionError(f"{method}/{family} should be rejected")


def test_family_detection_for_resnet_vit_and_swin_names():
    from torchvision.models import resnet18
    from torchvision.models.vision_transformer import VisionTransformer

    resnet = resnet18(weights=None)
    vit = VisionTransformer(
        image_size=32,
        patch_size=8,
        num_layers=1,
        num_heads=2,
        hidden_dim=32,
        mlp_dim=64,
        num_classes=3,
    )

    class SwinTransformer(nn.Module):
        def forward(self, x):
            return x

    assert detect_backbone_family(resnet, "resnet18") == "resnet"
    assert detect_backbone_family(vit, "vit_tiny") == "vit"
    assert detect_backbone_family(SwinTransformer(), "swin_t") == "swin"


def test_torchvision_vit_head_replacement_and_original_lora_identity():
    from torchvision.models.vision_transformer import VisionTransformer

    model = VisionTransformer(
        image_size=32,
        patch_size=8,
        num_layers=2,
        num_heads=2,
        hidden_dim=32,
        mlp_dim=64,
        num_classes=1000,
    )
    assert _replace_classifier_head(model, 7, keep_pretrained_head=False)
    assert model.heads.head.out_features == 7

    x = torch.randn(2, 3, 32, 32)
    model.eval()
    with torch.no_grad():
        before = model(x)

    count = apply_lora_transformer(model, rank=2, alpha=4.0)
    assert count == 2
    assert sum(isinstance(module, LoRAMultiheadAttention) for module in model.modules()) == 2
    with torch.no_grad():
        after = model(x)
    assert torch.allclose(before, after, atol=1e-6)

    model.train()
    loss = model(x).square().mean()
    loss.backward()
    trainable_lora_grads = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if any(token in name for token in ("q_a", "q_b", "v_a", "v_b"))
    ]
    assert trainable_lora_grads
    assert any(grad is not None for grad in trainable_lora_grads)



def test_torchvision_swin_lora_supports_functional_qkv_access():
    from torchvision.models import swin_t

    model = swin_t(weights=None, num_classes=4)
    x = torch.randn(1, 3, 32, 32)
    model.eval()
    with torch.no_grad():
        before = model(x)

    count = apply_lora_transformer(model, rank=2, alpha=4.0)
    assert count > 0
    assert any(isinstance(module, LoRAQKVLinear) for module in model.modules())
    with torch.no_grad():
        after = model(x)
    assert after.shape == (1, 4)
    assert torch.allclose(before, after, atol=1e-6)


def test_torchvision_swin_trso_builds_with_bhwc_blocks():
    parser = get_args_parser()
    args = parser.parse_args([
        "--backbone", "swin_t",
        "--weights", "none",
        "--pretrained", "False",
        "--tuning_method", "trso",
        "--nb_classes", "4",
        "--input_size", "32",
        "--device", "cpu",
        "--trso_calibration", "False",
    ])
    args = canonicalize_args(args)
    model, adapter_ids = build_model_for_experiment(args)
    model = set_trainability_policy(model, args, extra_adapter_param_ids=adapter_ids)
    assert sum(getattr(module, "is_trso_adapter", False) for module in model.modules()) == 12
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 32, 32))
    assert output.shape == (1, 4)


def test_trso_torchvision_backbone_matrix_runs():
    cases = [
        ("convnext_tiny", 32, "cnn"),
        ("efficientnet_b0", 64, "cnn"),
        ("mobilenet_v3_small", 64, "cnn"),
        ("vit_b_16", 32, "vit"),
    ]
    parser = get_args_parser()
    for backbone, input_size, family in cases:
        args = parser.parse_args([
            "--backbone", backbone,
            "--weights", "none",
            "--pretrained", "False",
            "--tuning_method", "trso",
            "--nb_classes", "3",
            "--input_size", str(input_size),
            "--device", "cpu",
            "--trso_calibration", "False",
        ])
        args = canonicalize_args(args)
        model, _ = build_model_for_experiment(args)
        assert args.backbone_family == family
        assert any(getattr(module, "is_trso_adapter", False) for module in model.modules())
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, input_size, input_size))
        assert output.shape == (1, 3)


def test_transformer_bitfit_trains_biases_and_task_head_only():
    from torchvision.models.vision_transformer import VisionTransformer

    model = VisionTransformer(
        image_size=32,
        patch_size=8,
        num_layers=1,
        num_heads=2,
        hidden_dim=32,
        mlp_dim=64,
        num_classes=3,
    )
    args = SimpleNamespace(tuning_method="bitfit", bitfit_train_head=True, weight_decay=1e-4)
    set_trainability_policy(model, args)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert "heads.head.weight" in trainable and "heads.head.bias" in trainable
    assert any(name.endswith(".bias") and not name.startswith("heads.") for name in trainable)
    assert all(name.endswith(".bias") or name.startswith("heads.") for name in trainable)


def test_visual_prompting_trains_prompt_only_and_keeps_pretrained_head():
    from torchvision.models import resnet18
    from models.tuning_modules.prompter import VisualPromptingClassifier

    backbone = resnet18(weights=None, num_classes=1000)
    model = VisualPromptingClassifier(backbone, num_classes=3, prompt_size=2, image_size=32)
    args = SimpleNamespace(tuning_method="prompt", weight_decay=1e-4)
    set_trainability_policy(model, args)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable and all(name.startswith("tuning_module.") for name in trainable)
    assert "backbone.fc.weight" not in trainable and "backbone.fc.bias" not in trainable
    assert model.backbone.fc.out_features == 1000
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 32, 32))
    assert output.shape == (1, 3)

def test_fake_dataset_supports_all_task_types():
    cases = [
        (TASK_SINGLE_LABEL, torch.int64),
        (TASK_MULTILABEL, torch.float32),
        (TASK_REGRESSION, torch.float32),
    ]
    for task, dtype in cases:
        args = _dataset_args(task=task, classes=4)
        dataset, output_dim = build_dataset_split(args, "train")
        image, target = dataset[0]
        assert image.shape == (3, 32, 32)
        assert output_dim == 4
        if task == TASK_SINGLE_LABEL:
            assert isinstance(target, int)
        else:
            assert target.dtype == dtype
            assert target.shape == (4,)
        assert args.task_type == task


def test_imagefolder_three_way_split_is_disjoint(tmp_path):
    for class_name in ("a", "b"):
        folder = tmp_path / class_name
        folder.mkdir(parents=True)
        for index in range(20):
            Image.new("RGB", (12, 12), color=(index, index, index)).save(folder / f"{index}.png")

    args = _dataset_args(tmp_path=tmp_path, task=TASK_SINGLE_LABEL, classes=2)
    args.dataset = "imagefolder"
    train, _ = build_dataset_split(args, "train")
    val, _ = build_dataset_split(args, "val")
    test, _ = build_dataset_split(args, "test")

    train_indices, val_indices, test_indices = map(lambda ds: set(ds.indices), (train, val, test))
    assert train_indices.isdisjoint(val_indices)
    assert train_indices.isdisjoint(test_indices)
    assert val_indices.isdisjoint(test_indices)
    assert len(train_indices | val_indices | test_indices) == 40



def test_vtab_official_list_files_are_routed(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for index in range(6):
        Image.new("RGB", (12, 12), color=(index * 10, 0, 0)).save(image_dir / f"{index}.png")

    (tmp_path / "train800.txt").write_text("images/0.png 0\nimages/1.png 1\n", encoding="utf-8")
    (tmp_path / "val200.txt").write_text("images/2.png 0\nimages/3.png 1\n", encoding="utf-8")
    (tmp_path / "train800val200.txt").write_text(
        "images/0.png 0\nimages/1.png 1\nimages/2.png 0\nimages/3.png 1\n",
        encoding="utf-8",
    )
    (tmp_path / "test.txt").write_text("images/4.png 0\nimages/5.png 1\n", encoding="utf-8")

    args = _dataset_args(tmp_path=tmp_path, task=TASK_SINGLE_LABEL, classes=2)
    args.dataset = "vtab"
    train, train_dim = build_dataset_split(args, "train")
    val, val_dim = build_dataset_split(args, "val")
    test, test_dim = build_dataset_split(args, "test")
    assert (len(train), len(val), len(test)) == (2, 2, 2)
    assert train_dim == val_dim == test_dim == 2


def test_csv_router_supports_multilabel_and_regression(tmp_path):
    Image.new("RGB", (12, 12), color=(10, 20, 30)).save(tmp_path / "a.png")
    Image.new("RGB", (12, 12), color=(30, 20, 10)).save(tmp_path / "b.png")

    for split in ("train", "val", "test"):
        (tmp_path / f"{split}_multi.csv").write_text(
            "image,label\na.png,cat;tree\nb.png,tree\n",
            encoding="utf-8",
        )
        (tmp_path / f"{split}_reg.csv").write_text(
            "image,label\na.png,0.1;0.2\nb.png,0.3;0.4\n",
            encoding="utf-8",
        )

    multi_args = _dataset_args(tmp_path=tmp_path, task=TASK_MULTILABEL, classes=2)
    multi_args.dataset = "csv"
    multi_args.image_column = "image"
    multi_args.label_column = "label"
    multi_args.label_separator = ";"
    multi_args.train_csv = str(tmp_path / "train_multi.csv")
    multi_args.val_csv = str(tmp_path / "val_multi.csv")
    multi_args.test_csv = str(tmp_path / "test_multi.csv")
    multi, output_dim = build_dataset_split(multi_args, "train")
    _, target = multi[0]
    assert output_dim == 2 and target.shape == (2,) and target.sum().item() == 2

    reg_args = _dataset_args(tmp_path=tmp_path, task=TASK_REGRESSION, classes=2)
    reg_args.dataset = "csv"
    reg_args.image_column = "image"
    reg_args.label_column = "label"
    reg_args.label_separator = ";"
    reg_args.train_csv = str(tmp_path / "train_reg.csv")
    reg_args.val_csv = str(tmp_path / "val_reg.csv")
    reg_args.test_csv = str(tmp_path / "test_reg.csv")
    regression, output_dim = build_dataset_split(reg_args, "train")
    _, target = regression[0]
    assert output_dim == 2 and torch.allclose(target, torch.tensor([0.1, 0.2]))


def test_generic_fewshot_metadata_route(tmp_path):
    image_dir = tmp_path / "data" / "images"
    annotation_dir = tmp_path / "annotations"
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir()
    for index in range(6):
        Image.new("RGB", (12, 12), color=(0, index * 10, 0)).save(image_dir / f"{index}.png")

    (annotation_dir / "train_meta.list.num_shot_2.seed_7").write_text(
        "0.png 0\n1.png 0\n2.png 1\n3.png 1\n",
        encoding="utf-8",
    )
    (annotation_dir / "val_meta.list").write_text("4.png 0\n", encoding="utf-8")
    (annotation_dir / "test_meta.list").write_text("5.png 1\n", encoding="utf-8")

    args = _dataset_args(tmp_path=tmp_path, task=TASK_SINGLE_LABEL, classes=2)
    args.dataset = "fewshot"
    args.fs_shot = 2
    train, output_dim = build_dataset_split(args, "train")
    val, _ = build_dataset_split(args, "val")
    test, _ = build_dataset_split(args, "test")
    assert (len(train), len(val), len(test), output_dim) == (4, 1, 1, 2)


def test_visual_decathlon_legacy_module_fails_clearly():
    from datasets.vdd import VisualDecathlonDataset

    try:
        VisualDecathlonDataset("unused")
    except NotImplementedError as exc:
        assert "Official Visual Decathlon is not implemented" in str(exc)
    else:
        raise AssertionError("Dormant Visual Decathlon loader must fail clearly.")


def test_celeba_landmark_wrapper_normalizes_coordinates():
    from datasets.build import CelebALandmarks
    from torchvision import transforms as T

    class DummyCelebA(torch.utils.data.Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, index):
            del index
            image = Image.new("RGB", (101, 201))
            landmarks = torch.tensor([0, 0, 100, 200, 50, 100, 25, 50, 75, 150])
            return image, landmarks

    dataset = CelebALandmarks(DummyCelebA(), T.Compose([T.Resize((32, 32)), T.ToTensor()]))
    image, target = dataset[0]
    assert image.shape == (3, 32, 32)
    assert target.shape == (10,)
    assert torch.all((0.0 <= target) & (target <= 1.0))
    assert torch.allclose(target[:4], torch.tensor([0.0, 0.0, 1.0, 1.0]))

class TinyTaskModel(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, output_dim))

    def forward(self, x):
        return self.net(x)


def test_shared_evaluator_supports_all_task_types():
    from torch.utils.data import DataLoader

    expected_metrics = {
        TASK_SINGLE_LABEL: "acc1",
        TASK_MULTILABEL: "map",
        TASK_REGRESSION: "mae",
    }
    for task, metric in expected_metrics.items():
        args = _dataset_args(task=task, classes=3)
        dataset, output_dim = build_dataset_split(args, "val")
        loader = DataLoader(dataset, batch_size=2)
        stats = evaluate(loader, TinyTaskModel(output_dim), torch.device("cpu"), task_type=task)
        assert metric in stats
        assert torch.isfinite(torch.tensor(stats[metric]))


def test_ssf_torchvision_swin_and_convnext_internal_insertion_runs():
    from torchvision.models import swin_t, convnext_tiny
    from models.tuning_modules.ssf import apply_ssf, set_ssf_trainability

    for factory in (swin_t, convnext_tiny):
        model = factory(weights=None, num_classes=3)
        records = apply_ssf(model, init_std=0.02)
        assert len(records) > 10
        set_ssf_trainability(model)
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, 32, 32))
        assert output.shape == (1, 3)
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        assert any("ssf_" in name or "qkv_scale" in name for name in trainable)
        assert any(name.startswith(("head.", "classifier.")) for name in trainable)
