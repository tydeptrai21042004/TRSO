from types import SimpleNamespace

import torch

from datasets.build import build_dataset_split


def _args():
    return SimpleNamespace(
        dataset="fake",
        data_path="./data",
        input_size=32,
        train_aug="none",
        imagenet_norm=True,
        imagenet_default_mean_and_std=True,
        nb_classes=7,
        seed=13,
        fake_train_size=11,
        fake_val_size=5,
        fake_test_size=3,
    )


def test_fake_dataset_split_sizes_shapes_and_classes():
    args = _args()
    expected_sizes = {"train": 11, "val": 5, "test": 3}

    for split, expected_size in expected_sizes.items():
        dataset, num_classes = build_dataset_split(args, split)
        image, target = dataset[0]

        assert len(dataset) == expected_size
        assert num_classes == 7
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 32, 32)
        assert 0 <= int(target) < num_classes


def test_fake_dataset_is_deterministic_per_split_and_separated_across_splits():
    args = _args()
    train_a, _ = build_dataset_split(args, "train")
    train_b, _ = build_dataset_split(args, "train")
    val, _ = build_dataset_split(args, "val")

    image_a, target_a = train_a[0]
    image_b, target_b = train_b[0]
    image_val, target_val = val[0]

    assert torch.equal(image_a, image_b)
    assert target_a == target_b
    assert not (torch.equal(image_a, image_val) and target_a == target_val)
