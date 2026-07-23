"""Dataset and task router for transfer-learning experiments.

Design goals:
- explicit, disjoint train/validation/test protocols;
- single-label, multi-label, and regression tasks;
- official splits when available;
- stratified local splits when an official validation split is absent;
- generic ImageFolder and CSV datasets for new benchmarks without code edits.
"""
from __future__ import annotations

import csv
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import datasets
from torchvision.datasets import CocoDetection


TASK_SINGLE_LABEL = "single_label"
TASK_MULTILABEL = "multilabel"
TASK_REGRESSION = "regression"


def _get_bool(args, name: str, default: bool) -> bool:
    return bool(getattr(args, name, default))


def _set_task(args, task_type: str, output_dim: int) -> None:
    previous = getattr(args, "task_type", None)
    requested = str(getattr(args, "task", "auto") or "auto").lower()
    if requested not in ("auto", task_type):
        raise ValueError(f"Dataset defines task '{task_type}', but --task={requested!r} was requested.")
    if previous not in (None, "auto", task_type):
        raise ValueError(f"Inconsistent task types across splits: {previous!r} vs {task_type!r}.")
    args.task_type = task_type
    args.nb_classes = int(output_dim)


def _interpolation(args):
    name = str(getattr(args, "train_interpolation", "bicubic") or "bicubic").lower()
    return {
        "nearest": T.InterpolationMode.NEAREST,
        "bilinear": T.InterpolationMode.BILINEAR,
        "bicubic": T.InterpolationMode.BICUBIC,
        "lanczos": T.InterpolationMode.LANCZOS,
    }.get(name, T.InterpolationMode.BICUBIC)


def _img_transforms(args, is_train: bool):
    size = int(getattr(args, "input_size", 224))
    train_aug = str(getattr(args, "train_aug", "standard")).lower()
    use_norm = _get_bool(args, "imagenet_norm", _get_bool(args, "imagenet_default_mean_and_std", True))
    interpolation = _interpolation(args)
    crop_ratio = float(getattr(args, "crop_ratio", 0.875) or 0.875)

    tfms: List[Callable] = []
    if is_train and train_aug == "standard":
        tfms.extend([
            T.RandomResizedCrop(size, scale=(0.8, 1.0), interpolation=interpolation),
            T.RandomHorizontalFlip(),
        ])
        color_jitter = float(getattr(args, "color_jitter", 0.0) or 0.0)
        if color_jitter > 0:
            tfms.append(T.ColorJitter(color_jitter, color_jitter, color_jitter, 0.0))
        aa = str(getattr(args, "aa", "") or "").lower()
        if aa not in ("", "none", "null"):
            # Torchvision RandAugment is a stable fallback for timm-style AA strings.
            tfms.append(T.RandAugment())
    elif is_train:
        tfms.append(T.Resize((size, size), interpolation=interpolation))
    else:
        resize_size = max(size, int(round(size / max(crop_ratio, 1e-6))))
        tfms.extend([
            T.Resize(resize_size, interpolation=interpolation),
            T.CenterCrop(size),
        ])

    tfms.extend([
        T.ToTensor(),
        T.Lambda(lambda x: x.expand(3, -1, -1) if x.shape[0] == 1 else x[:3]),
    ])
    if use_norm:
        tfms.append(T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    if is_train and float(getattr(args, "reprob", 0.0) or 0.0) > 0:
        tfms.append(T.RandomErasing(p=float(args.reprob), value="random"))
    return T.Compose(tfms)


def _extract_labels(dataset: Dataset) -> Optional[List[int]]:
    for attr in ("targets", "labels", "_labels", "y"):
        values = getattr(dataset, attr, None)
        if values is not None:
            if isinstance(values, torch.Tensor):
                values = values.tolist()
            try:
                return [int(v) for v in values]
            except Exception:
                pass
    samples = getattr(dataset, "samples", None)
    if samples is not None:
        return [int(label) for _, label in samples]
    return None


def _random_partition_indices(length: int, ratios: Sequence[float], seed: int) -> List[List[int]]:
    if length <= 0:
        raise ValueError("Cannot split an empty dataset.")
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(length, generator=generator).tolist()
    counts = [int(round(length * float(r))) for r in ratios[:-1]]
    if sum(counts) >= length:
        counts = [max(0, min(c, length - 1)) for c in counts]
    counts.append(length - sum(counts))
    parts, offset = [], 0
    for count in counts:
        parts.append(order[offset: offset + count])
        offset += count
    return parts


def _stratified_partition_indices(labels: Sequence[int], ratios: Sequence[float], seed: int) -> List[List[int]]:
    groups: Dict[int, List[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        groups[int(label)].append(index)
    outputs: List[List[int]] = [[] for _ in ratios]
    generator = torch.Generator().manual_seed(int(seed))
    for label in sorted(groups):
        indices = groups[label]
        permutation = torch.randperm(len(indices), generator=generator).tolist()
        shuffled = [indices[i] for i in permutation]
        raw_counts = [int(round(len(shuffled) * float(r))) for r in ratios[:-1]]
        # Keep at least one training sample whenever possible.
        max_non_train = max(0, len(shuffled) - 1)
        while sum(raw_counts) > max_non_train and raw_counts:
            largest = max(range(len(raw_counts)), key=lambda i: raw_counts[i])
            raw_counts[largest] -= 1
        counts = raw_counts + [len(shuffled) - sum(raw_counts)]
        offset = 0
        for part, count in enumerate(counts):
            outputs[part].extend(shuffled[offset: offset + count])
            offset += count
    return outputs


def _partition_dataset(
    base: Dataset,
    split: str,
    seed: int,
    ratios: Tuple[float, float, float] = (0.1, 0.1, 0.8),
    labels: Optional[Sequence[int]] = None,
) -> Subset:
    """Return val/test/train partitions; ratios are ordered val,test,train."""
    labels = list(labels) if labels is not None else _extract_labels(base)
    partitions = (
        _stratified_partition_indices(labels, ratios, seed)
        if labels is not None else _random_partition_indices(len(base), ratios, seed)
    )
    mapping = {"val": 0, "test": 1, "train": 2}
    return Subset(base, partitions[mapping[split]])


def _train_val_subset(base: Dataset, split: str, seed: int, val_ratio: float = 0.2) -> Subset:
    if split not in ("train", "val"):
        raise ValueError("_train_val_subset supports train/val only.")
    labels = _extract_labels(base)
    ratios = (float(val_ratio), 1.0 - float(val_ratio))
    parts = (
        _stratified_partition_indices(labels, ratios, seed)
        if labels is not None else _random_partition_indices(len(base), ratios, seed)
    )
    return Subset(base, parts[1] if split == "train" else parts[0])


def _single_label(args, dataset: Dataset, output_dim: int):
    _set_task(args, TASK_SINGLE_LABEL, output_dim)
    return dataset, output_dim


def _multilabel(args, dataset: Dataset, output_dim: int):
    _set_task(args, TASK_MULTILABEL, output_dim)
    return dataset, output_dim


def _regression(args, dataset: Dataset, output_dim: int):
    _set_task(args, TASK_REGRESSION, output_dim)
    return dataset, output_dim


# ---------------------------------------------------------------------------
# Standard torchvision datasets
# ---------------------------------------------------------------------------
def _build_train_test_dataset(cls, args, split: str, classes: int, **kwargs):
    if split in ("train", "val"):
        base = cls(root=args.data_path, train=True, download=bool(getattr(args, "download", False)), transform=_img_transforms(args, split == "train"), **kwargs)
        return _single_label(args, _train_val_subset(base, split, getattr(args, "split_seed", getattr(args, "seed", 42))), classes)
    base = cls(root=args.data_path, train=False, download=bool(getattr(args, "download", False)), transform=_img_transforms(args, False), **kwargs)
    return _single_label(args, base, classes)


def _build_cifar10(args, split): return _build_train_test_dataset(datasets.CIFAR10, args, split, 10)
def _build_cifar100(args, split): return _build_train_test_dataset(datasets.CIFAR100, args, split, 100)
def _build_mnist(args, split): return _build_train_test_dataset(datasets.MNIST, args, split, 10)
def _build_fashion_mnist(args, split): return _build_train_test_dataset(datasets.FashionMNIST, args, split, 10)
def _build_kmnist(args, split): return _build_train_test_dataset(datasets.KMNIST, args, split, 10)
def _build_usps(args, split): return _build_train_test_dataset(datasets.USPS, args, split, 10)


def _build_emnist(args, split):
    return _build_train_test_dataset(datasets.EMNIST, args, split, 62, split="byclass")


def _build_qmnist(args, split):
    transform = _img_transforms(args, split == "train")
    if split in ("train", "val"):
        base = datasets.QMNIST(root=args.data_path, what="train", download=bool(getattr(args, "download", False)), transform=transform)
        return _single_label(args, _train_val_subset(base, split, getattr(args, "split_seed", getattr(args, "seed", 42))), 10)
    return _single_label(args, datasets.QMNIST(root=args.data_path, what="test", download=bool(getattr(args, "download", False)), transform=transform), 10)


def _build_svhn(args, split):
    tv_split = "train" if split in ("train", "val") else "test"
    base = datasets.SVHN(root=args.data_path, split=tv_split, download=bool(getattr(args, "download", False)), transform=_img_transforms(args, split == "train"))
    if split in ("train", "val"):
        base = _train_val_subset(base, split, getattr(args, "split_seed", getattr(args, "seed", 42)))
    return _single_label(args, base, 10)


def _official_train_test(cls, args, split, classes, train_name="train", test_name="test", split_kw="split", **kwargs):
    name = train_name if split in ("train", "val") else test_name
    base = cls(root=args.data_path, download=bool(getattr(args, "download", False)), transform=_img_transforms(args, split == "train"), **{split_kw: name}, **kwargs)
    if split in ("train", "val"):
        base = _train_val_subset(base, split, getattr(args, "split_seed", getattr(args, "seed", 42)))
    return _single_label(args, base, classes)


def _build_stl10(args, split): return _official_train_test(datasets.STL10, args, split, 10)
def _build_food101(args, split): return _official_train_test(datasets.Food101, args, split, 101)
def _build_cars(args, split): return _official_train_test(datasets.StanfordCars, args, split, 196)
def _build_gtsrb(args, split): return _official_train_test(datasets.GTSRB, args, split, 43)


def _build_pets(args, split):
    name = "trainval" if split in ("train", "val") else "test"
    base = datasets.OxfordIIITPet(root=args.data_path, split=name, download=bool(getattr(args, "download", False)), transform=_img_transforms(args, split == "train"))
    if split in ("train", "val"):
        base = _train_val_subset(base, split, getattr(args, "split_seed", getattr(args, "seed", 42)))
    return _single_label(args, base, 37)


def _build_flowers102(args, split):
    return _single_label(args, datasets.Flowers102(root=args.data_path, split=split, download=bool(getattr(args, "download", False)), transform=_img_transforms(args, split == "train")), 102)


def _build_dtd(args, split):
    partition = int(getattr(args, "dtd_partition", 1))
    return _single_label(args, datasets.DTD(root=args.data_path, split=split, partition=partition, download=bool(getattr(args, "download", False)), transform=_img_transforms(args, split == "train")), 47)


def _build_fgvc_aircraft(args, split):
    name = "trainval" if split in ("train", "val") else "test"
    base = datasets.FGVCAircraft(root=args.data_path, split=name, download=bool(getattr(args, "download", False)), transform=_img_transforms(args, split == "train"))
    if split in ("train", "val"):
        base = _train_val_subset(base, split, getattr(args, "split_seed", getattr(args, "seed", 42)))
    return _single_label(args, base, 100)


def _three_way_no_official_split(cls, args, split, classes, **kwargs):
    base = cls(root=args.data_path, download=bool(getattr(args, "download", False)), transform=_img_transforms(args, split == "train"), **kwargs)
    subset = _partition_dataset(base, split, getattr(args, "split_seed", getattr(args, "seed", 42)), ratios=(0.1, 0.1, 0.8))
    return _single_label(args, subset, classes)


def _build_caltech101(args, split): return _three_way_no_official_split(datasets.Caltech101, args, split, 101)
def _build_eurosat(args, split): return _three_way_no_official_split(datasets.EuroSAT, args, split, 10)
def _build_sun397(args, split): return _three_way_no_official_split(datasets.SUN397, args, split, 397)


def _build_fer2013(args, split):
    name = "train" if split in ("train", "val") else "test"
    try:
        base = datasets.FER2013(root=args.data_path, split=name, transform=_img_transforms(args, split == "train"))
    except RuntimeError as exc:
        raise RuntimeError(
            "FER2013 cannot be downloaded automatically by torchvision. Place the official "
            "train.csv and test.csv files in the torchvision FER2013 directory."
        ) from exc
    if split in ("train", "val"):
        base = _train_val_subset(base, split, getattr(args, "split_seed", getattr(args, "seed", 42)))
    return _single_label(args, base, 7)


def _build_pcam(args, split):
    return _single_label(args, datasets.PCAM(root=args.data_path, split=split, download=bool(getattr(args, "download", False)), transform=_img_transforms(args, split == "train")), 2)


def _build_country211(args, split):
    name = {"train": "train", "val": "valid", "test": "test"}[split]
    base = datasets.Country211(root=args.data_path, split=name, download=bool(getattr(args, "download", False)), transform=_img_transforms(args, split == "train"))
    return _single_label(args, base, 211)


def _build_rendered_sst2(args, split):
    name = {"train": "train", "val": "val", "test": "test"}[split]
    base = datasets.RenderedSST2(root=args.data_path, split=name, download=bool(getattr(args, "download", False)), transform=_img_transforms(args, split == "train"))
    return _single_label(args, base, 2)


def _build_places365(args, split):
    # Places365 exposes train-standard and val. Use train-standard for train/val
    # and reserve the official validation images as the final test set.
    if split in ("train", "val"):
        base = datasets.Places365(root=args.data_path, split="train-standard", small=bool(getattr(args, "places_small", True)), download=bool(getattr(args, "download", False)), transform=_img_transforms(args, split == "train"))
        base = _train_val_subset(base, split, getattr(args, "split_seed", getattr(args, "seed", 42)), val_ratio=0.05)
    else:
        base = datasets.Places365(root=args.data_path, split="val", small=bool(getattr(args, "places_small", True)), download=bool(getattr(args, "download", False)), transform=_img_transforms(args, False))
    return _single_label(args, base, 365)


def _build_inaturalist(args, split):
    target_type = str(getattr(args, "inat_target_type", "full"))
    if split in ("train", "val"):
        base = datasets.INaturalist(root=args.data_path, version="2021_train", target_type=target_type, download=bool(getattr(args, "download", False)), transform=_img_transforms(args, split == "train"))
        base = _train_val_subset(base, split, getattr(args, "split_seed", getattr(args, "seed", 42)), val_ratio=0.05)
    else:
        base = datasets.INaturalist(root=args.data_path, version="2021_valid", target_type=target_type, download=bool(getattr(args, "download", False)), transform=_img_transforms(args, False))
    source = base.dataset if isinstance(base, Subset) else base
    if target_type == "full":
        output_dim = len(getattr(source, "all_categories", []))
    else:
        output_dim = len(getattr(source, "categories_index", {}).get(target_type, {}))
    if output_dim <= 0:
        labels = _extract_labels(source)
        output_dim = int(max(labels) + 1) if labels else int(getattr(args, "nb_classes", 10000))
    return _single_label(args, base, output_dim)


# ---------------------------------------------------------------------------
# Folder and metadata datasets
# ---------------------------------------------------------------------------
def _find_split_dir(root: Path, names: Sequence[str]) -> Optional[Path]:
    for name in names:
        path = root / name
        if path.is_dir():
            return path
    return None


def _build_imagefolder(args, split):
    root = Path(args.data_path)
    split_names = {
        "train": ("train", "training"),
        "val": ("val", "valid", "validation"),
        "test": ("test", "testing"),
    }
    target_dir = _find_split_dir(root, split_names[split])
    train_dir = _find_split_dir(root, split_names["train"])
    val_dir = _find_split_dir(root, split_names["val"])
    test_dir = _find_split_dir(root, split_names["test"])

    if target_dir is not None and (split == "train" or split == "test" or (val_dir is not None and test_dir is not None)):
        base = datasets.ImageFolder(target_dir, transform=_img_transforms(args, split == "train"))
        return _single_label(args, base, len(base.classes))

    if train_dir is not None and test_dir is not None:
        if split == "test":
            base = datasets.ImageFolder(test_dir, transform=_img_transforms(args, False))
        else:
            base = datasets.ImageFolder(train_dir, transform=_img_transforms(args, split == "train"))
            base = _train_val_subset(base, split, getattr(args, "split_seed", getattr(args, "seed", 42)))
        classes = len((base.dataset if isinstance(base, Subset) else base).classes)
        return _single_label(args, base, classes)

    if train_dir is not None and val_dir is not None:
        # Common ImageNet protocol: carve validation from train, use official val as test.
        if split == "test":
            base = datasets.ImageFolder(val_dir, transform=_img_transforms(args, False))
        else:
            base = datasets.ImageFolder(train_dir, transform=_img_transforms(args, split == "train"))
            base = _train_val_subset(base, split, getattr(args, "split_seed", getattr(args, "seed", 42)), val_ratio=float(getattr(args, "val_ratio", 0.1)))
        classes = len((base.dataset if isinstance(base, Subset) else base).classes)
        return _single_label(args, base, classes)

    # Root itself may be a class-folder dataset with no official splits.
    base = datasets.ImageFolder(root, transform=_img_transforms(args, split == "train"))
    subset = _partition_dataset(base, split, getattr(args, "split_seed", getattr(args, "seed", 42)), ratios=(0.1, 0.1, 0.8))
    return _single_label(args, subset, len(base.classes))


class CUB200Dataset(Dataset):
    """CUB-200-2011 official train/test metadata with a train-derived val split."""

    def __init__(self, root: str, split: str, transform, seed: int = 42, val_ratio: float = 0.1):
        base = Path(root)
        if (base / "CUB_200_2011").is_dir():
            base = base / "CUB_200_2011"
        required = [base / "images.txt", base / "image_class_labels.txt", base / "train_test_split.txt", base / "images"]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("CUB-200-2011 files are missing: " + ", ".join(missing))
        paths = {int(i): p for i, p in (line.strip().split(maxsplit=1) for line in (base / "images.txt").read_text().splitlines())}
        labels = {int(i): int(y) - 1 for i, y in (line.split() for line in (base / "image_class_labels.txt").read_text().splitlines())}
        train_flags = {int(i): int(v) for i, v in (line.split() for line in (base / "train_test_split.txt").read_text().splitlines())}
        train_ids = [i for i in paths if train_flags[i] == 1]
        test_ids = [i for i in paths if train_flags[i] == 0]
        if split in ("train", "val"):
            train_labels = [labels[i] for i in train_ids]
            val_indices, train_indices = _stratified_partition_indices(train_labels, (val_ratio, 1.0 - val_ratio), seed)
            selected = [train_ids[j] for j in (train_indices if split == "train" else val_indices)]
        else:
            selected = test_ids
        self.samples = [(str(base / "images" / paths[i]), labels[i]) for i in selected]
        self.targets = [label for _, label in self.samples]
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image) if self.transform else image, target


def _build_cub200(args, split):
    return _single_label(args, CUB200Dataset(args.data_path, split, _img_transforms(args, split == "train"), getattr(args, "split_seed", getattr(args, "seed", 42)), float(getattr(args, "val_ratio", 0.1))), 200)


class ListFileDataset(Dataset):
    """Image classification dataset described by ``relative_path label`` lines."""

    def __init__(self, root: str, list_file: str, transform):
        self.root = Path(root)
        self.transform = transform
        path = self.root / list_file
        if not path.is_file():
            raise FileNotFoundError(f"Required split list is missing: {path}")
        self.samples = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                relative, label = line.rsplit(maxsplit=1)
            except ValueError as exc:
                raise ValueError(f"Malformed line {line_number} in {path}: {line!r}") from exc
            self.samples.append((str(self.root / relative), int(label)))
        if not self.samples:
            raise ValueError(f"No samples were found in {path}")
        self.targets = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, target


def _build_vtab(args, split):
    split_file = {"train": "train800.txt", "val": "val200.txt", "test": "test.txt"}[split]
    dataset = ListFileDataset(args.data_path, split_file, _img_transforms(args, split == "train"))
    # Infer a stable output dimension across all available official list files.
    labels = []
    for filename in ("train800.txt", "val200.txt", "train800val200.txt", "test.txt"):
        path = Path(args.data_path) / filename
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    labels.append(int(line.rsplit(maxsplit=1)[1]))
    output_dim = max(labels) + 1 if labels else max(dataset.targets) + 1
    return _single_label(args, dataset, output_dim)


def _build_fewshot(args, split):
    from .fewshot import FewShotDataset, infer_fewshot_class_count

    dataset = FewShotDataset(
        root=args.data_path,
        split=split,
        transform=_img_transforms(args, split == "train"),
        shot=int(getattr(args, "fs_shot", 16)),
        seed=int(getattr(args, "seed", 0)),
    )
    output_dim = infer_fewshot_class_count(
        args.data_path,
        shot=int(getattr(args, "fs_shot", 16)),
        seed=int(getattr(args, "seed", 0)),
    )
    return _single_label(args, dataset, output_dim)


class BirdMetadataDataset(Dataset):
    """CUB/NABirds-style metadata with official train/test flags."""

    def __init__(self, root: str, split: str, transform, seed: int, val_ratio: float):
        base = Path(root)
        required = [base / "images.txt", base / "image_class_labels.txt", base / "train_test_split.txt", base / "images"]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Bird benchmark files are missing: " + ", ".join(missing))
        paths = {int(i): p for i, p in (line.strip().split(maxsplit=1) for line in (base / "images.txt").read_text().splitlines())}
        labels = {int(i): int(y) - 1 for i, y in (line.split() for line in (base / "image_class_labels.txt").read_text().splitlines())}
        flags = {int(i): int(v) for i, v in (line.split() for line in (base / "train_test_split.txt").read_text().splitlines())}
        train_ids = [identifier for identifier in paths if flags[identifier] == 1]
        test_ids = [identifier for identifier in paths if flags[identifier] == 0]
        if split in ("train", "val"):
            train_labels = [labels[identifier] for identifier in train_ids]
            val_indices, train_indices = _stratified_partition_indices(train_labels, (val_ratio, 1.0 - val_ratio), seed)
            chosen = train_indices if split == "train" else val_indices
            selected = [train_ids[index] for index in chosen]
        else:
            selected = test_ids
        self.samples = [(str(base / "images" / paths[identifier]), labels[identifier]) for identifier in selected]
        self.targets = [label for _, label in self.samples]
        self.num_classes = max(labels.values()) + 1
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, target


def _build_nabirds(args, split):
    dataset = BirdMetadataDataset(
        args.data_path,
        split,
        _img_transforms(args, split == "train"),
        int(getattr(args, "split_seed", getattr(args, "seed", 42))),
        float(getattr(args, "val_ratio", 0.1)),
    )
    return _single_label(args, dataset, dataset.num_classes)


class StanfordDogsDataset(Dataset):
    """Stanford Dogs official train/test lists with a stratified train-derived val split."""

    def __init__(self, root: str, split: str, transform, seed: int, val_ratio: float):
        from scipy.io import loadmat

        root_path = Path(root)
        list_path = root_path / ("train_list.mat" if split in ("train", "val") else "test_list.mat")
        images_root = root_path / "Images"
        if not list_path.is_file() or not images_root.is_dir():
            raise FileNotFoundError(f"Stanford Dogs requires {list_path} and {images_root}")
        metadata = loadmat(list_path)
        raw_files = metadata.get("file_list")
        raw_labels = metadata.get("labels")
        if raw_files is None or raw_labels is None:
            raise KeyError(f"{list_path} must contain file_list and labels")

        def decode(entry):
            value = entry
            while hasattr(value, "shape") and getattr(value, "size", 1) == 1 and not isinstance(value, str):
                value = value.item()
            return str(value)

        files = [decode(entry) for entry in raw_files.squeeze()]
        labels = [int(value) - 1 for value in raw_labels.squeeze().tolist()]
        indices = list(range(len(files)))
        if split in ("train", "val"):
            val_indices, train_indices = _stratified_partition_indices(labels, (val_ratio, 1.0 - val_ratio), seed)
            indices = train_indices if split == "train" else val_indices
        self.samples = [(str(images_root / files[index]), labels[index]) for index in indices]
        self.targets = [target for _, target in self.samples]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, target


def _build_stanford_dogs(args, split):
    dataset = StanfordDogsDataset(
        args.data_path,
        split,
        _img_transforms(args, split == "train"),
        int(getattr(args, "split_seed", getattr(args, "seed", 42))),
        float(getattr(args, "val_ratio", 0.1)),
    )
    return _single_label(args, dataset, 120)


class CSVDataset(Dataset):
    def __init__(self, csv_path: str, root: str, task_type: str, transform, image_column: str, label_column: str, separator: str, class_to_idx=None):
        self.root = Path(root)
        self.task_type = task_type
        self.transform = transform
        self.image_column = image_column
        self.label_column = label_column
        self.separator = separator
        with open(csv_path, newline="", encoding="utf-8-sig") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError(f"CSV dataset is empty: {csv_path}")
        if image_column not in self.rows[0] or label_column not in self.rows[0]:
            raise KeyError(f"CSV must contain columns {image_column!r} and {label_column!r}.")
        if task_type == TASK_SINGLE_LABEL:
            labels = sorted({row[label_column].strip() for row in self.rows})
            self.class_to_idx = class_to_idx or {label: idx for idx, label in enumerate(labels)}
            self.output_dim = len(self.class_to_idx)
        elif task_type == TASK_MULTILABEL:
            labels = sorted({label.strip() for row in self.rows for label in row[label_column].split(separator) if label.strip()})
            self.class_to_idx = class_to_idx or {label: idx for idx, label in enumerate(labels)}
            self.output_dim = len(self.class_to_idx)
        else:
            self.class_to_idx = None
            first = [v.strip() for v in self.rows[0][label_column].split(separator)]
            self.output_dim = len(first)

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = Path(row[self.image_column])
        if not path.is_absolute():
            path = self.root / path
        with Image.open(path) as image:
            image = image.convert("RGB")
        image = self.transform(image) if self.transform else image
        raw = row[self.label_column].strip()
        if self.task_type == TASK_SINGLE_LABEL:
            target = self.class_to_idx[raw]
        elif self.task_type == TASK_MULTILABEL:
            target = torch.zeros(self.output_dim, dtype=torch.float32)
            for label in raw.split(self.separator):
                label = label.strip()
                if label:
                    target[self.class_to_idx[label]] = 1.0
        else:
            target = torch.tensor([float(v.strip()) for v in raw.split(self.separator)], dtype=torch.float32)
        return image, target


def _build_csv(args, split):
    csv_path = getattr(args, f"{split}_csv", "")
    if not csv_path:
        raise ValueError(f"--{split}_csv is required for --dataset csv.")
    task_type = str(getattr(args, "task", TASK_SINGLE_LABEL)).lower()
    if task_type == "auto":
        task_type = TASK_SINGLE_LABEL
    class_to_idx = getattr(args, "csv_class_to_idx", None)
    dataset = CSVDataset(
        csv_path=csv_path,
        root=args.data_path,
        task_type=task_type,
        transform=_img_transforms(args, split == "train"),
        image_column=str(getattr(args, "image_column", "image")),
        label_column=str(getattr(args, "label_column", "label")),
        separator=str(getattr(args, "label_separator", ";")),
        class_to_idx=class_to_idx,
    )
    if split == "train" and dataset.class_to_idx is not None:
        args.csv_class_to_idx = dataset.class_to_idx
    if task_type == TASK_SINGLE_LABEL:
        return _single_label(args, dataset, dataset.output_dim)
    if task_type == TASK_MULTILABEL:
        return _multilabel(args, dataset, dataset.output_dim)
    return _regression(args, dataset, dataset.output_dim)


# ---------------------------------------------------------------------------
# Multi-label benchmarks
# ---------------------------------------------------------------------------
class CocoMultiLabel(Dataset):
    def __init__(self, img_root: str, ann_file: str, transform):
        self.base = CocoDetection(img_root, ann_file, transform=None, target_transform=None)
        self.transform = transform
        cat_ids = sorted(self.base.coco.getCatIds())
        self.catid2idx = {cid: i for i, cid in enumerate(cat_ids)}
        self.num_classes = len(cat_ids)
        self.keep = []
        for index, image_id in enumerate(self.base.ids):
            ann_ids = self.base.coco.getAnnIds(imgIds=[image_id], iscrowd=None)
            if ann_ids:
                self.keep.append(index)

    def __len__(self): return len(self.keep)

    def __getitem__(self, index):
        image, annotations = self.base[self.keep[index]]
        if self.transform:
            image = self.transform(image)
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        for annotation in annotations:
            category = annotation.get("category_id")
            if category in self.catid2idx:
                target[self.catid2idx[category]] = 1.0
        return image, target


class CocoMajorityLabel(CocoMultiLabel):
    """Legacy single-label reduction using the most frequent instance category."""

    def __getitem__(self, index):
        image, annotations = self.base[self.keep[index]]
        if self.transform:
            image = self.transform(image)
        counts = Counter(
            annotation.get("category_id")
            for annotation in annotations
            if annotation.get("category_id") in self.catid2idx
        )
        if not counts:
            return image, 0
        category_id, _ = max(counts.items(), key=lambda item: (item[1], -self.catid2idx[item[0]]))
        return image, int(self.catid2idx[category_id])


def _build_coco(args, split):
    split_dir = "train2017" if split in ("train", "val") else "val2017"
    img_root = os.path.join(args.data_path, split_dir)
    ann_file = os.path.join(args.data_path, "annotations", f"instances_{split_dir}.json")
    if not os.path.isdir(img_root) or not os.path.isfile(ann_file):
        raise FileNotFoundError(f"COCO requires {img_root} and {ann_file}.")
    mode = str(getattr(args, "coco_task", "multilabel")).lower()
    cls = CocoMajorityLabel if mode == "majority" else CocoMultiLabel
    base = cls(img_root, ann_file, _img_transforms(args, split == "train"))
    if split in ("train", "val"):
        base = _train_val_subset(base, split, getattr(args, "split_seed", getattr(args, "seed", 42)))
    if mode == "majority":
        return _single_label(args, base, (base.dataset if isinstance(base, Subset) else base).num_classes)
    return _multilabel(args, base, (base.dataset if isinstance(base, Subset) else base).num_classes)


VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)


class VOCMultiLabel(Dataset):
    def __init__(self, base: Dataset): self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, index):
        image, annotation = self.base[index]
        target = torch.zeros(len(VOC_CLASSES), dtype=torch.float32)
        objects = annotation.get("annotation", {}).get("object", [])
        if isinstance(objects, dict): objects = [objects]
        mapping = {name: i for i, name in enumerate(VOC_CLASSES)}
        for obj in objects:
            name = obj.get("name")
            if name in mapping: target[mapping[name]] = 1.0
        return image, target


def _build_voc2007(args, split):
    image_set = {"train": "train", "val": "val", "test": "test"}[split]
    base = datasets.VOCDetection(root=args.data_path, year="2007", image_set=image_set, download=bool(getattr(args, "download", False)), transform=_img_transforms(args, split == "train"))
    return _multilabel(args, VOCMultiLabel(base), 20)


class CelebAAttributes(Dataset):
    def __init__(self, base): self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, index):
        image, attributes = self.base[index]
        return image, attributes.float().clamp_min(0)


class CelebALandmarks(Dataset):
    """CelebA landmarks with resize-safe normalized coordinates.

    The underlying dataset returns the original PIL image and pixel-space
    coordinates. We normalize x/y by the original image dimensions before a
    deterministic direct resize, avoiding classification crops/flips that would
    invalidate the regression targets.
    """

    def __init__(self, base, transform):
        self.base = base
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, landmarks = self.base[index]
        width, height = image.size
        target = landmarks.float().reshape(-1, 2)
        target[:, 0] = target[:, 0] / max(1.0, float(width - 1))
        target[:, 1] = target[:, 1] / max(1.0, float(height - 1))
        image = self.transform(image) if self.transform is not None else image
        return image, target.flatten()


def _landmark_regression_transform(args):
    size = int(getattr(args, "input_size", 224))
    mean, std = _normalization(args)
    return T.Compose([
        T.Resize((size, size), interpolation=_interpolation(args)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def _build_celeba(args, split):
    name = {"train": "train", "val": "valid", "test": "test"}[split]
    task = str(getattr(args, "celeba_task", "attributes")).lower()
    if task == "landmarks":
        base = datasets.CelebA(
            root=args.data_path,
            split=name,
            target_type="landmarks",
            download=bool(getattr(args, "download", False)),
            transform=None,
        )
        return _regression(args, CelebALandmarks(base, _landmark_regression_transform(args)), 10)
    base = datasets.CelebA(
        root=args.data_path,
        split=name,
        target_type="attr",
        download=bool(getattr(args, "download", False)),
        transform=_img_transforms(args, split == "train"),
    )
    return _multilabel(args, CelebAAttributes(base), 40)


# ---------------------------------------------------------------------------
# Synthetic CI datasets for every supported task
# ---------------------------------------------------------------------------
class _SyntheticTaskDataset(Dataset):
    def __init__(self, size, image_size, output_dim, task_type, seed, transform):
        self.size, self.image_size, self.output_dim = int(size), int(image_size), int(output_dim)
        self.task_type, self.seed, self.transform = task_type, int(seed), transform
    def __len__(self): return self.size
    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(self.seed + index)
        image = T.ToPILImage()(torch.rand(3, self.image_size, self.image_size, generator=generator))
        image = self.transform(image) if self.transform else image
        if self.task_type == TASK_SINGLE_LABEL:
            target = int(torch.randint(self.output_dim, (1,), generator=generator).item())
        elif self.task_type == TASK_MULTILABEL:
            target = (torch.rand(self.output_dim, generator=generator) > 0.7).float()
            if target.sum() == 0: target[0] = 1.0
        else:
            target = torch.randn(self.output_dim, generator=generator)
        return image, target


def _build_fake(args, split):
    sizes = {"train": int(getattr(args, "fake_train_size", 32)), "val": int(getattr(args, "fake_val_size", 16)), "test": int(getattr(args, "fake_test_size", 16))}
    task_type = str(getattr(args, "task", "auto")).lower()
    if task_type == "auto": task_type = TASK_SINGLE_LABEL
    output_dim = int(getattr(args, "nb_classes", 10))
    dataset = _SyntheticTaskDataset(sizes[split], int(getattr(args, "input_size", 224)), output_dim, task_type, int(getattr(args, "seed", 0)) + {"train": 0, "val": 100000, "test": 200000}[split], _img_transforms(args, split == "train"))
    if task_type == TASK_MULTILABEL: return _multilabel(args, dataset, output_dim)
    if task_type == TASK_REGRESSION: return _regression(args, dataset, output_dim)
    return _single_label(args, dataset, output_dim)


_BUILDERS = {
    ("fake", "fakedata", "synthetic"): _build_fake,
    ("csv", "csv_dataset"): _build_csv,
    ("imagefolder", "folder", "imagenet", "imagenet1k", "imagenet_a", "imagenet_r", "imagenet_sketch", "tiny_imagenet", "tinyimagenet"): _build_imagefolder,
    ("cub200", "cub_200", "cub_200_2011", "cub"): _build_cub200,
    ("nabirds", "na_birds"): _build_nabirds,
    ("stanford_dogs", "stanforddogs", "dogs"): _build_stanford_dogs,
    ("vtab", "vtab1k", "vtab_1k"): _build_vtab,
    ("fewshot", "few_shot", "metadata_fewshot"): _build_fewshot,
    ("cifar10",): _build_cifar10,
    ("cifar100", "cifar_100"): _build_cifar100,
    ("mnist",): _build_mnist,
    ("fashion_mnist", "fashionmnist"): _build_fashion_mnist,
    ("emnist", "emnist_byclass"): _build_emnist,
    ("kmnist",): _build_kmnist,
    ("qmnist",): _build_qmnist,
    ("usps",): _build_usps,
    ("svhn",): _build_svhn,
    ("stl10",): _build_stl10,
    ("food101", "food_101"): _build_food101,
    ("oxfordiiitpet", "oxford_iiit_pet", "pets", "oxford_pets"): _build_pets,
    ("flowers102", "oxfordflowers102", "oxford_flowers102"): _build_flowers102,
    ("stanford_cars", "stanfordcars", "cars"): _build_cars,
    ("caltech101", "caltech_101"): _build_caltech101,
    ("dtd", "describable_textures", "textures"): _build_dtd,
    ("eurosat", "euro_sat"): _build_eurosat,
    ("fgvc_aircraft", "fgvca", "aircraft"): _build_fgvc_aircraft,
    ("sun397", "sun_397"): _build_sun397,
    ("gtsrb", "traffic_signs", "german_traffic_signs"): _build_gtsrb,
    ("fer2013", "fer_2013"): _build_fer2013,
    ("pcam", "patch_camelyon"): _build_pcam,
    ("country211", "country_211"): _build_country211,
    ("rendered_sst2", "renderedsst2"): _build_rendered_sst2,
    ("places365", "places_365"): _build_places365,
    ("inaturalist", "inat", "inat2021"): _build_inaturalist,
    ("coco", "coco2017", "mscoco", "mscoco2017"): _build_coco,
    ("voc2007", "pascal_voc", "pascal_voc2007"): _build_voc2007,
    ("celeba", "celeba_attributes"): _build_celeba,
}


def available_datasets() -> List[str]:
    return sorted({aliases[0] for aliases in _BUILDERS})


def _resolve_builder(name: str) -> Callable:
    normalized = (name or "").lower().replace("-", "_")
    for aliases, function in _BUILDERS.items():
        if normalized in aliases:
            return function
    raise NotImplementedError(f"Unsupported dataset '{normalized}'. Available: {', '.join(available_datasets())}")


def build_dataset_split(args, split: str) -> Tuple[Dataset, int]:
    split = split.lower()
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be train/val/test, got {split!r}")
    return _resolve_builder(args.dataset)(args, split)


def build_dataset(args, is_train: bool) -> Tuple[Dataset, int]:
    return build_dataset_split(args, "train" if is_train else "val")


__all__ = [
    "TASK_SINGLE_LABEL", "TASK_MULTILABEL", "TASK_REGRESSION",
    "available_datasets", "build_dataset", "build_dataset_split",
]
