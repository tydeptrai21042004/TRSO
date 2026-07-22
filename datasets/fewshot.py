"""Generic metadata-list few-shot dataset.

Expected files under ``<root>/annotations``:

* ``train_meta.list.num_shot_<shot>.seed_<seed>``
* ``val_meta.list``
* ``test_meta.list``

Each non-empty line must end with an integer class label. The preceding text is
interpreted as a relative image path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image
from torch.utils.data import Dataset


class FewShotDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        target_transform=None,
        shot: int = 16,
        seed: int = 0,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        split = str(split).lower()
        if split == "train":
            filename = f"train_meta.list.num_shot_{int(shot)}.seed_{int(seed)}"
        elif split == "val":
            filename = "val_meta.list"
        elif split == "test":
            filename = "test_meta.list"
        else:
            raise ValueError(f"split must be train/val/test, got {split!r}")

        list_path = self.root / "annotations" / filename
        if not list_path.is_file():
            raise FileNotFoundError(
                f"Few-shot split list is missing: {list_path}. "
                "Create the metadata file or use --dataset imagefolder/csv."
            )

        self.samples = []
        for line_number, line in enumerate(list_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                relative_path, raw_label = line.rsplit(maxsplit=1)
                label = int(raw_label)
            except ValueError as exc:
                raise ValueError(
                    f"Malformed line {line_number} in {list_path}: {line!r}; "
                    "expected '<relative image path> <integer label>'."
                ) from exc
            image_path = self._resolve_image_path(relative_path)
            self.samples.append((str(image_path), label))

        if not self.samples:
            raise ValueError(f"No samples were found in {list_path}.")
        self.targets = [label for _, label in self.samples]

    def _resolve_image_path(self, relative_path: str) -> Path:
        relative = Path(relative_path)
        candidates = [
            relative if relative.is_absolute() else self.root / relative,
            self.root / "data" / "images" / relative,
            self.root / "images" / relative,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        # Preserve the most direct path in the eventual error message.
        raise FileNotFoundError(
            f"Few-shot image {relative_path!r} was not found. Checked: "
            + ", ".join(str(path) for path in candidates)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


def infer_fewshot_class_count(root: str, shot: int = 16, seed: int = 0) -> int:
    """Infer the output dimension from every available metadata split."""
    annotation_root = Path(root) / "annotations"
    filenames = [
        f"train_meta.list.num_shot_{int(shot)}.seed_{int(seed)}",
        "val_meta.list",
        "test_meta.list",
    ]
    labels = []
    for filename in filenames:
        path = annotation_root / filename
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                labels.append(int(line.rsplit(maxsplit=1)[1]))
    if not labels:
        raise FileNotFoundError(f"No few-shot metadata lists were found under {annotation_root}.")
    return max(labels) + 1


__all__ = ["FewShotDataset", "infer_fewshot_class_count"]
