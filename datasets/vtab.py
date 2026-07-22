"""Compatibility exports for the active VTAB list-file implementation."""
from __future__ import annotations

from .build import ListFileDataset


class VTABDataset(ListFileDataset):
    """Backward-compatible wrapper around the active VTAB list loader."""

    def __init__(self, root, train=True, transform=None, target_transform=None, is_tuning=True, **kwargs):
        del target_transform, kwargs
        if is_tuning:
            filename = "train800.txt" if train else "val200.txt"
        else:
            filename = "train800val200.txt" if train else "test.txt"
        super().__init__(root=root, list_file=filename, transform=transform)


__all__ = ["VTABDataset", "ListFileDataset"]
