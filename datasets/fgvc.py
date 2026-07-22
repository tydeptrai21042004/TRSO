"""Compatibility aliases for fine-grained datasets in the active router.

New code should use ``datasets.build.build_dataset_split`` with dataset keys
``cub200``, ``stanford_cars``, ``stanford_dogs``, ``nabirds`` or
``fgvc_aircraft``.
"""
from __future__ import annotations

from .build import BirdMetadataDataset, CUB200Dataset, StanfordDogsDataset


# Historical import names retained without duplicating a second dataset stack.
CUB2011 = CUB200Dataset
DogsDataset = StanfordDogsDataset

__all__ = [
    "CUB2011",
    "DogsDataset",
    "CUB200Dataset",
    "BirdMetadataDataset",
    "StanfordDogsDataset",
]
