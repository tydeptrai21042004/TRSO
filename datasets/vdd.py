"""Visual Decathlon compatibility notice.

The old module contained incomplete COCO-path formatting and undefined state.
It is intentionally not routed as a supported benchmark. Use the generic
ImageFolder/CSV routes, or add the official Visual Decathlon annotation and
scoring protocol before claiming support.
"""
from __future__ import annotations


class VisualDecathlonDataset:
    def __init__(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            "Official Visual Decathlon is not implemented. The previous dormant "
            "loader was removed because it used malformed annotation paths and "
            "did not implement the benchmark protocol."
        )


# Historical name retained as a fail-fast alias.
ImageFolder = VisualDecathlonDataset

__all__ = ["VisualDecathlonDataset", "ImageFolder"]
