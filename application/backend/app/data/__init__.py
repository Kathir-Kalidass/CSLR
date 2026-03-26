"""Lazy exports for dataset modules.

Avoid importing heavyweight backend dependencies when callers only need the
iSign dataset helpers.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "CSLRVideoDataset",
    "collate_fn",
    "ISignDataset",
    "isign_collate_fn",
    "build_dataloaders",
]

_EXPORT_MAP = {
    "CSLRVideoDataset": (".video_dataset", "CSLRVideoDataset"),
    "collate_fn": (".video_dataset", "collate_fn"),
    "ISignDataset": (".isign_dataset", "ISignDataset"),
    "isign_collate_fn": (".isign_dataset", "isign_collate_fn"),
    "build_dataloaders": (".isign_dataset", "build_dataloaders"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
