from __future__ import annotations

from typing import Dict, Type

_DATASET_REGISTRY: Dict[str, Type] = {}


def register_dataset(name: str):
    def deco(cls: Type):
        _DATASET_REGISTRY[name] = cls
        return cls
    return deco


def get_dataset(name: str):
    cls = _DATASET_REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"Unknown dataset: {name}")
    return cls
