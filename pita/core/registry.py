from __future__ import annotations

from typing import Dict, Type


_ALGO_REGISTRY: Dict[str, Type] = {}


def register_algorithm(name: str):
    def decorator(cls: Type):
        _ALGO_REGISTRY[name] = cls
        return cls

    return decorator


def get_algorithm_registry() -> Dict[str, Type]:
    return dict(_ALGO_REGISTRY)
