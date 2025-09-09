from __future__ import annotations

from typing import Dict, Type


class AlgorithmBase:
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self, cfg, ref_model: str, cls_model: str, dataset: str, output_dir):
        raise NotImplementedError


_ALGO_REGISTRY: Dict[str, Type[AlgorithmBase]] = {}


def register_algorithm(name: str):
    def decorator(cls: Type[AlgorithmBase]):
        _ALGO_REGISTRY[name] = cls
        return cls

    return decorator


def get_algorithm_registry() -> Dict[str, Type[AlgorithmBase]]:
    return dict(_ALGO_REGISTRY)
