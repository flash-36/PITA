from __future__ import annotations

from typing import Any, Dict

from pita.core.registry import register_algorithm
from .base import PostTrainingAlgorithms


@register_algorithm("RLHF")
class RLHFAlgorithm(PostTrainingAlgorithms):
    ALGO_KEY = "RLHF"

    def run(
        self, cfg, ref_model: str, cls_model: str, dataset: str, output_dir
    ) -> Dict[str, Any]:
        return {
            "algo": "RLHF",
            "ref_model": ref_model,
            "cls_model": cls_model,
            "dataset": dataset,
            "metrics": {"dummy_reward": 0.0},
        }
