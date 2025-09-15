from __future__ import annotations

from typing import Any, Dict

from pita.core.registry import register_algorithm
from .base import PostTrainingAlgorithms


@register_algorithm("DPO")
class DPOAlgorithm(PostTrainingAlgorithms):
    ALGO_KEY = "DPO"

    def run(
        self, cfg, ref_model: str, cls_model: str, dataset: str, output_dir
    ) -> Dict[str, Any]:
        return {
            "algo": "DPO",
            "ref_model": ref_model,
            "cls_model": cls_model,
            "dataset": dataset,
            "metrics": {"dummy_loss": 0.0},
        }
