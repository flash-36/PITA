from __future__ import annotations

from typing import Any, Dict

from pita.core.registry import register_algorithm
from .base import AlgorithmBase


@register_algorithm("BEST_OF_N")
class BestOfNAlgorithm(AlgorithmBase):
    ALGO_KEY = "BEST_OF_N"

    def generate_data(
        self,
        cfg,
        ref_model: str,
        dataset: str,
        family: str,
        round_idx: int,
        cls_model: str | None = None,
    ) -> None:
        # No data collection required for Best-of-N
        pass

    def run(
        self,
        cfg,
        ref_model: str,
        cls_model: str,
        dataset: str,
        family: str,
        output_dir,
        round_idx: int,
    ) -> Dict[str, Any]:
        return {
            "algo": "BEST_OF_N",
            "ref_model": ref_model,
            "cls_model": cls_model,
            "dataset": dataset,
            "metrics": {"dummy": 0.0},
        }
