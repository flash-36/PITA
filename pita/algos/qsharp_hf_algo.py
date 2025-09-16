from __future__ import annotations

from typing import Any, Dict, Tuple

from pita.core.registry import register_algorithm
from .base import ValueGuidedAlgorithms


@register_algorithm("Q#HF")
class QSharpHFAlgorithm(ValueGuidedAlgorithms):
    ALGO_KEY = "Q#HF"

    def run(
        self, cfg, ref_model: str, cls_model: str, dataset: str, output_dir
    ) -> Dict[str, Any]:
        return {
            "algo": "Q#HF",
            "ref_model": ref_model,
            "cls_model": cls_model,
            "dataset": dataset,
            "metrics": {"dummy_reward": 0.0},
        }

    def score_samples(self, ex, y_a: str, y_b: str) -> Tuple[float, float, int]:
        return 0.0, 0.0, 0