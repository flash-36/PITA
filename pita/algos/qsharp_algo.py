from __future__ import annotations

from typing import Any, Dict, Tuple

from pita.core.registry import register_algorithm
from .base import ValueGuidedAlgorithms


@register_algorithm("Q#")
class QSharpAlgorithm(ValueGuidedAlgorithms):
    ALGO_KEY = "Q#"

    def run(
        self, cfg, ref_model: str, cls_model: str, dataset: str, family: str, output_dir
    ) -> Dict[str, Any]:
        return {
            "algo": "Q#",
            "ref_model": ref_model,
            "cls_model": cls_model,
            "dataset": dataset,
            "metrics": {"dummy": 0.0},
        }

    def score_samples(self, ex, y_a: str, y_b: str) -> Tuple[float, float, int]:
        ds = getattr(self, "_dataset", None)
        if ds is None:
            return 0.0, 0.0, None
        score_a = 1.0 if ds.is_correct(ex.answer, y_a) else 0.0
        score_b = 1.0 if ds.is_correct(ex.answer, y_b) else 0.0
        return score_a, score_b, None
