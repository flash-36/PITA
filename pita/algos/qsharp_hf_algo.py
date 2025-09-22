from __future__ import annotations

from typing import Any, Dict, Tuple
from pathlib import Path

from pita.core.registry import register_algorithm
from .base import ValueGuidedAlgorithms


@register_algorithm("QSharpHF")
class QSharpHFAlgorithm(ValueGuidedAlgorithms):
    ALGO_KEY = "QSharpHF"

    def generate_data(
        self,
        cfg,
        ref_model: str,
        dataset: str,
        family: str,
        round_idx: int,
        cls_model: str | None = None,
    ) -> None:
        return None

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
            "algo": "QSharpHF",
            "ref_model": ref_model,
            "cls_model": cls_model,
            "dataset": dataset,
            "metrics": {"dummy_reward": 0.0},
        }
