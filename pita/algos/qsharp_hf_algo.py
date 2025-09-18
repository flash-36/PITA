from __future__ import annotations

from typing import Any, Dict, Tuple
from pathlib import Path

from hydra.utils import get_original_cwd

from pita.core.registry import register_algorithm
from .base import ValueGuidedAlgorithms


@register_algorithm("Q#HF")
class QSharpHFAlgorithm(ValueGuidedAlgorithms):
    ALGO_KEY = "Q#HF"

    def generate_data(self, cfg, ref_model: str, dataset: str, family: str) -> None:
        orig_root = Path(get_original_cwd())
        family_cap = str(family).capitalize()
        pita_hf_dir = (
            orig_root / "outputs" / "datasets" / "PITA" / f"{dataset}_{family_cap}.hf"
        )
        assert (
            pita_hf_dir.exists()
        ), f"Missing PITA dataset: {pita_hf_dir} to be used for Q#HF"

    def run(
        self, cfg, ref_model: str, cls_model: str, dataset: str, family: str, output_dir
    ) -> Dict[str, Any]:
        return {
            "algo": "Q#HF",
            "ref_model": ref_model,
            "cls_model": cls_model,
            "dataset": dataset,
            "metrics": {"dummy_reward": 0.0},
        }
