from __future__ import annotations

from typing import Any, Dict

from pita.core.registry import AlgorithmBase, register_algorithm


@register_algorithm("Q#HF")
class QSharpHFAlgorithm(AlgorithmBase):
    def run(self, cfg, ref_model: str, cls_model: str, dataset: str, output_dir) -> Dict[str, Any]:
        return {
            "algo": "Q#HF",
            "ref_model": ref_model,
            "cls_model": cls_model,
            "dataset": dataset,
            "metrics": {"dummy_reward": 0.0},
        }
