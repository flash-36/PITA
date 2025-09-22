from __future__ import annotations

from typing import Any, Dict, Tuple, Optional
import torch
from pita.models import RewardScorer

from pita.core.registry import register_algorithm
from .base import ValueGuidedAlgorithms


@register_algorithm("PITA")
class PITAAlgorithm(ValueGuidedAlgorithms):
    ALGO_KEY = "PITA"

    def generate_data(
        self,
        cfg,
        ref_model: str,
        dataset: str,
        family: str,
        round_idx: int,
        cls_model: Optional[str] = None,
    ) -> None:
        ds_cfg = cfg.datasets[dataset]
        rm_model = str(ds_cfg.reward_model)
        device = 0 if torch.cuda.is_available() else -1
        self._reward = RewardScorer(
            rm_model,
            bt_sampling=bool(cfg.common.bt_sampling),
            bt_beta=float(cfg.common.bt_beta),
            device=device,
        )
        return super().generate_data(
            cfg=cfg,
            ref_model=ref_model,
            dataset=dataset,
            family=family,
            round_idx=round_idx,
            cls_model=cls_model,
        )

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
            "algo": "PITA",
            "ref_model": ref_model,
            "cls_model": cls_model,
            "dataset": dataset,
            "metrics": {"dummy": 0.0},
        }

    def score_samples(self, ex, y_a: str, y_b: str) -> Tuple[float, float, int]:
        r_a, r_b, preferred = self._reward.score_pair(ex.question, y_a, y_b)
        return r_a, r_b, preferred
