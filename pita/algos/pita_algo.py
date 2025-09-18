from __future__ import annotations

from typing import Any, Dict, Tuple
import math
import random

import torch
from transformers import AutoTokenizer, pipeline

from pita.core.registry import register_algorithm
from .base import ValueGuidedAlgorithms
from pita.models.hf import HFModel, GenerationConfig
from pita.datasets.registry import get_dataset


@register_algorithm("PITA")
class PITAAlgorithm(ValueGuidedAlgorithms):
    ALGO_KEY = "PITA"

    def generate_data(self, cfg, ref_model: str, dataset: str, family: str) -> None:
        ds_cfg = cfg.datasets[dataset]
        rm_model = str(ds_cfg.reward_model)
        device = 0 if torch.cuda.is_available() else -1
        self._bt_sampling = bool(cfg.common.bt_sampling)
        self._rm_tokenizer = AutoTokenizer.from_pretrained(
            rm_model, use_fast=True, trust_remote_code=True
        )
        self._rm_pipe = pipeline(
            "text-classification",
            model=rm_model,
            tokenizer=self._rm_tokenizer,
            device=device,
            model_kwargs={"torch_dtype": torch.bfloat16},
        )
        return super().generate_data(cfg, ref_model, dataset, family)

    def run(
        self, cfg, ref_model: str, cls_model: str, dataset: str, family: str, output_dir
    ) -> Dict[str, Any]:
        return {
            "algo": "PITA",
            "ref_model": ref_model,
            "cls_model": cls_model,
            "dataset": dataset,
            "metrics": {"dummy": 0.0},
        }

    def score_samples(self, ex, y_a: str, y_b: str) -> Tuple[float, float, int]:
        rm_pipe = self._rm_pipe
        rm_tok = self._rm_tokenizer

        msgs_a = [
            {"role": "user", "content": ex.question},
            {"role": "assistant", "content": y_a},
        ]
        msgs_b = [
            {"role": "user", "content": ex.question},
            {"role": "assistant", "content": y_b},
        ]
        texts = [
            rm_tok.apply_chat_template(
                msgs_a, tokenize=False, add_generation_prompt=False
            ),
            rm_tok.apply_chat_template(
                msgs_b, tokenize=False, add_generation_prompt=False
            ),
        ]

        outs = rm_pipe(texts, top_k=None, function_to_apply="none", batch_size=2)
        r_a = (
            float(outs[0][0]["score"])
            if isinstance(outs[0], list)
            else float(outs[0]["score"])
        )
        r_b = (
            float(outs[1][0]["score"])
            if isinstance(outs[1], list)
            else float(outs[1]["score"])
        )

        if self._bt_sampling:
            beta = float(self.cfg.bt_beta)
            p_a = 1.0 / (1.0 + math.exp(-beta * (r_a - r_b)))
            preferred = 0 if random.random() < p_a else 1
        else:
            preferred = 0 if r_a >= r_b else 1
        return r_a, r_b, preferred
