from __future__ import annotations

from typing import Tuple

import torch
from transformers import AutoTokenizer, pipeline
from pita.core.prompts import build_reward_model_prompt
import math
import random


class RewardScorer:
    def __init__(
        self,
        model_id: str,
        *,
        bt_sampling: bool,
        bt_beta: float,
        device: int | str | torch.device,
        dtype: str,
        batch_size: int,
    ) -> None:
        self._model_id = str(model_id)
        self._bt_sampling = bool(bt_sampling)
        self._bt_beta = float(bt_beta)
        self._batch_size = int(batch_size)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=True, trust_remote_code=True
        )
        dev = device
        if isinstance(dev, str) and dev == "auto":
            dev = 0 if torch.cuda.is_available() else -1
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }[dtype]
        self._pipe = pipeline(
            "text-classification",
            model=model_id,
            tokenizer=self._tokenizer,
            device=dev,
            model_kwargs={"torch_dtype": torch_dtype},
        )

    @property
    def tokenizer(self):
        return self._tokenizer

    def score_pair(self, question: str, y_a: str, y_b: str) -> Tuple[float, float, int]:
        if "distilbert-imdb" in self._model_id:
            texts = [y_a, y_b]
            outs = self._pipe(
                texts, top_k=None, function_to_apply="none", batch_size=self._batch_size
            )
            r_a = [d for d in outs[0] if d["label"] == "POSITIVE"][0]["score"]
            r_b = [d for d in outs[1] if d["label"] == "POSITIVE"][0]["score"]
        else:
            texts = build_reward_model_prompt(
                question=question, y_a=y_a, y_b=y_b, tokenizer=self._tokenizer
            )
            outs = self._pipe(
                texts, top_k=None, function_to_apply="none", batch_size=self._batch_size
            )
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
            beta = self._bt_beta
            p_a = 1.0 / (1.0 + math.exp(-beta * (r_a - r_b)))
            preferred = 0 if random.random() < p_a else 1
        else:
            preferred = 0 if r_a >= r_b else 1
        return r_a, r_b, preferred

    def score_single(self, question: str, y: str) -> float:
        # IMDb reward model scores the text directly
        if "distilbert-imdb" in self._model_id:
            outs = self._pipe(
                [y], top_k=None, function_to_apply="none", batch_size=self._batch_size
            )
            # outs[0] is a list of label scores
            return [d for d in outs[0] if d["label"] == "POSITIVE"][0]["score"]
        # Otherwise, score the (question, answer) pair using the reward prompt
        texts = build_reward_model_prompt(
            question=question, y_a=y, y_b=y, tokenizer=self._tokenizer
        )
        # Feed only the first constructed input to avoid duplicate compute
        outs = self._pipe(
            [texts[0]],
            top_k=None,
            function_to_apply="none",
            batch_size=self._batch_size,
        )
        if isinstance(outs[0], list):
            return float(outs[0][0]["score"])  # type: ignore[index]
        return float(outs[0]["score"])  # type: ignore[index]
