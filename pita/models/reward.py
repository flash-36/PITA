from __future__ import annotations

from typing import Tuple, List

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
            truncation=True,  # Truncate sequences that exceed model max_length
            max_length=512,  # Standard max length for most reward models
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

    def score_batch(
        self, pairs: List[Tuple[str, str, str]]
    ) -> List[Tuple[float, float, int]]:
        """Score multiple (question, y_a, y_b) pairs in batch for efficiency.

        Args:
            pairs: List of (question, y_a, y_b) tuples to score

        Returns:
            List of (score_a, score_b, preferred) tuples
        """
        if not pairs:
            return []

        # Build all texts for batching
        all_texts = []
        if "distilbert-imdb" in self._model_id:
            # For IMDb, we just score y_a and y_b directly
            for _, y_a, y_b in pairs:
                all_texts.extend([y_a, y_b])
        else:
            # For other models, build reward model prompts
            for question, y_a, y_b in pairs:
                texts = build_reward_model_prompt(
                    question=question, y_a=y_a, y_b=y_b, tokenizer=self._tokenizer
                )
                all_texts.extend(texts)

        # Batch score all texts at once
        outs = self._pipe(
            all_texts, top_k=None, function_to_apply="none", batch_size=self._batch_size
        )

        # Parse results and compute preferences
        results = []
        for i in range(len(pairs)):
            if "distilbert-imdb" in self._model_id:
                # Extract POSITIVE scores for y_a and y_b
                r_a = [d for d in outs[i * 2] if d["label"] == "POSITIVE"][0]["score"]
                r_b = [d for d in outs[i * 2 + 1] if d["label"] == "POSITIVE"][0][
                    "score"
                ]
            else:
                # Extract scores from reward model outputs
                r_a = (
                    float(outs[i * 2][0]["score"])
                    if isinstance(outs[i * 2], list)
                    else float(outs[i * 2]["score"])
                )
                r_b = (
                    float(outs[i * 2 + 1][0]["score"])
                    if isinstance(outs[i * 2 + 1], list)
                    else float(outs[i * 2 + 1]["score"])
                )

            # Compute preference
            if self._bt_sampling:
                beta = self._bt_beta
                p_a = 1.0 / (1.0 + math.exp(-beta * (r_a - r_b)))
                preferred = 0 if random.random() < p_a else 1
            else:
                preferred = 0 if r_a >= r_b else 1

            results.append((r_a, r_b, preferred))

        return results

    def score_batch_single(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Score multiple (question, y) pairs in batch for efficiency.

        Args:
            pairs: List of (question, y) tuples to score

        Returns:
            List of reward scores
        """
        if not pairs:
            return []

        # Build all texts for batching
        all_texts = []
        if "distilbert-imdb" in self._model_id:
            # For IMDb, we just score y directly
            for _, y in pairs:
                all_texts.append(y)
        else:
            # For other models, build reward model prompts
            for question, y in pairs:
                texts = build_reward_model_prompt(
                    question=question, y_a=y, y_b=y, tokenizer=self._tokenizer
                )
                # Use only the first text to avoid duplicate compute
                all_texts.append(texts[0])

        # Batch score all texts at once
        outs = self._pipe(
            all_texts, top_k=None, function_to_apply="none", batch_size=self._batch_size
        )

        # Parse results
        results = []
        for i in range(len(pairs)):
            if "distilbert-imdb" in self._model_id:
                # Extract POSITIVE score
                r = [d for d in outs[i] if d["label"] == "POSITIVE"][0]["score"]
            else:
                # Extract score from reward model output
                r = (
                    float(outs[i][0]["score"])
                    if isinstance(outs[i], list)
                    else float(outs[i]["score"])
                )
            results.append(r)

        return results
