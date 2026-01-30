from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Dict, Any, List
import re

from datasets import load_dataset
from .utils import extract_boxed_last, eq_math

from .registry import register_dataset


@dataclass
class Sample:
    question: str
    answer: str


@register_dataset("AIME2025")
class AIME2025:
    def __init__(self, hf_config: str, split: str, question_key: str, answer_key: str):
        self.ds = load_dataset("opencompass/AIME2025", hf_config, split=split)
        self.q_key = question_key
        self.a_key = answer_key

    def __len__(self) -> int:
        return len(self.ds)

    def iter(self) -> Iterable[Sample]:
        for ex in self.ds:
            q = ex[self.q_key]
            a = ex[self.a_key]
            yield Sample(question=str(q), answer=str(a))

    @staticmethod
    def hydrate_prompt(question: str) -> str:
        return (
            f"Problem:\n\n{question} Write your answer inside \\boxed{{}}.\n\nSolution:"
        )

    @staticmethod
    def is_correct(gold: str, pred_text: str) -> bool:
        pred_clean = extract_boxed_last(pred_text)
        return eq_math(pred_clean, gold)
