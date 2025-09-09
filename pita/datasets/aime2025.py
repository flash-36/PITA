from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Dict, Any, List
import re

from datasets import load_dataset

from .registry import register_dataset


@dataclass
class Sample:
    question: str
    answer: str


@register_dataset("AIME")
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
            "You are a careful mathematician. Solve the following problem.\n"
            "Provide ONLY the final numeric answer on the last line after the word Answer:.\n\n"
            f"Problem: {question}\n\nAnswer:"
        )

    @staticmethod
    def extract_numeric(text: str) -> str:
        # Extract the last integer-like token (AIME answers are typically integers 0-999)
        candidates = re.findall(r"-?\d+", text)
        return candidates[-1] if candidates else ""
