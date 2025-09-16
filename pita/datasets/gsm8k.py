from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset

from .registry import register_dataset


@dataclass
class Sample:
    question: str
    answer: str


@register_dataset("GSM8K")
class GSM8K:
    def __init__(self, hf_config: str, split: str, question_key: str, answer_key: str):
        self.ds = load_dataset("openai/gsm8k", hf_config, split=split)
        self.q_key = question_key
        self.a_key = answer_key

    def __len__(self) -> int:
        return len(self.ds)

    def iter(self) -> Iterable[Sample]:
        for ex in self.ds:
            yield Sample(question=str(ex[self.q_key]), answer=str(ex[self.a_key]))

    @staticmethod
    def hydrate_prompt(question: str) -> str:
        return (
            f"Problem:\n\n{question} Write your answer inside \\boxed{{}}.\n\nSolution:"
        )

    @staticmethod
    def extract_numeric(text: str) -> str:
        import re

        m = re.search(r"\\boxed\{([^}]*)\}", text)
        if m:
            return m.group(1).strip()
        return ""
