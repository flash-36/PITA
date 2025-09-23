from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset

from .registry import register_dataset


@dataclass
class Sample:
    question: str
    answer: str


@register_dataset("TLDR")
class TLDR:
    def __init__(self, hf_config: str, split: str, question_key: str, answer_key: str):
        self.ds = load_dataset("trl-lib/tldr", split=split)
        self.q_key = question_key
        self.a_key = answer_key

    def __len__(self) -> int:
        return len(self.ds)

    def iter(self) -> Iterable[Sample]:
        for ex in self.ds:
            yield Sample(
                question=str(ex[self.q_key]), answer=str(ex.get(self.a_key, ""))
            )

    @staticmethod
    def hydrate_prompt(question: str) -> str:
        return (
            "Summarize the following content briefly and clearly.\n\n"
            f"Content:\n\n{question}\n\nSummary:"
        )

    @staticmethod
    def is_correct(gold: str, pred_text: str) -> bool:
        return True
