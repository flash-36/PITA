from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset

from .registry import register_dataset


@dataclass
class Sample:
    question: str
    answer: str


@register_dataset("IMDBGen")
class IMDBGen:
    def __init__(self, hf_config: str, split: str, question_key: str, answer_key: str):
        self.ds = load_dataset("stanfordnlp/imdb", split=split)
        self.q_key = question_key
        self.a_key = answer_key

    def __len__(self) -> int:
        return len(self.ds)

    def iter(self) -> Iterable[Sample]:
        for ex in self.ds:
            text = str(ex[self.q_key])
            prefix = " ".join(text.split()[:60])
            yield Sample(question=prefix, answer=str(ex.get(self.a_key, "")))

    @staticmethod
    def hydrate_prompt(question: str) -> str:
        return (
            "Continue this movie review in a very positive, enthusiastic tone.\n\n"
            f"Review start:\n\n{question}\n\nPositive continuation:"
        )

    @staticmethod
    def is_correct(gold: str, pred_text: str) -> bool:
        return True
