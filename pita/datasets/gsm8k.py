from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import re
from datasets import load_dataset
from .utils import extract_boxed_last, eq_math, extract_final_answer

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
    def is_correct(gold: str, pred_text: str) -> bool:
        def clean_gold(a: str) -> str:
            g = extract_boxed_last(a)
            if g:
                return g
            m = re.search(r"####\s*([^\n]+)", a)
            return m.group(1).strip() if m else a.strip()

        gold_clean = clean_gold(gold)
        pred_clean = extract_final_answer(pred_text)
        return eq_math(pred_clean, gold_clean)
