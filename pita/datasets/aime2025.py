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
            f"Problem:\n\n{question} Write your answer inside \\boxed{{}}.\n\nSolution:"
        )



    @staticmethod
    def is_correct(gold: str, pred_text: str) -> bool:
        from math_verify import parse, verify

        def extract_boxed_last(s: str) -> str:
            matches = list(re.finditer(r"\\boxed\{", s))
            if not matches:
                return ""
            start = matches[-1].end()
            stack = 1
            i = start
            while i < len(s) and stack > 0:
                if s[i] == "{":
                    stack += 1
                elif s[i] == "}":
                    stack -= 1
                i += 1
            return s[start : i - 1].strip() if stack == 0 else ""

        def eq(u: str, v: str) -> bool:
            if not u or not v:
                return False
            if u == v:
                return True
            return verify(parse("$" + v + "$"), parse("$" + u + "$"))

        pred_clean = extract_boxed_last(pred_text)
        return eq(pred_clean, gold)
