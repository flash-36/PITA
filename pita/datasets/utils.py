from __future__ import annotations

import re
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


def eq_math(u: str, v: str) -> bool:
    if not u or not v:
        return False
    if u == v:
        return True
    return verify(parse("$" + v + "$"), parse("$" + u + "$"))


def extract_final_answer(s: str) -> str:
    a = extract_boxed_last(s)
    if a:
        return a
    m = re.search(r"####\s*([^\n]+)", s)
    if m:
        return m.group(1).strip()
    dollars = re.findall(r"\$([^$]+)\$", s)
    if dollars:
        cand = dollars[-1].strip()
        cand = re.sub(r"<<[^>]+>>", "", cand)
        inner_nums = re.findall(r"-?\d+(?:\.\d+)?", cand)
        if inner_nums:
            return inner_nums[-1]
        return cand
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    if nums:
        return nums[-1]
    return ""
