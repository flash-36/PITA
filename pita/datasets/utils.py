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
