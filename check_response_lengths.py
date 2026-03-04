#!/usr/bin/env python3
"""Check y_a and y_b response lengths in pilot run CSVs to assess 512 token truncation impact."""

import pandas as pd
from pathlib import Path

BASE = Path("/mnt/shared-scratch/Shakkottai_S/ujwald36/repos/PITA/outputs/2026-03-03/17-03-15/datasets/PITA")
FILES = ["MATH_Qwen3_r1.csv", "TLDR_Qwen3_r1.csv", "IMDBGen_Qwen3_r1.csv"]


def stats(name, lengths):
    tokens = [l // 4 for l in lengths]
    return {
        "min": min(tokens),
        "max": max(tokens),
        "mean": sum(tokens) / len(tokens),
        "median": sorted(tokens)[len(tokens) // 2],
    }


def report(df, col):
    lengths = df[col].fillna("").astype(str).str.len()
    s = stats(col, lengths)
    print(f"  {col}: min={s['min']:.0f} max={s['max']:.0f} mean={s['mean']:.1f} median={s['median']:.0f} tokens")


for f in FILES:
    path = BASE / f
    if not path.exists():
        continue
    df = pd.read_csv(path)
    print(f"\n{f} (n={len(df)})")
    report(df, "y_a")
    report(df, "y_b")
    over_512 = (df["y_a"].fillna("").astype(str).str.len() // 4 > 512).sum()
    over_512 += (df["y_b"].fillna("").astype(str).str.len() // 4 > 512).sum()
    print(f"  responses >512 tokens: {over_512}")
