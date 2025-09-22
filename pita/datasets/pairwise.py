from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from pathlib import Path

from torch.utils.data import Dataset as TorchDataset
from datasets import load_from_disk, Dataset


@dataclass
class PairExample:
    prompt: str
    chosen: str
    rejected: str


class PreferencePairDataset(TorchDataset):
    def __init__(self, hf_dir: Path):
        self.hf_dir = Path(hf_dir)
        self.ds: Dataset = load_from_disk(str(self.hf_dir))

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> PairExample:
        ex = self.ds[int(idx)]
        prompt = str(ex["prompt"])
        y_a = str(ex["y_a"])
        y_b = str(ex["y_b"])
        preferred = int(ex["preferred"])
        if preferred == 0:
            chosen, rejected = y_a, y_b
        else:
            chosen, rejected = y_b, y_a
        return PairExample(prompt=prompt, chosen=chosen, rejected=rejected)
