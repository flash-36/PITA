from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator
from pathlib import Path

from datasets import Dataset as HFDataset
from torch.utils.data import Dataset


@dataclass
class GRPOSample:
    prompt: str
    response: str
    reward: float
    group_id: int
    question: str
    answer: str


class GRPODataset(Dataset):
    def __init__(self, hf_dataset_path: str | Path | HFDataset):
        if isinstance(hf_dataset_path, (str, Path)):
            self.ds = HFDataset.load_from_disk(str(hf_dataset_path))
        else:
            self.ds = hf_dataset_path

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> GRPOSample:
        row = self.ds[idx]
        return GRPOSample(
            prompt=row["prompt"],
            response=row["response"],
            reward=float(row["reward"]),
            group_id=int(row["group_id"]),
            question=row["question"],
            answer=row["answer"],
        )

    def iter(self) -> Iterator[GRPOSample]:
        for i in range(len(self)):
            yield self[i]
