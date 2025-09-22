from __future__ import annotations

from pathlib import Path
from typing import Iterable, Dict, Any, Optional, Tuple
import json
from hydra.core.hydra_config import HydraConfig
from datasets import Dataset, load_from_disk, concatenate_datasets


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_subdir(root: Path, parts: Iterable[str]) -> Path:
    sub = root
    for p in parts:
        sub = sub / str(p)
    return ensure_dir(sub)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_run_root() -> Path:
    return Path(HydraConfig.get().runtime.output_dir)


def get_snapshot_paths(
    algo_key: str, dataset: str, family: str, round_idx: int
) -> Tuple[Optional[Path], Path, Path]:
    run_root = get_run_root()
    ds_root = run_root / "datasets" / str(algo_key)
    ds_root.mkdir(parents=True, exist_ok=True)
    family_cap = str(family).capitalize()
    r = int(round_idx) + 1
    prev_hf = ds_root / f"{dataset}_{family_cap}_r{r-1}.hf" if r > 1 else None
    cur_hf = ds_root / f"{dataset}_{family_cap}_r{r}.hf"
    cur_csv = ds_root / f"{dataset}_{family_cap}_r{r}.csv"
    return prev_hf, cur_hf, cur_csv


def merge_and_save_hf(
    prev_hf: Optional[Path], new_ds: Dataset, cur_hf: Path, cur_csv: Path
) -> None:
    if prev_hf is not None and prev_hf.exists():
        old_ds = load_from_disk(str(prev_hf))
        merged = concatenate_datasets([old_ds, new_ds])
    else:
        merged = new_ds
    merged.save_to_disk(str(cur_hf))
    merged.to_csv(str(cur_csv))
