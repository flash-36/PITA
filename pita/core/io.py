from __future__ import annotations

from pathlib import Path
from typing import Iterable, Dict, Any
import json
from hydra.core.hydra_config import HydraConfig


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
