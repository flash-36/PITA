from __future__ import annotations

from pathlib import Path
from typing import Iterable, Dict, Any, Optional, Tuple
import json
import shutil
from hydra.core.hydra_config import HydraConfig
from datasets import Dataset, load_from_disk, concatenate_datasets
from loguru import logger


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


def check_disk_space(path: Path, required_gb: float = 5.0) -> bool:
    """Check if there's enough disk space available.

    Args:
        path: Path to check disk space for
        required_gb: Required free space in GB

    Returns:
        True if enough space available, False otherwise
    """
    try:
        stat = shutil.disk_usage(path)
        free_gb = stat.free / (1024**3)

        if free_gb < required_gb:
            logger.warning(
                f"⚠️  Low disk space: {free_gb:.2f} GB free (< {required_gb:.2f} GB recommended)"
            )
            return False
        return True
    except Exception as e:
        logger.warning(f"⚠️  Could not check disk space: {e}")
        return True  # Don't block execution if we can't check


def get_snapshot_paths(
    algo_key: str,
    dataset: str,
    family: str,
    round_idx: int,
    run_root: Optional[Path] = None,
) -> Tuple[Optional[Path], Path, Path]:
    if run_root is None:
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


def check_data_generated(
    algo_key: str,
    dataset: str,
    family: str,
    round_idx: int,
    run_root: Path,
) -> bool:
    """Check if data generation is complete for a job."""
    _, cur_hf, _ = get_snapshot_paths(algo_key, dataset, family, round_idx, run_root)
    return cur_hf.exists()


def check_model_trained(
    algo_key: str,
    dataset: str,
    family: str,
    round_idx: int,
    run_root: Path,
) -> bool:
    """Check if model training is complete for a job.

    For HF-based algorithms, checks for config.json which is only present after
    a full model save. For other algorithms (PITA, QSharp, QSharp-HF), checks
    for classifier.pt which is the trained classifier checkpoint.
    """
    family_cap = str(family).capitalize()
    r = int(round_idx) + 1
    model_path = run_root / "models" / algo_key / f"{dataset}_{family_cap}_r{r}"
    if not model_path.exists():
        return False

    # Check for HF model files (DPO, GRPO, GRPO-HF after final training)
    has_hf_model = (model_path / "config.json").exists()
    # Check for classifier checkpoint (PITA, QSharp, QSharp-HF)
    has_classifier = (model_path / "classifier.pt").exists()

    return has_hf_model or has_classifier


def check_eval_completed(
    algo_key: str,
    dataset: str,
    family: str,
    round_idx: int,
    run_root: Path,
) -> bool:
    """Check if evaluation is complete for a job."""
    r = int(round_idx) + 1
    result_path = (
        run_root
        / "results"
        / algo_key
        / str(family)
        / dataset
        / f"r{r}"
        / "result.json"
    )
    return result_path.exists()


def check_base_eval_completed(
    family: str,
    dataset: str,
    run_root: Path,
) -> bool:
    """Check if base model evaluation is complete."""
    result_path = (
        run_root / "results" / "base_model" / str(family) / dataset / "results.json"
    )
    return result_path.exists()


def check_base_eval_cot8_completed(
    family: str,
    dataset: str,
    run_root: Path,
) -> bool:
    """Check if base model 8-shot CoT evaluation is complete."""
    result_path = (
        run_root
        / "results"
        / "base_model_cot8"
        / str(family)
        / dataset
        / "results.json"
    )
    return result_path.exists()


def mark_phase_complete(
    phase_name: str,
    algo_key: str,
    dataset: str,
    family: str,
    round_idx: int,
    run_root: Path,
) -> None:
    """Mark a training phase as complete by creating a marker file."""
    family_cap = str(family).capitalize()
    r = int(round_idx) + 1
    phase_marker_dir = run_root / "models" / algo_key / f"{dataset}_{family_cap}_r{r}"
    phase_marker_dir.mkdir(parents=True, exist_ok=True)
    marker_file = phase_marker_dir / f".{phase_name}_complete"
    marker_file.touch()


def check_phase_complete(
    phase_name: str,
    algo_key: str,
    dataset: str,
    family: str,
    round_idx: int,
    run_root: Path,
) -> bool:
    """Check if a training phase is complete by checking for marker file."""
    family_cap = str(family).capitalize()
    r = int(round_idx) + 1
    marker_file = (
        run_root
        / "models"
        / algo_key
        / f"{dataset}_{family_cap}_r{r}"
        / f".{phase_name}_complete"
    )
    return marker_file.exists()


def get_checkpoint_dir(
    algo_key: str,
    dataset: str,
    family: str,
    round_idx: int,
    run_root: Path,
) -> Path:
    """Get checkpoint directory for data generation."""
    family_cap = str(family).capitalize()
    r = int(round_idx) + 1
    checkpoint_dir = (
        run_root / "checkpoints" / algo_key / f"{dataset}_{family_cap}_r{r}"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def save_checkpoint(
    checkpoint_dir: Path,
    key: str,
    records: list,
) -> None:
    """Save checkpoint atomically. Key can be e.g. 'worker_0' or 'chunk_5'."""
    checkpoint_file = checkpoint_dir / f"{key}.json"
    temp_file = checkpoint_dir / f"{key}.json.tmp"

    with open(temp_file, "w") as f:
        json.dump(records, f)
    temp_file.replace(checkpoint_file)


def load_all_checkpoints(checkpoint_dir: Path) -> list:
    """Load all checkpoints (worker_*.json and chunk_*.json), skipping corrupted ones."""
    all_records = []
    if checkpoint_dir.exists():
        for checkpoint_file in sorted(checkpoint_dir.glob("*.json")):
            try:
                with open(checkpoint_file, "r") as f:
                    records = json.load(f)
                    all_records.extend(records)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Corrupted checkpoint {checkpoint_file}: {e}, skipping")
                continue
    return all_records


def get_saved_chunk_indices(checkpoint_dir: Path) -> set:
    """Return set of chunk indices that have been checkpointed."""
    indices = set()
    if checkpoint_dir.exists():
        for f in checkpoint_dir.glob("chunk_*.json"):
            idx = int(f.stem.split("_", 1)[1])
            indices.add(idx)
    return indices


def clear_checkpoints(checkpoint_dir: Path) -> None:
    """Clear all checkpoints in the directory."""
    if checkpoint_dir.exists():
        import shutil

        shutil.rmtree(checkpoint_dir)
