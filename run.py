from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, List
import random

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from loguru import logger

from pita.core.io import create_subdir, save_json
from pita.core.registry import get_algorithm_registry
import pita.algos  # trigger registration imports
from pita.plotting.hooks import plot_after_run
from pita.models.hf import HFModel, GenerationConfig
from pita.models.registry import resolve_family_pair
from datasets import Dataset


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("Loaded config:\n{}", OmegaConf.to_yaml(cfg, resolve=True))

    exp_name = cfg.experiment.name
    if exp_name in (None, "", "???"):
        raise ValueError("experiment.name must be set (non-empty)")

    run_root = Path(os.getcwd())
    logger.info("Run directory: {}", run_root)

    all_results: Dict[str, Any] = {}

    model_pairs = []
    if isinstance(cfg.model_pairs, list):
        for family in cfg.model_pairs:
            ref_alias, cls_alias = resolve_family_pair(str(family))
            model_pairs.append((ref_alias, cls_alias))
    else:
        raise ValueError(
            "model_pairs must be a list of valid model family names. \nSupported families: llama, gemma"
        )

    datasets = list(cfg.get("datasets", {}).keys())
    algo_registry = get_algorithm_registry()

    for algo_key, algo_cfg in cfg.algos.items():
        algo_cls = algo_registry.get(algo_key)
        if algo_cls is None:
            raise KeyError(f"Algorithm '{algo_key}' is not registered")
        algo = algo_cls(algo_cfg)

        for ref_name, clf_name in model_pairs:
            for dataset_name in datasets:
                # Algorithm-specific data collection (skips if already present)
                if hasattr(algo, "generate_data"):
                    family_name = (
                        ref_name.split("-")[0] if "-" in ref_name else ref_name
                    )
                    algo.generate_data(
                        cfg=cfg,
                        ref_model=ref_name,
                        dataset=dataset_name,
                        family=family_name,
                    )
                logger.info(
                    "Running algo={} on dataset={} with pair={} vs {}",
                    algo_key,
                    dataset_name,
                    ref_name,
                    clf_name,
                )
                out_dir = create_subdir(
                    run_root,
                    ["results", algo_key, f"{ref_name}_vs_{clf_name}", dataset_name],
                )
                result = algo.run(
                    cfg=cfg,
                    ref_model=ref_name,
                    cls_model=clf_name,
                    dataset=dataset_name,
                    output_dir=out_dir,
                )
                save_json(out_dir / "result.json", result or {})
                all_results.setdefault(algo_key, {}).setdefault(
                    f"{ref_name}_vs_{clf_name}", {}
                )[dataset_name] = result

    figs_dir = create_subdir(run_root, ["figures"])  # separate from raw results
    plot_after_run(cfg=cfg, results=all_results, output_dir=figs_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
