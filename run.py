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
import pita.datasets  # trigger dataset registration
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
    for family in cfg.model_pairs:
        ref_model_alias, value_model_alias = resolve_family_pair(str(family))
        model_pairs.append((str(family), ref_model_alias, value_model_alias))

    datasets = list(cfg.get("datasets", {}).keys())
    algo_registry = get_algorithm_registry()

    for algo_key, algo_cfg in cfg.algos.items():
        algo_cls = algo_registry.get(algo_key)
        if algo_cls is None:
            raise KeyError(f"Algorithm '{algo_key}' is not registered")
        algo = algo_cls(algo_cfg)

        for family_name, ref_model_alias, value_model_alias in model_pairs:
            for dataset_name in datasets:
                # Algorithm-specific data collection (skips if already present)
                logger.info(
                    "Generating data for algo={} on dataset={} with model_pair={} and {}",
                    algo_key,
                    dataset_name,
                    ref_model_alias,
                    value_model_alias,
                )
                algo.generate_data(
                    cfg=cfg,
                    ref_model=ref_model_alias,
                    dataset=dataset_name,
                    family=family_name,
                )
                # logger.info(
                #     "Running algo={} on dataset={} with model_pair={} and {}",
                #     algo_key,
                #     dataset_name,
                #     ref_model_alias,
                #     value_model_alias,
                # )
                # out_dir = create_subdir(
                #     run_root,
                #     [
                #         "results",
                #         algo_key,
                #         f"{ref_model_alias}_vs_{value_model_alias}",
                #         dataset_name,
                #     ],
                # )
                # result = algo.run(
                #     cfg=cfg,
                #     ref_model=ref_model_alias,
                #     cls_model=value_model_alias,
                #     dataset=dataset_name,
                #     output_dir=out_dir,
                # )
                # save_json(out_dir / "result.json", result or {})
                # all_results.setdefault(algo_key, {}).setdefault(
                #     f"{ref_model_alias}_vs_{value_model_alias}", {}
                # )[dataset_name] = result

    figs_dir = create_subdir(run_root, ["figures"])  # separate from raw results
    plot_after_run(cfg=cfg, results=all_results, output_dir=figs_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
