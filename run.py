from __future__ import annotations

from typing import Dict, Any

import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from tqdm import tqdm
import random
import numpy as np
import torch

from pita.core.io import create_subdir, save_json, get_run_root
from pita.core.registry import get_algorithm_registry
import pita.algos  # trigger registration imports
import pita.datasets  # trigger dataset registration
from pita.plotting.hooks import plot_after_run
from pita.models.catalog import resolve_family_pair


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run_root = get_run_root()
    logger.add(
        str(run_root / "run.log"),
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
        level="INFO",
        enqueue=True,
    )
    logger.info("🧾 Loaded config:\n{}", OmegaConf.to_yaml(cfg, resolve=True))

    exp_name = cfg.experiment.name
    if exp_name in (None, "", "???"):
        raise ValueError("experiment.name must be set (non-empty)")

    # Global seeding for reproducibility
    seed = int(cfg.experiment.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    logger.info("📂 Run directory: {}", run_root)

    all_results: Dict[str, Any] = {}

    model_pairs = []
    for family in cfg.model_pairs:
        ref_model_alias, value_model_alias = resolve_family_pair(str(family))
        model_pairs.append((str(family), ref_model_alias, value_model_alias))

    # Use configured training datasets subset
    datasets = list(cfg.training.datasets)
    algo_registry = get_algorithm_registry()

    rounds = int(cfg.rounds_of_training)
    for algo_key, algo_cfg in cfg.algos.items():
        algo_cls = algo_registry.get(algo_key)
        if algo_cls is None:
            raise KeyError(f"Algorithm '{algo_key}' is not registered")
        algo = algo_cls(algo_cfg)

        for family_name, ref_model_alias, value_model_alias in model_pairs:
            for dataset_name in datasets:
                for round_idx in tqdm(
                    range(rounds),
                    total=rounds,
                    desc=f"{algo_key}:{dataset_name}:{family_name}",
                ):
                    logger.info("🔁 Round {}/{}", round_idx + 1, rounds)
                    ref_for_train = algo.resolve_ref_for_round(
                        run_root=run_root,
                        dataset=dataset_name,
                        family=family_name,
                        ref_model_alias=ref_model_alias,
                        round_idx=round_idx,
                    )

                    logger.info(
                        "🧪 Generating data (round {}) algo={} dataset={} gen_model={} cls={}",
                        round_idx + 1,
                        algo_key,
                        dataset_name,
                        ref_for_train,
                        value_model_alias,
                    )
                    algo.generate_data(
                        cfg=cfg,
                        ref_model=ref_for_train,
                        cls_model=value_model_alias,
                        dataset=dataset_name,
                        family=family_name,
                        round_idx=round_idx,
                    )
                    logger.info(
                        "🏋️ Training (round {}) algo={} dataset={} ref_model={} cls={}",
                        round_idx + 1,
                        algo_key,
                        dataset_name,
                        ref_model_alias,
                        value_model_alias,
                    )
                    out_dir = create_subdir(
                        run_root,
                        [
                            "results",
                            algo_key,
                            f"{family_name}",
                            dataset_name,
                            f"r{round_idx + 1}",
                        ],
                    )
                    result = algo.run(
                        cfg=cfg,
                        ref_model=ref_for_train,
                        cls_model=value_model_alias,
                        dataset=dataset_name,
                        family=family_name,
                        output_dir=out_dir,
                        round_idx=round_idx,
                    )
                    save_json(out_dir / "result.json", result or {})
                    all_results.setdefault(algo_key, {}).setdefault(
                        f"{family_name}", {}
                    ).setdefault(dataset_name, {})[f"r{round_idx + 1}"] = result

    figs_dir = create_subdir(run_root, ["figures"])  # separate from raw results
    plot_after_run(cfg=cfg, results=all_results, output_dir=figs_dir)

    logger.info("✅ Done.")


if __name__ == "__main__":
    main()
