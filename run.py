from __future__ import annotations

from typing import Dict, Any

import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger

from pita.core.io import create_subdir, save_json, get_run_root
from pita.core.registry import get_algorithm_registry
import pita.algos  # trigger registration imports
import pita.datasets  # trigger dataset registration
from pita.plotting.hooks import plot_after_run
from pita.models.registry import resolve_family_pair


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("Loaded config:\n{}", OmegaConf.to_yaml(cfg, resolve=True))

    exp_name = cfg.experiment.name
    if exp_name in (None, "", "???"):
        raise ValueError("experiment.name must be set (non-empty)")

    run_root = get_run_root()
    logger.info("Run directory: {}", run_root)

    all_results: Dict[str, Any] = {}

    model_pairs = []
    for family in cfg.model_pairs:
        ref_model_alias, value_model_alias = resolve_family_pair(str(family))
        model_pairs.append((str(family), ref_model_alias, value_model_alias))

    datasets = list(cfg.get("datasets", {}).keys())
    algo_registry = get_algorithm_registry()

    rounds = int(cfg.get("rounds_of_training"))
    for algo_key, algo_cfg in cfg.algos.items():
        algo_cls = algo_registry.get(algo_key)
        if algo_cls is None:
            raise KeyError(f"Algorithm '{algo_key}' is not registered")
        algo = algo_cls(algo_cfg)

        for family_name, ref_model_alias, value_model_alias in model_pairs:
            for dataset_name in datasets:
                family_cap = str(family_name).capitalize()
                ckpt_dir = (
                    run_root / "models" / algo_key / f"{dataset_name}_{family_cap}"
                )
                for round_idx in range(rounds):
                    logger.info("=== Round {}/{} ===", round_idx + 1, rounds)
                    prev_ckpt_dir = (
                        run_root
                        / "models"
                        / algo_key
                        / f"{dataset_name}_{family_cap}_r{round_idx}"
                        if round_idx > 0
                        else ckpt_dir
                    )
                    gen_model = (
                        str(prev_ckpt_dir)
                        if (round_idx > 0 and prev_ckpt_dir.exists())
                        else ref_model_alias
                    )
                    ref_for_train = gen_model

                    logger.info(
                        "Generating data (round {}) algo={} dataset={} gen_model={} cls={}",
                        round_idx + 1,
                        algo_key,
                        dataset_name,
                        gen_model,
                        value_model_alias,  # TODO for PITA like take care of this
                    )
                    algo.generate_data(
                        cfg=cfg,
                        ref_model=gen_model,
                        dataset=dataset_name,
                        family=family_name,
                        round_idx=round_idx,
                    )
                    logger.info(
                        "Training (round {}) algo={} dataset={} ref_model={} cls={}",
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
                            f"{ref_model_alias}_vs_{value_model_alias}",
                            dataset_name,
                            f"r{round_idx + 1}",
                        ],
                    )
                    result = algo.run(
                        cfg=cfg,
                        ref_model=ref_for_train,
                        cls_model=value_model_alias,  # TODO for PITA like take care of this
                        dataset=dataset_name,
                        family=family_name,
                        output_dir=out_dir,
                        round_idx=round_idx,
                    )
                    save_json(out_dir / "result.json", result or {})
                    all_results.setdefault(algo_key, {}).setdefault(
                        f"{ref_model_alias}_vs_{value_model_alias}", {}
                    )[dataset_name] = result

    figs_dir = create_subdir(run_root, ["figures"])  # separate from raw results
    plot_after_run(cfg=cfg, results=all_results, output_dir=figs_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
