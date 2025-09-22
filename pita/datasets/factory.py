from __future__ import annotations

from omegaconf import DictConfig
from pita.datasets.registry import get_dataset


def build_train_dataset(cfg: DictConfig, dataset_name: str):
    ds_cfg = cfg.datasets[dataset_name]
    ds_cls = get_dataset(dataset_name)
    return ds_cls(
        hf_config=str(ds_cfg.train_hf_config),
        split=str(ds_cfg.train_split),
        question_key=str(ds_cfg.question_key),
        answer_key=str(ds_cfg.answer_key),
    )


def build_test_dataset(cfg: DictConfig, dataset_name: str):
    ds_cfg = cfg.datasets[dataset_name]
    if "test_hf_config" in ds_cfg and "test_split" in ds_cfg:
        ds_cls = get_dataset(dataset_name)
        return ds_cls(
            hf_config=str(ds_cfg.test_hf_config),
            split=str(ds_cfg.test_split),
            question_key=str(ds_cfg.question_key),
            answer_key=str(ds_cfg.answer_key),
        )
    return None
