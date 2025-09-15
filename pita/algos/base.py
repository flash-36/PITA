from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import random
import logging

from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from datasets import Dataset
from tqdm import tqdm
from itertools import islice

from pita.models.hf import HFModel, GenerationConfig
from pita.datasets.registry import get_dataset


logger = logging.getLogger(__name__)


class AlgorithmBase:
    def __init__(self, cfg):
        self.cfg = cfg

    @property
    def algo_key(self) -> str:
        return self.ALGO_KEY

    def generate_data(self, cfg, ref_model: str, dataset: str, family: str) -> None:
        return None

    def run(self, cfg, ref_model: str, cls_model: str, dataset: str, output_dir):
        raise NotImplementedError


class ValueGuidedAlgorithms(AlgorithmBase):
    def _build_model(self, cfg: DictConfig, model_alias: str) -> HFModel:
        gen_cfg = GenerationConfig(
            max_new_tokens=int(cfg.common.max_new_tokens),
            temperature=float(cfg.common.temperature),
            top_p=float(cfg.common.top_p),
            use_chat_template=bool(cfg.common.use_chat_template),
            dtype=str(cfg.common.dtype),
        )
        return HFModel(model_alias, gen_cfg)

    def _build_dataset(self, cfg: DictConfig, dataset_name: str):
        ds_cfg = cfg.datasets[dataset_name]
        ds_cls = get_dataset(dataset_name)
        return ds_cls(
            hf_config=str(ds_cfg.hf_config),
            split=str(ds_cfg.split),
            question_key=str(ds_cfg.question_key),
            answer_key=str(ds_cfg.answer_key),
        )

    def generate_data(
        self, cfg: DictConfig, ref_model: str, dataset: str, family: str
    ) -> None:
        orig_root = Path(get_original_cwd())
        ds_root = orig_root / "outputs" / "datasets" / self.algo_key
        ds_root.mkdir(parents=True, exist_ok=True)

        family_cap = str(family).capitalize()
        hf_dir = ds_root / f"{dataset}_{family_cap}.hf"
        if hf_dir.exists():
            return

        random.seed(int(cfg.collection.seed))
        model = self._build_model(cfg, ref_model)
        ds = self._build_dataset(cfg, dataset)
        gen_cfg = model.gen_cfg

        samples_per_example = int(cfg.collection.samples_per_example)

        records: List[Dict[str, Any]] = []
        max_examples = int(cfg.collection.get("max_examples", 0) or 0)
        limit = max_examples if max_examples > 0 else len(ds)
        for ex in tqdm(
            islice(ds.iter(), limit), total=limit, desc=f"{self.algo_key}:{dataset}"
        ):
            prompt = ds.hydrate_prompt(ex.question)
            rollout = model.roll_in(prompt, max_roll_tokens=gen_cfg.max_new_tokens)
            rollout_text = rollout["context_text"]

            prompt_token_count = len(model.tokenizer(prompt)["input_ids"])
            full_token_count = len(model.tokenizer(rollout_text)["input_ids"])
            rollout_token_count = full_token_count - prompt_token_count

            if rollout_token_count < 2:
                logger.info(
                    "Skipping example due to short rollout: dataset=%s, family=%s, prompt_tokens=%d, rollout_tokens=%d",
                    dataset,
                    str(family),
                    prompt_token_count,
                    rollout_token_count,
                )
                continue

            for _ in range(samples_per_example):
                cutoff_tokens = random.randint(1, rollout_token_count - 1)

                rollout_token_ids = model.tokenizer(rollout_text)["input_ids"]
                context_token_ids = rollout_token_ids[
                    : prompt_token_count + cutoff_tokens
                ]
                context_text = model.tokenizer.decode(
                    context_token_ids, skip_special_tokens=True
                )

                remaining_token_budget = gen_cfg.max_new_tokens - cutoff_tokens
                y_ref = model.continue_from_context(
                    context_text, max_new_tokens=remaining_token_budget, greedy=True
                )
                y_sample = model.continue_from_context(
                    context_text, max_new_tokens=remaining_token_budget, greedy=False
                )

                records.append(
                    {
                        "dataset": dataset,
                        "model_family": str(family),
                        "model_alias": ref_model,
                        "question": ex.question,
                        "answer": ex.answer,
                        "prompt": prompt,
                        "t": cutoff_tokens,
                        "context": context_text,
                        "y_sample": y_sample,
                        "y_ref": y_ref,
                        "pref_sample_gt_ref": None,
                    }
                )

        if not records:
            return
        hf_ds = Dataset.from_list(records)
        hf_ds.save_to_disk(str(hf_dir))


class PostTrainingAlgorithms(AlgorithmBase):
    def _build_model(self, cfg: DictConfig, model_alias: str) -> HFModel:
        gen_cfg = GenerationConfig(
            max_new_tokens=int(cfg.common.max_new_tokens),
            temperature=float(cfg.common.temperature),
            top_p=float(cfg.common.top_p),
            use_chat_template=bool(cfg.common.use_chat_template),
            dtype=str(cfg.common.dtype),
        )
        return HFModel(model_alias, gen_cfg)

    def _build_dataset(self, cfg: DictConfig, dataset_name: str):
        ds_cfg = cfg.datasets[dataset_name]
        ds_cls = get_dataset(dataset_name)
        return ds_cls(
            hf_config=str(ds_cfg.hf_config),
            split=str(ds_cfg.split),
            question_key=str(ds_cfg.question_key),
            answer_key=str(ds_cfg.answer_key),
        )

    def generate_data(
        self, cfg: DictConfig, ref_model: str, dataset: str, family: str
    ) -> None:
        orig_root = Path(get_original_cwd())
        ds_root = orig_root / "outputs" / "datasets" / self.algo_key
        ds_root.mkdir(parents=True, exist_ok=True)

        family_cap = str(family).capitalize()
        hf_dir = ds_root / f"{dataset}_{family_cap}.hf"
        if hf_dir.exists():
            return

        random.seed(int(cfg.collection.seed))
        model = self._build_model(cfg, ref_model)
        ds = self._build_dataset(cfg, dataset)
        gen_cfg = model.gen_cfg

        samples_per_example = int(cfg.collection.samples_per_example)

        records: List[Dict[str, Any]] = []
        max_examples = int(cfg.collection.get("max_examples", 0) or 0)
        limit = max_examples if max_examples > 0 else len(ds)
        for ex in tqdm(
            islice(ds.iter(), limit), total=limit, desc=f"{self.algo_key}:{dataset}"
        ):
            prompt = ds.hydrate_prompt(ex.question)
            built = model.build_prompt(prompt)
            for _ in range(samples_per_example):
                y_a = model.continue_from_context(
                    built, max_new_tokens=gen_cfg.max_new_tokens, greedy=False
                )
                y_b = model.continue_from_context(
                    built, max_new_tokens=gen_cfg.max_new_tokens, greedy=False
                )
                records.append(
                    {
                        "dataset": dataset,
                        "model_family": str(family),
                        "model_alias": ref_model,
                        "question": ex.question,
                        "answer": ex.answer,
                        "prompt": prompt,
                        "y_a": y_a,
                        "y_b": y_b,
                        "preferred": None,
                    }
                )

        if not records:
            return
        hf_ds = Dataset.from_list(records)
        hf_ds.save_to_disk(str(hf_dir))
