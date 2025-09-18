from __future__ import annotations

from pathlib import Path
import os
from typing import Any, Dict, List, Tuple
import random
import logging

from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from datasets import Dataset
from tqdm import tqdm
from itertools import islice
import torch
from transformers import AutoTokenizer, pipeline

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

    def run(
        self, cfg, ref_model: str, cls_model: str, dataset: str, family: str, output_dir
    ):
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
        run_root = Path(os.getcwd())
        ds_root = run_root / "datasets" / self.algo_key
        ds_root.mkdir(parents=True, exist_ok=True)

        family_cap = str(family).capitalize()
        hf_dir = ds_root / f"{dataset}_{family_cap}.hf"
        csv_path = ds_root / f"{dataset}_{family_cap}.csv"
        if hf_dir.exists():
            return

        random.seed(int(cfg.collection.seed))
        model = self._build_model(cfg, ref_model)
        ds = self._build_dataset(cfg, dataset)
        self._dataset = ds
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

            built_prompt = rollout["prompt"]
            prompt_token_count = len(model.tokenizer(built_prompt)["input_ids"])
            context_token_ids = rollout["context_ids"].tolist()
            full_token_count = len(context_token_ids)
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

                context_prefix_ids = context_token_ids[
                    : prompt_token_count + cutoff_tokens
                ]
                context_text = model.tokenizer.decode(
                    context_prefix_ids, skip_special_tokens=True
                )
                solution_prefix_ids = context_prefix_ids[prompt_token_count:]
                solution_prefix_text = model.tokenizer.decode(
                    solution_prefix_ids, skip_special_tokens=True
                )

                remaining_token_budget = gen_cfg.max_new_tokens - cutoff_tokens
                y_ref = model.continue_from_context(
                    context_text, max_new_tokens=remaining_token_budget, greedy=True
                )
                y_sample = model.continue_from_context(
                    context_text, max_new_tokens=remaining_token_budget, greedy=False
                )

                y_a = solution_prefix_text + y_ref
                y_b = solution_prefix_text + y_sample

                score_a, score_b, preferred = self.score_samples(ex, y_a, y_b)

                records.append(
                    {
                        "question": ex.question,
                        "answer": ex.answer,
                        "prompt": prompt,
                        "t": cutoff_tokens,
                        "context": prompt + solution_prefix_text,
                        "y_a": y_a,
                        "y_b": y_b,
                        "score_a": score_a,
                        "score_b": score_b,
                        "preferred": preferred,
                    }
                )

        if not records:
            return
        hf_ds = Dataset.from_list(records)
        hf_ds.save_to_disk(str(hf_dir))
        hf_ds.to_csv(str(csv_path))

    def score_samples(self, ex, y_a: str, y_b: str) -> Tuple[float, float, int]:
        raise NotImplementedError


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
        run_root = Path(os.getcwd())
        ds_root = run_root / "datasets" / self.algo_key
        ds_root.mkdir(parents=True, exist_ok=True)

        family_cap = str(family).capitalize()
        hf_dir = ds_root / f"{dataset}_{family_cap}.hf"
        csv_path = ds_root / f"{dataset}_{family_cap}.csv"
        if hf_dir.exists():
            return

        random.seed(int(cfg.collection.seed))
        model = self._build_model(cfg, ref_model)
        ds = self._build_dataset(cfg, dataset)
        gen_cfg = model.gen_cfg

        # reward model setup for preference scoring
        ds_cfg = cfg.datasets[dataset]
        rm_model = str(ds_cfg.reward_model)
        device = 0 if torch.cuda.is_available() else -1
        self._bt_sampling = bool(cfg.common.bt_sampling)
        self._rm_tokenizer = AutoTokenizer.from_pretrained(
            rm_model, use_fast=True, trust_remote_code=True
        )
        self._rm_pipe = pipeline(
            "text-classification",
            model=rm_model,
            tokenizer=self._rm_tokenizer,
            device=device,
            model_kwargs={"torch_dtype": torch.bfloat16},
        )

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
                score_a, score_b, preferred = self.score_samples(ex, y_a, y_b)
                records.append(
                    {
                        "question": ex.question,
                        "answer": ex.answer,
                        "prompt": prompt,
                        "y_a": y_a,
                        "y_b": y_b,
                        "score_a": score_a,
                        "score_b": score_b,
                        "preferred": preferred,
                    }
                )

        if not records:
            return
        hf_ds = Dataset.from_list(records)
        hf_ds.save_to_disk(str(hf_dir))
        hf_ds.to_csv(str(csv_path))

    def score_samples(self, ex, y_a: str, y_b: str) -> Tuple[float, float, int]:
        import math

        rm_pipe = self._rm_pipe
        rm_tok = self._rm_tokenizer

        msgs_a = [
            {"role": "user", "content": ex.question},
            {"role": "assistant", "content": y_a},
        ]
        msgs_b = [
            {"role": "user", "content": ex.question},
            {"role": "assistant", "content": y_b},
        ]
        texts = [
            rm_tok.apply_chat_template(
                msgs_a, tokenize=False, add_generation_prompt=False
            ),
            rm_tok.apply_chat_template(
                msgs_b, tokenize=False, add_generation_prompt=False
            ),
        ]

        outs = rm_pipe(texts, top_k=None, function_to_apply="none", batch_size=2)
        r_a = (
            float(outs[0][0]["score"])
            if isinstance(outs[0], list)
            else float(outs[0]["score"])
        )
        r_b = (
            float(outs[1][0]["score"])
            if isinstance(outs[1], list)
            else float(outs[1]["score"])
        )

        if self._bt_sampling:
            beta = float(self.cfg.bt_beta)
            p_a = 1.0 / (1.0 + math.exp(-beta * (r_a - r_b)))
            preferred = 0 if random.random() < p_a else 1
        else:
            preferred = 0 if r_a >= r_b else 1
        return r_a, r_b, preferred
