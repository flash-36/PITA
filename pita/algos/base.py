from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
import random
from loguru import logger

from omegaconf import DictConfig
from datasets import Dataset
from tqdm import tqdm
from itertools import islice
import torch
from pita.core.prompts import build_instruction_prompt

from pita.models.hf import HFModel, GenerationConfig
from pita.models.guided import GuidedHFModel, GuidanceConfig
from pita.models.value_classifier import ValueClassifier
from pita.datasets import build_train_dataset, build_test_dataset
from pita.core.io import get_run_root, get_snapshot_paths, merge_and_save_hf
from pita.models import RewardScorer


class AlgorithmBase:
    def __init__(self, cfg):
        self.cfg = cfg

    @property
    def algo_key(self) -> str:
        return self.ALGO_KEY

    def family_cap(self, family: str) -> str:
        return str(family).capitalize()

    def get_prev_ckpt_dir(
        self, run_root: Path, dataset: str, family: str, round_idx: int
    ) -> Path:
        family_cap = self.family_cap(family)
        return (
            run_root
            / "models"
            / self.algo_key
            / f"{dataset}_{family_cap}_r{int(round_idx)}"
        )

    def resolve_ref_for_round(
        self,
        run_root: Path,
        dataset: str,
        family: str,
        ref_model_alias: str,
        round_idx: int,
    ) -> str:
        if int(round_idx) > 0:
            prev_ckpt_dir = self.get_prev_ckpt_dir(run_root, dataset, family, round_idx)
            if prev_ckpt_dir.exists():
                return str(prev_ckpt_dir)
        return ref_model_alias

    def _build_dataset(self, cfg: DictConfig, dataset_name: str):
        return build_train_dataset(cfg, dataset_name)

    def _build_test_dataset(self, cfg: DictConfig, dataset_name: str):
        return build_test_dataset(cfg, dataset_name)

    def generate_data(
        self,
        cfg,
        ref_model: str,
        dataset: str,
        family: str,
        round_idx: int,
        cls_model: Optional[str] = None,
    ) -> None:
        return None

    def run(
        self,
        cfg,
        ref_model: str,
        cls_model: str,
        dataset: str,
        family: str,
        output_dir,
        round_idx: int,
    ):
        raise NotImplementedError

    def _build_model(self, cfg: DictConfig, model_alias: str) -> HFModel:
        gen_cfg = GenerationConfig(
            max_new_tokens=int(cfg.common.max_new_tokens),
            temperature=float(cfg.common.temperature),
            top_p=float(cfg.common.top_p),
            use_chat_template=bool(cfg.common.use_chat_template),
            dtype=str(cfg.common.dtype),
            attn_impl=str(getattr(cfg.common, "attn_impl", "eager")),
        )
        return HFModel(model_alias, gen_cfg)

    def get_ckpt_dir(
        self, run_root: Path, dataset: str, family: str, round_idx: int
    ) -> Path:
        family_cap = self.family_cap(family)
        return (
            run_root
            / "models"
            / self.algo_key
            / f"{dataset}_{family_cap}_r{int(round_idx)+1}"
        )


class ValueGuidedAlgorithms(AlgorithmBase):
    def maybe_load_classifier_from_prev_round(
        self,
        classifier: ValueClassifier,
        *,
        run_root: Path,
        dataset: str,
        family: str,
        round_idx: int,
        device: torch.device,
    ) -> bool:
        if int(round_idx) <= 0:
            return False
        prev_ckpt_dir = self.get_prev_ckpt_dir(run_root, dataset, family, round_idx)
        ckpt = prev_ckpt_dir / "classifier.pt"
        if ckpt.exists():
            state = torch.load(str(ckpt), map_location=device)
            classifier.load_state_dict(state)
            return True
        return False

    def generate_data(
        self,
        cfg: DictConfig,
        ref_model: str,
        dataset: str,
        family: str,
        round_idx: int,
        cls_model: Optional[str] = None,
    ) -> None:
        snap_hf_prev, snap_hf, snap_csv = get_snapshot_paths(
            self.algo_key, dataset, family, round_idx
        )

        random.seed(int(cfg.collection.seed))
        # Build generator model: round 1 uses ref-only, later rounds can use guided
        ref_hf = self._build_model(cfg, ref_model)
        if int(round_idx) > 0 and cls_model is not None:
            ckpt_loaded = False
            prev_classifier = ValueClassifier(
                cls_model,
                tokenizer=ref_hf.tokenizer,
                device=ref_hf.model.device,
            )
            run_root = get_run_root()
            ckpt_loaded = self.maybe_load_classifier_from_prev_round(
                prev_classifier,
                run_root=run_root,
                dataset=dataset,
                family=family,
                round_idx=round_idx,
                device=ref_hf.model.device,
            )
            if ckpt_loaded:
                model = self.build_guided_with(
                    cfg, ref=ref_hf, classifier=prev_classifier
                )
            else:
                model = ref_hf
        else:
            model = ref_hf
        ds = self._build_dataset(cfg, dataset)
        self._dataset = ds
        gen_cfg = model.gen_cfg

        samples_per_example = int(cfg.collection.samples_per_example)

        records: List[Dict[str, Any]] = []
        max_examples = int(cfg.collection.max_examples or 0)
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
                    "Skipping example due to short rollout: dataset={} family={} prompt_tokens={} rollout_tokens={}",
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
        new_ds = Dataset.from_list(records)
        merge_and_save_hf(snap_hf_prev, new_ds, snap_hf, snap_csv)

    def score_samples(self, ex, y_a: str, y_b: str) -> Tuple[float, float, int]:
        raise NotImplementedError

    def _build_guided_model(
        self,
        cfg: DictConfig,
        ref_model_alias: str,
        cls_model_alias: str,
    ) -> GuidedHFModel:
        ref_model = self._build_model(cfg, ref_model_alias)
        cls = ValueClassifier(
            cls_model_alias,
            tokenizer=ref_model.tokenizer,
            device=ref_model.model.device,
        )
        g = GuidanceConfig(
            eta=float(cfg.common.guidance.eta),
            mode=str(cfg.common.guidance.mode),
            top_k=int(cfg.common.guidance.top_k),
            use_cache=bool(cfg.common.guidance.use_cache),
        )
        return GuidedHFModel(ref_model, cls, g)

    def build_guided_with(
        self,
        cfg: DictConfig,
        *,
        ref: HFModel,
        classifier: ValueClassifier,
    ) -> GuidedHFModel:
        g = GuidanceConfig(
            eta=float(cfg.common.guidance.eta),
            mode=str(cfg.common.guidance.mode),
            top_k=int(cfg.common.guidance.top_k),
            use_cache=bool(cfg.common.guidance.use_cache),
        )
        return GuidedHFModel(ref, classifier, g)


class PostTrainingAlgorithms(AlgorithmBase):

    def generate_data(
        self,
        cfg: DictConfig,
        ref_model: str,
        dataset: str,
        family: str,
        round_idx: int,
        cls_model: Optional[str] = None,
    ) -> None:
        snap_hf_prev, snap_hf, snap_csv = get_snapshot_paths(
            self.algo_key, dataset, family, round_idx
        )

        random.seed(int(cfg.collection.seed))
        model = self._build_model(cfg, ref_model)
        ds = self._build_dataset(cfg, dataset)
        gen_cfg = model.gen_cfg

        # reward model setup for preference scoring
        ds_cfg = cfg.datasets[dataset]
        rm_model = str(ds_cfg.reward_model)
        device = 0 if torch.cuda.is_available() else -1
        self._reward = RewardScorer(
            rm_model,
            bt_sampling=bool(cfg.common.bt_sampling),
            bt_beta=float(cfg.common.bt_beta),
            device=device,
        )

        samples_per_example = int(cfg.collection.samples_per_example)

        records: List[Dict[str, Any]] = []
        max_examples = int(cfg.collection.max_examples or 0)
        limit = max_examples if max_examples > 0 else len(ds)
        for ex in tqdm(
            islice(ds.iter(), limit), total=limit, desc=f"{self.algo_key}:{dataset}"
        ):
            prompt = ds.hydrate_prompt(ex.question)
            built = build_instruction_prompt(
                prompt,
                tokenizer=model.tokenizer,
                use_chat_template=model.gen_cfg.use_chat_template,
            )
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
        new_ds = Dataset.from_list(records)
        merge_and_save_hf(snap_hf_prev, new_ds, snap_hf, snap_csv)

    def score_samples(self, ex, y_a: str, y_b: str) -> Tuple[float, float, int]:
        r_a, r_b, preferred = self._reward.score_pair(ex.question, y_a, y_b)
        return r_a, r_b, preferred

    # Evaluation handled directly in algorithms
