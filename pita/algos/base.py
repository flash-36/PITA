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
            attn_impl=str(cfg.common.attn_impl),
            gradient_checkpointing=bool(cfg.common.gradient_checkpointing),
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
                loss_type=str(self.cfg.loss_type),
                num_atoms=int(self.cfg.num_atoms),
                V_min=float(self.cfg.V_min),
                V_max=float(self.cfg.V_max),
                attn_impl=str(cfg.common.attn_impl),
                dtype=str(cfg.common.amp_dtype),
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
        gb = int(cfg.common.gen_batch_size)
        it = islice(ds.iter(), limit)
        pbar = tqdm(total=limit, desc=f"{self.algo_key}:{dataset}:{family}")
        while True:
            chunk = list(islice(it, gb))
            if not chunk:
                break
            prompts = [ds.hydrate_prompt(ex.question) for ex in chunk]
            rollouts = model.roll_in_batch(
                prompts, max_roll_tokens=gen_cfg.max_new_tokens
            )

            # Group by remaining token budget to batch continuations
            budget_to_items: Dict[int, List[Tuple[Any, str, str, str]]] = {}
            for ex, prompt, rollout in zip(chunk, prompts, rollouts):
                built_prompt = rollout["prompt"]
                context_token_ids = rollout["context_ids"].tolist()
                full_token_count = len(context_token_ids)
                padded_in_len = int(
                    rollout.get(
                        "padded_in_len", len(model.tokenizer(built_prompt)["input_ids"])
                    )
                )
                rollout_token_count = full_token_count - padded_in_len
                if rollout_token_count < 2:
                    continue
                for _ in range(samples_per_example):
                    cutoff_tokens = random.randint(1, rollout_token_count - 1)
                    context_prefix_ids = context_token_ids[
                        : padded_in_len + cutoff_tokens
                    ]
                    context_text = model.tokenizer.decode(
                        context_prefix_ids, skip_special_tokens=True
                    )
                    solution_prefix_ids = context_prefix_ids[padded_in_len:]
                    solution_prefix_text = model.tokenizer.decode(
                        solution_prefix_ids, skip_special_tokens=True
                    )
                    remaining = int(gen_cfg.max_new_tokens - cutoff_tokens)
                    budget_to_items.setdefault(remaining, []).append(
                        (ex, prompt, solution_prefix_text, context_text)
                    )

            total_items = sum(len(v) for v in budget_to_items.values())
            cont_bar = tqdm(
                total=total_items, desc=f"{self.algo_key}:{dataset}:cont", leave=False
            )
            for remaining, items in budget_to_items.items():
                contexts = [ctx for (_, _, _, ctx) in items]
                y_refs = model.continue_from_context_batch(
                    contexts, max_new_tokens=remaining, greedy=True
                )
                y_samps = model.continue_from_context_batch(
                    contexts, max_new_tokens=remaining, greedy=False
                )
                for (ex, prompt, sol_prefix, _), y_ref, y_sample in zip(
                    items, y_refs, y_samps
                ):
                    y_a = sol_prefix + y_ref
                    y_b = sol_prefix + y_sample
                    score_a, score_b, preferred = self.score_samples(ex, y_a, y_b)
                    records.append(
                        {
                            "question": ex.question,
                            "answer": ex.answer,
                            "prompt": prompt,
                            "t": None,
                            "context": prompt + sol_prefix,
                            "y_a": y_a,
                            "y_b": y_b,
                            "score_a": score_a,
                            "score_b": score_b,
                            "preferred": preferred,
                        }
                    )
                cont_bar.update(len(items))
            cont_bar.close()
            pbar.update(len(chunk))

        pbar.close()
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
            eta=float(self.cfg.guidance.eta),
            mode=str(self.cfg.guidance.mode),
            top_k=int(self.cfg.guidance.top_k),
            use_cache=bool(self.cfg.guidance.use_cache),
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
            eta=float(self.cfg.guidance.eta),
            mode=str(self.cfg.guidance.mode),
            top_k=int(self.cfg.guidance.top_k),
            use_cache=bool(self.cfg.guidance.use_cache),
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
            dtype=str(cfg.common.dtype),
            batch_size=int(cfg.collection.reward_batch_size),
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
