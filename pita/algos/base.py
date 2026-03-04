from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
import math
import random
from loguru import logger

from omegaconf import DictConfig
from datasets import Dataset
from tqdm import tqdm
from itertools import islice
import torch

from pita.models.hf import HFModel, GenerationConfig
from pita.models.guided import GuidedHFModel, GuidanceConfig
from pita.models.value_classifier import ValueClassifier
from pita.datasets import build_train_dataset, build_test_dataset
from pita.core.io import get_run_root, get_snapshot_paths, merge_and_save_hf, save_checkpoint, get_saved_chunk_indices
from pita.core.compute_tracker import get_compute_tracker


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
        run_root: Optional[Path] = None,
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
        run_root: Optional[Any] = None,
    ):
        raise NotImplementedError

    def _build_model(self, cfg: DictConfig, model_alias: str) -> HFModel:
        gen_cfg = GenerationConfig(
            max_new_tokens=int(cfg.generation.max_new_tokens),
            temperature=float(cfg.generation.temperature),
            top_p=float(cfg.generation.top_p),
            repetition_penalty=float(cfg.generation.repetition_penalty),
            use_chat_template=bool(cfg.generation.use_chat_template),
            dtype=str(cfg.system.dtype),
            attn_impl=str(cfg.system.attn_impl),
            gradient_checkpointing=bool(cfg.training.gradient_checkpointing),
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
    """Value-guided algorithms (PITA, QSharp, QSharp-HF)."""

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
        run_root: Optional[Path] = None,
    ) -> None:
        from pita.core.io import (
            get_checkpoint_dir,
            load_all_checkpoints,
            clear_checkpoints,
        )

        if run_root is None:
            run_root = get_run_root()

        snap_hf_prev, snap_hf, snap_csv = get_snapshot_paths(
            self.algo_key, dataset, family, round_idx, run_root=run_root
        )

        checkpoint_dir = get_checkpoint_dir(
            self.algo_key, dataset, family, round_idx, run_root
        )

        existing_records = load_all_checkpoints(checkpoint_dir)
        if existing_records:
            logger.info(
                f"💾 Found {len(existing_records)} records from previous checkpoints"
            )

        tracker = get_compute_tracker()
        with tracker.track_phase(f"data_generation_{dataset}"):
            records = self._generate_data_sequential(
                cfg, ref_model, dataset, family, round_idx, cls_model, run_root,
                checkpoint_dir=checkpoint_dir,
            )

        all_records = existing_records + records

        if not all_records:
            return

        logger.info(f"💾 Saving {len(all_records)} total records")
        new_ds = Dataset.from_list(all_records)
        merge_and_save_hf(snap_hf_prev, new_ds, snap_hf, snap_csv)

        clear_checkpoints(checkpoint_dir)
        logger.info(f"🧹 Cleared checkpoints")

    def _generate_data_sequential(
        self,
        cfg: DictConfig,
        ref_model: str,
        dataset: str,
        family: str,
        round_idx: int,
        cls_model: Optional[str] = None,
        run_root: Optional[Path] = None,
        checkpoint_dir: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """Sequential data generation with per-chunk checkpointing."""
        random.seed(int(cfg.data_collection.seed))

        completed_chunks = get_saved_chunk_indices(checkpoint_dir) if checkpoint_dir else set()
        if completed_chunks:
            logger.info(f"⏩ Found {len(completed_chunks)} completed chunk checkpoints, will skip them")

        ref_hf = self._build_model(cfg, ref_model)
        if int(round_idx) > 0 and cls_model is not None:
            prev_classifier = ValueClassifier(
                cls_model,
                tokenizer=ref_hf.tokenizer,
                device=ref_hf.model.device,
                loss_type=str(self.cfg.loss_type),
                num_atoms=int(self.cfg.num_atoms),
                V_min=float(self.cfg.V_min),
                V_max=float(self.cfg.V_max),
                attn_impl=str(cfg.system.attn_impl),
                dtype=str(cfg.system.amp_dtype),
                gradient_checkpointing=bool(cfg.training.gradient_checkpointing),
            )
            if run_root is None:
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

        samples_per_example = int(cfg.data_collection.samples_per_example)
        batch_size = int(cfg.algos[self.algo_key].batch_size)
        chunk_size = int(getattr(cfg.data_collection, "chunk_size", 100))

        limit = int(getattr(cfg.datasets[dataset], "train_size_cap", 0) or 0)
        limit = limit if limit > 0 else len(ds)

        examples = []
        for ex in islice(ds.iter(), limit):
            prompt = ds.hydrate_prompt(ex.question)
            examples.append({"ex": ex, "prompt": prompt})

        num_chunks = (len(examples) + chunk_size - 1) // chunk_size
        logger.info(
            f"🔄 Processing {len(examples)} examples in {num_chunks} chunks of {chunk_size} (batch_size={batch_size})"
        )

        all_records: List[Dict[str, Any]] = []

        for chunk_idx, chunk_start in enumerate(range(0, len(examples), chunk_size)):
            chunk_end = min(chunk_start + chunk_size, len(examples))

            if chunk_idx in completed_chunks:
                logger.info(f"⏩ Skipping chunk {chunk_idx + 1}/{num_chunks} (already checkpointed)")
                continue

            chunk_examples = examples[chunk_start:chunk_end]
            logger.info(
                f"📦 Processing chunk {chunk_idx + 1}/{num_chunks} "
                f"(examples {chunk_start+1}-{chunk_end}/{len(examples)})"
            )

            prompts = [item["prompt"] for item in chunk_examples]
            rollouts = model.roll_in_batch(
                prompts, gen_cfg.max_new_tokens, batch_size=batch_size
            )

            contexts_to_continue = []
            eos_set = set(model.eos_token_ids)
            for item, rollout in zip(chunk_examples, rollouts):
                built_prompt = rollout["prompt"]
                prompt_token_count = len(model.tokenizer(built_prompt, add_special_tokens=False)["input_ids"])
                context_token_ids = rollout["context_ids"].tolist()

                # Truncate at first EOS token (after the prompt) to avoid
                # sampling cutoff points in the EOS-padding zone.
                gen_ids = context_token_ids[prompt_token_count:]
                first_eos = next(
                    (i for i, tid in enumerate(gen_ids) if tid in eos_set),
                    len(gen_ids),
                )
                context_token_ids = context_token_ids[: prompt_token_count + first_eos]
                rollout_token_count = first_eos

                if rollout_token_count < 2:
                    continue

                for _ in range(samples_per_example):
                    cutoff_tokens = random.randint(1, rollout_token_count - 1)
                    context_prefix_ids = context_token_ids[
                        : prompt_token_count + cutoff_tokens
                    ]
                    context_text = model.tokenizer.decode(
                        context_prefix_ids, skip_special_tokens=False
                    )
                    solution_prefix_ids = context_prefix_ids[prompt_token_count:]
                    solution_prefix_text = model.tokenizer.decode(
                        solution_prefix_ids, skip_special_tokens=False
                    )
                    remaining_token_budget = gen_cfg.max_new_tokens - cutoff_tokens

                    if remaining_token_budget >= 1:
                        contexts_to_continue.append(
                            {
                                "ex": item["ex"],
                                "prompt": item["prompt"],
                                "context_text": context_text,
                                "solution_prefix_text": solution_prefix_text,
                                "cutoff_tokens": cutoff_tokens,
                                "remaining_token_budget": remaining_token_budget,
                            }
                        )

            if not contexts_to_continue:
                logger.info(
                    f"⏭️  Chunk {chunk_idx + 1}: No valid contexts, skipping"
                )
                if checkpoint_dir:
                    save_checkpoint(checkpoint_dir, f"chunk_{chunk_idx}", [])
                del rollouts
                continue

            all_contexts = [item["context_text"] for item in contexts_to_continue]
            min_remaining_tokens = min(
                item["remaining_token_budget"] for item in contexts_to_continue
            )

            greedy_continuations = model.continue_from_context_batch(
                all_contexts, min_remaining_tokens, greedy=True, batch_size=batch_size
            )
            sampled_continuations = model.continue_from_context_batch(
                all_contexts, min_remaining_tokens, greedy=False, batch_size=batch_size
            )

            chunk_records = []
            for ctx, y_ref, y_sample in zip(
                contexts_to_continue, greedy_continuations, sampled_continuations
            ):
                y_a = ctx["solution_prefix_text"] + y_ref
                y_b = ctx["solution_prefix_text"] + y_sample
                score_a, score_b, preferred = self.score_samples(ctx["ex"], y_a, y_b)
                chunk_records.append(
                    {
                        "question": ctx["ex"].question,
                        "answer": ctx["ex"].answer,
                        "prompt": ctx["prompt"],
                        "t": ctx["cutoff_tokens"],
                        "context": ctx["prompt"] + ctx["solution_prefix_text"],
                        "y_a": y_a,
                        "y_b": y_b,
                        "score_a": score_a,
                        "score_b": score_b,
                        "preferred": preferred,
                    }
                )

            all_records.extend(chunk_records)

            if checkpoint_dir:
                save_checkpoint(checkpoint_dir, f"chunk_{chunk_idx}", chunk_records)

            del rollouts, contexts_to_continue, all_contexts
            del greedy_continuations, sampled_continuations
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(
                f"✅ Chunk {chunk_idx + 1}/{num_chunks} saved: {len(chunk_records)} records ({len(all_records)} new total)"
            )

        logger.info(f"🎉 All chunks processed: {len(all_records)} new records")
        return all_records

    def score_samples(self, ex, y_a: str, y_b: str) -> Tuple[float, float, int]:
        """Score a pair of responses using RM + correctness bonus, then BT sample.

        For verifiable datasets (those with is_correct), a correctness_bonus is
        added to correct responses before BT sampling so that correctness
        dominates the preference signal while the RM provides tie-breaking.
        """
        r_a, r_b = self._reward.score_batch_single(
            [(ex.question, y_a), (ex.question, y_b)]
        )

        ds = getattr(self, "_dataset", None)
        if ds is not None and hasattr(ds, "is_correct"):
            bonus = float(self._correctness_bonus)
            c_a = bonus if ds.is_correct(ex.answer, y_a) else 0.0
            c_b = bonus if ds.is_correct(ex.answer, y_b) else 0.0
            composite_a, composite_b = r_a + c_a, r_b + c_b
        else:
            composite_a, composite_b = r_a, r_b

        beta = self._reward._bt_beta
        p_a = 1.0 / (1.0 + math.exp(-beta * (composite_a - composite_b)))
        preferred = 0 if random.random() < p_a else 1

        return composite_a, composite_b, preferred

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
            loss_type=str(self.cfg.loss_type),
            num_atoms=int(self.cfg.num_atoms),
            V_min=float(self.cfg.V_min),
            V_max=float(self.cfg.V_max),
            attn_impl=str(cfg.system.attn_impl),
            dtype=str(cfg.system.amp_dtype),
            gradient_checkpointing=bool(cfg.training.gradient_checkpointing),
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


