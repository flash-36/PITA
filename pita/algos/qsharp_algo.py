from __future__ import annotations

from typing import Any, Dict, Tuple, List, Optional
from datasets import Dataset
from tqdm import tqdm

from loguru import logger
import torch
import random
from itertools import islice

from pita.core.registry import register_algorithm
from .base import ValueGuidedAlgorithms
from pita.eval.evaluate import evaluate_pass1_maj8, evaluate_avg_reward
from pita.core.io import get_run_root, get_snapshot_paths, merge_and_save_hf
from pita.trainers import QSharpTrainer
from datasets import load_from_disk
from pita.models.value_classifier import ValueClassifier
from pita.core.prompts import build_instruction_prompt
from pita.datasets.convert import convert_qsharp_rows_to_classifier_dataset
from pita.models import RewardScorer


@register_algorithm("QSharp")
class QSharpAlgorithm(ValueGuidedAlgorithms):
    ALGO_KEY = "QSharp"

    def generate_data(
        self,
        cfg,
        ref_model: str,
        dataset: str,
        family: str,
        round_idx: int,
        cls_model: Optional[str] = None,
    ) -> None:
        # For non-verifiable datasets, set up a reward model for scoring
        if dataset in {"TLDR", "IMDBGen"}:
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

        snap_hf_prev, snap_hf, snap_csv = get_snapshot_paths(
            self.algo_key, dataset, family, round_idx
        )

        random.seed(int(cfg.collection.seed))
        # Build generator model: round 1 uses ref-only, later rounds can use guided
        ref_hf = self._build_model(cfg, ref_model)

        if int(round_idx) > 0 and cls_model is not None:
            verifiable = {"AIME", "GSM8K", "MATH"}
            loss_type = "bce" if dataset in verifiable else "mle"
            prev_classifier = ValueClassifier(
                cls_model,
                tokenizer=ref_hf.tokenizer,
                device=ref_hf.model.device,
                loss_type=loss_type,
                num_atoms=int(self.cfg.num_atoms),
                V_min=float(self.cfg.V_min),
                V_max=float(self.cfg.V_max),
                attn_impl=str(cfg.common.attn_impl),
                dtype=str(cfg.common.amp_dtype),
                gradient_checkpointing=bool(cfg.common.gradient_checkpointing),
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

    def resolve_ref_for_round(
        self,
        run_root,
        dataset: str,
        family: str,
        ref_model_alias: str,
        round_idx: int,
    ) -> str:
        return ref_model_alias

    def run(
        self,
        cfg,
        ref_model: str,
        cls_model: str,
        dataset: str,
        family: str,
        output_dir,
        round_idx: int,
    ) -> Dict[str, Any]:
        logger.info("🚀 QSharp start: dataset={} family={}", dataset, family)

        run_root = get_run_root()
        _, hf_dir, _ = get_snapshot_paths(self.algo_key, dataset, family, round_idx)
        if not hf_dir.exists():
            raise FileNotFoundError(f"Missing QSharp dataset: {hf_dir}")
        ds = load_from_disk(str(hf_dir))

        ref = self._build_model(cfg, ref_model)
        verifiable = {"AIME", "GSM8K", "MATH"}
        loss_type = "bce" if dataset in verifiable else "mle"
        classifier = ValueClassifier(
            cls_model,
            tokenizer=ref.tokenizer,
            device=ref.model.device,
            loss_type=loss_type,
            num_atoms=int(self.cfg.num_atoms),
            V_min=float(self.cfg.V_min),
            V_max=float(self.cfg.V_max),
            attn_impl=str(cfg.common.attn_impl),
            dtype=str(cfg.common.amp_dtype),
            gradient_checkpointing=bool(cfg.common.gradient_checkpointing),
        )
        self.maybe_load_classifier_from_prev_round(
            classifier,
            run_root=run_root,
            dataset=dataset,
            family=family,
            round_idx=round_idx,
            device=ref.model.device,
        )

        trainer = QSharpTrainer(
            classifier=classifier,
            tokenizer=ref.tokenizer,
            batch_size=int(self.cfg.batch_size),
            num_workers=int(self.cfg.num_workers),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
            grad_clip=float(self.cfg.grad_clip),
            pad_token_id=ref.pad_token_id,
            micro_batch_size=int(cfg.common.micro_batch_size),
            amp_dtype=str(cfg.common.amp_dtype),
            clear_cache_interval=int(cfg.common.clear_cache_interval),
        )
        # Convert dataset rows to classifier training examples if needed
        # Expecting keys: input_ids, target_ids, rewards, loss_weights
        sample = ds[0] if len(ds) > 0 else None
        need_convert = sample is not None and not all(
            k in sample for k in ("input_ids", "target_ids", "rewards", "loss_weights")
        )
        if need_convert:
            ds = convert_qsharp_rows_to_classifier_dataset(
                ds,
                tokenizer=ref.tokenizer,
                use_chat_template=ref.gen_cfg.use_chat_template,
            )

        loader = trainer.create_loader(ds)
        stats = trainer.train(loader, num_epochs=int(self.cfg.epochs))

        ckpt_dir = self.get_ckpt_dir(run_root, dataset, family, round_idx)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(classifier.state_dict(), str(ckpt_dir / "classifier.pt"))

        # Reload classifier for eval to mirror DPO pattern
        reloaded = ValueClassifier(
            cls_model,
            tokenizer=ref.tokenizer,
            device=ref.model.device,
            loss_type=loss_type,
            num_atoms=int(self.cfg.num_atoms),
            V_min=float(self.cfg.V_min),
            V_max=float(self.cfg.V_max),
            attn_impl=str(cfg.common.attn_impl),
            dtype=str(cfg.common.amp_dtype),
            gradient_checkpointing=bool(cfg.common.gradient_checkpointing),
        )
        state = torch.load(
            str(ckpt_dir / "classifier.pt"), map_location=ref.model.device
        )
        reloaded.load_state_dict(state)

        guided = self.build_guided_with(cfg, ref=ref, classifier=reloaded)

        # Evaluate across multiple etas if provided
        eval_targets = list(
            getattr(cfg.evaluation.datasets_by_train, dataset, [dataset])
        )
        eval_etas = list(
            getattr(cfg.evaluation, "etas", [float(self.cfg.guidance.eta)])
        )

        eval_map: Dict[str, Dict[str, Dict[str, float]]] = {}
        for eval_ds in eval_targets:
            by_eta: Dict[str, Dict[str, float]] = {}
            for eta in eval_etas:
                try:
                    guided.guidance.eta = float(eta)
                except Exception:
                    guided.guidance.eta = float(self.cfg.guidance.eta)
                save_dir = output_dir / f"eval_{eval_ds}_eta{guided.guidance.eta}"
                if eval_ds in {"TLDR", "IMDBGen"}:
                    metrics = evaluate_avg_reward(
                        cfg, guided, eval_ds, ref_model=ref, save_dir=save_dir
                    )
                else:
                    metrics = evaluate_pass1_maj8(
                        cfg, guided, eval_ds, ref_model=ref, save_dir=save_dir
                    )
                by_eta[str(guided.guidance.eta)] = metrics
            eval_map[eval_ds] = by_eta

        primary_eta = float(
            getattr(self.cfg.guidance, "eta", eval_etas[0] if eval_etas else 1.0)
        )
        primary_eta_str = str(primary_eta)
        primary_ds = (
            dataset
            if dataset in eval_map
            else (next(iter(eval_map.keys())) if eval_map else None)
        )
        if primary_ds is not None:
            pmap = eval_map[primary_ds]
            primary_metrics = pmap.get(primary_eta_str) or (
                next(iter(pmap.values())) if pmap else {}
            )
        else:
            primary_metrics = {}

        result = {
            "algo": "QSharp",
            "ref_model": ref_model,
            "cls_model": cls_model,
            "dataset": dataset,
            "metrics": {
                "trained_steps": int(stats.get("steps", 0)),
                "loss": float(stats.get("loss", 0.0) or 0.0),
                **primary_metrics,
            },
            "eval": eval_map,
        }
        if "avg_reward" in primary_metrics:
            logger.info(
                "✅ QSharp done: steps={} loss={:.4f} avg_reward={:.4f}",
                int(result["metrics"]["trained_steps"]),
                float(result["metrics"]["loss"]),
                float(primary_metrics.get("avg_reward", 0.0) or 0.0),
            )
        else:
            logger.info(
                "✅ QSharp done: steps={} loss={:.4f} pass@1={:.3f} maj@8={:.3f}",
                int(result["metrics"]["trained_steps"]),
                float(result["metrics"]["loss"]),
                float(primary_metrics.get("pass@1", 0.0) or 0.0),
                float(primary_metrics.get("maj@8", 0.0) or 0.0),
            )
        return result

    def score_samples(
        self, ex, y_a: str, y_b: str
    ) -> Tuple[float, float, Optional[int]]:
        ds = getattr(self, "_dataset", None)
        # Verifiable datasets expose an is_correct method
        if ds is not None and hasattr(ds, "is_correct"):
            score_a = 1.0 if ds.is_correct(ex.answer, y_a) else 0.0
            score_b = 1.0 if ds.is_correct(ex.answer, y_b) else 0.0
            return score_a, score_b, None
        # Otherwise, fall back to reward model scoring if available
        if hasattr(self, "_reward") and self._reward is not None:
            r_a, r_b, _ = self._reward.score_pair(ex.question, y_a, y_b)
            return r_a, r_b, None
        raise ValueError("No reward model available")
