from __future__ import annotations

from typing import Any, Dict, List, Optional
from loguru import logger
from pathlib import Path
import random
from itertools import islice

import torch
from datasets import Dataset
from tqdm import tqdm

from pita.trainers.grpo_trainer import GRPOTrainer
from pita.datasets.grpo_dataset import GRPODataset
from pita.models import RewardScorer
from pita.core.prompts import build_instruction_prompt

from pita.core.registry import register_algorithm
from .base import PostTrainingAlgorithms
from pita.core.io import get_run_root, get_snapshot_paths, merge_and_save_hf
from pita.eval.evaluate import evaluate_pass1_maj8, evaluate_avg_reward


@register_algorithm("GRPO")
class GRPOAlgorithm(PostTrainingAlgorithms):
    ALGO_KEY = "GRPO"

    def generate_data(
        self,
        cfg,
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

        ds_cfg = cfg.datasets[dataset]
        rm_model = str(ds_cfg.reward_model)
        device = 0 if torch.cuda.is_available() else -1
        self._reward = RewardScorer(
            rm_model,
            bt_sampling=False,
            bt_beta=float(cfg.common.bt_beta),
            device=device,
            dtype=str(cfg.common.dtype),
            batch_size=int(cfg.collection.reward_batch_size),
        )

        samples_per_prompt = int(self.cfg.samples_per_prompt)
        records: List[Dict[str, Any]] = []
        max_examples = int(cfg.collection.max_examples or 0)
        limit = max_examples if max_examples > 0 else len(ds)

        for group_id, ex in enumerate(
            tqdm(
                islice(ds.iter(), limit), total=limit, desc=f"{self.algo_key}:{dataset}"
            )
        ):
            prompt = ds.hydrate_prompt(ex.question)
            built = build_instruction_prompt(
                prompt,
                tokenizer=model.tokenizer,
                use_chat_template=model.gen_cfg.use_chat_template,
            )

            for _ in range(samples_per_prompt):
                response = model.continue_from_context(
                    built, max_new_tokens=gen_cfg.max_new_tokens, greedy=False
                )
                reward = self._reward.score_single(ex.question, response)

                records.append(
                    {
                        "question": ex.question,
                        "answer": ex.answer,
                        "prompt": prompt,
                        "response": response,
                        "reward": float(reward),
                        "group_id": int(group_id),
                    }
                )

        if not records:
            return
        new_ds = Dataset.from_list(records)
        merge_and_save_hf(snap_hf_prev, new_ds, snap_hf, snap_csv)

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
        logger.info(
            "GRPO start: dataset={} family={}",
            dataset,
            family,
        )

        run_root = get_run_root()
        _, hf_dir, _ = get_snapshot_paths(self.algo_key, dataset, family, round_idx)
        if not hf_dir.exists():
            raise FileNotFoundError(f"Missing GRPO dataset: {hf_dir}")
        ds = GRPODataset(hf_dir)

        policy = self._build_model(cfg, ref_model)
        reference = self._build_model(cfg, ref_model)

        trainer = GRPOTrainer(
            policy=policy,
            reference=reference,
            grpo_cfg=self.cfg,
            use_chat_template=bool(cfg.common.use_chat_template),
            micro_batch_size=int(cfg.common.micro_batch_size),
            amp_dtype=str(cfg.common.amp_dtype),
            clear_cache_interval=int(cfg.common.clear_cache_interval),
        )

        loader = trainer.create_loader(ds, shuffle=True)
        stats = trainer.train(loader, epochs=int(self.cfg.epochs))

        trainer.policy.eval()
        ckpt_dir = self.get_ckpt_dir(run_root, dataset, family, round_idx)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if trainer.tokenizer.pad_token is None:
            trainer.tokenizer.pad_token = trainer.tokenizer.eos_token

        trainer.policy.save_pretrained(str(ckpt_dir))
        trainer.tokenizer.save_pretrained(str(ckpt_dir))

        del policy.model
        del policy
        del trainer
        torch.cuda.empty_cache()

        eval_model = self._build_model(cfg, str(ckpt_dir))
        eval_map: Dict[str, Dict[str, float]] = {}
        eval_targets = list(
            getattr(cfg.evaluation.datasets_by_train, dataset, [dataset])
        )
        for eval_ds in eval_targets:
            torch.cuda.empty_cache()
            logger.info(f"📊 Evaluating on {eval_ds}...")
            if eval_ds in {"TLDR", "IMDBGen"}:
                metrics = evaluate_avg_reward(
                    cfg,
                    eval_model,
                    eval_ds,
                    ref_model=reference,
                    save_dir=output_dir / f"eval_{eval_ds}",
                )
            else:
                metrics = evaluate_pass1_maj8(
                    cfg,
                    eval_model,
                    eval_ds,
                    ref_model=reference,
                    save_dir=output_dir / f"eval_{eval_ds}",
                )
            eval_map[eval_ds] = metrics
            logger.info(f"✓ {eval_ds} metrics: {metrics}")

        primary_metrics = eval_map.get(dataset) or (
            next(iter(eval_map.values())) if eval_map else {}
        )

        result = {
            "algo": "GRPO",
            "ref_model": ref_model,
            "cls_model": cls_model,
            "dataset": dataset,
            "metrics": {
                "trained_steps": int(stats.get("steps", 0)),
                "loss": float(stats.get("loss", 0.0) or 0.0),
                "avg_reward": float(stats.get("avg_reward", 0.0) or 0.0),
                "avg_kl": float(stats.get("avg_kl", 0.0) or 0.0),
                **primary_metrics,
            },
            "eval": eval_map,
        }

        if "avg_reward" in primary_metrics:
            logger.info(
                "✅ GRPO done: steps={} loss={:.4f} train_reward={:.4f} eval_reward={:.4f} kl={:.4f}",
                int(result["metrics"]["trained_steps"]),
                float(result["metrics"]["loss"]),
                float(stats.get("avg_reward", 0.0) or 0.0),
                float(primary_metrics.get("avg_reward", 0.0) or 0.0),
                float(stats.get("avg_kl", 0.0) or 0.0),
            )
        else:
            logger.info(
                "✅ GRPO done: steps={} loss={:.4f} train_reward={:.4f} pass@1={:.3f} maj@8={:.3f} kl={:.4f}",
                int(result["metrics"]["trained_steps"]),
                float(result["metrics"]["loss"]),
                float(stats.get("avg_reward", 0.0) or 0.0),
                float(primary_metrics.get("pass@1", 0.0) or 0.0),
                float(primary_metrics.get("maj@8", 0.0) or 0.0),
                float(stats.get("avg_kl", 0.0) or 0.0),
            )
        return result
