from __future__ import annotations

from typing import Any, Dict, Optional
from loguru import logger
import torch

from pita.trainers import DPOTrainer
from pita.datasets import PreferencePairDataset

from pita.core.registry import register_algorithm
from .base import PostTrainingAlgorithms
from pita.core.io import get_run_root, get_snapshot_paths
from pita.eval.evaluate import evaluate_pass1_maj8, evaluate_avg_reward


@register_algorithm("DPO")
class DPOAlgorithm(PostTrainingAlgorithms):
    ALGO_KEY = "DPO"

    def run(
        self,
        cfg,
        ref_model: str,
        cls_model: str,  # Not actually used
        dataset: str,
        family: str,
        output_dir,
        round_idx: int,
        run_root: Optional[Any] = None,
    ) -> Dict[str, Any]:
        logger.info(
            "DPO start: dataset={} family={}",
            dataset,
            family,
        )

        # Load dataset
        if run_root is None:
            run_root = get_run_root()
        _, hf_dir, _ = get_snapshot_paths(
            self.algo_key, dataset, family, round_idx, run_root=run_root
        )
        if not hf_dir.exists():
            raise FileNotFoundError(f"Missing DPO dataset: {hf_dir}")
        ds = PreferencePairDataset(hf_dir)

        policy = self._build_model(cfg, ref_model)
        reference = self._build_model(cfg, ref_model)

        trainer = DPOTrainer(
            policy=policy,
            reference=reference,
            dpo_cfg=self.cfg,
            use_chat_template=bool(cfg.generation.use_chat_template),
            micro_batch_size=int(cfg.training.micro_batch_size),
            amp_dtype=str(cfg.system.amp_dtype),
            clear_cache_interval=int(cfg.system.clear_cache_interval),
            grad_accumulation_steps=int(
                getattr(self.cfg, "gradient_accumulation_steps", 1) or 1
            ),
            warmup_steps=int(getattr(self.cfg, "warmup_steps", 0) or 0),
            save_dir=str(output_dir),
            ckpt_freq=int(getattr(self.cfg, "ckpt_freq", -1) or -1),
            eval_freq=int(getattr(self.cfg, "eval_freq", -1) or -1),
        )

        loader = trainer.create_loader(ds, shuffle=True)
        stats = trainer.train(loader, epochs=int(self.cfg.epochs))

        ckpt_dir = self.get_ckpt_dir(run_root, dataset, family, round_idx)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # Save model checkpoint
        # DPOTrainer doesn't use accelerator, so we save directly
        try:
            unwrapped = trainer.unwrap_policy()
        except Exception:
            unwrapped = trainer.policy
        unwrapped.save_pretrained(str(ckpt_dir))
        trainer.tokenizer.save_pretrained(str(ckpt_dir))

        # Evaluate on one or more datasets using the trained policy
        eval_map: Dict[str, Dict[str, float]] = {}
        eval_model = self._build_model(cfg, str(ckpt_dir))
        eval_targets = list(
            getattr(cfg.evaluation.datasets_by_train, dataset, [dataset])
        )
        for eval_ds in eval_targets:
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
            # Clear memory after each evaluation to prevent fragmentation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Cleanup models to free memory
        del policy, reference, trainer, eval_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        primary_metrics = eval_map.get(dataset) or (
            next(iter(eval_map.values())) if eval_map else {}
        )

        result = {
            "algo": "DPO",
            "ref_model": ref_model,
            "cls_model": cls_model,
            "dataset": dataset,
            "metrics": {
                "trained_steps": int(stats.get("steps", 0)),
                "loss": float(stats.get("loss", 0.0) or 0.0),
                "acc": float(stats.get("acc", 0.0) or 0.0),
                **primary_metrics,
            },
            "eval": eval_map,
        }
        if "avg_reward" in primary_metrics:
            logger.info(
                "✅ DPO done: steps={} loss={:.4f} acc={:.4f} avg_reward={:.4f}",
                int(result["metrics"]["trained_steps"]),
                float(result["metrics"]["loss"]),
                float(result["metrics"].get("acc", 0.0) or 0.0),
                float(primary_metrics.get("avg_reward", 0.0) or 0.0),
            )
        else:
            logger.info(
                "✅ DPO done: steps={} loss={:.4f} acc={:.4f} pass@1={:.3f} maj@8={:.3f}",
                int(result["metrics"]["trained_steps"]),
                float(result["metrics"]["loss"]),
                float(result["metrics"].get("acc", 0.0) or 0.0),
                float(primary_metrics.get("pass@1", 0.0) or 0.0),
                float(primary_metrics.get("maj@8", 0.0) or 0.0),
            )
        return result
