from __future__ import annotations

from typing import Any, Dict
from loguru import logger

from pita.trainers import DPOTrainer
from pita.datasets import PreferencePairDataset

from pita.core.registry import register_algorithm
from .base import PostTrainingAlgorithms
from pita.core.io import get_run_root, get_snapshot_paths
from pita.eval.evaluate import evaluate_pass1_maj8


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
    ) -> Dict[str, Any]:
        logger.info(
            "DPO start: dataset={} family={}",
            dataset,
            family,
        )

        # Load dataset
        run_root = get_run_root()
        _, hf_dir, _ = get_snapshot_paths(self.algo_key, dataset, family, round_idx)
        if not hf_dir.exists():
            raise FileNotFoundError(f"Missing DPO dataset: {hf_dir}")
        ds = PreferencePairDataset(hf_dir)

        policy = self._build_model(cfg, ref_model)
        reference = self._build_model(cfg, ref_model)

        trainer = DPOTrainer(
            policy=policy,
            reference=reference,
            dpo_cfg=self.cfg,
            use_chat_template=bool(cfg.common.use_chat_template),
        )

        loader = trainer.create_loader(ds, shuffle=True)
        stats = trainer.train(loader, epochs=int(self.cfg.epochs))

        ckpt_dir = self.get_ckpt_dir(run_root, dataset, family, round_idx)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        trainer.policy.save_pretrained(str(ckpt_dir))
        trainer.tokenizer.save_pretrained(str(ckpt_dir))

        # Evaluate pass@1 and maj@8 on test split using the trained policy
        eval_model = self._build_model(cfg, str(ckpt_dir))
        eval_metrics = evaluate_pass1_maj8(
            cfg, eval_model, dataset, save_dir=output_dir
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
                **eval_metrics,
            },
        }
        logger.info(
            "✅ DPO done: steps={} loss={:.4f} acc={:.4f} pass@1={:.3f} maj@8={:.3f}",
            int(result["metrics"]["trained_steps"]),
            float(result["metrics"]["loss"]),
            float(result["metrics"]["acc"]),
            float(result["metrics"].get("pass@1", 0.0) or 0.0),
            float(result["metrics"].get("maj@8", 0.0) or 0.0),
        )
        return result
