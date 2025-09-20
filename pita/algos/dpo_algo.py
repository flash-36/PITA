from __future__ import annotations

from typing import Any, Dict
from pathlib import Path
from loguru import logger

import os

from pita.trainers import PreferencePairDataset, DPOTrainer

from pita.core.registry import register_algorithm
from .base import PostTrainingAlgorithms
from pita.core.io import get_run_root


@register_algorithm("DPO")
class DPOAlgorithm(PostTrainingAlgorithms):
    ALGO_KEY = "DPO"

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
            "DPO start: dataset=%s family=%s",
            dataset,
            family,
        )

        # Load dataset
        run_root = get_run_root()
        family_cap = str(family).capitalize()
        ds_root = run_root / "datasets" / self.algo_key
        hf_dir = ds_root / f"{dataset}_{family_cap}_r{int(round_idx)+1}.hf"
        if not hf_dir.exists():
            raise FileNotFoundError(f"Missing DPO dataset: {hf_dir}")
        ds = PreferencePairDataset(hf_dir)

        policy = self._build_model(cfg, cls_model)
        reference = self._build_model(cfg, ref_model)

        trainer = DPOTrainer(
            policy=policy,
            reference=reference,
            dpo_cfg=self.cfg,
            use_chat_template=bool(cfg.common.use_chat_template),
        )

        loader = trainer.create_loader(ds, shuffle=True)
        stats = trainer.train(loader, epochs=int(self.cfg.epochs))

        r_suffix = f"_r{int(round_idx)+1}"
        ckpt_dir = (
            run_root / "models" / self.algo_key / f"{dataset}_{family_cap}{r_suffix}"
        )
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        try:
            trainer.policy.save_pretrained(str(ckpt_dir))
            trainer.tokenizer.save_pretrained(str(ckpt_dir))
        except Exception as e:
            logger.warning("Failed to save model: %s", e)

        # Evaluate pass@1 and maj@8 on test split using the trained policy
        eval_model = self._build_model(cfg, str(ckpt_dir))
        eval_metrics = self.evaluate_pass1_maj8(
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
            "✅ DPO done: steps=%d loss=%.4f acc=%.4f pass@1=%.3f maj@8=%.3f",
            int(result["metrics"]["trained_steps"]),
            float(result["metrics"]["loss"]),
            float(result["metrics"]["acc"]),
            float(result["metrics"].get("pass@1", 0.0) or 0.0),
            float(result["metrics"].get("maj@8", 0.0) or 0.0),
        )
        return result
