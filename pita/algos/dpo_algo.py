from __future__ import annotations

from typing import Any, Dict
from pathlib import Path
import logging

import os

from pita.trainers import PreferencePairDataset, DPOTrainer

from pita.core.registry import register_algorithm
from .base import PostTrainingAlgorithms


@register_algorithm("DPO")
class DPOAlgorithm(PostTrainingAlgorithms):
    ALGO_KEY = "DPO"

    def run(
        self, cfg, ref_model: str, cls_model: str, dataset: str, family: str, output_dir
    ) -> Dict[str, Any]:
        logger = logging.getLogger(__name__)

        # Load dataset
        run_root = Path(os.getcwd())
        family_cap = str(family).capitalize()
        ds_root = run_root / "datasets" / self.algo_key
        hf_dir = ds_root / f"{dataset}_{family_cap}.hf"
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
        stats = trainer.train(
            loader, max_steps=int(self.cfg.max_steps), epochs=int(self.cfg.epochs)
        )

        # Save checkpoints under Hydra run folder: models/<ALGO>/<ref_vs_cls>/<dataset>
        pair_name = f"{ref_model}_vs_{cls_model}"
        ckpt_dir = run_root / "models" / self.algo_key / pair_name / dataset
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        try:
            trainer.policy.save_pretrained(str(ckpt_dir))
            trainer.tokenizer.save_pretrained(str(ckpt_dir))
        except Exception as e:
            logger.warning("Failed to save model: {}", e)

        return {
            "algo": "DPO",
            "ref_model": ref_model,
            "cls_model": cls_model,
            "dataset": dataset,
            "metrics": {
                "trained_steps": int(stats.get("steps", 0)),
                "loss": float(stats.get("loss", 0.0) or 0.0),
                "acc": float(stats.get("acc", 0.0) or 0.0),
            },
        }
