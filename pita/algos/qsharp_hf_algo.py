from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from loguru import logger
from pathlib import Path

import torch
from datasets import load_from_disk

from pita.trainers import PITATrainer
from pita.models.value_classifier import ValueClassifier
from pita.models import RewardScorer
from pita.datasets.convert import convert_pita_rows_to_classifier_dataset

from pita.core.registry import register_algorithm
from .base import ValueGuidedAlgorithms
from pita.core.io import get_run_root, get_snapshot_paths
from pita.eval.evaluate import evaluate_pass1_maj8, evaluate_avg_reward


@register_algorithm("QSharp-HF")
class QSharpHFAlgorithm(ValueGuidedAlgorithms):
    ALGO_KEY = "QSharp-HF"

    def generate_data(
        self,
        cfg,
        ref_model: str,
        dataset: str,
        family: str,
        round_idx: int,
        cls_model: Optional[str] = None,
    ) -> None:
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

        super().generate_data(cfg, ref_model, dataset, family, round_idx, cls_model)

    def score_samples(
        self, ex, y_a: str, y_b: str
    ) -> Tuple[float, float, Optional[int]]:
        ds = getattr(self, "_dataset", None)
        if ds is not None and hasattr(ds, "is_correct"):
            score_a = 1.0 if ds.is_correct(ex.answer, y_a) else 0.0
            score_b = 1.0 if ds.is_correct(ex.answer, y_b) else 0.0
            return score_a, score_b, None
        if hasattr(self, "_reward") and self._reward is not None:
            r_a, r_b, _ = self._reward.score_pair(ex.question, y_a, y_b)
            return r_a, r_b, None
        raise ValueError("No reward model available")

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
            "🎯 QSharp-HF start: dataset={} family={}",
            dataset,
            family,
        )

        run_root = get_run_root()
        _, hf_dir, _ = get_snapshot_paths(self.algo_key, dataset, family, round_idx)
        if not hf_dir.exists():
            raise FileNotFoundError(f"Missing QSharp-HF dataset: {hf_dir}")

        logger.info("📚 Training proxy reward model from preferences...")
        ref = self._build_model(cfg, ref_model)

        verifiable = {"AIME", "GSM8K", "MATH"}
        loss_type = "bce" if dataset in verifiable else str(self.cfg.loss_type)

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

        prev_classifier_loaded = self.maybe_load_classifier_from_prev_round(
            classifier,
            run_root=run_root,
            dataset=dataset,
            family=family,
            round_idx=round_idx,
            device=ref.model.device,
        )

        cls_trainer = PITATrainer(
            classifier,
            tokenizer=ref.tokenizer,
            batch_size=int(self.cfg.batch_size),
            num_workers=int(self.cfg.num_workers),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
            grad_clip=float(self.cfg.grad_clip),
            pad_token_id=int(ref.pad_token_id),
            micro_batch_size=int(cfg.common.micro_batch_size),
            amp_dtype=str(cfg.common.amp_dtype),
            clear_cache_interval=int(cfg.common.clear_cache_interval),
        )

        ds_raw = load_from_disk(str(hf_dir))
        ds_converted = convert_pita_rows_to_classifier_dataset(
            ds_raw,
            tokenizer=ref.tokenizer,
            use_chat_template=ref.gen_cfg.use_chat_template,
        )

        cls_loader = cls_trainer.create_loader(ds_converted)
        cls_stats = cls_trainer.train(cls_loader, num_epochs=int(self.cfg.epochs))

        ckpt_dir = self.get_ckpt_dir(run_root, dataset, family, round_idx)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(classifier.state_dict(), str(ckpt_dir / "classifier.pt"))

        logger.info("🔮 Evaluating with proxy reward model guidance...")
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

        eval_targets = list(
            getattr(cfg.evaluation.datasets_by_train, dataset, [dataset])
        )
        eval_etas = list(
            getattr(cfg.evaluation, "etas", [float(self.cfg.guidance.eta)])
        )

        eval_map: Dict[str, Dict[str, Dict[str, float]]] = {}
        for eval_ds in eval_targets:
            torch.cuda.empty_cache()
            by_eta: Dict[str, Dict[str, float]] = {}
            for eta in eval_etas:
                try:
                    guided.guidance.eta = float(eta)
                except Exception:
                    guided.guidance.eta = float(self.cfg.guidance.eta)
                logger.info(f"📊 Evaluating {eval_ds} at eta={guided.guidance.eta}...")
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
                logger.info(f"✓ {eval_ds} eta={guided.guidance.eta} metrics: {metrics}")
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
            "algo": "QSharp-HF",
            "ref_model": ref_model,
            "cls_model": cls_model,
            "dataset": dataset,
            "metrics": {
                "trained_steps": int(cls_stats.get("steps", 0)),
                "loss": float(cls_stats.get("loss", 0.0) or 0.0),
                **primary_metrics,
            },
            "eval": eval_map,
        }

        if "avg_reward" in primary_metrics:
            logger.info(
                "✅ QSharp-HF done: steps={} loss={:.4f} avg_reward={:.4f}",
                int(result["metrics"]["trained_steps"]),
                float(result["metrics"]["loss"]),
                float(primary_metrics.get("avg_reward", 0.0) or 0.0),
            )
        else:
            logger.info(
                "✅ QSharp-HF done: steps={} loss={:.4f} pass@1={:.3f} maj@8={:.3f}",
                int(result["metrics"]["trained_steps"]),
                float(result["metrics"]["loss"]),
                float(primary_metrics.get("pass@1", 0.0) or 0.0),
                float(primary_metrics.get("maj@8", 0.0) or 0.0),
            )
        return result
