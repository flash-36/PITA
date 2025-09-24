from __future__ import annotations

from typing import Any, Dict, Tuple, Optional
from loguru import logger
import torch

from pita.core.registry import register_algorithm
from .base import ValueGuidedAlgorithms
from pita.eval.evaluate import evaluate_pass1_maj8, evaluate_avg_reward
from pita.core.io import get_run_root, get_snapshot_paths
from pita.trainers import PITATrainer
from datasets import load_from_disk
from pita.models.value_classifier import ValueClassifier
from pita.datasets.convert import convert_pita_rows_to_classifier_dataset
from pita.models import RewardScorer


@register_algorithm("PITA")
class PITAAlgorithm(ValueGuidedAlgorithms):
    ALGO_KEY = "PITA"

    def generate_data(
        self,
        cfg,
        ref_model: str,
        dataset: str,
        family: str,
        round_idx: int,
        cls_model: Optional[str] = None,
    ) -> None:
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
        return super().generate_data(
            cfg=cfg,
            ref_model=ref_model,
            dataset=dataset,
            family=family,
            round_idx=round_idx,
            cls_model=cls_model,
        )

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
        logger.info("🚀 PITA start: dataset={} family={}", dataset, family)

        run_root = get_run_root()
        _, hf_dir, _ = get_snapshot_paths(self.algo_key, dataset, family, round_idx)
        if not hf_dir.exists():
            raise FileNotFoundError(f"Missing PITA dataset: {hf_dir}")
        ds = load_from_disk(str(hf_dir))

        ref = self._build_model(cfg, ref_model)
        classifier = ValueClassifier(
            cls_model,
            tokenizer=ref.tokenizer,
            device=ref.model.device,
            loss_type=str(self.cfg.loss_type),
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

        trainer = PITATrainer(
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
        sample = ds[0] if len(ds) > 0 else None
        need_convert = sample is not None and not all(
            k in sample
            for k in ("input_ids", "chosen_target_ids", "rejected_target_ids")
        )
        if need_convert:
            ds = convert_pita_rows_to_classifier_dataset(
                ds,
                tokenizer=ref.tokenizer,
                use_chat_template=ref.gen_cfg.use_chat_template,
            )

        loader = trainer.create_loader(ds)
        stats = trainer.train(loader, num_epochs=int(self.cfg.epochs))

        ckpt_dir = self.get_ckpt_dir(run_root, dataset, family, round_idx)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(classifier.state_dict(), str(ckpt_dir / "classifier.pt"))

        reloaded = ValueClassifier(
            cls_model,
            tokenizer=ref.tokenizer,
            device=ref.model.device,
            loss_type=str(self.cfg.loss_type),
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

        # Primary metrics: use configured eta if present, otherwise the first eta
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
            "algo": "PITA",
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
                "✅ PITA done: steps={} loss={:.4f} avg_reward={:.4f}",
                int(result["metrics"]["trained_steps"]),
                float(result["metrics"]["loss"]),
                float(primary_metrics.get("avg_reward", 0.0) or 0.0),
            )
        else:
            logger.info(
                "✅ PITA done: steps={} loss={:.4f} pass@1={:.3f} maj@8={:.3f}",
                int(result["metrics"]["trained_steps"]),
                float(result["metrics"]["loss"]),
                float(primary_metrics.get("pass@1", 0.0) or 0.0),
                float(primary_metrics.get("maj@8", 0.0) or 0.0),
            )
        return result

    def score_samples(
        self, ex, y_a: str, y_b: str
    ) -> Tuple[float, float, Optional[int]]:
        r_a, r_b, preferred = self._reward.score_pair(ex.question, y_a, y_b)
        return r_a, r_b, preferred
