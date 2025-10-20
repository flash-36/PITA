from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from loguru import logger
from pathlib import Path

import torch
from datasets import load_from_disk

from pita.trainers import PITATrainer, QSharpTrainer
from pita.models.value_classifier import ValueClassifier
from pita.models import RewardScorer
from pita.datasets.convert import (
    convert_pita_rows_to_classifier_dataset,
    convert_qsharp_rows_to_classifier_dataset,
)
from pita.core.prompts import build_instruction_prompt
from datasets import Dataset
from tqdm import tqdm

from pita.core.registry import register_algorithm
from .base import ValueGuidedAlgorithms
from pita.core.io import (
    get_run_root,
    get_snapshot_paths,
    mark_phase_complete,
    check_phase_complete,
)
from pita.core.compute_tracker import get_compute_tracker, reset_compute_tracker
from pita.eval.evaluate import evaluate_pass1_maj8, evaluate_avg_reward


@register_algorithm("QSharp-HF")
class QSharpHFAlgorithm(ValueGuidedAlgorithms):
    ALGO_KEY = "QSharp-HF"

    def resolve_ref_for_round(
        self,
        run_root,
        dataset: str,
        family: str,
        ref_model_alias: str,
        round_idx: int,
    ) -> str:
        """QSharp-HF doesn't save fine-tuned models, only classifiers.

        Always return the base reference model since we don't update the policy.
        """
        return ref_model_alias

    def generate_data(
        self,
        cfg,
        ref_model: str,
        dataset: str,
        family: str,
        round_idx: int,
        cls_model: Optional[str] = None,
        run_root: Optional[Any] = None,
    ) -> None:
        if dataset in {"TLDR", "IMDBGen"}:
            ds_cfg = cfg.datasets[dataset]
            rm_model = str(ds_cfg.reward_model)
            device = 0 if torch.cuda.is_available() else -1
            self._reward = RewardScorer(
                rm_model,
                bt_sampling=bool(cfg.data_collection.bradley_terry_sampling),
                bt_beta=float(cfg.data_collection.bradley_terry_beta),
                device=device,
                dtype=str(cfg.system.dtype),
                batch_size=int(cfg.data_collection.reward_batch_size),
            )

        super().generate_data(
            cfg, ref_model, dataset, family, round_idx, cls_model, run_root
        )

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

    def _rescore_with_proxy_rm(
        self, ds_raw: Dataset, proxy_rm: ValueClassifier, ref, use_chat_template: bool
    ) -> Dataset:
        proxy_rm.eval()
        rescored_rows = []
        for ex in tqdm(ds_raw, desc="QSharp-HF:rescore"):
            prompt = ex.get("prompt", "")
            context = ex.get("context") or prompt
            built = build_instruction_prompt(
                prompt, tokenizer=ref.tokenizer, use_chat_template=use_chat_template
            )
            sol_prefix = context[len(prompt) :] if len(context) >= len(prompt) else ""
            context_built = built + sol_prefix

            y_a = ex.get("y_a", "")
            y_b = ex.get("y_b", "")

            def score_response(response: str) -> float:
                full_text = context_built + response
                tokenized = ref.tokenizer(full_text, return_tensors="pt")
                input_ids = tokenized["input_ids"].to(proxy_rm.backbone.device)
                attention_mask = tokenized["attention_mask"].to(
                    proxy_rm.backbone.device
                )
                with torch.no_grad():
                    output = proxy_rm(
                        input_ids=input_ids, attention_mask=attention_mask
                    )
                    if proxy_rm.loss_type == "mle":
                        probs = torch.softmax(output.logits.squeeze(0), dim=0)
                        reward = float(
                            (probs * proxy_rm.atoms.to(probs.device)).sum().item()
                        )
                    else:
                        reward = float(output.logits.squeeze().item())
                return reward

            proxy_score_a = score_response(y_a)
            proxy_score_b = score_response(y_b)

            rescored_rows.append(
                {
                    **ex,
                    "score_a": proxy_score_a,
                    "score_b": proxy_score_b,
                }
            )

        return Dataset.from_list(rescored_rows)

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
    ) -> Dict[str, Any]:
        logger.info(
            "🎯 QSharp-HF start: dataset={} family={}",
            dataset,
            family,
        )

        reset_compute_tracker()
        tracker = get_compute_tracker()

        if run_root is None:
            run_root = get_run_root()
        _, hf_dir, _ = get_snapshot_paths(
            self.algo_key, dataset, family, round_idx, run_root=run_root
        )
        if not hf_dir.exists():
            raise FileNotFoundError(f"Missing QSharp-HF dataset: {hf_dir}")

        ckpt_dir = self.get_ckpt_dir(run_root, dataset, family, round_idx)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        ds_raw = load_from_disk(str(hf_dir))
        ref = self._build_model(cfg, ref_model)

        # Phase 1: Proxy RM Training
        if check_phase_complete(
            "proxy_rm_training", self.algo_key, dataset, family, round_idx, run_root
        ):
            logger.info(
                "⏭️  Phase 1: Proxy RM training already complete, loading checkpoint..."
            )
            proxy_loss_type = str(self.cfg.proxy_loss_type)
            proxy_rm = ValueClassifier(
                cls_model,
                tokenizer=ref.tokenizer,
                device=ref.model.device,
                loss_type=proxy_loss_type,
                num_atoms=int(self.cfg.num_atoms),
                V_min=float(self.cfg.V_min),
                V_max=float(self.cfg.V_max),
                attn_impl=str(cfg.system.attn_impl),
                dtype=str(cfg.system.amp_dtype),
                gradient_checkpointing=bool(cfg.training.gradient_checkpointing),
            )
            proxy_rm.load_state_dict(
                torch.load(str(ckpt_dir / "proxy_rm.pt"), map_location=ref.model.device)
            )
        else:
            logger.info("📚 Phase 1: Training proxy reward model from preferences...")
            proxy_loss_type = str(self.cfg.proxy_loss_type)

            proxy_rm = ValueClassifier(
                cls_model,
                tokenizer=ref.tokenizer,
                device=ref.model.device,
                loss_type=proxy_loss_type,
                num_atoms=int(self.cfg.num_atoms),
                V_min=float(self.cfg.V_min),
                V_max=float(self.cfg.V_max),
                attn_impl=str(cfg.system.attn_impl),
                dtype=str(cfg.system.amp_dtype),
                gradient_checkpointing=bool(cfg.training.gradient_checkpointing),
            )

            self.maybe_load_classifier_from_prev_round(
                proxy_rm,
                run_root=run_root,
                dataset=dataset,
                family=family,
                round_idx=round_idx,
                device=ref.model.device,
            )

            proxy_trainer = PITATrainer(
                proxy_rm,
                tokenizer=ref.tokenizer,
                batch_size=int(self.cfg.batch_size),
                max_batch_num_tokens=int(getattr(self.cfg, "max_batch_num_tokens", -1)),
                num_workers=int(self.cfg.num_workers),
                lr=float(self.cfg.proxy_lr),
                weight_decay=float(self.cfg.weight_decay),
                grad_clip=float(self.cfg.grad_clip),
                pad_token_id=int(ref.pad_token_id),
                micro_batch_size=int(cfg.training.micro_batch_size),
                amp_dtype=str(cfg.system.amp_dtype),
                clear_cache_interval=int(cfg.system.clear_cache_interval),
            )

            ds_pref = convert_pita_rows_to_classifier_dataset(
                ds_raw,
                tokenizer=ref.tokenizer,
                use_chat_template=ref.gen_cfg.use_chat_template,
            )

            proxy_loader = proxy_trainer.create_loader(ds_pref)
            with tracker.track_phase(f"proxy_rm_training_{dataset}"):
                proxy_stats = proxy_trainer.train(
                    proxy_loader, num_epochs=int(self.cfg.proxy_epochs)
                )

            torch.save(proxy_rm.state_dict(), str(ckpt_dir / "proxy_rm.pt"))
            mark_phase_complete(
                "proxy_rm_training", self.algo_key, dataset, family, round_idx, run_root
            )

            del proxy_trainer, proxy_loader, ds_pref
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Phase 2: Data Rescoring
        rescored_data_path = ckpt_dir / "ds_rescored.hf"
        if check_phase_complete(
            "data_rescoring", self.algo_key, dataset, family, round_idx, run_root
        ):
            logger.info("⏭️  Phase 2: Data rescoring already complete, loading...")
            ds_rescored = Dataset.load_from_disk(str(rescored_data_path))
        else:
            logger.info("💪 Phase 2: Re-scoring dataset with proxy reward model...")
            with tracker.track_phase(f"rescoring_{dataset}"):
                ds_rescored = self._rescore_with_proxy_rm(
                    ds_raw, proxy_rm, ref, ref.gen_cfg.use_chat_template
                )
            ds_rescored.save_to_disk(str(rescored_data_path))
            mark_phase_complete(
                "data_rescoring", self.algo_key, dataset, family, round_idx, run_root
            )

        del proxy_rm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Phase 3: ValueClassifier Training
        if check_phase_complete(
            "classifier_training", self.algo_key, dataset, family, round_idx, run_root
        ):
            logger.info(
                "⏭️  Phase 3: Classifier training already complete, loading checkpoint..."
            )
            qsharp_loss_type = str(self.cfg.loss_type)
            qsharp_classifier = ValueClassifier(
                cls_model,
                tokenizer=ref.tokenizer,
                device=ref.model.device,
                loss_type=qsharp_loss_type,
                num_atoms=int(self.cfg.num_atoms),
                V_min=float(self.cfg.V_min),
                V_max=float(self.cfg.V_max),
                attn_impl=str(cfg.system.attn_impl),
                dtype=str(cfg.system.amp_dtype),
                gradient_checkpointing=bool(cfg.training.gradient_checkpointing),
            )
            qsharp_classifier.load_state_dict(
                torch.load(
                    str(ckpt_dir / "classifier.pt"), map_location=ref.model.device
                )
            )
            qsharp_stats = {"steps": 0, "loss": 0.0}
        else:
            logger.info("🎓 Phase 3: Training ValueClassifier with proxy rewards...")
            qsharp_loss_type = str(self.cfg.loss_type)

            qsharp_classifier = ValueClassifier(
                cls_model,
                tokenizer=ref.tokenizer,
                device=ref.model.device,
                loss_type=qsharp_loss_type,
                num_atoms=int(self.cfg.num_atoms),
                V_min=float(self.cfg.V_min),
                V_max=float(self.cfg.V_max),
                attn_impl=str(cfg.system.attn_impl),
                dtype=str(cfg.system.amp_dtype),
                gradient_checkpointing=bool(cfg.training.gradient_checkpointing),
            )

            qsharp_trainer = QSharpTrainer(
                classifier=qsharp_classifier,
                tokenizer=ref.tokenizer,
                batch_size=int(self.cfg.batch_size),
                max_batch_num_tokens=int(getattr(self.cfg, "max_batch_num_tokens", -1)),
                num_workers=int(self.cfg.num_workers),
                lr=float(self.cfg.lr),
                weight_decay=float(self.cfg.weight_decay),
                grad_clip=float(self.cfg.grad_clip),
                pad_token_id=ref.pad_token_id,
                micro_batch_size=int(cfg.training.micro_batch_size),
                amp_dtype=str(cfg.system.amp_dtype),
                clear_cache_interval=int(cfg.system.clear_cache_interval),
            )

            ds_qsharp = convert_qsharp_rows_to_classifier_dataset(
                ds_rescored,
                tokenizer=ref.tokenizer,
                use_chat_template=ref.gen_cfg.use_chat_template,
            )

            qsharp_loader = qsharp_trainer.create_loader(ds_qsharp)
            with tracker.track_phase(f"value_classifier_training_{dataset}"):
                qsharp_stats = qsharp_trainer.train(
                    qsharp_loader, num_epochs=int(self.cfg.epochs)
                )

            torch.save(qsharp_classifier.state_dict(), str(ckpt_dir / "classifier.pt"))
            mark_phase_complete(
                "classifier_training",
                self.algo_key,
                dataset,
                family,
                round_idx,
                run_root,
            )

            del qsharp_trainer, qsharp_loader, ds_qsharp
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del ds_rescored, ds_raw
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("🔮 Evaluating with trained ValueClassifier guidance...")
        reloaded = ValueClassifier(
            cls_model,
            tokenizer=ref.tokenizer,
            device=ref.model.device,
            loss_type=qsharp_loss_type,
            num_atoms=int(self.cfg.num_atoms),
            V_min=float(self.cfg.V_min),
            V_max=float(self.cfg.V_max),
            attn_impl=str(cfg.system.attn_impl),
            dtype=str(cfg.system.amp_dtype),
            gradient_checkpointing=bool(cfg.training.gradient_checkpointing),
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
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
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

        compute_metrics = tracker.get_metrics()
        result = {
            "algo": "QSharp-HF",
            "ref_model": ref_model,
            "cls_model": cls_model,
            "dataset": dataset,
            "metrics": {
                "trained_steps": int(qsharp_stats.get("steps", 0)),
                "loss": float(qsharp_stats.get("loss", 0.0) or 0.0),
                **primary_metrics,
            },
            "eval": eval_map,
            "compute": compute_metrics,
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

        del ref, reloaded, guided
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result
