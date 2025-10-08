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
from pita.trainers.pita_trainer import PITATrainer
from pita.datasets.grpo_dataset import GRPODataset
from pita.datasets.convert import convert_pita_rows_to_classifier_dataset
from pita.models import RewardScorer
from pita.models.value_classifier import ValueClassifier
from pita.core.prompts import build_instruction_prompt

from pita.core.registry import register_algorithm
from .base import PostTrainingAlgorithms
from pita.core.io import get_run_root, get_snapshot_paths, merge_and_save_hf
from pita.eval.evaluate import evaluate_pass1_maj8, evaluate_avg_reward


@register_algorithm("GRPO-HF")
class GRPOHFAlgorithm(PostTrainingAlgorithms):
    ALGO_KEY = "GRPO-HF"

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
        snap_hf_prev, snap_hf, snap_csv = get_snapshot_paths(
            self.algo_key, dataset, family, round_idx, run_root=run_root
        )

        random.seed(int(cfg.data_collection.seed))
        model = self._build_model(cfg, ref_model)
        ds = self._build_dataset(cfg, dataset)
        gen_cfg = model.gen_cfg

        ds_cfg = cfg.datasets[dataset]
        rm_model = str(ds_cfg.reward_model)
        device = 0 if torch.cuda.is_available() else -1
        self._reward = RewardScorer(
            rm_model,
            bt_sampling=False,
            bt_beta=float(cfg.data_collection.bradley_terry_beta),
            device=device,
            dtype=str(cfg.system.dtype),
            batch_size=int(cfg.data_collection.reward_batch_size),
        )

        samples_per_prompt = int(self.cfg.samples_per_prompt)
        records: List[Dict[str, Any]] = []
        max_examples = int(cfg.data_collection.max_examples or 0)
        limit = max_examples if max_examples > 0 else len(ds)

        for ex in tqdm(
            islice(ds.iter(), limit), total=limit, desc=f"{self.algo_key}:{dataset}"
        ):
            prompt = ds.hydrate_prompt(ex.question)
            built = build_instruction_prompt(
                prompt,
                tokenizer=model.tokenizer,
                use_chat_template=model.gen_cfg.use_chat_template,
            )

            responses = []
            rewards = []
            for _ in range(samples_per_prompt):
                response = model.continue_from_context(
                    built, max_new_tokens=gen_cfg.max_new_tokens, greedy=False
                )
                reward = self._reward.score_single(ex.question, response)
                responses.append(response)
                rewards.append(float(reward))

            best_idx = max(range(len(rewards)), key=lambda i: rewards[i])
            worst_idx = min(range(len(rewards)), key=lambda i: rewards[i])

            records.append(
                {
                    "question": ex.question,
                    "answer": ex.answer,
                    "prompt": prompt,
                    "y_a": responses[best_idx],
                    "y_b": responses[worst_idx],
                    "score_a": rewards[best_idx],
                    "score_b": rewards[worst_idx],
                    "preferred": 0,
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
        run_root: Optional[Any] = None,
    ) -> Dict[str, Any]:
        logger.info(
            "🎯 GRPO-HF start: dataset={} family={}",
            dataset,
            family,
        )

        if run_root is None:
            run_root = get_run_root()
        _, hf_dir, _ = get_snapshot_paths(
            self.algo_key, dataset, family, round_idx, run_root=run_root
        )
        if not hf_dir.exists():
            raise FileNotFoundError(f"Missing GRPO-HF dataset: {hf_dir}")

        logger.info("📚 Training proxy reward model...")
        ref = self._build_model(cfg, ref_model)
        classifier = ValueClassifier(
            cls_model,
            tokenizer=ref.tokenizer,
            device=ref.model.device,
            loss_type=str(self.cfg.loss_type),
            num_atoms=int(self.cfg.num_atoms),
            V_min=float(self.cfg.V_min),
            V_max=float(self.cfg.V_max),
            attn_impl=str(cfg.system.attn_impl),
            dtype=str(cfg.system.amp_dtype),
            gradient_checkpointing=bool(cfg.training.gradient_checkpointing),
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
            lr=float(self.cfg.cls_lr),
            weight_decay=float(self.cfg.weight_decay),
            grad_clip=float(self.cfg.grad_clip),
            pad_token_id=int(ref.pad_token_id),
            micro_batch_size=int(cfg.training.micro_batch_size),
            amp_dtype=str(cfg.system.amp_dtype),
            clear_cache_interval=int(cfg.system.clear_cache_interval),
        )

        from datasets import load_from_disk

        ds_raw = load_from_disk(str(hf_dir))
        ds_converted = convert_pita_rows_to_classifier_dataset(
            ds_raw,
            tokenizer=ref.tokenizer,
            use_chat_template=ref.gen_cfg.use_chat_template,
        )

        cls_loader = cls_trainer.create_loader(ds_converted)
        cls_stats = cls_trainer.train(cls_loader, num_epochs=int(self.cfg.cls_epochs))

        ckpt_dir = self.get_ckpt_dir(run_root, dataset, family, round_idx)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(classifier.state_dict(), str(ckpt_dir / "proxy_reward_model.pt"))

        logger.info("💪 Re-scoring samples with proxy reward model...")
        grpo_ds = self._rescore_samples_with_proxy(ds_raw, classifier, ref)

        # Free up memory before loading policy and reference models
        logger.info("🧹 Clearing classifier and ref model to free memory...")
        del classifier, ref, cls_trainer, cls_loader, ds_raw, ds_converted
        torch.cuda.empty_cache()

        logger.info("🏋️ Training policy with GRPO...")
        policy = self._build_model(cfg, ref_model)
        reference = self._build_model(cfg, ref_model)

        trainer = GRPOTrainer(
            policy=policy,
            reference=reference,
            grpo_cfg=self.cfg,
            use_chat_template=bool(cfg.generation.use_chat_template),
            micro_batch_size=int(cfg.training.micro_batch_size),
            amp_dtype=str(cfg.system.amp_dtype),
            clear_cache_interval=int(cfg.system.clear_cache_interval),
        )

        loader = trainer.create_loader(grpo_ds, shuffle=True)
        stats = trainer.train(loader, epochs=int(self.cfg.epochs))

        trainer.policy.eval()
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
            # Also clear after evaluation to release memory
            torch.cuda.empty_cache()

        primary_metrics = eval_map.get(dataset) or (
            next(iter(eval_map.values())) if eval_map else {}
        )

        result = {
            "algo": "GRPO-HF",
            "ref_model": ref_model,
            "cls_model": cls_model,
            "dataset": dataset,
            "metrics": {
                "trained_steps": int(stats.get("steps", 0)),
                "loss": float(stats.get("loss", 0.0) or 0.0),
                "avg_reward": float(stats.get("avg_reward", 0.0) or 0.0),
                "avg_kl": float(stats.get("avg_kl", 0.0) or 0.0),
                "cls_loss": float(cls_stats.get("loss", 0.0) or 0.0),
                **primary_metrics,
            },
            "eval": eval_map,
        }

        if "avg_reward" in primary_metrics:
            logger.info(
                "✅ GRPO-HF done: steps={} loss={:.4f} train_reward={:.4f} eval_reward={:.4f} kl={:.4f} cls_loss={:.4f}",
                int(result["metrics"]["trained_steps"]),
                float(result["metrics"]["loss"]),
                float(stats.get("avg_reward", 0.0) or 0.0),
                float(primary_metrics.get("avg_reward", 0.0) or 0.0),
                float(stats.get("avg_kl", 0.0) or 0.0),
                float(cls_stats.get("loss", 0.0) or 0.0),
            )
        else:
            logger.info(
                "✅ GRPO-HF done: steps={} loss={:.4f} train_reward={:.4f} pass@1={:.3f} maj@8={:.3f} kl={:.4f} cls_loss={:.4f}",
                int(result["metrics"]["trained_steps"]),
                float(result["metrics"]["loss"]),
                float(stats.get("avg_reward", 0.0) or 0.0),
                float(primary_metrics.get("pass@1", 0.0) or 0.0),
                float(primary_metrics.get("maj@8", 0.0) or 0.0),
                float(stats.get("avg_kl", 0.0) or 0.0),
                float(cls_stats.get("loss", 0.0) or 0.0),
            )

        # Cleanup models to free memory
        del reference, eval_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    def _rescore_samples_with_proxy(
        self, ds_raw: Dataset, classifier: ValueClassifier, ref
    ):
        classifier.eval()
        records: List[Dict[str, Any]] = []

        group_id = 0
        prev_question = None

        for row in tqdm(ds_raw, desc=f"{self.algo_key}:proxy_scoring"):
            question = row["question"]
            prompt = row["prompt"]

            if question != prev_question:
                group_id += 1
                prev_question = question

            built = build_instruction_prompt(
                prompt,
                tokenizer=ref.tokenizer,
                use_chat_template=ref.gen_cfg.use_chat_template,
            )

            response_a = row["y_a"]
            response_b = row["y_b"]

            for response in [response_a, response_b]:
                full_text = built + response
                tokenized = ref.tokenizer(full_text, return_tensors="pt")
                input_ids = tokenized["input_ids"].to(classifier.backbone.device)
                attention_mask = tokenized["attention_mask"].to(
                    classifier.backbone.device
                )

                with torch.no_grad():
                    output = classifier(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    if classifier.loss_type == "mle":
                        probs = torch.softmax(output.logits.squeeze(0), dim=0)
                        reward = float(
                            (probs * classifier.atoms.to(probs.device)).sum().item()
                        )
                    else:
                        reward = float(output.logits.squeeze().item())

                records.append(
                    {
                        "question": question,
                        "answer": row["answer"],
                        "prompt": prompt,
                        "response": response,
                        "reward": reward,
                        "group_id": int(group_id - 1),
                    }
                )

        grpo_ds_list = Dataset.from_list(records)
        return GRPODataset(grpo_ds_list)

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
        ckpt = prev_ckpt_dir / "proxy_reward_model.pt"
        if ckpt.exists():
            state = torch.load(str(ckpt), map_location=device)
            classifier.load_state_dict(state)
            return True
        return False

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
