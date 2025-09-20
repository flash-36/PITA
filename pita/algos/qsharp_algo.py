from __future__ import annotations

from typing import Any, Dict, Tuple, List, Optional

from loguru import logger
import torch

from pita.core.registry import register_algorithm
from .base import ValueGuidedAlgorithms, PostTrainingAlgorithms
from pita.core.io import get_run_root
from pita.trainers import ValueClassifierTrainer
from datasets import load_from_disk
from pita.models.value_classifier import ValueClassifier
from pita.core.prompts import build_instruction_prompt


@register_algorithm("Q#")
class QSharpAlgorithm(ValueGuidedAlgorithms):
    ALGO_KEY = "Q#"

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
            "Q# start: dataset=%s family=%s",
            dataset,
            family,
        )

        run_root = get_run_root()
        family_cap = str(family).capitalize()
        ds_root = run_root / "datasets" / self.algo_key
        hf_dir = ds_root / f"{dataset}_{family_cap}_r{int(round_idx)+1}.hf"
        if not hf_dir.exists():
            raise FileNotFoundError(f"Missing Q# dataset: {hf_dir}")
        ds = load_from_disk(str(hf_dir))

        ref = self._build_model(cfg, ref_model)
        classifier = ValueClassifier(
            cls_model,
            tokenizer=ref.tokenizer,
            device=ref.model.device,
        )

        trainer = ValueClassifierTrainer(
            classifier=classifier,
            tokenizer=ref.tokenizer,
            batch_size=int(self.cfg.batch_size),
            num_workers=int(self.cfg.num_workers),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
            grad_clip=float(self.cfg.grad_clip),
            dtype=ref.gen_cfg.dtype,
            pad_token_id=ref.pad_token_id,
        )
        # Convert dataset rows to classifier training examples if needed
        # Expecting keys: input_ids, target_ids, rewards, loss_weights
        sample = ds[0] if len(ds) > 0 else None
        need_convert = sample is not None and not all(
            k in sample for k in ("input_ids", "target_ids", "rewards", "loss_weights")
        )
        if need_convert:
            from datasets import Dataset
            from tqdm import tqdm

            rows: List[Dict[str, Any]] = []
            for ex in tqdm(ds, desc="Q#:convert_ds"):
                prompt = ex.get("prompt", "")
                context = ex.get("context") or prompt
                # Build instruction-formatted context to match rollout formatting
                built = build_instruction_prompt(
                    prompt,
                    tokenizer=ref.tokenizer,
                    use_chat_template=ref.gen_cfg.use_chat_template,
                )
                # Extract solution prefix from saved context (without chat template)
                sol_prefix = (
                    context[len(prompt) :] if len(context) >= len(prompt) else ""
                )
                context_built = built + sol_prefix

                # Tokenize context
                ti = ref.tokenizer(context_built, add_special_tokens=False)["input_ids"]

                # Prepare targets for y_a and y_b by removing prefix tokens
                y_a = ex.get("y_a", "")
                y_b = ex.get("y_b", "")
                prefix_ids = ref.tokenizer(sol_prefix, add_special_tokens=False)[
                    "input_ids"
                ]

                def to_target_ids(text: str) -> List[int]:
                    full = ref.tokenizer(text, add_special_tokens=False)["input_ids"]
                    return full[len(prefix_ids) :]

                to_a = to_target_ids(y_a)
                to_b = to_target_ids(y_b)

                score_a = float(ex.get("score_a", 0.0))
                score_b = float(ex.get("score_b", 0.0))

                rows.append(
                    {
                        "input_ids": ti,
                        "target_ids": to_a,
                        "rewards": score_a,
                        "loss_weights": 1.0,
                    }
                )
                rows.append(
                    {
                        "input_ids": ti,
                        "target_ids": to_b,
                        "rewards": score_b,
                        "loss_weights": 1.0,
                    }
                )
            ds = Dataset.from_list(rows)

        loader = trainer.create_loader(ds)
        stats = trainer.train(loader, num_epochs=int(self.cfg.epochs))

        r_suffix = f"_r{int(round_idx)+1}"
        ckpt_dir = (
            run_root / "models" / self.algo_key / f"{dataset}_{family_cap}{r_suffix}"
        )
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        try:
            torch.save(classifier.state_dict(), str(ckpt_dir / "classifier.pt"))
        except Exception as e:
            logger.warning("Failed to save classifier: %s", e)

        # Reload classifier for eval to mirror DPO pattern
        reloaded = ValueClassifier(
            cls_model,
            tokenizer=ref.tokenizer,
            device=ref.model.device,
        )
        try:
            state = torch.load(
                str(ckpt_dir / "classifier.pt"), map_location=ref.model.device
            )
            reloaded.load_state_dict(state)
        except Exception as e:
            logger.warning("Failed to reload classifier, using in-memory: %s", e)
            reloaded = classifier

        guided = self.build_guided_with(cfg, ref=ref, classifier=reloaded)
        eval_metrics = PostTrainingAlgorithms.evaluate_pass1_maj8(
            self, cfg, guided, dataset, save_dir=output_dir
        )

        result = {
            "algo": "Q#",
            "ref_model": ref_model,
            "cls_model": cls_model,
            "dataset": dataset,
            "metrics": {
                "trained_steps": int(stats.get("steps", 0)),
                "loss": float(stats.get("loss", 0.0) or 0.0),
                **eval_metrics,
            },
        }
        logger.info(
            "✅ Q# done: steps=%d loss=%.4f pass@1=%.3f maj@8=%.3f",
            int(result["metrics"]["trained_steps"]),
            float(result["metrics"]["loss"]),
            float(result["metrics"].get("pass@1", 0.0) or 0.0),
            float(result["metrics"].get("maj@8", 0.0) or 0.0),
        )
        return result

    def score_samples(
        self, ex, y_a: str, y_b: str
    ) -> Tuple[float, float, Optional[int]]:
        ds = getattr(self, "_dataset", None)
        score_a = 1.0 if ds and ds.is_correct(ex.answer, y_a) else 0.0
        score_b = 1.0 if ds and ds.is_correct(ex.answer, y_b) else 0.0
        return score_a, score_b, None
