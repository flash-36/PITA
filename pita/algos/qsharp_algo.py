from __future__ import annotations

from typing import Any, Dict, Tuple, List, Optional
from datasets import Dataset
from tqdm import tqdm

from loguru import logger
import torch

from pita.core.registry import register_algorithm
from .base import ValueGuidedAlgorithms
from pita.eval.evaluate import evaluate_pass1_maj8
from pita.core.io import get_run_root, get_snapshot_paths
from pita.trainers import QSharpTrainer
from datasets import load_from_disk
from pita.models.value_classifier import ValueClassifier
from pita.core.prompts import build_instruction_prompt
from pita.datasets.convert import convert_qsharp_rows_to_classifier_dataset


@register_algorithm("QSharp")
class QSharpAlgorithm(ValueGuidedAlgorithms):
    ALGO_KEY = "QSharp"

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
        logger.info("🚀 QSharp start: dataset={} family={}", dataset, family)

        run_root = get_run_root()
        _, hf_dir, _ = get_snapshot_paths(self.algo_key, dataset, family, round_idx)
        if not hf_dir.exists():
            raise FileNotFoundError(f"Missing QSharp dataset: {hf_dir}")
        ds = load_from_disk(str(hf_dir))

        ref = self._build_model(cfg, ref_model)
        classifier = ValueClassifier(
            cls_model,
            tokenizer=ref.tokenizer,
            device=ref.model.device,
        )
        self.maybe_load_classifier_from_prev_round(
            classifier,
            run_root=run_root,
            dataset=dataset,
            family=family,
            round_idx=round_idx,
            device=ref.model.device,
        )

        trainer = QSharpTrainer(
            classifier=classifier,
            tokenizer=ref.tokenizer,
            batch_size=int(self.cfg.batch_size),
            num_workers=int(self.cfg.num_workers),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
            grad_clip=float(self.cfg.grad_clip),
            pad_token_id=ref.pad_token_id,
        )
        # Convert dataset rows to classifier training examples if needed
        # Expecting keys: input_ids, target_ids, rewards, loss_weights
        sample = ds[0] if len(ds) > 0 else None
        need_convert = sample is not None and not all(
            k in sample for k in ("input_ids", "target_ids", "rewards", "loss_weights")
        )
        if need_convert:
            ds = convert_qsharp_rows_to_classifier_dataset(
                ds,
                tokenizer=ref.tokenizer,
                use_chat_template=ref.gen_cfg.use_chat_template,
            )

        loader = trainer.create_loader(ds)
        stats = trainer.train(loader, num_epochs=int(self.cfg.epochs))

        ckpt_dir = self.get_ckpt_dir(run_root, dataset, family, round_idx)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(classifier.state_dict(), str(ckpt_dir / "classifier.pt"))

        # Reload classifier for eval to mirror DPO pattern
        reloaded = ValueClassifier(
            cls_model,
            tokenizer=ref.tokenizer,
            device=ref.model.device,
        )
        state = torch.load(
            str(ckpt_dir / "classifier.pt"), map_location=ref.model.device
        )
        reloaded.load_state_dict(state)

        guided = self.build_guided_with(cfg, ref=ref, classifier=reloaded)
        eval_metrics = evaluate_pass1_maj8(cfg, guided, dataset, save_dir=output_dir)

        result = {
            "algo": "QSharp",
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
            "✅ QSharp done: steps={} loss={:.4f} pass@1={:.3f} maj@8={:.3f}",
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
