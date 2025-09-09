from __future__ import annotations

from typing import Any, Dict

from pita.core.registry import AlgorithmBase, register_algorithm
from pita.models.hf import HFModel, GenerationConfig
from pita.datasets.registry import get_dataset


@register_algorithm("PITA")
class PITAAlgorithm(AlgorithmBase):
    def run(
        self, cfg, ref_model: str, cls_model: str, dataset: str, output_dir
    ) -> Dict[str, Any]:
        # Resolve dataset via registry and pass args from cfg.datasets[dataset]
        dataset_cls = get_dataset(dataset)
        ds_args = dict(cfg.datasets.get(dataset, {}))
        ds = dataset_cls(**ds_args)

        # Initialize reference model (guidance via classifier can be added later)
        gen_cfg = GenerationConfig(
            max_new_tokens=int(cfg.common.max_new_tokens),
            temperature=float(cfg.common.temperature),
            top_p=float(cfg.common.top_p),
            use_chat_template=bool(cfg.common.use_chat_template),
            dtype=str(cfg.common.dtype),
        )
        model = HFModel(ref_model, gen_cfg)

        total = 0
        correct = 0
        preds = []
        # AIME-style hydration and numeric extraction for now
        from pita.datasets.aime2025 import (
            AIME2025,
        )  # local import to avoid hard dependency at module import

        for sample in ds.iter():
            prompt = AIME2025.hydrate_prompt(sample.question)
            completion = model.generate_text(model.build_prompt(prompt))
            pred = AIME2025.extract_numeric(completion)
            gold = AIME2025.extract_numeric(sample.answer)
            is_correct = int(pred == gold and pred != "")
            correct += is_correct
            total += 1
            preds.append({"pred": pred, "gold": gold, "ok": bool(is_correct)})

        acc = (correct / total) if total else 0.0
        return {
            "algo": "PITA",
            "dataset": dataset,
            "ref_model": ref_model,
            "cls_model": cls_model,
            "num_samples": total,
            "metrics": {"accuracy": acc},
            "predictions": preds[:50],
        }
