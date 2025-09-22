from __future__ import annotations

from typing import Dict, List, Optional
from collections import Counter
from pathlib import Path

from omegaconf import DictConfig
from datasets import Dataset
from tqdm import tqdm
from itertools import islice

from pita.models.hf import HFModel
from pita.core.io import ensure_dir
from pita.core.prompts import build_instruction_prompt
from pita.datasets.utils import extract_final_answer
from pita.datasets import build_test_dataset


def evaluate_pass1_maj8(
    cfg: DictConfig,
    model: HFModel,
    dataset: str,
    save_dir: Optional[Path] = None,
) -> Dict[str, float]:
    ds = build_test_dataset(cfg, dataset)
    if ds is None:
        return {}

    max_examples = int(cfg.collection.max_examples or 0)
    limit = max_examples if max_examples > 0 else len(ds)

    total = 0
    pass1 = 0
    maj8 = 0
    rows: List[Dict[str, str]] = []
    for ex in tqdm(islice(ds.iter(), limit), total=limit, desc=f"Eval:{dataset}"):
        prompt = ds.hydrate_prompt(ex.question)
        built = build_instruction_prompt(
            prompt,
            tokenizer=model.tokenizer,
            use_chat_template=model.gen_cfg.use_chat_template,
        )
        preds: List[str] = model.generate_n(built, 8)

        if ds.is_correct(ex.answer, preds[0]):
            pass1 += 1

        norm = [extract_final_answer(p) for p in preds]
        counts = Counter(norm)
        if counts:
            top_str, top_cnt = max(counts.items(), key=lambda kv: kv[1])
            if list(counts.values()).count(top_cnt) == 1:
                rep_idx = norm.index(top_str)
                if ds.is_correct(ex.answer, preds[rep_idx]):
                    maj8 += 1

        total += 1

        rows.append(
            {
                "question": ex.question,
                "answer": ex.answer,
                **{f"pred_{i+1}": preds[i] for i in range(8)},
            }
        )

    if save_dir is not None:
        ensure_dir(Path(save_dir))
        ds_out = Dataset.from_list(rows)
        ds_out.save_to_disk(str(Path(save_dir) / "eval_predictions.hf"))
        ds_out.to_csv(str(Path(save_dir) / "eval_predictions.csv"))

    return {
        "pass@1": float(pass1 / max(1, total)),
        "maj@8": float(maj8 / max(1, total)),
        "num_examples": int(total),
    }
