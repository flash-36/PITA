from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import random

from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from datasets import Dataset

from pita.models.hf import HFModel, GenerationConfig
from pita.models.registry import resolve_family_pair
from pita.datasets.registry import get_dataset


def collect_datasets(cfg: DictConfig) -> None:
    if not cfg.get("collection", {}).get("enabled", True):
        return

    orig_root = Path(get_original_cwd())
    ds_root = orig_root / "outputs" / "datasets"
    ds_root.mkdir(parents=True, exist_ok=True)

    gen_cfg = GenerationConfig(
        max_new_tokens=int(cfg.common.max_new_tokens),
        temperature=float(cfg.common.temperature),
        top_p=float(cfg.common.top_p),
        use_chat_template=bool(cfg.common.use_chat_template),
        dtype=str(cfg.common.dtype),
    )
    random.seed(int(cfg.collection.seed))

    datasets = list(cfg.get("datasets", {}).keys())
    fam_list = cfg.model_pairs if isinstance(cfg.model_pairs, list) else []

    for ds_name in datasets:
        for ref_family in fam_list:
            family_cap = str(ref_family).capitalize()
            parquet_path = ds_root / f"{ds_name}_{family_cap}.parquet"
            if parquet_path.exists():
                continue

            ref_alias, _ = resolve_family_pair(str(ref_family))
            model = HFModel(ref_alias, gen_cfg)

            ds_cfg = cfg.datasets[ds_name]
            ds_cls = get_dataset(ds_name)
            dataset = ds_cls(
                hf_config=str(ds_cfg.hf_config),
                split=str(ds_cfg.split),
                question_key=str(ds_cfg.question_key),
                answer_key=str(ds_cfg.answer_key),
            )

            N = int(cfg.collection.contexts_per_example)
            T = int(cfg.collection.rollout_max_steps)
            M = int(cfg.collection.responses_per_context)

            records: List[Dict[str, Any]] = []
            for ex in dataset.iter():
                prompt = dataset.hydrate_prompt(ex.question)
                for _ in range(N):
                    t = random.randint(1, T)
                    ctx = model.roll_in(prompt, max_roll_tokens=t)
                    s_t = ctx["context_text"]
                    y_ref = model.continue_from_context(
                        s_t, max_new_tokens=gen_cfg.max_new_tokens, greedy=True
                    )
                    for _m in range(M):
                        y_i = model.continue_from_context(
                            s_t, max_new_tokens=gen_cfg.max_new_tokens, greedy=False
                        )
                        records.append(
                            {
                                "dataset": ds_name,
                                "model_family": str(ref_family),
                                "model_alias": ref_alias,
                                "question": ex.question,
                                "answer": ex.answer,
                                "prompt": prompt,
                                "t": t,
                                "context": s_t,
                                "y_sample": y_i,
                                "y_ref": y_ref,
                                "pref_sample_gt_ref": None,
                            }
                        )

            hf_ds = Dataset.from_list(records)
            hf_dir = ds_root / f"{ds_name}_{family_cap}.hf"
            hf_ds.save_to_disk(str(hf_dir))
            hf_ds.to_parquet(str(parquet_path))

