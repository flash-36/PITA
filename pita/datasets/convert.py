from __future__ import annotations

from typing import Any, Callable, Dict, List

from datasets import Dataset
from loguru import logger
from tqdm import tqdm

from pita.core.prompts import build_instruction_prompt


def _build_reward_normalizer(ds: Any) -> Callable[[float], float]:
    """Build a min-max normalizer from a dataset's score_a/score_b columns.

    Returns the identity function when rewards are already binary {0, 1}.
    """
    all_scores = [float(ex.get("score_a", 0.0)) for ex in ds] + [
        float(ex.get("score_b", 0.0)) for ex in ds
    ]
    s_min, s_max = min(all_scores), max(all_scores)
    s_mean = sum(all_scores) / len(all_scores)
    is_binary = all(s in (0.0, 1.0) for s in all_scores)

    logger.info(
        f"Reward stats: min={s_min:.4f} max={s_max:.4f} mean={s_mean:.4f} "
        f"n={len(all_scores)} binary={is_binary}"
    )

    if is_binary or s_max <= s_min:
        return lambda s: s
    return lambda s: (s - s_min) / (s_max - s_min)


def convert_qsharp_rows_to_classifier_dataset(
    ds: Any,
    *,
    tokenizer: Any,
    use_chat_template: bool,
) -> Dataset:
    normalize = _build_reward_normalizer(ds)

    rows: List[Dict[str, Any]] = []
    for ex in tqdm(ds, desc="Q#:convert_ds"):
        prompt = ex.get("prompt", "")
        context = ex.get("context") or prompt
        built = build_instruction_prompt(
            prompt, tokenizer=tokenizer, use_chat_template=use_chat_template
        )
        sol_prefix = context[len(prompt) :] if len(context) >= len(prompt) else ""
        context_built = built + sol_prefix

        ti = tokenizer(context_built, add_special_tokens=False)["input_ids"]

        y_a = ex.get("y_a", "")
        y_b = ex.get("y_b", "")
        prefix_ids = tokenizer(sol_prefix, add_special_tokens=False)["input_ids"]

        def to_target_ids(text: str) -> List[int]:
            full = tokenizer(text, add_special_tokens=False)["input_ids"]
            return full[len(prefix_ids) :]

        to_a = to_target_ids(y_a)
        to_b = to_target_ids(y_b)
        if len(to_a) == 0 or len(to_b) == 0:
            continue

        score_a = normalize(float(ex.get("score_a", 0.0)))
        score_b = normalize(float(ex.get("score_b", 0.0)))

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
    return Dataset.from_list(rows)


def convert_pita_rows_to_classifier_dataset(
    ds: Any,
    *,
    tokenizer: Any,
    use_chat_template: bool,
) -> Dataset:
    rows: List[Dict[str, Any]] = []
    for ex in tqdm(ds, desc="PITA:convert_ds"):
        prompt = ex.get("prompt", "")
        context = ex.get("context") or prompt
        built = build_instruction_prompt(
            prompt, tokenizer=tokenizer, use_chat_template=use_chat_template
        )
        sol_prefix = context[len(prompt) :] if len(context) >= len(prompt) else ""
        context_built = built + sol_prefix

        ti = tokenizer(context_built, add_special_tokens=False)["input_ids"]

        y_a = ex.get("y_a", "")
        y_b = ex.get("y_b", "")
        prefix_ids = tokenizer(sol_prefix, add_special_tokens=False)["input_ids"]

        def to_target_ids(text: str) -> List[int]:
            full = tokenizer(text, add_special_tokens=False)["input_ids"]
            return full[len(prefix_ids) :]

        to_a = to_target_ids(y_a)
        to_b = to_target_ids(y_b)
        if len(to_a) == 0 or len(to_b) == 0:
            continue

        preferred = ex.get("preferred", None)
        if preferred == 0:
            chosen_ids, rejected_ids = to_a, to_b
        elif preferred == 1:
            chosen_ids, rejected_ids = to_b, to_a
        else:
            score_a = float(ex.get("score_a", 0.0))
            score_b = float(ex.get("score_b", 0.0))
            chosen_ids, rejected_ids = (
                (to_a, to_b) if score_a >= score_b else (to_b, to_a)
            )

        rows.append(
            {
                "input_ids": ti,
                "chosen_target_ids": chosen_ids,
                "rejected_target_ids": rejected_ids,
            }
        )
    return Dataset.from_list(rows)
