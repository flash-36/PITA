from __future__ import annotations

from typing import Any, Dict, List, Optional
from collections import Counter
from pathlib import Path

from omegaconf import DictConfig
from datasets import Dataset
from tqdm import tqdm
from itertools import islice
import torch
import torch.nn.functional as F
from loguru import logger

from pita.models.hf import HFModel
from pita.models.guided import GuidedHFModel
from pita.models import RewardScorer
from pita.core.io import ensure_dir
from pita.core.prompts import build_instruction_prompt
from pita.datasets.utils import extract_final_answer
from pita.datasets import build_test_dataset


@torch.inference_mode()
def _kl_divergence(
    policy_logits: torch.Tensor, ref_logits: torch.Tensor
) -> torch.Tensor:
    """Compute token-wise KL(policy || ref) along last dimension.

    Shapes:
      policy_logits: [*, V]
      ref_logits:    [*, V]
    Returns:
      kl:            [*]
    """
    p_logprob = F.log_softmax(policy_logits, dim=-1)
    q_logprob = F.log_softmax(ref_logits, dim=-1)
    p_prob = p_logprob.exp()
    return torch.sum(p_prob * (p_logprob - q_logprob), dim=-1)


@torch.inference_mode()
def _compute_traj_kl_for_text(
    policy: Any,
    ref: Optional[HFModel],
    *,
    prompt_text: str,
    continuation_text: str,
) -> float:
    """Compute trajectory KL(policy || ref) over the continuation tokens for a single example.

    If policy is a GuidedHFModel, the policy distribution is obtained by applying its
    logits processor step-wise to the reference model logits. Otherwise, the policy
    logits are taken from the policy HFModel forward pass.
    """
    # Resolve reference model
    if isinstance(policy, GuidedHFModel):
        ref_model: HFModel = policy.ref
    else:
        if ref is None:
            return 0.0
        ref_model = ref

    tok = ref_model.tokenizer
    device = ref_model.model.device

    ids_prompt = tok(prompt_text, return_tensors="pt").to(device)
    ids_cont = tok(continuation_text, return_tensors="pt", add_special_tokens=False).to(
        device
    )
    concat_input_ids = torch.cat(
        [ids_prompt["input_ids"], ids_cont["input_ids"]], dim=1
    )
    attention_mask = concat_input_ids.ne(
        getattr(tok, "pad_token_id", tok.eos_token_id)
    ).long()

    # Forward through reference once to get base logits at each position
    ref_out = ref_model.model(
        input_ids=concat_input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    # Slice logits corresponding to predicting continuation tokens
    prompt_len = ids_prompt["input_ids"].shape[1]
    ref_logits_steps = ref_out.logits[:, prompt_len - 1 : -1, :]

    if ref_logits_steps.numel() == 0:
        return 0.0

    # Build policy logits across steps
    if isinstance(policy, GuidedHFModel):
        proc = policy._build_processor()
        policy._reset_state(proc)
        kl_sum = 0.0
        steps = ref_logits_steps.shape[1]
        for t in range(steps):
            # Prefix up to current step (position indexing aligns with logits index)
            prefix_len = prompt_len + t
            prefix_ids = concat_input_ids[:, :prefix_len]
            base_scores = ref_logits_steps[:, t, :]
            guided_scores = proc(prefix_ids, base_scores)
            kl_t = _kl_divergence(guided_scores, base_scores).mean()
            kl_sum += float(kl_t.item())
        return kl_sum
    else:
        # Policy is a plain HFModel
        policy_model: HFModel = policy
        pol_out = policy_model.model(
            input_ids=concat_input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        pol_logits_steps = pol_out.logits[:, prompt_len - 1 : -1, :]
        # Token-wise KL then sum over continuation
        kl_steps = _kl_divergence(pol_logits_steps, ref_logits_steps)
        return float(kl_steps.sum().item())


def evaluate_pass1_maj8(
    cfg: DictConfig,
    model: HFModel,
    dataset: str,
    ref_model: Optional[HFModel] = None,
    save_dir: Optional[Path] = None,
) -> Dict[str, float]:
    ds = build_test_dataset(cfg, dataset)
    if ds is None:
        return {}

    max_examples = int(cfg.data_collection.max_examples or 0)
    limit = max_examples if max_examples > 0 else len(ds)

    total = 0
    pass1 = 0
    maj8 = 0
    sum_kl = 0.0
    has_verifier = hasattr(ds, "is_correct")
    rows: List[Dict[str, str]] = []
    for ex in tqdm(islice(ds.iter(), limit), total=limit, desc=f"Eval:{dataset}"):
        prompt = ds.hydrate_prompt(ex.question)
        built = build_instruction_prompt(
            prompt,
            tokenizer=model.tokenizer,
            use_chat_template=model.gen_cfg.use_chat_template,
        )

        num_samples = int(cfg.evaluation.num_samples)
        preds: List[str] = model.generate_n(built, num_samples)

        if has_verifier and ds.is_correct(ex.answer, preds[0]):
            pass1 += 1

        if has_verifier:
            norm = [extract_final_answer(p) for p in preds]
            counts = Counter(norm)
            if counts:
                top_str, top_cnt = max(counts.items(), key=lambda kv: kv[1])
                if list(counts.values()).count(top_cnt) == 1:
                    rep_idx = norm.index(top_str)
                    if ds.is_correct(ex.answer, preds[rep_idx]):
                        maj8 += 1

        total += 1

        # KL over the first sample continuation
        try:
            sum_kl += _compute_traj_kl_for_text(
                model,
                ref_model,
                prompt_text=built,
                continuation_text=preds[0],
            )
        except Exception:
            sum_kl += 0.0

        rows.append(
            {
                "question": ex.question,
                "answer": ex.answer,
                **{f"pred_{i+1}": preds[i] for i in range(len(preds))},
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
        "avg_kl": float(sum_kl / max(1, total)),
        "num_examples": int(total),
    }


def evaluate_avg_reward(
    cfg: DictConfig,
    model: HFModel,
    dataset: str,
    ref_model: Optional[HFModel] = None,
    save_dir: Optional[Path] = None,
) -> Dict[str, float]:
    ds = build_test_dataset(cfg, dataset)
    if ds is None:
        return {}

    # Reward model setup
    ds_cfg = cfg.datasets[dataset]
    rm_model = str(ds_cfg.reward_model)
    scorer = RewardScorer(
        rm_model,
        bt_sampling=bool(cfg.data_collection.bradley_terry_sampling),
        bt_beta=float(cfg.data_collection.bradley_terry_beta),
        device="auto",
        dtype=str(cfg.system.dtype),
        batch_size=int(cfg.data_collection.reward_batch_size),
    )

    max_examples = int(cfg.data_collection.max_examples or 0)
    limit = max_examples if max_examples > 0 else len(ds)
    batch_size = int(cfg.evaluation.batch_size)

    # Collect all examples first
    examples = list(islice(ds.iter(), limit))
    logger.info(f"🔄 Evaluating {len(examples)} examples with batch_size={batch_size}")

    # PHASE 1: Build all prompts
    prompts_data = []
    for ex in examples:
        prompt = ds.hydrate_prompt(ex.question)
        built = build_instruction_prompt(
            prompt,
            tokenizer=model.tokenizer,
            use_chat_template=model.gen_cfg.use_chat_template,
        )
        prompts_data.append({"ex": ex, "built": built})

    # PHASE 2: Batch generate all predictions
    logger.info("🔄 Batching generation...")
    all_prompts = [item["built"] for item in prompts_data]
    preds = model.continue_from_context_batch(
        all_prompts, model.gen_cfg.max_new_tokens, greedy=False, batch_size=batch_size
    )

    # PHASE 3: Batch score all predictions
    logger.info("✅ Batching reward scoring...")
    score_pairs = [
        (item["ex"].question, pred) for item, pred in zip(prompts_data, preds)
    ]
    rewards = scorer.score_batch_single(score_pairs)

    # PHASE 4: Compute KL and assemble results
    total = 0
    sum_reward = 0.0
    sum_kl = 0.0
    rows: List[Dict[str, str]] = []

    for item, pred, r in tqdm(
        zip(prompts_data, preds, rewards), total=len(examples), desc=f"EvalR:{dataset}"
    ):
        sum_reward += r
        total += 1

        # KL over the produced continuation
        try:
            sum_kl += _compute_traj_kl_for_text(
                model,
                ref_model,
                prompt_text=item["built"],
                continuation_text=pred,
            )
        except Exception:
            sum_kl += 0.0

        rows.append(
            {
                "question": item["ex"].question,
                "answer": item["ex"].answer,
                "pred": pred,
                "reward": r,
            }
        )

    if save_dir is not None:
        ensure_dir(Path(save_dir))
        ds_out = Dataset.from_list(rows)
        ds_out.save_to_disk(str(Path(save_dir) / "eval_predictions.hf"))
        ds_out.to_csv(str(Path(save_dir) / "eval_predictions.csv"))

    avg_reward = float(sum_reward / max(1, total))
    return {
        "avg_reward": avg_reward,
        "avg_kl": float(sum_kl / max(1, total)),
        "num_examples": int(total),
    }
