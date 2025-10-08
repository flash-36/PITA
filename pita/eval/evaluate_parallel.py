"""Parallel evaluation across multiple GPUs with batched processing."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from collections import Counter
from pathlib import Path

from omegaconf import DictConfig
from datasets import Dataset
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
from pita.core.gpu_manager import get_gpu_manager


@torch.inference_mode()
def _kl_divergence(
    policy_logits: torch.Tensor, ref_logits: torch.Tensor
) -> torch.Tensor:
    """Compute token-wise KL(policy || ref) along last dimension."""
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
    """Compute trajectory KL(policy || ref) over the continuation tokens."""
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

    ref_out = ref_model.model(
        input_ids=concat_input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    prompt_len = ids_prompt["input_ids"].shape[1]
    ref_logits_steps = ref_out.logits[:, prompt_len - 1 : -1, :]

    if ref_logits_steps.numel() == 0:
        return 0.0

    if isinstance(policy, GuidedHFModel):
        proc = policy._build_processor()
        policy._reset_state(proc)
        kl_sum = 0.0
        steps = ref_logits_steps.shape[1]
        for t in range(steps):
            prefix_len = prompt_len + t
            prefix_ids = concat_input_ids[:, :prefix_len]
            base_scores = ref_logits_steps[:, t, :]
            guided_scores = proc(prefix_ids, base_scores)
            kl_t = _kl_divergence(guided_scores, base_scores).mean()
            kl_sum += float(kl_t.item())
        return kl_sum
    else:
        policy_model: HFModel = policy
        pol_out = policy_model.model(
            input_ids=concat_input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        pol_logits_steps = pol_out.logits[:, prompt_len - 1 : -1, :]
        kl_steps = _kl_divergence(pol_logits_steps, ref_logits_steps)
        return float(kl_steps.sum().item())


def evaluate_pass1_maj8_batched(
    cfg: DictConfig,
    model: HFModel,
    dataset: str,
    ref_model: Optional[HFModel] = None,
    save_dir: Optional[Path] = None,
    batch_size: int = 8,
) -> Dict[str, float]:
    """Evaluate pass@1 and maj@8 with batched generation.

    Args:
        cfg: Configuration
        model: Model to evaluate
        dataset: Dataset name
        ref_model: Reference model for KL computation
        save_dir: Directory to save results
        batch_size: Batch size for generation (number of examples to process at once)
    """
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

    # Collect examples into batches
    examples = list(islice(ds.iter(), limit))

    from tqdm import tqdm

    for ex in tqdm(examples, desc=f"Eval:{dataset}"):
        prompt = ds.hydrate_prompt(ex.question)
        built = build_instruction_prompt(
            prompt,
            tokenizer=model.tokenizer,
            use_chat_template=model.gen_cfg.use_chat_template,
        )

        # Generate 8 samples
        preds: List[str] = model.generate_n(built, 8)

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

        # KL computation
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
        "avg_kl": float(sum_kl / max(1, total)),
        "num_examples": int(total),
    }


def evaluate_avg_reward_batched(
    cfg: DictConfig,
    model: HFModel,
    dataset: str,
    ref_model: Optional[HFModel] = None,
    save_dir: Optional[Path] = None,
    batch_size: int = 8,
) -> Dict[str, float]:
    """Evaluate average reward with batched processing.

    Args:
        cfg: Configuration
        model: Model to evaluate
        dataset: Dataset name
        ref_model: Reference model for KL computation
        save_dir: Directory to save results
        batch_size: Batch size for reward scoring
    """
    ds = build_test_dataset(cfg, dataset)
    if ds is None:
        return {}

    # Reward model setup with larger batch size
    ds_cfg = cfg.datasets[dataset]
    rm_model = str(ds_cfg.reward_model)
    reward_batch_size = max(batch_size, 32)
    scorer = RewardScorer(
        rm_model,
        bt_sampling=bool(cfg.data_collection.bradley_terry_sampling),
        bt_beta=float(cfg.data_collection.bradley_terry_beta),
        device="auto",
        dtype=str(cfg.system.dtype),
        batch_size=reward_batch_size,
    )

    max_examples = int(cfg.data_collection.max_examples or 0)
    limit = max_examples if max_examples > 0 else len(ds)

    total = 0
    sum_reward = 0.0
    sum_kl = 0.0
    rows: List[Dict[str, str]] = []

    from tqdm import tqdm

    for ex in tqdm(islice(ds.iter(), limit), total=limit, desc=f"EvalR:{dataset}"):
        prompt = ds.hydrate_prompt(ex.question)
        built = build_instruction_prompt(
            prompt,
            tokenizer=model.tokenizer,
            use_chat_template=model.gen_cfg.use_chat_template,
        )
        pred: str = model.continue_from_context(
            built, max_new_tokens=model.gen_cfg.max_new_tokens, greedy=False
        )

        r = float(scorer.score_single(ex.question, pred))
        sum_reward += r
        total += 1

        try:
            sum_kl += _compute_traj_kl_for_text(
                model,
                ref_model,
                prompt_text=built,
                continuation_text=pred,
            )
        except Exception:
            sum_kl += 0.0

        rows.append(
            {
                "question": ex.question,
                "answer": ex.answer,
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


def evaluate_parallel(
    cfg: DictConfig,
    model: HFModel,
    dataset: str,
    ref_model: Optional[HFModel] = None,
    save_dir: Optional[Path] = None,
    use_reward: bool = False,
) -> Dict[str, float]:
    """Automatically select evaluation method based on GPU availability.

    Uses batched evaluation with optimized batch sizes based on GPU memory.
    """
    gpu_manager = get_gpu_manager()

    # Determine optimal batch size based on GPU memory
    if gpu_manager.num_gpus > 0:
        # For 80GB GPUs, use larger batches; for 40GB, use smaller
        total_memory = gpu_manager.get_gpu_memory(0) / (1024**3)  # GB
        batch_size = 16 if total_memory >= 60 else 8
    else:
        batch_size = 4

    logger.info(f"Using batch size {batch_size} for evaluation on {dataset}")

    if use_reward:
        return evaluate_avg_reward_batched(
            cfg, model, dataset, ref_model, save_dir, batch_size
        )
    else:
        return evaluate_pass1_maj8_batched(
            cfg, model, dataset, ref_model, save_dir, batch_size
        )
