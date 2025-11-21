from __future__ import annotations

from typing import Any, Dict, List, Optional
from collections import Counter
from pathlib import Path
import time

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


@torch.inference_mode()
def _compute_traj_kl_batched(
    policy: Any,
    ref: Optional[HFModel],
    *,
    prompt_texts: List[str],
    continuation_texts: List[str],
    batch_size: int = 8,
) -> List[float]:
    """Compute trajectory KL for multiple (prompt, continuation) pairs in batches."""
    if isinstance(policy, GuidedHFModel):
        ref_model: HFModel = policy.ref
        is_guided = True
    else:
        if ref is None:
            return [0.0] * len(prompt_texts)
        ref_model = ref
        is_guided = False
        policy_model: HFModel = policy

    tok = ref_model.tokenizer
    device = ref_model.model.device
    kl_results = []

    # Check if guided model to decide on progress bar strategy
    from pita.models.guided import GuidedHFModel
    is_guided_model = isinstance(policy, GuidedHFModel)
    
    outer_pbar = tqdm(
        range(0, len(prompt_texts), batch_size), 
        desc="Computing KL", 
        leave=False,
        disable=is_guided_model  # Disable outer bar for guided models (we'll use inner token bar)
    )
    
    for i in outer_pbar:
        batch_prompts = prompt_texts[i : i + batch_size]
        batch_conts = continuation_texts[i : i + batch_size]

        batch_prompt_lens = []
        batch_input_ids = []
        batch_attention_masks = []

        for prompt_text, cont_text in zip(batch_prompts, batch_conts):
            ids_prompt = tok(prompt_text, return_tensors="pt")
            ids_cont = tok(cont_text, return_tensors="pt", add_special_tokens=False)
            concat_input_ids = torch.cat(
                [ids_prompt["input_ids"], ids_cont["input_ids"]], dim=1
            )
            attention_mask = concat_input_ids.ne(
                getattr(tok, "pad_token_id", tok.eos_token_id)
            ).long()

            batch_prompt_lens.append(ids_prompt["input_ids"].shape[1])
            batch_input_ids.append(concat_input_ids.squeeze(0))
            batch_attention_masks.append(attention_mask.squeeze(0))

        max_len = max(x.shape[0] for x in batch_input_ids)
        pad_id = getattr(tok, "pad_token_id", tok.eos_token_id)

        padded_input_ids = []
        padded_attention_masks = []
        for input_ids, attention_mask in zip(batch_input_ids, batch_attention_masks):
            pad_len = max_len - input_ids.shape[0]
            padded_input_ids.append(F.pad(input_ids, (0, pad_len), value=pad_id))
            padded_attention_masks.append(F.pad(attention_mask, (0, pad_len), value=0))

        batched_input_ids = torch.stack(padded_input_ids).to(device)
        batched_attention_masks = torch.stack(padded_attention_masks).to(device)

        ref_out = ref_model.model(
            input_ids=batched_input_ids,
            attention_mask=batched_attention_masks,
            use_cache=False,
        )

        if is_guided:
            # Calculate total steps for progress tracking
            total_steps = sum(
                ref_out.logits[idx, prompt_len - 1 : -1, :].shape[0]
                for idx, prompt_len in enumerate(batch_prompt_lens)
                if ref_out.logits[idx, prompt_len - 1 : -1, :].numel() > 0
            )
            
            # Single progress bar for all examples in this batch
            use_pbar = total_steps > 200
            pbar = tqdm(
                total=total_steps, 
                desc=f"  KL tokens (batch {(i//batch_size)+1})", 
                leave=False, 
                unit="tok",
                position=0  # Force to top position
            ) if use_pbar else None
            
            for idx, prompt_len in enumerate(batch_prompt_lens):
                ref_logits_steps = ref_out.logits[idx : idx + 1, prompt_len - 1 : -1, :]
                if ref_logits_steps.numel() == 0:
                    kl_results.append(0.0)
                    continue

                proc = policy._build_processor()
                policy._reset_state(proc)
                kl_sum = 0.0
                steps = ref_logits_steps.shape[1]
                concat_input_ids = batched_input_ids[idx : idx + 1]
                
                for t in range(steps):
                    if pbar:
                        pbar.update(1)
                    prefix_len = prompt_len + t
                    prefix_ids = concat_input_ids[:, :prefix_len]
                    base_scores = ref_logits_steps[:, t, :]
                    guided_scores = proc(prefix_ids, base_scores)
                    kl_t = _kl_divergence(guided_scores, base_scores).mean()
                    kl_sum += float(kl_t.item())
                kl_results.append(kl_sum)
            
            if pbar:
                pbar.close()
        else:
            pol_out = policy_model.model(
                input_ids=batched_input_ids,
                attention_mask=batched_attention_masks,
                use_cache=False,
            )

            for idx, prompt_len in enumerate(batch_prompt_lens):
                ref_logits_steps = ref_out.logits[idx : idx + 1, prompt_len - 1 : -1, :]
                pol_logits_steps = pol_out.logits[idx : idx + 1, prompt_len - 1 : -1, :]

                if ref_logits_steps.numel() == 0:
                    kl_results.append(0.0)
                else:
                    kl_steps = _kl_divergence(pol_logits_steps, ref_logits_steps)
                    kl_results.append(float(kl_steps.sum().item()))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return kl_results


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
    num_samples = int(cfg.evaluation.num_samples)
    batch_size = int(cfg.evaluation.batch_size)
    has_verifier = hasattr(ds, "is_correct")

    examples = list(islice(ds.iter(), limit))
    logger.info(
        f"🔄 Evaluating {len(examples)} examples with batch_size={batch_size}, {num_samples} samples each"
    )

    prompts_data = []
    for ex in examples:
        prompt = ds.hydrate_prompt(ex.question)
        built = build_instruction_prompt(
            prompt,
            tokenizer=model.tokenizer,
            use_chat_template=model.gen_cfg.use_chat_template,
        )
        prompts_data.append({"ex": ex, "built": built})

    all_prompts = []
    prompt_to_example = []
    for idx, item in enumerate(prompts_data):
        for _ in range(num_samples):
            all_prompts.append(item["built"])
            prompt_to_example.append(idx)

    logger.info(f"\n🔄 Generating {len(all_prompts)} predictions...")
    gen_start = time.time()
    all_preds = model.continue_from_context_batch(
        all_prompts, model.gen_cfg.max_new_tokens, greedy=False, batch_size=batch_size
    )
    gen_elapsed = time.time() - gen_start
    logger.info(f"✅ Generation complete ({gen_elapsed:.1f}s)")

    grouped_preds = [[] for _ in range(len(examples))]
    for ex_idx, pred in zip(prompt_to_example, all_preds):
        grouped_preds[ex_idx].append(pred)

    logger.info(f"\n📊 Computing metrics and KL divergence...")
    from pita.models.guided import GuidedHFModel
    if isinstance(model, GuidedHFModel):
        logger.info(f"⚠️  KL computation for guided models processes tokens step-by-step - this may take several minutes")
    kl_start = time.time()
    total = 0
    pass1 = 0
    maj8 = 0
    rows: List[Dict[str, str]] = []

    pass1_prompts = []
    pass1_conts = []

    for item, preds in zip(prompts_data, grouped_preds):
        if has_verifier and ds.is_correct(item["ex"].answer, preds[0]):
            pass1 += 1

        if has_verifier:
            norm = [extract_final_answer(p) for p in preds]
            counts = Counter(norm)
            if counts:
                top_str, top_cnt = max(counts.items(), key=lambda kv: kv[1])
                if list(counts.values()).count(top_cnt) == 1:
                    rep_idx = norm.index(top_str)
                    if ds.is_correct(item["ex"].answer, preds[rep_idx]):
                        maj8 += 1

        total += 1
        pass1_prompts.append(item["built"])
        pass1_conts.append(preds[0])

        rows.append(
            {
                "question": item["ex"].question,
                "answer": item["ex"].answer,
                **{f"pred_{i+1}": preds[i] for i in range(len(preds))},
            }
        )

    kl_values = _compute_traj_kl_batched(
        model,
        ref_model,
        prompt_texts=pass1_prompts,
        continuation_texts=pass1_conts,
        batch_size=batch_size,
    )
    sum_kl = sum(kl_values)
    kl_elapsed = time.time() - kl_start
    logger.info(f"✅ KL computation complete ({kl_elapsed:.1f}s)")

    if save_dir is not None:
        ensure_dir(Path(save_dir))
        ds_out = Dataset.from_list(rows)
        ds_out.save_to_disk(str(Path(save_dir) / "eval_predictions.hf"))
        ds_out.to_csv(str(Path(save_dir) / "eval_predictions.csv"))

    eval_elapsed = time.time() - eval_start
    results = {
        "pass@1": float(pass1 / max(1, total)),
        "maj@8": float(maj8 / max(1, total)),
        "avg_kl": float(sum_kl / max(1, total)),
        "num_examples": int(total),
    }
    
    logger.info(f"\n✅ Evaluation complete ({eval_elapsed:.1f}s)")
    logger.info(f"   pass@1: {results['pass@1']:.3f}")
    logger.info(f"   maj@8: {results['maj@8']:.3f}")
    logger.info(f"   avg_kl: {results['avg_kl']:.3f}")
    
    return results


def evaluate_avg_reward(
    cfg: DictConfig,
    model: HFModel,
    dataset: str,
    ref_model: Optional[HFModel] = None,
    save_dir: Optional[Path] = None,
) -> Dict[str, float]:
    logger.info(f"\n{'▸'*50}")
    logger.info(f"📊 Evaluating: {dataset}")
    logger.info(f"   Metric: average reward")
    logger.info(f"{'▸'*50}")
    eval_start = time.time()
    
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

    # PHASE 4: Compute KL in batches
    logger.info("✅ Computing KL divergence...")
    all_prompts = [item["built"] for item in prompts_data]
    kl_values = _compute_traj_kl_batched(
        model,
        ref_model,
        prompt_texts=all_prompts,
        continuation_texts=preds,
        batch_size=batch_size,
    )

    # PHASE 5: Assemble results
    total = 0
    sum_reward = 0.0
    sum_kl = sum(kl_values)
    rows: List[Dict[str, str]] = []

    for item, pred, r in zip(prompts_data, preds, rewards):
        sum_reward += r
        total += 1

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
