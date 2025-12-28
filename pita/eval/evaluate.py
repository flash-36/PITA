from __future__ import annotations

from typing import Any, Dict, List, Optional
from collections import Counter
from pathlib import Path
import time
import json
import random

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
from pita.core.logging_context import logging_context
from pita.datasets.utils import extract_final_answer
from pita.datasets import build_test_dataset


def _get_eval_checkpoint_path(save_dir: Path) -> Path:
    return save_dir / ".eval_checkpoint.json"


def _load_eval_checkpoint(save_dir: Path) -> Dict[str, Any]:
    ckpt_path = _get_eval_checkpoint_path(save_dir)
    if ckpt_path.exists():
        with open(ckpt_path, "r") as f:
            return json.load(f)
    return {}


def _save_eval_checkpoint(save_dir: Path, data: Dict[str, Any]) -> None:
    ensure_dir(save_dir)
    ckpt_path = _get_eval_checkpoint_path(save_dir)
    temp_path = ckpt_path.with_suffix(".json.tmp")
    with open(temp_path, "w") as f:
        json.dump(data, f)
    temp_path.replace(ckpt_path)


def _get_kl_checkpoint_path(save_dir: Path) -> Path:
    return save_dir / ".kl_checkpoint.json"


def _load_kl_checkpoint(save_dir: Path) -> Dict[str, Any]:
    ckpt_path = _get_kl_checkpoint_path(save_dir)
    if ckpt_path.exists():
        with open(ckpt_path, "r") as f:
            return json.load(f)
    return {}


def _save_kl_checkpoint(save_dir: Path, data: Dict[str, Any]) -> None:
    ensure_dir(save_dir)
    ckpt_path = _get_kl_checkpoint_path(save_dir)
    temp_path = ckpt_path.with_suffix(".json.tmp")
    with open(temp_path, "w") as f:
        json.dump(data, f)
    temp_path.replace(ckpt_path)


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
    sample_ratio: float = 1.0,
    save_dir: Optional[Path] = None,
) -> List[float]:
    """Compute trajectory KL for multiple (prompt, continuation) pairs in batches.
    
    Args:
        sample_ratio: Fraction of examples to compute KL for (0.1 = 10%). Results are
            scaled up to estimate full KL.
        save_dir: Directory to save/load KL checkpoints for resumability.
    """
    n_total = len(prompt_texts)
    
    # Sample subset if ratio < 1.0
    if sample_ratio < 1.0 and n_total > 0:
        n_sample = max(1, int(n_total * sample_ratio))
        random.seed(42)  # Deterministic sampling
        sample_indices = sorted(random.sample(range(n_total), n_sample))
        prompt_texts = [prompt_texts[i] for i in sample_indices]
        continuation_texts = [continuation_texts[i] for i in sample_indices]
        logger.info(f"📊 KL sampling: {n_sample}/{n_total} examples ({sample_ratio*100:.0f}%)")
    
    if isinstance(policy, GuidedHFModel):
        ref_model: HFModel = policy.ref
        is_guided = True
    else:
        if ref is None:
            return [0.0] * n_total
        ref_model = ref
        is_guided = False
        policy_model: HFModel = policy

    tok = ref_model.tokenizer
    device = ref_model.model.device
    
    # Load checkpoint if available
    kl_results: List[float] = []
    start_batch = 0
    if save_dir is not None:
        ckpt = _load_kl_checkpoint(Path(save_dir))
        if ckpt.get("kl_values"):
            kl_results = ckpt["kl_values"]
            start_batch = ckpt.get("completed_batches", 0)
            if start_batch > 0:
                logger.info(f"📂 Loaded KL checkpoint: {len(kl_results)} values, resuming from batch {start_batch+1}")

    is_guided_model = isinstance(policy, GuidedHFModel)
    total_batches = (len(prompt_texts) + batch_size - 1) // batch_size

    outer_pbar = tqdm(
        range(0, len(prompt_texts), batch_size),
        desc="Computing KL",
        leave=False,
        disable=is_guided_model,
    )

    for batch_idx, i in enumerate(outer_pbar):
        # Skip already computed batches
        if batch_idx < start_batch:
            continue
            
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

        batch_kl_values = []
        if is_guided:
            total_steps = sum(
                ref_out.logits[idx, prompt_len - 1 : -1, :].shape[0]
                for idx, prompt_len in enumerate(batch_prompt_lens)
                if ref_out.logits[idx, prompt_len - 1 : -1, :].numel() > 0
            )

            use_pbar = total_steps > 200
            pbar = (
                tqdm(
                    total=total_steps,
                    desc=f"  KL tokens (batch {batch_idx+1}/{total_batches})",
                    leave=False,
                    unit="tok",
                    position=0,
                )
                if use_pbar
                else None
            )

            for idx, prompt_len in enumerate(batch_prompt_lens):
                ref_logits_steps = ref_out.logits[idx : idx + 1, prompt_len - 1 : -1, :]
                if ref_logits_steps.numel() == 0:
                    batch_kl_values.append(0.0)
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
                batch_kl_values.append(kl_sum)

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
                    batch_kl_values.append(0.0)
                else:
                    kl_steps = _kl_divergence(pol_logits_steps, ref_logits_steps)
                    batch_kl_values.append(float(kl_steps.sum().item()))

        # Add batch results and save checkpoint
        kl_results.extend(batch_kl_values)
        if save_dir is not None:
            _save_kl_checkpoint(Path(save_dir), {
                "kl_values": kl_results,
                "completed_batches": batch_idx + 1,
                "total_batches": total_batches,
                "sample_ratio": sample_ratio,
            })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Clean up checkpoint on completion
    if save_dir is not None:
        ckpt_path = _get_kl_checkpoint_path(Path(save_dir))
        if ckpt_path.exists():
            ckpt_path.unlink()

    # Scale up if we sampled
    if sample_ratio < 1.0:
        avg_kl = sum(kl_results) / max(1, len(kl_results))
        return [avg_kl] * n_total
    
    return kl_results


def evaluate_pass1_maj8(
    cfg: DictConfig,
    model: HFModel,
    dataset: str,
    ref_model: Optional[HFModel] = None,
    save_dir: Optional[Path] = None,
) -> Dict[str, float]:
    with logging_context(stage="EVAL"):
        eval_start = time.time()
        ds = build_test_dataset(cfg, dataset)
        if ds is None:
            return {}

        # Use evaluation.max_examples if set, otherwise fall back to data_collection.max_examples
        eval_max = int(getattr(cfg.evaluation, "max_examples", 0) or 0)
        data_max = int(cfg.data_collection.max_examples or 0)
        max_examples = eval_max if eval_max > 0 else data_max
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

        # Check for checkpoint if save_dir provided
        checkpoint = {}
        completed_indices = set()
        if save_dir is not None:
            checkpoint = _load_eval_checkpoint(Path(save_dir))
            if checkpoint.get("predictions"):
                completed_indices = set(checkpoint.get("completed_indices", []))
                logger.info(
                    f"📂 Loaded checkpoint with {len(completed_indices)} completed examples"
                )

        # Build prompts for remaining examples
        all_prompts = []
        prompt_to_example = []
        for idx, item in enumerate(prompts_data):
            if idx in completed_indices:
                continue
            for _ in range(num_samples):
                all_prompts.append(item["built"])
                prompt_to_example.append(idx)

        # Initialize grouped_preds from checkpoint
        grouped_preds = [[] for _ in range(len(examples))]
        if checkpoint.get("predictions"):
            for idx_str, preds in checkpoint["predictions"].items():
                grouped_preds[int(idx_str)] = preds

        if all_prompts:
            logger.info(
                f"\n🔄 Generating {len(all_prompts)} predictions ({len(completed_indices)} already cached)..."
            )
            gen_start = time.time()

            # Generate in chunks and checkpoint periodically
            chunk_size = batch_size * num_samples * 10
            for chunk_start in range(0, len(all_prompts), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(all_prompts))
                chunk_prompts = all_prompts[chunk_start:chunk_end]
                chunk_mapping = prompt_to_example[chunk_start:chunk_end]

                chunk_preds = model.continue_from_context_batch(
                    chunk_prompts,
                    model.gen_cfg.max_new_tokens,
                    greedy=False,
                    batch_size=batch_size,
                )

                # Group predictions
                for ex_idx, pred in zip(chunk_mapping, chunk_preds):
                    grouped_preds[ex_idx].append(pred)
                    if len(grouped_preds[ex_idx]) == num_samples:
                        completed_indices.add(ex_idx)

                # Save checkpoint
                if save_dir is not None:
                    checkpoint_data = {
                        "predictions": {
                            str(i): preds
                            for i, preds in enumerate(grouped_preds)
                            if preds
                        },
                        "completed_indices": list(completed_indices),
                        "num_samples": num_samples,
                    }
                    _save_eval_checkpoint(Path(save_dir), checkpoint_data)

            gen_elapsed = time.time() - gen_start
            logger.info(f"✅ Generation complete ({gen_elapsed:.1f}s)")
        else:
            logger.info(
                f"✅ All {len(examples)} examples already generated from checkpoint"
            )

        logger.info(f"\n📊 Computing metrics and KL divergence...")
        from pita.models.guided import GuidedHFModel

        if isinstance(model, GuidedHFModel):
            logger.info(
                f"⚠️  KL computation for guided models processes tokens step-by-step - this may take several minutes"
            )
        kl_start = time.time()
        total = 0
        pass1 = 0
        maj8 = 0
        rows: List[Dict[str, str]] = []

        pass1_prompts = []
        pass1_conts = []

        for item, preds in zip(prompts_data, grouped_preds):
            if not preds:
                continue
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

        kl_sample_ratio = float(getattr(cfg.evaluation, "kl_sample_ratio", 1.0) or 1.0)
        kl_values = _compute_traj_kl_batched(
            model,
            ref_model,
            prompt_texts=pass1_prompts,
            continuation_texts=pass1_conts,
            batch_size=batch_size,
            sample_ratio=kl_sample_ratio,
            save_dir=Path(save_dir) if save_dir else None,
        )
        sum_kl = sum(kl_values)
        kl_elapsed = time.time() - kl_start
        logger.info(f"✅ KL computation complete ({kl_elapsed:.1f}s)")

        if save_dir is not None:
            ensure_dir(Path(save_dir))
            ds_out = Dataset.from_list(rows)
            ds_out.save_to_disk(str(Path(save_dir) / "eval_predictions.hf"))
            ds_out.to_csv(str(Path(save_dir) / "eval_predictions.csv"))
            # Clean up checkpoint after successful completion
            ckpt_path = _get_eval_checkpoint_path(Path(save_dir))
            if ckpt_path.exists():
                ckpt_path.unlink()

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
    with logging_context(stage="EVAL"):
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

        # Use evaluation.max_examples if set, otherwise fall back to data_collection.max_examples
        eval_max = int(getattr(cfg.evaluation, "max_examples", 0) or 0)
        data_max = int(cfg.data_collection.max_examples or 0)
        max_examples = eval_max if eval_max > 0 else data_max
        limit = max_examples if max_examples > 0 else len(ds)
        batch_size = int(cfg.evaluation.batch_size)

        # Collect all examples first
        examples = list(islice(ds.iter(), limit))
        logger.info(
            f"🔄 Evaluating {len(examples)} examples with batch_size={batch_size}"
        )

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

        # Check for checkpoint
        checkpoint = {}
        if save_dir is not None:
            checkpoint = _load_eval_checkpoint(Path(save_dir))
            if checkpoint.get("predictions"):
                logger.info(
                    f"📂 Loaded checkpoint with {len(checkpoint['predictions'])} predictions"
                )

        # PHASE 2: Generate predictions (with checkpointing)
        preds = checkpoint.get("predictions", [])
        if len(preds) < len(prompts_data):
            remaining_prompts = [item["built"] for item in prompts_data[len(preds) :]]
            logger.info(
                f"🔄 Generating {len(remaining_prompts)} predictions ({len(preds)} already cached)..."
            )

            # Generate in chunks and checkpoint
            chunk_size = batch_size * 20
            for chunk_start in range(0, len(remaining_prompts), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(remaining_prompts))
                chunk_prompts = remaining_prompts[chunk_start:chunk_end]

                chunk_preds = model.continue_from_context_batch(
                    chunk_prompts,
                    model.gen_cfg.max_new_tokens,
                    greedy=False,
                    batch_size=batch_size,
                )
                preds.extend(chunk_preds)

                # Save checkpoint
                if save_dir is not None:
                    checkpoint_data = {"predictions": preds}
                    _save_eval_checkpoint(Path(save_dir), checkpoint_data)

            logger.info(f"✅ Generation complete")
        else:
            logger.info(
                f"✅ All {len(prompts_data)} predictions loaded from checkpoint"
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
        kl_sample_ratio = float(getattr(cfg.evaluation, "kl_sample_ratio", 1.0) or 1.0)
        kl_values = _compute_traj_kl_batched(
            model,
            ref_model,
            prompt_texts=all_prompts,
            continuation_texts=preds,
            batch_size=batch_size,
            sample_ratio=kl_sample_ratio,
            save_dir=Path(save_dir) if save_dir else None,
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
            # Clean up checkpoint after successful completion
            ckpt_path = _get_eval_checkpoint_path(Path(save_dir))
            if ckpt_path.exists():
                ckpt_path.unlink()

        avg_reward = float(sum_reward / max(1, total))
        return {
            "avg_reward": avg_reward,
            "avg_kl": float(sum_kl / max(1, total)),
            "num_examples": int(total),
        }
