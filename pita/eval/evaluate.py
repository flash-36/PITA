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
    kl = p_prob * (p_logprob - q_logprob)
    # When p=0 and q=0, we get 0 * (-inf - -inf) = 0 * NaN = NaN
    # The correct contribution when p=0 is 0, so replace NaN with 0
    kl = torch.nan_to_num(kl, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.sum(kl, dim=-1)


@torch.inference_mode()
def _compute_kl_fast_with_scores(
    ref_model: HFModel,
    prompt_texts: List[str],
    continuation_texts: List[str],
    saved_scores: List[torch.Tensor],
    batch_size: int = 8,
    sample_ratio: float = 1.0,
) -> List[float]:
    """Fast KL computation using pre-saved guided scores.

    This is ~600x faster than recomputing guided logits for each token.

    Args:
        ref_model: The reference model to compute ref logits
        prompt_texts: List of prompt strings
        continuation_texts: List of continuation strings (generated text)
        saved_scores: List of saved guided scores from generation, each [seq_len, vocab_size]
        batch_size: Batch size for ref model forward pass
        sample_ratio: Fraction of examples to compute KL for
    """
    n_total = len(prompt_texts)

    # Sample subset if ratio < 1.0
    if sample_ratio < 1.0 and n_total > 0:
        n_sample = max(1, int(n_total * sample_ratio))
        random.seed(42)
        sample_indices = sorted(random.sample(range(n_total), n_sample))
        prompt_texts = [prompt_texts[i] for i in sample_indices]
        continuation_texts = [continuation_texts[i] for i in sample_indices]
        saved_scores = [saved_scores[i] for i in sample_indices]
        logger.info(
            f"📊 KL sampling: {n_sample}/{n_total} examples ({sample_ratio*100:.0f}%)"
        )

    tok = ref_model.tokenizer
    device = ref_model.model.device
    kl_results: List[float] = []

    for i in tqdm(range(0, len(prompt_texts), batch_size), desc="Fast KL", leave=False):
        batch_prompts = prompt_texts[i : i + batch_size]
        batch_conts = continuation_texts[i : i + batch_size]
        batch_scores = saved_scores[i : i + batch_size]

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
        batch_pad_lens = []
        for input_ids, attention_mask in zip(batch_input_ids, batch_attention_masks):
            pad_len = max_len - input_ids.shape[0]
            batch_pad_lens.append(pad_len)
            padded_input_ids.append(F.pad(input_ids, (pad_len, 0), value=pad_id))
            padded_attention_masks.append(F.pad(attention_mask, (pad_len, 0), value=0))

        batched_input_ids = torch.stack(padded_input_ids).to(device)
        batched_attention_masks = torch.stack(padded_attention_masks).to(device)

        # Single forward pass through ref model
        ref_out = ref_model.model(
            input_ids=batched_input_ids,
            attention_mask=batched_attention_masks,
            use_cache=False,
        )

        # Compute KL for each example using saved scores
        for idx, prompt_len in enumerate(batch_prompt_lens):
            pad_len = batch_pad_lens[idx]
            ref_logits_steps = ref_out.logits[
                idx, pad_len + prompt_len - 1 : -1, :
            ]  # [seq_len, vocab]

            if ref_logits_steps.numel() == 0:
                kl_results.append(0.0)
                continue

            # Get saved guided scores for this example
            guided_scores = batch_scores[idx].to(device)  # [seq_len, vocab]

            # Align lengths (generation might have different length than tokenized continuation)
            min_len = min(guided_scores.shape[0], ref_logits_steps.shape[0])
            if min_len == 0:
                kl_results.append(0.0)
                continue

            guided_scores = guided_scores[:min_len]
            ref_logits_steps = ref_logits_steps[:min_len]

            # Compute KL(guided || ref) for all tokens at once
            kl_per_token = _kl_divergence(guided_scores, ref_logits_steps)
            kl_results.append(float(kl_per_token.sum().item()))

        del ref_out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Scale up if we sampled
    if sample_ratio < 1.0:
        avg_kl = sum(kl_results) / max(1, len(kl_results))
        return [avg_kl] * n_total

    return kl_results


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
        logger.info(
            f"📊 KL sampling: {n_sample}/{n_total} examples ({sample_ratio*100:.0f}%)"
        )

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
                logger.info(
                    f"📂 Loaded KL checkpoint: {len(kl_results)} values, resuming from batch {start_batch+1}"
                )

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
        batch_pad_lens = []
        for input_ids, attention_mask in zip(batch_input_ids, batch_attention_masks):
            pad_len = max_len - input_ids.shape[0]
            batch_pad_lens.append(pad_len)
            padded_input_ids.append(F.pad(input_ids, (pad_len, 0), value=pad_id))
            padded_attention_masks.append(F.pad(attention_mask, (pad_len, 0), value=0))

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
                ref_out.logits[idx, batch_pad_lens[idx] + prompt_len - 1 : -1, :].shape[0]
                for idx, prompt_len in enumerate(batch_prompt_lens)
                if ref_out.logits[idx, batch_pad_lens[idx] + prompt_len - 1 : -1, :].numel() > 0
            )

            # Single progress bar for all examples in this batch
            use_pbar = total_steps > 200
            pbar = (
                tqdm(
                    total=total_steps,
                    desc=f"  KL tokens (batch {batch_idx+1}/{total_batches})",
                    leave=False,
                    unit="tok",
                    position=0,  # Force to top position
                )
                if use_pbar
                else None
            )

            for idx, prompt_len in enumerate(batch_prompt_lens):
                pad_len = batch_pad_lens[idx]
                ref_logits_steps = ref_out.logits[idx : idx + 1, pad_len + prompt_len - 1 : -1, :]
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
                    prefix_len = pad_len + prompt_len + t
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
                pad_len = batch_pad_lens[idx]
                ref_logits_steps = ref_out.logits[idx : idx + 1, pad_len + prompt_len - 1 : -1, :]
                pol_logits_steps = pol_out.logits[idx : idx + 1, pad_len + prompt_len - 1 : -1, :]

                if ref_logits_steps.numel() == 0:
                    batch_kl_values.append(0.0)
                else:
                    kl_steps = _kl_divergence(pol_logits_steps, ref_logits_steps)
                    batch_kl_values.append(float(kl_steps.sum().item()))

        # Add batch results and save checkpoint
        kl_results.extend(batch_kl_values)
        if save_dir is not None:
            _save_kl_checkpoint(
                Path(save_dir),
                {
                    "kl_values": kl_results,
                    "completed_batches": batch_idx + 1,
                    "total_batches": total_batches,
                    "sample_ratio": sample_ratio,
                },
            )

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
    batch_size: Optional[int] = None,
    is_cot8: bool = False,
) -> Dict[str, float]:
    with logging_context(stage="EVAL"):
        eval_start = time.time()

        # Check if evaluation already completed (results.json exists)
        if save_dir is not None:
            results_json = Path(save_dir) / "results.json"
            if results_json.exists():
                with open(results_json) as f:
                    cached_results = json.load(f)
                logger.info(
                    f"✅ Evaluation already complete, loaded from {results_json}"
                )
                return cached_results

        ds = build_test_dataset(cfg, dataset)
        if ds is None:
            return {}

        from pita.eval.cot_examples import get_8shot_prompt

        limit = int(getattr(cfg.datasets[dataset], "test_size_cap", 0) or 0)
        limit = limit if limit > 0 else len(ds)
        num_samples = int(cfg.evaluation.num_samples)
        batch_size = (
            batch_size if batch_size is not None else int(cfg.evaluation.batch_size)
        )
        has_verifier = hasattr(ds, "is_correct")

        examples = list(islice(ds.iter(), limit))
        logger.info(
            f"🔄 Evaluating {len(examples)} examples with batch_size={batch_size}, {num_samples} samples each"
        )

        prompts_data = []
        for ex in examples:
            if is_cot8:
                prompt = get_8shot_prompt(dataset, ex.question)
            else:
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
        loaded_from_csv = False
        grouped_preds = [[] for _ in range(len(examples))]

        if save_dir is not None:
            # First try loading from checkpoint
            checkpoint = _load_eval_checkpoint(Path(save_dir))
            if checkpoint.get("predictions"):
                completed_indices = set(checkpoint.get("completed_indices", []))
                logger.info(
                    f"📂 Loaded checkpoint with {len(completed_indices)} completed examples"
                )
                for idx_str, preds in checkpoint["predictions"].items():
                    grouped_preds[int(idx_str)] = preds
            else:
                # Fallback: try loading from eval_predictions.csv if it exists
                csv_path = Path(save_dir) / "eval_predictions.csv"
                if csv_path.exists():
                    import csv

                    with open(csv_path) as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                    if rows:
                        # Get pred columns
                        pred_cols = sorted(
                            [c for c in rows[0].keys() if c.startswith("pred_")]
                        )
                        if pred_cols:
                            for idx, row in enumerate(rows):
                                if idx < len(grouped_preds):
                                    grouped_preds[idx] = [row[pc] for pc in pred_cols]
                            completed_indices = set(range(len(rows)))
                            loaded_from_csv = True
                            logger.info(
                                f"📂 Loaded {len(rows)} predictions from {csv_path}"
                            )

        # Build prompts for remaining examples: separate first samples from extra samples
        first_prompts_data = []  # List of (ex_idx, prompt_text)
        extra_prompts_data = []  # List of (ex_idx, prompt_text)

        for idx, item in enumerate(prompts_data):
            if idx in completed_indices:
                continue

            n_existing = len(grouped_preds[idx])
            if n_existing == 0:
                first_prompts_data.append((idx, item["built"]))
                start_n = 1
            else:
                start_n = n_existing

            for _ in range(start_n, num_samples):
                extra_prompts_data.append((idx, item["built"]))

        # Track saved scores for fast KL computation (only first sample per example)
        saved_scores_map: Dict[int, torch.Tensor] = {}
        use_fast_kl = isinstance(model, GuidedHFModel)

        # Determine which examples need scores for KL (respect kl_sample_ratio)
        kl_sample_ratio = float(getattr(cfg.evaluation, "kl_sample_ratio", 1.0) or 1.0)
        n_examples = len(examples)
        if kl_sample_ratio < 1.0 and n_examples > 0:
            n_kl_samples = max(1, int(n_examples * kl_sample_ratio))
            random.seed(42)
            kl_sample_indices = set(random.sample(range(n_examples), n_kl_samples))
        else:
            kl_sample_indices = set(range(n_examples))

        # PHASE 1: Generate first samples (with scores if needed)
        if first_prompts_data:
            logger.info(
                f"\n🔄 Phase 1: Generating first samples for {len(first_prompts_data)} examples..."
            )
            gen_start = time.time()

            # Use small chunk size for score-heavy generation to avoid RAM peak
            score_chunk_size = batch_size * 2

            for chunk_start in range(0, len(first_prompts_data), score_chunk_size):
                chunk_end = min(chunk_start + score_chunk_size, len(first_prompts_data))
                chunk_items = first_prompts_data[chunk_start:chunk_end]
                chunk_prompts = [item[1] for item in chunk_items]
                chunk_indices = [item[0] for item in chunk_items]

                needs_scores = use_fast_kl and any(
                    idx in kl_sample_indices for idx in chunk_indices
                )

                if needs_scores:
                    chunk_preds, chunk_scores = model.continue_from_context_batch(
                        chunk_prompts,
                        model.gen_cfg.max_new_tokens,
                        greedy=False,
                        batch_size=batch_size,
                        return_scores=True,
                    )
                    for i, (ex_idx, pred, scores) in enumerate(
                        zip(chunk_indices, chunk_preds, chunk_scores)
                    ):
                        grouped_preds[ex_idx].append(pred)
                        if ex_idx in kl_sample_indices and scores is not None:
                            saved_scores_map[ex_idx] = scores
                else:
                    chunk_preds = model.continue_from_context_batch(
                        chunk_prompts,
                        model.gen_cfg.max_new_tokens,
                        greedy=False,
                        batch_size=batch_size,
                    )
                    for ex_idx, pred in zip(chunk_indices, chunk_preds):
                        grouped_preds[ex_idx].append(pred)

                # Checkpoint periodically
                if save_dir is not None:
                    checkpoint_data = {
                        "predictions": {
                            str(i): p for i, p in enumerate(grouped_preds) if p
                        },
                        "completed_indices": [
                            i
                            for i, p in enumerate(grouped_preds)
                            if len(p) == num_samples
                        ],
                        "num_samples": num_samples,
                    }
                    _save_eval_checkpoint(Path(save_dir), checkpoint_data)

                import gc

                gc.collect()

        # PHASE 2: Generate extra samples (never need scores)
        if extra_prompts_data:
            logger.info(
                f"\n🔄 Phase 2: Generating remaining {len(extra_prompts_data)} samples..."
            )

            # Larger chunk size is fine here since no scores are returned
            extra_chunk_size = batch_size * 20

            for chunk_start in range(0, len(extra_prompts_data), extra_chunk_size):
                chunk_end = min(chunk_start + extra_chunk_size, len(extra_prompts_data))
                chunk_items = extra_prompts_data[chunk_start:chunk_end]
                chunk_prompts = [item[1] for item in chunk_items]
                chunk_indices = [item[0] for item in chunk_items]

                chunk_preds = model.continue_from_context_batch(
                    chunk_prompts,
                    model.gen_cfg.max_new_tokens,
                    greedy=False,
                    batch_size=batch_size,
                    return_scores=False,
                )

                for ex_idx, pred in zip(chunk_indices, chunk_preds):
                    grouped_preds[ex_idx].append(pred)
                    if len(grouped_preds[ex_idx]) == num_samples:
                        completed_indices.add(ex_idx)

                # Checkpoint periodically
                if save_dir is not None:
                    checkpoint_data = {
                        "predictions": {
                            str(i): p for i, p in enumerate(grouped_preds) if p
                        },
                        "completed_indices": list(completed_indices),
                        "num_samples": num_samples,
                    }
                    _save_eval_checkpoint(Path(save_dir), checkpoint_data)

                import gc

                gc.collect()

            logger.info(f"✅ Generation complete")
        else:
            logger.info(f"✅ All {len(examples)} examples already generated")

        logger.info(f"\n📊 Computing metrics and KL divergence...")

        kl_start = time.time()
        total = 0
        pass1 = 0
        maj8 = 0
        rows: List[Dict[str, str]] = []

        pass1_prompts = []
        pass1_conts = []
        pass1_scores = []

        for idx, (item, preds) in enumerate(zip(prompts_data, grouped_preds)):
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

            # Get first sample's scores for KL (if available)
            if idx in saved_scores_map:
                pass1_scores.append(saved_scores_map[idx])
            else:
                pass1_scores.append(None)

            rows.append(
                {
                    "question": item["ex"].question,
                    "answer": item["ex"].answer,
                    **{f"pred_{i+1}": preds[i] for i in range(len(preds))},
                }
            )

        kl_sample_ratio = float(getattr(cfg.evaluation, "kl_sample_ratio", 1.0) or 1.0)

        # Use fast KL if we have saved scores
        if use_fast_kl and pass1_scores and any(s is not None for s in pass1_scores):
            # Filter to only examples with scores
            valid_indices = [i for i, s in enumerate(pass1_scores) if s is not None]
            if valid_indices:
                logger.info(
                    f"🚀 Using fast KL computation with {len(valid_indices)} saved score sets"
                )
                ref_for_kl = (
                    model.ref if isinstance(model, GuidedHFModel) else ref_model
                )
                kl_values = _compute_kl_fast_with_scores(
                    ref_for_kl,
                    [pass1_prompts[i] for i in valid_indices],
                    [pass1_conts[i] for i in valid_indices],
                    [pass1_scores[i] for i in valid_indices],
                    batch_size=batch_size,
                    sample_ratio=kl_sample_ratio,
                )
                # Map back to full list
                full_kl_values = [0.0] * len(pass1_prompts)
                for i, kl in zip(valid_indices, kl_values):
                    full_kl_values[i] = kl
                kl_values = full_kl_values
            else:
                logger.warning(
                    "⚠️  No saved scores available, falling back to slow KL computation"
                )
                kl_values = _compute_traj_kl_batched(
                    model,
                    ref_model,
                    prompt_texts=pass1_prompts,
                    continuation_texts=pass1_conts,
                    batch_size=batch_size,
                    sample_ratio=kl_sample_ratio,
                    save_dir=Path(save_dir) if save_dir else None,
                )
        else:
            if isinstance(model, GuidedHFModel):
                logger.info(
                    f"⚠️  KL computation for guided models processes tokens step-by-step - this may take several minutes"
                )
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

        # Save results to JSON for caching
        if save_dir is not None:
            results_json = Path(save_dir) / "results.json"
            with open(results_json, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"💾 Saved results to {results_json}")

        return results


def evaluate_avg_reward(
    cfg: DictConfig,
    model: HFModel,
    dataset: str,
    ref_model: Optional[HFModel] = None,
    save_dir: Optional[Path] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, float]:
    with logging_context(stage="EVAL"):
        # Check if evaluation already completed (results.json exists)
        if save_dir is not None:
            results_json = Path(save_dir) / "results.json"
            if results_json.exists():
                with open(results_json) as f:
                    cached_results = json.load(f)
                logger.info(
                    f"✅ Evaluation already complete, loaded from {results_json}"
                )
                return cached_results

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

        limit = int(getattr(cfg.datasets[dataset], "test_size_cap", 0) or 0)
        limit = limit if limit > 0 else len(ds)
        batch_size = (
            batch_size if batch_size is not None else int(cfg.evaluation.batch_size)
        )

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
        loaded_from_csv = False
        if save_dir is not None:
            checkpoint = _load_eval_checkpoint(Path(save_dir))
            if checkpoint.get("predictions"):
                logger.info(
                    f"📂 Loaded checkpoint with {len(checkpoint['predictions'])} predictions"
                )
            else:
                # Fallback: try loading from eval_predictions.csv if it exists
                csv_path = Path(save_dir) / "eval_predictions.csv"
                if csv_path.exists():
                    import csv

                    with open(csv_path) as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                    if rows and "pred" in rows[0]:
                        checkpoint["predictions"] = [row["pred"] for row in rows]
                        loaded_from_csv = True
                        logger.info(
                            f"📂 Loaded {len(rows)} predictions from {csv_path}"
                        )

        # PHASE 2: Generate predictions (with checkpointing)
        preds = checkpoint.get("predictions", [])
        saved_scores: List[Optional[torch.Tensor]] = [None] * len(preds)
        use_fast_kl = isinstance(model, GuidedHFModel)

        # Determine which examples need scores for KL (respect kl_sample_ratio)
        kl_sample_ratio = float(getattr(cfg.evaluation, "kl_sample_ratio", 1.0) or 1.0)
        n_examples = len(prompts_data)
        if kl_sample_ratio < 1.0 and n_examples > 0:
            n_kl_samples = max(1, int(n_examples * kl_sample_ratio))
            random.seed(42)
            kl_sample_indices = set(random.sample(range(n_examples), n_kl_samples))
        else:
            kl_sample_indices = set(range(n_examples))

        if len(preds) < len(prompts_data):
            remaining_prompts = [item["built"] for item in prompts_data[len(preds) :]]
            start_idx = len(preds)
            logger.info(
                f"🔄 Generating {len(remaining_prompts)} predictions ({len(preds)} already cached)..."
            )
            if use_fast_kl:
                logger.info(
                    f"📊 Using fast KL computation (saving scores for {len(kl_sample_indices)} examples)"
                )

            # Use small chunk size for score-heavy generation to avoid RAM peak
            score_chunk_size = batch_size * 2

            for chunk_start in range(0, len(remaining_prompts), score_chunk_size):
                chunk_end = min(chunk_start + score_chunk_size, len(remaining_prompts))
                chunk_prompts = remaining_prompts[chunk_start:chunk_end]
                global_start = start_idx + chunk_start

                # Check if any example in this chunk needs scores for KL
                chunk_needs_scores = use_fast_kl and any(
                    (global_start + i) in kl_sample_indices
                    for i in range(len(chunk_prompts))
                )

                if chunk_needs_scores:
                    chunk_preds, chunk_scores = model.continue_from_context_batch(
                        chunk_prompts,
                        model.gen_cfg.max_new_tokens,
                        greedy=False,
                        batch_size=batch_size,
                        return_scores=True,
                    )
                    preds.extend(chunk_preds)
                    # Only save scores for KL-sampled examples
                    for i, score in enumerate(chunk_scores):
                        if (global_start + i) in kl_sample_indices:
                            saved_scores.append(score)
                        else:
                            saved_scores.append(None)
                else:
                    chunk_preds = model.continue_from_context_batch(
                        chunk_prompts,
                        model.gen_cfg.max_new_tokens,
                        greedy=False,
                        batch_size=batch_size,
                    )
                    preds.extend(chunk_preds)
                    saved_scores.extend([None] * len(chunk_preds))

                # Save checkpoint
                if save_dir is not None:
                    checkpoint_data = {"predictions": preds}
                    _save_eval_checkpoint(Path(save_dir), checkpoint_data)

                import gc

                gc.collect()

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

        # Use fast KL if we have saved scores
        if use_fast_kl and saved_scores and any(s is not None for s in saved_scores):
            valid_indices = [i for i, s in enumerate(saved_scores) if s is not None]
            if valid_indices:
                logger.info(
                    f"🚀 Using fast KL computation with {len(valid_indices)} saved score sets"
                )
                ref_for_kl = (
                    model.ref if isinstance(model, GuidedHFModel) else ref_model
                )
                kl_values = _compute_kl_fast_with_scores(
                    ref_for_kl,
                    [all_prompts[i] for i in valid_indices],
                    [preds[i] for i in valid_indices],
                    [saved_scores[i] for i in valid_indices],
                    batch_size=batch_size,
                    sample_ratio=kl_sample_ratio,
                )
                # Map back to full list
                full_kl_values = [0.0] * len(all_prompts)
                for i, kl in zip(valid_indices, kl_values):
                    full_kl_values[i] = kl
                kl_values = full_kl_values
            else:
                logger.warning(
                    "⚠️  No saved scores available, falling back to slow KL computation"
                )
                kl_values = _compute_traj_kl_batched(
                    model,
                    ref_model,
                    prompt_texts=all_prompts,
                    continuation_texts=preds,
                    batch_size=batch_size,
                    sample_ratio=kl_sample_ratio,
                    save_dir=Path(save_dir) if save_dir else None,
                )
        else:
            if isinstance(model, GuidedHFModel):
                logger.info(
                    f"⚠️  KL computation for guided models processes tokens step-by-step - this may take several minutes"
                )
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
        results = {
            "avg_reward": avg_reward,
            "avg_kl": float(sum_kl / max(1, total)),
            "num_examples": int(total),
        }

        # Save results to JSON for caching
        if save_dir is not None:
            results_json = Path(save_dir) / "results.json"
            with open(results_json, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"💾 Saved results to {results_json}")

        return results
