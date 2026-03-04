from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, Any, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.logits_process import LogitsProcessorList
from loguru import logger

from pita.models.catalog import resolve_model_id
from pita.core.prompts import build_instruction_prompt


def _is_device_assert(e: BaseException) -> bool:
    """Check if exception is a CUDA device-side assert."""
    msg = str(e).lower()
    return (
        "device-side assert triggered" in msg
        or "device side assert triggered" in msg
        or "cuda error: device-side assert triggered" in msg
        or "_assert_async_cuda_kernel" in msg
        or "probability tensor contains either" in msg
    )


@dataclass
class GenerationConfig:
    max_new_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: float
    use_chat_template: bool
    dtype: str
    attn_impl: str
    gradient_checkpointing: bool


class HFModel:
    def __init__(self, name_or_id: str, gen_cfg: GenerationConfig):
        model_id = resolve_model_id(name_or_id)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            trust_remote_code=True,
        )
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            if getattr(self.tokenizer, "eos_token", None) is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                raise ValueError("Pad token not found")
        self.tokenizer.padding_side = "left"
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }[gen_cfg.dtype]
        # Use device_map="auto" for inference-only single-process runs.
        # In distributed training (e.g., Accelerate/DDP), avoid device_map and let the trainer place/shard.
        in_dist = (
            os.environ.get("LOCAL_RANK") is not None
            or os.environ.get("RANK") is not None
        )
        device_map = None if in_dist else "auto"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            attn_implementation=gen_cfg.attn_impl,
            low_cpu_mem_usage=True,
        )
        if getattr(gen_cfg, "gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable()
        self.model.eval()
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        import warnings
        warnings.filterwarnings(
            "ignore",
            message=".*right-padding was detected.*",
            category=UserWarning,
        )
        self.gen_cfg = gen_cfg
        self.eos_token_ids = self._compute_eos_token_ids()
        self.pad_token_id = self._compute_pad_token_id()

    def _compute_eos_token_ids(self) -> List[int]:
        eos_ids: List[int] = []
        base_eos = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(base_eos, int):
            eos_ids.append(base_eos)
        elif isinstance(base_eos, (list, tuple)):
            eos_ids.extend(base_eos)

        # Also pull from the model's generation config
        gen_eos = getattr(self.model.generation_config, "eos_token_id", None)
        if isinstance(gen_eos, int) and gen_eos not in eos_ids:
            eos_ids.append(gen_eos)
        elif isinstance(gen_eos, (list, tuple)):
            eos_ids.extend(eid for eid in gen_eos if eid not in eos_ids)

        unk_id = getattr(self.tokenizer, "unk_token_id", None)
        for tok in [
            "<|endoftext|>",
            "<|end_of_text|>",
            "<|end|>",
            "<|im_end|>",
            "<|eot_id|>",
            "<|end_of_turn|>",
            "<end_of_turn>",
            "</s>",
        ]:
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if (
                isinstance(tid, int)
                and tid >= 0
                and tid != unk_id
                and tid not in eos_ids
            ):
                eos_ids.append(tid)

        logger.info(
            f"🔍 Computed EOS token IDs: {eos_ids} for tokens: {[self.tokenizer.decode([eid]) for eid in eos_ids]}"
        )
        return list(set(eos_ids))

    def _compute_pad_token_id(self) -> int:
        pad = getattr(self.tokenizer, "pad_token_id", None)
        if isinstance(pad, int):
            return pad
        raise ValueError("Pad token id not found")

    def _gen_kwargs(
        self,
        *,
        greedy: bool,
        logits_processor: Optional[LogitsProcessorList],
        max_new_tokens: Optional[int] = None,
    ):
        temperature = 1.0 if greedy else self.gen_cfg.temperature
        top_p = 1.0 if greedy else self.gen_cfg.top_p

        res = {
            "max_new_tokens": (
                self.gen_cfg.max_new_tokens
                if max_new_tokens is None
                else int(max_new_tokens)
            ),
            "do_sample": (not greedy),
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": self.gen_cfg.repetition_penalty,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_ids,
            "logits_processor": logits_processor,
        }
        logger.debug(
            f"🔍 _gen_kwargs: greedy={greedy}, do_sample={res['do_sample']}, temp={res['temperature']}, top_p={res['top_p']}, rep_pen={res['repetition_penalty']}"
        )
        return res

    def _generate_with_fallback(
        self, inputs: Dict[str, torch.Tensor], gen_kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        """Generate with fp32 greedy fallback on device-assert."""
        try:
            out = self.model.generate(**inputs, **gen_kwargs)
            # logger.debug(f"🔍 generate done. Output shape: {out.shape if hasattr(out, 'shape') else 'dict'}")
            return out
        except RuntimeError as e:
            if _is_device_assert(e):
                logger.warning(
                    f"🔥 Device-assert during generation, falling back to greedy fp32: {e}"
                )
                original_dtype = self.model.dtype
                self.model.to(torch.float32)
                fallback_kwargs = {
                    **gen_kwargs,
                    "do_sample": False,
                    "temperature": 1.0,
                    "top_p": 1.0,
                }
                fallback_kwargs.pop("logits_processor", None)
                try:
                    result = self.model.generate(**inputs, **fallback_kwargs)
                    return result
                finally:
                    self.model.to(original_dtype)
            raise

    @torch.inference_mode()
    def generate_text(
        self,
        prompt: str,
        *,
        greedy: bool = False,
        logits_processor: Optional[LogitsProcessorList] = None,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        gen_kwargs = self._gen_kwargs(greedy=greedy, logits_processor=logits_processor)
        output = self._generate_with_fallback(inputs, gen_kwargs)
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    @torch.inference_mode()
    def generate_n(
        self,
        prompt: str,
        n: int,
        *,
        greedy: bool = False,
        logits_processor: Optional[LogitsProcessorList] = None,
        batch_size: int = 1,
        max_new_tokens: Optional[int] = None,
    ) -> List[str]:
        """Generate n sequences, batched to avoid OOM with guided generation.

        For guided generation with a classifier, batch_size=1 is safest since both
        the reference model and classifier run in parallel, doubling memory usage.
        """
        n = int(n)
        batch_size = min(int(batch_size), n)

        all_outputs = []
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        seq_len = inputs["input_ids"].shape[1]

        # Generate in batches
        remaining = n
        while remaining > 0:
            current_batch = min(batch_size, remaining)
            gen_kwargs = self._gen_kwargs(
                greedy=greedy,
                logits_processor=logits_processor,
                max_new_tokens=max_new_tokens,
            )
            gen_kwargs["num_return_sequences"] = current_batch
            out = self._generate_with_fallback(inputs, gen_kwargs)

            # Decode this batch
            batch_texts = [
                self.tokenizer.decode(seq[seq_len:], skip_special_tokens=True).strip()
                for seq in out
            ]
            all_outputs.extend(batch_texts)
            remaining -= current_batch

            # Delete generation output to free memory immediately
            del out

            # Clear cache between batches to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return all_outputs

    @torch.inference_mode()
    def roll_in(
        self,
        full_prompt: str,
        max_roll_tokens: int,
        logits_processor: Optional[LogitsProcessorList] = None,
    ) -> Dict[str, Any]:
        """Greedy rollout for t steps to obtain context s_t.

        Returns dict with keys: prompt, context_ids, context_text.
        """
        built = build_instruction_prompt(
            full_prompt,
            tokenizer=self.tokenizer,
            use_chat_template=self.gen_cfg.use_chat_template,
        )
        logger.debug(f"🛠️  roll_in: built prompt: {built[:200]}...")
        ids = self.tokenizer(built, return_tensors="pt").to(self.model.device)
        gen_kwargs = self._gen_kwargs(
            greedy=True,
            logits_processor=logits_processor,
            max_new_tokens=max_roll_tokens,
        )
        out = self._generate_with_fallback(ids, gen_kwargs)
        context_tokens = out[0]
        context_text = self.tokenizer.decode(context_tokens, skip_special_tokens=False)
        return {
            "prompt": built,
            "context_ids": context_tokens,
            "context_text": context_text,
        }

    @torch.inference_mode()
    def continue_from_ids(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        greedy: bool,
        logits_processor: Optional[LogitsProcessorList] = None,
    ) -> str:
        """Continue generation directly from token IDs to avoid decode/re-tokenize issues."""
        last_token = input_ids[0, -1].item()
        if last_token in self.eos_token_ids:
            return ""

        gen_kwargs = self._gen_kwargs(
            greedy=greedy,
            logits_processor=logits_processor,
            max_new_tokens=max_new_tokens,
        )
        # Ensure we don't add special tokens again
        inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        out = self._generate_with_fallback(inputs, gen_kwargs)
        new_tokens = out[0][input_ids.shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    @torch.inference_mode()
    def continue_from_context(
        self,
        context_text: str,
        max_new_tokens: int,
        greedy: bool,
        logits_processor: Optional[LogitsProcessorList] = None,
    ) -> str:
        # If context already ends with EOS, don't generate more
        ids = self.tokenizer(
            context_text, return_tensors="pt", add_special_tokens=False
        ).to(self.model.device)
        return self.continue_from_ids(ids["input_ids"], max_new_tokens, greedy, logits_processor)

    @torch.inference_mode()
    def roll_in_batch(
        self,
        prompts: List[str],
        max_roll_tokens: int,
        logits_processor: Optional[LogitsProcessorList] = None,
        batch_size: int = 8,
    ) -> List[Dict[str, Any]]:
        """Batch greedy rollout for multiple prompts."""
        import time

        built_prompts = [
            build_instruction_prompt(
                p,
                tokenizer=self.tokenizer,
                use_chat_template=self.gen_cfg.use_chat_template,
            )
            for p in prompts
        ]

        all_results = []
        num_batches = (len(built_prompts) + batch_size - 1) // batch_size

        total_tokens_generated = 0
        start_time = time.time()

        for batch_idx, i in enumerate(range(0, len(built_prompts), batch_size)):
            batch = built_prompts[i : i + batch_size]
            ids = self.tokenizer(batch, return_tensors="pt", padding=True).to(
                self.model.device
            )
            prompt_lengths = (ids["attention_mask"].sum(dim=1)).tolist()

            gen_kwargs = self._gen_kwargs(
                greedy=True,
                logits_processor=logits_processor,
                max_new_tokens=max_roll_tokens,
            )
            out = self._generate_with_fallback(ids, gen_kwargs)

            batch_tokens = 0
            num_input_tokens = ids["input_ids"].shape[1]
            for j, (seq, prompt_len) in enumerate(zip(out, prompt_lengths)):
                # Remove left-padding before saving context_ids and context_text
                start_idx = num_input_tokens - prompt_len
                clean_seq = seq[start_idx:]
                batch_tokens += len(seq) - num_input_tokens
                context_text = self.tokenizer.decode(
                    clean_seq, skip_special_tokens=False
                )
                all_results.append(
                    {
                        "prompt": batch[j],
                        "context_ids": clean_seq,
                        "context_text": context_text,
                    }
                )

            total_tokens_generated += batch_tokens
            elapsed = time.time() - start_time
            tokens_per_sec = total_tokens_generated / max(elapsed, 0.001)

            logger.info(
                f"  📦 roll_in batch {batch_idx+1}/{num_batches} | "
                f"{len(all_results)}/{len(built_prompts)} prompts | "
                f"{tokens_per_sec:.0f} tok/s"
            )

            del out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return all_results

    @torch.inference_mode()
    def continue_from_ids_batch(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        greedy: bool,
        logits_processor: Optional[LogitsProcessorList] = None,
        batch_size: int = 8,
        return_scores: bool = False,
    ) -> List[str] | tuple[List[str], List[torch.Tensor]]:
        """Batch continuation directly from token IDs."""
        import time
        all_results = []
        all_scores = [] if return_scores else None
        num_examples = input_ids.shape[0]
        num_batches = (num_examples + batch_size - 1) // batch_size
        mode = "greedy" if greedy else "sampled"

        total_tokens_generated = 0
        start_time = time.time()

        for batch_idx in range(0, num_examples, batch_size):
            batch_ids = input_ids[batch_idx : batch_idx + batch_size].to(self.model.device)
            # Input IDs are already padded/sliced correctly if coming from roll_in
            # But we might need to mask padding if batching diverse lengths
            attention_mask = (batch_ids != self.pad_token_id).long()
            
            # Check for EOS at end of each item in batch
            active_mask = []
            for item_ids in batch_ids:
                # Find last non-pad token
                non_pad = item_ids[item_ids != self.pad_token_id]
                if len(non_pad) > 0 and non_pad[-1].item() in self.eos_token_ids:
                    active_mask.append(False)
                else:
                    active_mask.append(True)

            if not any(active_mask):
                for _ in range(len(batch_ids)):
                    all_results.append("")
                    if return_scores: all_scores.append(None)
                continue

            gen_kwargs = self._gen_kwargs(
                greedy=greedy,
                logits_processor=logits_processor,
                max_new_tokens=max_new_tokens,
            )
            if return_scores:
                gen_kwargs["output_scores"] = True
                gen_kwargs["return_dict_in_generate"] = True

            # For now, generate the whole batch. Masked items will just generate a few tokens and stop.
            inputs = {"input_ids": batch_ids, "attention_mask": attention_mask}
            out = self._generate_with_fallback(inputs, gen_kwargs)

            if return_scores:
                sequences = out.sequences
                stacked_scores = torch.stack(out.scores, dim=0).transpose(0, 1) if out.scores else None
            else:
                sequences = out
                stacked_scores = None

            num_input_tokens = batch_ids.shape[1]
            for j in range(len(batch_ids)):
                if not active_mask[j]:
                    all_results.append("")
                    if return_scores: all_scores.append(None)
                    continue
                
                new_tokens = sequences[j, num_input_tokens:]
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                all_results.append(text)
                if return_scores:
                    all_scores.append(stacked_scores[j, :len(new_tokens), :].cpu() if stacked_scores is not None else None)
                total_tokens_generated += len(new_tokens)

            elapsed = time.time() - start_time
            tokens_per_sec = total_tokens_generated / max(elapsed, 0.001)
            logger.info(f"  📦 {mode} batch {batch_idx//batch_size + 1}/{num_batches} | {tokens_per_sec:.0f} tok/s")

        if return_scores: return all_results, all_scores
        return all_results

    @torch.inference_mode()
    def continue_from_context_batch(
        self,
        contexts: List[str],
        max_new_tokens: int,
        greedy: bool,
        logits_processor: Optional[LogitsProcessorList] = None,
        batch_size: int = 8,
        return_scores: bool = False,
    ) -> List[str] | tuple[List[str], List[torch.Tensor]]:
        """Batch continuation from multiple contexts.

        Args:
            return_scores: If True, also return the logits/scores at each generation step.
                          This is useful for fast KL computation.

        Returns:
            If return_scores=False: List of generated texts
            If return_scores=True: Tuple of (texts, scores) where scores[i] is [seq_len, vocab_size]
        """
        import time

        all_results = []
        all_scores = [] if return_scores else None
        num_batches = (len(contexts) + batch_size - 1) // batch_size
        mode = "greedy" if greedy else "sampled"

        total_tokens_generated = 0
        start_time = time.time()

        for batch_idx, i in enumerate(range(0, len(contexts), batch_size)):
            batch = contexts[i : i + batch_size]
            if batch_idx == 0:
                logger.debug(
                    f"🛠️  continue_from_context_batch: first context: {batch[0][:200]}..."
                )
            ids = self.tokenizer(
                batch, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(self.model.device)

            # Check if all items in batch already end with EOS
            # Note: with left padding, the last token is at the end of input_ids
            last_tokens = ids["input_ids"][:, -1]
            all_ended = all(t.item() in self.eos_token_ids for t in last_tokens)
            if all_ended:
                logger.debug(f"⏭️  continue_from_context_batch: all items in batch {batch_idx} end with EOS, skipping generation")
                for _ in range(len(batch)):
                    all_results.append("")
                    if return_scores:
                        all_scores.append(None)
                continue

            prompt_lengths = (ids["attention_mask"].sum(dim=1)).tolist()

            gen_kwargs = self._gen_kwargs(
                greedy=greedy,
                logits_processor=logits_processor,
                max_new_tokens=max_new_tokens,
            )

            if return_scores:
                gen_kwargs["output_scores"] = True
                gen_kwargs["return_dict_in_generate"] = True

            out = self._generate_with_fallback(ids, gen_kwargs)

            if return_scores:
                sequences = out.sequences
                if out.scores:
                    stacked_scores = torch.stack(out.scores, dim=0)
                    stacked_scores = stacked_scores.transpose(0, 1)
                else:
                    stacked_scores = None
            else:
                sequences = out
            
            # Debug: check if EOS was hit for the first item in batch
            if sequences.shape[0] > 0:
                last_tokens = sequences[0, -5:].tolist()
                logger.debug(f"🔍 Batch generation tail (ex 0): {last_tokens} | Decoded: '{self.tokenizer.decode(last_tokens)}' | EOS IDs: {self.eos_token_ids}")

            batch_tokens = 0
            num_input_tokens = ids["input_ids"].shape[1]
            for j, (seq, prompt_len) in enumerate(zip(sequences, prompt_lengths)):
                new_tokens = seq[num_input_tokens:]
                batch_tokens += len(new_tokens)
                text = self.tokenizer.decode(
                    new_tokens, skip_special_tokens=True
                ).strip()
                all_results.append(text)

                if return_scores:
                    if stacked_scores is not None:
                        num_new_tokens = len(new_tokens)
                        example_scores = stacked_scores[j, :num_new_tokens, :].cpu()
                        all_scores.append(example_scores)
                    else:
                        all_scores.append(None)

            total_tokens_generated += batch_tokens
            elapsed = time.time() - start_time
            tokens_per_sec = total_tokens_generated / max(elapsed, 0.001)
            remaining_batches = num_batches - batch_idx - 1
            eta_sec = (remaining_batches * batch_tokens) / max(tokens_per_sec, 0.001)

            # Log progress every batch
            logger.info(
                f"  📦 {mode} batch {batch_idx+1}/{num_batches} | "
                f"{len(all_results)}/{len(contexts)} examples | "
                f"{tokens_per_sec:.0f} tok/s | ETA {eta_sec:.0f}s"
            )

            del out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if return_scores:
            return all_results, all_scores
        return all_results
