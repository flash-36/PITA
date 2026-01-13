from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, Any, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.logits_process import LogitsProcessorList
from tqdm import tqdm
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
        self.gen_cfg = gen_cfg
        self.eos_token_ids = self._compute_eos_token_ids()
        self.pad_token_id = self._compute_pad_token_id()

    def _compute_eos_token_ids(self) -> List[int]:
        eos_ids: List[int] = []
        base_eos = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(base_eos, int):
            eos_ids.append(base_eos)
        unk_id = getattr(self.tokenizer, "unk_token_id", None)
        for tok in [
            "<end_of_turn>",
            "<|end_of_turn|>",
            "<|eot_id|>",
            "<|im_end|>",
            "<|end|>",
        ]:
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if (
                isinstance(tid, int)
                and tid >= 0
                and tid != unk_id
                and tid not in eos_ids
            ):
                eos_ids.append(tid)
        return eos_ids

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
        temperature = 1.0 if greedy else max(0.5, min(2.0, self.gen_cfg.temperature))
        top_p = 1.0 if greedy else max(0.9, min(0.99, self.gen_cfg.top_p))

        return {
            "max_new_tokens": (
                self.gen_cfg.max_new_tokens
                if max_new_tokens is None
                else int(max_new_tokens)
            ),
            "do_sample": (not greedy),
            "temperature": temperature,
            "top_p": top_p,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_ids,
            "logits_processor": logits_processor,
        }

    def _generate_with_fallback(
        self, inputs: Dict[str, torch.Tensor], gen_kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        """Generate with fp32 greedy fallback on device-assert."""
        try:
            return self.model.generate(**inputs, **gen_kwargs)
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
        ids = self.tokenizer(built, return_tensors="pt").to(self.model.device)
        gen_kwargs = self._gen_kwargs(
            greedy=True,
            logits_processor=logits_processor,
            max_new_tokens=max_roll_tokens,
        )
        out = self._generate_with_fallback(ids, gen_kwargs)
        context_tokens = out[0]
        context_text = self.tokenizer.decode(context_tokens, skip_special_tokens=True)
        return {
            "prompt": built,
            "context_ids": context_tokens,
            "context_text": context_text,
        }

    @torch.inference_mode()
    def continue_from_context(
        self,
        context_text: str,
        max_new_tokens: int,
        greedy: bool,
        logits_processor: Optional[LogitsProcessorList] = None,
    ) -> str:
        ids = self.tokenizer(context_text, return_tensors="pt").to(self.model.device)
        gen_kwargs = self._gen_kwargs(
            greedy=greedy,
            logits_processor=logits_processor,
            max_new_tokens=max_new_tokens,
        )
        out = self._generate_with_fallback(ids, gen_kwargs)
        new_tokens = out[0][ids["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    @torch.inference_mode()
    def roll_in_batch(
        self,
        prompts: List[str],
        max_roll_tokens: int,
        logits_processor: Optional[LogitsProcessorList] = None,
        batch_size: int = 8,
    ) -> List[Dict[str, Any]]:
        """Batch greedy rollout for multiple prompts."""
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

        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        for i in tqdm(
            range(0, len(built_prompts), batch_size),
            total=num_batches,
            desc="roll_in batches",
        ):
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

            for j, (seq, prompt_len) in enumerate(zip(out, prompt_lengths)):
                context_text = self.tokenizer.decode(seq, skip_special_tokens=True)
                all_results.append(
                    {
                        "prompt": batch[j],
                        "context_ids": seq,
                        "context_text": context_text,
                    }
                )

            del out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.tokenizer.padding_side = original_padding_side
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
        all_results = []
        all_scores = [] if return_scores else None
        num_batches = (len(contexts) + batch_size - 1) // batch_size
        mode = "greedy" if greedy else "sampled"

        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        for i in tqdm(
            range(0, len(contexts), batch_size),
            total=num_batches,
            desc=f"continue ({mode}) batches",
        ):
            batch = contexts[i : i + batch_size]
            ids = self.tokenizer(batch, return_tensors="pt", padding=True).to(
                self.model.device
            )

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
                # Stack scores: [num_steps, batch_size, vocab_size] -> per-example [num_steps, vocab_size]
                if out.scores:
                    stacked_scores = torch.stack(out.scores, dim=0)  # [num_steps, batch_size, vocab_size]
                    stacked_scores = stacked_scores.transpose(0, 1)  # [batch_size, num_steps, vocab_size]
                else:
                    stacked_scores = None
            else:
                sequences = out

            for j, (seq, prompt_len) in enumerate(zip(sequences, prompt_lengths)):
                new_tokens = seq[prompt_len:]
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
                        # Append None to maintain alignment with results
                        all_scores.append(None)

            del out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.tokenizer.padding_side = original_padding_side
        
        if return_scores:
            return all_results, all_scores
        return all_results
