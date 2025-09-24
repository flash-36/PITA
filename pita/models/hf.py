from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.logits_process import LogitsProcessorList

from pita.models.catalog import resolve_model_id
from pita.core.prompts import build_instruction_prompt


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
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
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
        return {
            "max_new_tokens": (
                self.gen_cfg.max_new_tokens
                if max_new_tokens is None
                else int(max_new_tokens)
            ),
            "do_sample": (not greedy),
            "temperature": (1.0 if greedy else self.gen_cfg.temperature),
            "top_p": (1.0 if greedy else self.gen_cfg.top_p),
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_ids,
            "logits_processor": logits_processor,
        }

    @torch.inference_mode()
    def generate_text(
        self,
        prompt: str,
        *,
        greedy: bool = False,
        logits_processor: Optional[LogitsProcessorList] = None,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **inputs,
            **self._gen_kwargs(greedy=greedy, logits_processor=logits_processor),
        )
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
    ) -> List[str]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            **self._gen_kwargs(greedy=greedy, logits_processor=logits_processor),
            num_return_sequences=int(n),
        )
        seq_len = inputs["input_ids"].shape[1]
        return [
            self.tokenizer.decode(seq[seq_len:], skip_special_tokens=True).strip()
            for seq in out
        ]

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
        out = self.model.generate(
            **ids,
            **self._gen_kwargs(
                greedy=True,
                logits_processor=logits_processor,
                max_new_tokens=max_roll_tokens,
            ),
        )
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
        out = self.model.generate(
            **ids,
            **self._gen_kwargs(
                greedy=greedy,
                logits_processor=logits_processor,
                max_new_tokens=max_new_tokens,
            ),
        )
        new_tokens = out[0][ids["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
