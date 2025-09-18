from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .registry import resolve_model_id
from pita.core.prompts import build_instruction_prompt


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.9
    use_chat_template: bool = True
    dtype: str = "bfloat16"


class HFModel:
    def __init__(self, name_or_id: str, gen_cfg: GenerationConfig):
        model_id = resolve_model_id(name_or_id)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            trust_remote_code=True,
        )
        added_special_tokens = False
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
        }.get(gen_cfg.dtype, torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
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
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(eos, list) and len(eos) > 0:
            return eos[0]
        return eos

    @torch.inference_mode()
    def generate_text(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=self.gen_cfg.max_new_tokens,
            do_sample=True,
            temperature=self.gen_cfg.temperature,
            top_p=self.gen_cfg.top_p,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_ids,
        )
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    @torch.inference_mode()
    def roll_in(self, full_prompt: str, max_roll_tokens: int) -> Dict[str, Any]:
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
            max_new_tokens=max_roll_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_ids,
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
        self, context_text: str, max_new_tokens: int, greedy: bool
    ) -> str:
        ids = self.tokenizer(context_text, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=not greedy,
            temperature=self.gen_cfg.temperature if not greedy else 1.0,
            top_p=self.gen_cfg.top_p if not greedy else 1.0,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_ids,
        )
        new_tokens = out[0][ids["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
