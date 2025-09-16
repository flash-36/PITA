from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .registry import resolve_model_id


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
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }.get(gen_cfg.dtype, torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.gen_cfg = gen_cfg

    def build_prompt(self, instruction: str) -> str:
        # Basic wrapper; can be extended per model with chat templates
        if self.gen_cfg.use_chat_template and hasattr(
            self.tokenizer, "apply_chat_template"
        ):
            messages = [
                {"role": "user", "content": instruction},
            ]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return instruction

    @torch.inference_mode()
    def generate_text(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=self.gen_cfg.max_new_tokens,
            do_sample=True,
            temperature=self.gen_cfg.temperature,
            top_p=self.gen_cfg.top_p,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    @torch.inference_mode()
    def roll_in(self, full_prompt: str, max_roll_tokens: int) -> Dict[str, Any]:
        """Greedy rollout for t steps to obtain context s_t.

        Returns dict with keys: prompt, context_ids, context_text.
        """
        built = self.build_prompt(full_prompt)
        ids = self.tokenizer(built, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **ids,
            max_new_tokens=max_roll_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        tokens = out[0]
        eos_id = self.tokenizer.eos_token_id
        prompt_len = ids["input_ids"].shape[1]
        end = tokens.shape[0]
        while (
            end > prompt_len and eos_id is not None and tokens[end - 1].item() == eos_id
        ):
            end -= 1
        context_tokens = tokens[:end]
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
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = out[0][ids["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
