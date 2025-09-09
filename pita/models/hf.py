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
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
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
        )
        self.gen_cfg = gen_cfg

    def build_prompt(self, instruction: str) -> str:
        # Basic wrapper; can be extended per model with chat templates
        if self.gen_cfg.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "user", "content": instruction},
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return instruction

    @torch.inference_mode()
    def generate_text(self, prompt: str) -> str:
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **input_ids,
            max_new_tokens=self.gen_cfg.max_new_tokens,
            do_sample=True,
            temperature=self.gen_cfg.temperature,
            top_p=self.gen_cfg.top_p,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        # Return only the completion after the prompt
        return text[len(prompt):].strip()
