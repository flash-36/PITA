from __future__ import annotations

from typing import Any, Dict

import torch
from transformers.generation.logits_process import LogitsProcessorList

from .hf import HFModel
from .logits import CustomValueGuidedLogitProcessor
from .value_classifier import ValueClassifier
from pita.core.prompts import build_instruction_prompt


class GuidedHFModel:
    def __init__(
        self, ref: HFModel, classifier: ValueClassifier, guidance: Any
    ) -> None:
        self.ref = ref
        self.classifier = classifier
        self.guidance = guidance
        self.tokenizer = ref.tokenizer
        self.gen_cfg = ref.gen_cfg
        self.eos_token_ids = ref.eos_token_ids
        self.pad_token_id = ref.pad_token_id

    @torch.inference_mode()
    def generate_text(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.ref.model.device)
        proc = CustomValueGuidedLogitProcessor(
            eta=self.guidance.eta,
            ref_model=self.ref.model,
            ref_model_tokenizer=self.tokenizer,
            value_classifier=self.classifier,
            inference_mode=self.guidance.mode,
            top_k=self.guidance.top_k,
            use_cache=self.guidance.use_cache,
        )
        proc.reset_classifier_state()
        outputs = self.ref.model.generate(
            **inputs,
            logits_processor=LogitsProcessorList([proc]),
            pad_token_id=self.pad_token_id,
            max_new_tokens=self.gen_cfg.max_new_tokens,
            do_sample=True,
            temperature=self.gen_cfg.temperature,
            top_p=self.gen_cfg.top_p,
            eos_token_id=self.eos_token_ids,
        )
        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    @torch.inference_mode()
    def roll_in(self, full_prompt: str, max_roll_tokens: int) -> Dict[str, Any]:
        built = build_instruction_prompt(
            full_prompt,
            tokenizer=self.tokenizer,
            use_chat_template=self.gen_cfg.use_chat_template,
        )
        ids = self.tokenizer(built, return_tensors="pt").to(self.ref.model.device)
        proc = CustomValueGuidedLogitProcessor(
            eta=self.guidance.eta,
            ref_model=self.ref.model,
            ref_model_tokenizer=self.tokenizer,
            value_classifier=self.classifier,
            inference_mode=self.guidance.mode,
            top_k=self.guidance.top_k,
            use_cache=self.guidance.use_cache,
        )
        proc.reset_classifier_state()
        out = self.ref.model.generate(
            **ids,
            logits_processor=LogitsProcessorList([proc]),
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
        ids = self.tokenizer(context_text, return_tensors="pt").to(
            self.ref.model.device
        )
        proc = CustomValueGuidedLogitProcessor(
            eta=self.guidance.eta,
            ref_model=self.ref.model,
            ref_model_tokenizer=self.tokenizer,
            value_classifier=self.classifier,
            inference_mode=self.guidance.mode,
            top_k=self.guidance.top_k,
            use_cache=self.guidance.use_cache,
        )
        proc.reset_classifier_state()
        out = self.ref.model.generate(
            **ids,
            logits_processor=LogitsProcessorList([proc]),
            max_new_tokens=max_new_tokens,
            do_sample=not greedy,
            temperature=self.gen_cfg.temperature if not greedy else 1.0,
            top_p=self.gen_cfg.top_p if not greedy else 1.0,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_ids,
        )
        new_tokens = out[0][ids["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
