from __future__ import annotations

from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import torch
from transformers.generation.logits_process import LogitsProcessorList

from pita.models.hf import HFModel
from pita.models.logits import CustomValueGuidedLogitProcessor
from pita.models.value_classifier import ValueClassifier
from pita.core.prompts import build_instruction_prompt


@dataclass
class GuidanceConfig:
    eta: float
    mode: str
    top_k: int
    use_cache: bool


class GuidedHFModel:
    def __init__(
        self, ref: HFModel, classifier: ValueClassifier, guidance: GuidanceConfig
    ) -> None:
        self.ref = ref
        self.classifier = classifier
        self.guidance = guidance
        self.tokenizer = ref.tokenizer
        self.gen_cfg = ref.gen_cfg
        self.eos_token_ids = ref.eos_token_ids
        self.pad_token_id = ref.pad_token_id

    @torch.inference_mode()
    def _build_processor(self) -> LogitsProcessorList:
        return LogitsProcessorList(
            [
                CustomValueGuidedLogitProcessor(
                    eta=self.guidance.eta,
                    ref_model=self.ref.model,
                    ref_model_tokenizer=self.tokenizer,
                    value_classifier=self.classifier,
                    inference_mode=self.guidance.mode,
                    top_k=self.guidance.top_k,
                    use_cache=self.guidance.use_cache,
                )
            ]
        )

    @staticmethod
    def _reset_state(proc: LogitsProcessorList) -> None:
        for p in proc:
            if hasattr(p, "reset_classifier_state"):
                p.reset_classifier_state()

    @torch.inference_mode()
    def generate_text(self, prompt: str) -> str:
        proc = self._build_processor()
        self._reset_state(proc)
        return self.ref.generate_text(prompt, logits_processor=proc)

    @torch.inference_mode()
    def generate_n(
        self,
        prompt: str,
        n: int,
        *,
        greedy: bool = False,
        batch_size: int = 1,
        max_new_tokens: Optional[int] = None,
    ) -> List[str]:
        """Generate n sequences with guidance. Uses batch_size=1 by default for memory efficiency."""
        all_outputs = []

        for i in range(n):
            proc = self._build_processor()
            self._reset_state(proc)
            outputs = self.ref.generate_n(
                prompt,
                1,
                greedy=greedy,
                logits_processor=proc,
                batch_size=1,
                max_new_tokens=max_new_tokens,
            )
            all_outputs.extend(outputs)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return all_outputs

    @torch.inference_mode()
    def roll_in(self, full_prompt: str, max_roll_tokens: int) -> Dict[str, Any]:
        proc = self._build_processor()
        self._reset_state(proc)
        return self.ref.roll_in(full_prompt, max_roll_tokens, logits_processor=proc)

    @torch.inference_mode()
    def continue_from_context(
        self, context_text: str, max_new_tokens: int, greedy: bool
    ) -> str:
        return self.ref.continue_from_context(
            context_text, max_new_tokens, greedy, logits_processor=None
        )

    @torch.inference_mode()
    def roll_in_batch(
        self, prompts: List[str], max_roll_tokens: int, batch_size: int = 8
    ) -> List[Dict[str, Any]]:
        """Batch greedy rollout with guidance."""
        proc = self._build_processor()
        self._reset_state(proc)
        return self.ref.roll_in_batch(
            prompts, max_roll_tokens, logits_processor=proc, batch_size=batch_size
        )

    @torch.inference_mode()
    def continue_from_context_batch(
        self,
        contexts: List[str],
        max_new_tokens: int,
        greedy: bool,
        batch_size: int = 8,
    ) -> List[str]:
        """Batch continuation without guidance."""
        return self.ref.continue_from_context_batch(
            contexts,
            max_new_tokens,
            greedy,
            logits_processor=None,
            batch_size=batch_size,
        )
