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
        batch_size: int = 8,
        max_new_tokens: Optional[int] = None,
    ) -> List[str]:
        """Generate n sequences with guidance using batched generation."""
        all_outputs = []
        remaining = n

        while remaining > 0:
            current_batch = min(batch_size, remaining)
            proc = self._build_processor()
            self._reset_state(proc)

            outputs = self.ref.generate_n(
                prompt,
                current_batch,
                greedy=greedy,
                logits_processor=proc,
                batch_size=current_batch,
                max_new_tokens=max_new_tokens,
            )
            all_outputs.extend(outputs)
            remaining -= current_batch

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
        proc = self._build_processor()
        self._reset_state(proc)
        return self.ref.continue_from_context(
            context_text, max_new_tokens, greedy, logits_processor=proc
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
        return_scores: bool = False,
    ) -> List[str] | tuple[List[str], List[torch.Tensor]]:
        """Batch continuation WITH guidance.

        Processes contexts in batches with classifier guidance applied at each step.
        The LogitsProcessor handles the entire batch in parallel for efficiency.

        Args:
            return_scores: If True, also return the guided logits/scores at each step.
                          These are the scores AFTER applying the classifier guidance.
        """
        all_results = []
        all_scores = [] if return_scores else None

        for i in range(0, len(contexts), batch_size):
            batch_contexts = contexts[i : i + batch_size]

            # Build ONE processor for this batch - it handles all examples in parallel
            proc = self._build_processor()
            self._reset_state(proc)

            if return_scores:
                texts, scores = self.ref.continue_from_context_batch(
                    batch_contexts,
                    max_new_tokens,
                    greedy,
                    logits_processor=proc,
                    batch_size=len(batch_contexts),
                    return_scores=True,
                )
                all_results.extend(texts)
                all_scores.extend(scores if scores else [None] * len(texts))
            else:
                texts = self.ref.continue_from_context_batch(
                    batch_contexts,
                    max_new_tokens,
                    greedy,
                    logits_processor=proc,
                    batch_size=len(batch_contexts),
                    return_scores=False,
                )
                all_results.extend(texts)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if return_scores:
            return all_results, all_scores
        return all_results
