from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor


def _log1p_exp(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


class CustomValueGuidedLogitProcessor(LogitsProcessor):
    def __init__(
        self,
        *,
        eta: float,
        ref_model: Any,
        ref_model_tokenizer: Any,
        value_classifier: Any,
        inference_mode: str,
        top_k: int,
        use_cache: bool = True,
        cd_baseline: int = 0,
    ) -> None:
        self.eta = float(eta)
        self.ref_model = ref_model
        self.ref_model_tokenizer = ref_model_tokenizer
        self.inference_mode = str(inference_mode)
        self.modify_top_k = int(top_k)
        self.cd_baseline = int(cd_baseline)
        self.value_classifier = value_classifier
        self.loss_type = getattr(value_classifier, "loss_type", "bce")
        self.use_cache = bool(use_cache)
        self.classifier_state = {
            "input_ids": None,
            "attention_mask": None,
            "use_cache": self.use_cache,
            "past_key_values": None,
            "first_pass": True,
        }

    def reset_classifier_state(self) -> None:
        self.classifier_state = {
            "input_ids": None,
            "attention_mask": None,
            "use_cache": self.use_cache,
            "past_key_values": None,
            "first_pass": True,
        }

    @torch.no_grad()
    def _get_classifier_values(
        self, input_ids: torch.Tensor, top_k_indices: torch.Tensor
    ) -> torch.Tensor:
        if self.classifier_state["first_pass"]:
            self.classifier_state["first_pass"] = False
            self.classifier_state["input_ids"] = input_ids
            pad_token_id = self.ref_model_tokenizer.pad_token_id
            attention_mask = input_ids.ne(pad_token_id).long()
            self.classifier_state["attention_mask"] = attention_mask.to(input_ids.dtype)
        else:
            attention_mask = torch.cat(
                [
                    self.classifier_state["attention_mask"],
                    torch.ones_like(input_ids[:, -1:], dtype=torch.long),
                ],
                dim=1,
            )
            if not self.classifier_state["use_cache"]:
                input_ids = torch.cat(
                    [self.classifier_state["input_ids"], input_ids[:, -1:]], dim=1
                )
            else:
                input_ids = input_ids[:, -1:]
            self.classifier_state["input_ids"] = input_ids
            self.classifier_state["attention_mask"] = attention_mask

        outputs = self.value_classifier.score_candidates(
            input_ids=input_ids,
            attention_mask=self.classifier_state["attention_mask"],
            use_cache=self.classifier_state["use_cache"],
            logit_indices=top_k_indices,
            past_key_values=self.classifier_state["past_key_values"],
        )
        if self.classifier_state["use_cache"]:
            self.classifier_state["past_key_values"] = outputs.past_key_values
        return outputs.logits

    def _modify_top_k(
        self,
        ref_logits: torch.Tensor,
        logit_offset: torch.Tensor,
        top_k_indices: torch.Tensor,
    ) -> torch.Tensor:
        return torch.scatter_add(
            ref_logits, 1, top_k_indices.to(ref_logits.device), logit_offset
        )

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if self.inference_mode == "disabled":
            return scores

        if self.modify_top_k == -1:
            top_k_indices = (
                torch.arange(scores.size(-1), device=scores.device)
                .unsqueeze(0)
                .expand(scores.size(0), -1)
            )
        else:
            _, top_k_indices = torch.topk(scores, self.modify_top_k, dim=-1)

        if self.loss_type == "mle":
            classifier_logits = self._get_classifier_values(
                input_ids, top_k_indices
            ).float()
            log_pmfs = F.log_softmax(classifier_logits, dim=-1)
            atoms = self.value_classifier.atoms.float()
            if atoms.device != log_pmfs.device:
                atoms = atoms.to(log_pmfs.device)
            logit_offset = torch.logsumexp(log_pmfs + self.eta * atoms, dim=-1)
            logit_offset = logit_offset - logit_offset.min(dim=-1, keepdim=True).values
            logit_offset = torch.nan_to_num(
                logit_offset, nan=0.0, posinf=0.0, neginf=0.0
            )
            return self._modify_top_k(scores, logit_offset, top_k_indices)

        classifier_logits = self._get_classifier_values(
            input_ids, top_k_indices
        ).float()
        if self.inference_mode == "expectation":
            if self.cd_baseline:
                logit_offset = self.eta * torch.sigmoid(classifier_logits)
            else:
                logit_offset = self.eta * classifier_logits
            logit_offset = torch.nan_to_num(
                logit_offset, nan=0.0, posinf=0.0, neginf=0.0
            )
            return self._modify_top_k(scores, logit_offset, top_k_indices)

        if self.inference_mode == "bernoulli":
            if self.cd_baseline:
                logit_offset = self.eta * torch.sigmoid(classifier_logits)
            else:
                log_numerator = _log1p_exp(self.eta + classifier_logits)
                log_denominator = _log1p_exp(classifier_logits)
                logit_offset = log_numerator - log_denominator
            logit_offset = torch.nan_to_num(
                logit_offset, nan=0.0, posinf=0.0, neginf=0.0
            )
            return self._modify_top_k(scores, logit_offset, top_k_indices)

        return scores
