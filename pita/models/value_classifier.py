from __future__ import annotations

from typing import Any, Optional, Union
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import SequenceClassifierOutput
from pita.models.catalog import resolve_model_id


class ValueClassifier(nn.Module):
    def __init__(
        self,
        name_or_id: str,
        tokenizer: Any,
        device: Optional[Union[str, torch.device]] = None,
        *,
        loss_type: str,
        num_atoms: int,
        V_min: float,
        V_max: float,
        attn_impl: str,
        dtype: str | None,
        gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        assert loss_type in ("bce", "mse", "mle")
        self.tokenizer = tokenizer
        model_id = resolve_model_id(name_or_id)
        torch_dtype = None
        if dtype:
            torch_dtype = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }.get(str(dtype), None)
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        if device is not None:
            self.backbone = self.backbone.to(device)
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        self.loss_type = loss_type
        hidden_size = int(getattr(self.backbone.config, "hidden_size"))

        if self.loss_type == "mle":
            self.num_atoms = int(num_atoms)
            self.V_min = float(V_min)
            self.V_max = float(V_max)
            self.atoms = torch.linspace(self.V_min, self.V_max, self.num_atoms).float()

        out_dim = 1 if self.loss_type != "mle" else self.num_atoms
        # Ensure the classifier head matches the backbone parameter dtype/device
        backbone_param = next(self.backbone.parameters())
        self.score = nn.Linear(
            hidden_size,
            out_dim,
            bias=True,
            device=backbone_param.device,
            dtype=backbone_param.dtype,
        )
        if device is not None:
            # Keep explicit device move for safety in case backbone moved after init
            self.score = self.score.to(device)

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    def _calculate_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_weights: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        bs, seqlen = loss_mask.shape
        denom = loss_mask.sum(dim=-1).clamp_min(1)
        if self.loss_type == "mse":
            relevant_logits = torch.sigmoid(logits).squeeze(-1)
            labels_expanded = labels.unsqueeze(1).expand(-1, seqlen)
            loss = F.mse_loss(
                relevant_logits,
                labels_expanded.to(relevant_logits.dtype),
                reduction="none",
            )
            loss = (loss * loss_mask).sum(dim=-1) / denom
        elif self.loss_type == "bce":
            logits = logits.squeeze(-1)
            labels_expanded = labels.unsqueeze(1).expand(-1, seqlen)
            loss = F.binary_cross_entropy_with_logits(
                logits, labels_expanded, reduction="none"
            )
            loss = (loss * loss_mask).sum(dim=-1) / denom
        else:
            log_pmfs = F.log_softmax(logits, dim=-1)
            label_indices = torch.round(labels * (self.num_atoms - 1)).long()
            label_indices = torch.clamp(label_indices, 0, self.num_atoms - 1)
            loss = -log_pmfs[torch.arange(bs), :, label_indices]
            loss = (loss * loss_mask).sum(dim=-1) / denom
        return (loss * loss_weights).mean()

    def compute_loss(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        labels: torch.FloatTensor,
        loss_weights: torch.FloatTensor,
        loss_mask: torch.Tensor,
        use_cache: Optional[bool] = None,
    ) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        logits = self.score(hidden)
        return self._calculate_loss(logits, labels, loss_weights, loss_mask)

    def score_candidates(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        logit_indices: torch.LongTensor,
        use_cache: Optional[bool] = None,
        past_key_values: Optional[Any] = None,
    ) -> Any:
        pad_id = getattr(self.tokenizer, "pad_token_id", 0)
        bs, seq_len = input_ids.shape
        top_k = logit_indices.size(1)
        base = input_ids
        expanded = (
            base.unsqueeze(1)
            .expand(bs, top_k, seq_len)
            .contiguous()
            .view(bs * top_k, seq_len)
        )
        cand = logit_indices.reshape(bs * top_k, 1)
        expanded_ids = torch.cat([expanded, cand.to(expanded.device)], dim=1)
        if attention_mask is not None:
            base_mask = (
                attention_mask.unsqueeze(1)
                .expand(bs, top_k, seq_len)
                .contiguous()
                .view(bs * top_k, seq_len)
            )
            expanded_mask = torch.cat(
                [
                    base_mask,
                    torch.ones(
                        (bs * top_k, 1), dtype=base_mask.dtype, device=base_mask.device
                    ),
                ],
                dim=1,
            )
        else:
            expanded_mask = expanded_ids.ne(pad_id).long()
        out = self.backbone(
            input_ids=expanded_ids,
            attention_mask=expanded_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = out.hidden_states[-1][:, -1]
        cand_logits = self.score(last_hidden)
        if self.loss_type == "mle":
            cand_logits = cand_logits.view(bs, top_k, self.num_atoms)
        else:
            cand_logits = cand_logits.view(bs, top_k)
        return SimpleNamespace(
            logits=cand_logits, past_key_values=getattr(out, "past_key_values", None)
        )

    def forward(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        loss_weights: Optional[torch.FloatTensor] = None,
        logit_indices: Optional[torch.LongTensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: bool = True,
    ) -> SequenceClassifierOutput:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        logits = self.score(hidden)
        if labels is not None:
            return SequenceClassifierOutput(
                loss=self._calculate_loss(logits, labels, loss_weights, loss_mask),
                logits=logits,
            )
        # inference
        if logit_indices is not None:
            return self.score_candidates(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=use_cache,
                past_key_values=past_key_values,
                logit_indices=logit_indices,
            )
        last = logits[:, -1]
        if self.loss_type == "mle":
            return SequenceClassifierOutput(
                logits=last.view(last.size(0), self.num_atoms)
            )
        return SequenceClassifierOutput(logits=last)
