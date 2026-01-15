from __future__ import annotations

from typing import Any, Optional, Union
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import SequenceClassifierOutput
from pita.models.catalog import resolve_model_id


def _expand_cache_for_candidates(cache: Any, bs: int, top_k: int) -> Any:
    """Expand a KV cache for parallel candidate scoring.

    Handles:
    - Legacy tuple format
    - DynamicCache
    - HybridCache (Gemma 3)
    """
    if cache is None:
        return None

    # Handle HybridCache (Gemma 3) specifically if possible
    cache_class_name = type(cache).__name__
    if cache_class_name == "HybridCache":
        try:
            from transformers.cache_utils import HybridCache
            
            # HybridCache needs the original config to maintain sliding window settings
            expanded = HybridCache(
                config=cache.config,
                batch_size=bs * top_k,
                max_cache_len=cache.max_cache_len,
                device=cache.key_cache[0].device if cache.key_cache else None,
                dtype=cache.key_cache[0].dtype if cache.key_cache else None,
            )
            # Copy over the current sequence length metadata
            if hasattr(cache, "seen_tokens"):
                expanded.seen_tokens = cache.seen_tokens
                
            for layer_idx in range(len(cache.key_cache)):
                key = cache.key_cache[layer_idx]
                value = cache.value_cache[layer_idx]
                if key.numel() == 0:
                    continue
                
                shape = list(key.shape)
                exp_key = (
                    key.unsqueeze(1)
                    .expand(bs, top_k, *shape[1:])
                    .reshape(bs * top_k, *shape[1:])
                )
                exp_value = (
                    value.unsqueeze(1)
                    .expand(bs, top_k, *shape[1:])
                    .reshape(bs * top_k, *shape[1:])
                )
                expanded.update(exp_key, exp_value, layer_idx)
            return expanded
        except (ImportError, AttributeError, RuntimeError):
            pass

    # Handle legacy tuple format (tuple of tuples)
    if isinstance(cache, tuple):
        try:
            from transformers.cache_utils import DynamicCache
            expanded = DynamicCache()
            for layer_idx, layer_kv in enumerate(cache):
                if isinstance(layer_kv, tuple) and len(layer_kv) == 2:
                    key, value = layer_kv
                    shape = list(key.shape)
                    exp_key = (
                        key.unsqueeze(1)
                        .expand(bs, top_k, *shape[1:])
                        .reshape(bs * top_k, *shape[1:])
                    )
                    exp_value = (
                        value.unsqueeze(1)
                        .expand(bs, top_k, *shape[1:])
                        .reshape(bs * top_k, *shape[1:])
                    )
                    expanded.update(exp_key, exp_value, layer_idx)
                else:
                    raise ImportError("Cannot convert to DynamicCache")
            return expanded
        except (ImportError, AttributeError):
            expanded_list = []
            for layer_kv in cache:
                if isinstance(layer_kv, tuple) and len(layer_kv) == 2:
                    key, value = layer_kv
                    shape = list(key.shape)
                    exp_key = (
                        key.unsqueeze(1)
                        .expand(bs, top_k, *shape[1:])
                        .reshape(bs * top_k, *shape[1:])
                    )
                    exp_value = (
                        value.unsqueeze(1)
                        .expand(bs, top_k, *shape[1:])
                        .reshape(bs * top_k, *shape[1:])
                    )
                    expanded_list.append((exp_key, exp_value))
                else:
                    expanded_list.append(layer_kv)
            return tuple(expanded_list)

    # Handle modern Cache objects (DynamicCache, etc.)
    key_cache = getattr(cache, "key_cache", None) or getattr(cache, "_key_cache", None)
    value_cache = getattr(cache, "value_cache", None) or getattr(
        cache, "_value_cache", None
    )

    if key_cache is not None and value_cache is not None:
        try:
            from transformers.cache_utils import DynamicCache
            expanded = DynamicCache()
            for layer_idx in range(len(key_cache)):
                key = key_cache[layer_idx]
                value = value_cache[layer_idx]
                shape = list(key.shape)
                exp_key = (
                    key.unsqueeze(1)
                    .expand(bs, top_k, *shape[1:])
                    .reshape(bs * top_k, *shape[1:])
                )
                exp_value = (
                    value.unsqueeze(1)
                    .expand(bs, top_k, *shape[1:])
                    .reshape(bs * top_k, *shape[1:])
                )
                expanded.update(exp_key, exp_value, layer_idx)
            return expanded
        except ImportError:
            pass

    if hasattr(cache, "to_legacy_cache"):
        legacy = cache.to_legacy_cache()
        return _expand_cache_for_candidates(legacy, bs, top_k)

    return cache


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
        assert loss_type in ("bce", "mse", "mle", "bradley_terry")
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
        backbone_param = next(self.backbone.parameters())
        self.score = nn.Linear(
            hidden_size,
            out_dim,
            bias=True,
            device=backbone_param.device,
            dtype=backbone_param.dtype,
        )
        if device is not None:
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
        elif self.loss_type == "bradley_terry":
            logits_squeezed = logits.squeeze(-1)
            reward_scores = (logits_squeezed * loss_mask).sum(dim=-1) / denom
            half_bs = bs // 2
            chosen_rewards = reward_scores[:half_bs]
            rejected_rewards = reward_scores[half_bs:]
            logits_diff = chosen_rewards - rejected_rewards
            loss = F.binary_cross_entropy_with_logits(
                logits_diff,
                torch.ones_like(logits_diff),
                reduction="none",
            )
            loss_weights_chosen = loss_weights[:half_bs]
            return (loss * loss_weights_chosen).mean()
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
        """Score candidate tokens for guided generation.

        This method uses a two-phase approach:
        1. Process/update the prefix to get prefix KV cache
        2. Score all candidates in parallel using the prefix cache
        """
        bs, seq_len = input_ids.shape
        top_k = logit_indices.size(1)

        # Phase 1: Process prefix to get/update prefix KV cache
        prefix_out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=past_key_values,
            output_hidden_states=False,
            return_dict=True,
        )
        prefix_pkv = prefix_out.past_key_values
        del prefix_out

        # Determine past_len for position_ids
        if prefix_pkv is None:
            past_len = 0
        elif hasattr(prefix_pkv, "get_seq_length"):
            past_len = prefix_pkv.get_seq_length()
        else:
            # Fallback for legacy tuple cache [layer][key/value][bs, heads, seq, dim]
            try:
                past_len = prefix_pkv[0][0].shape[-2]
            except (IndexError, TypeError, AttributeError):
                past_len = seq_len

        # Phase 2: Score all candidates in parallel using prefix cache
        expanded_pkv = _expand_cache_for_candidates(prefix_pkv, bs, top_k)

        # Candidate tokens as input
        candidate_ids = logit_indices.reshape(bs * top_k, 1)

        # Explicit position_ids for correct RoPE embeddings
        candidate_positions = torch.full(
            (bs * top_k, 1),
            past_len,
            dtype=torch.long,
            device=self.device,
        )

        # Check if this is a HybridCache (Gemma 3 with sliding window attention)
        cache_class_name = type(prefix_pkv).__name__
        is_hybrid_cache = cache_class_name == "HybridCache"

        if is_hybrid_cache:
            # For HybridCache: don't pass attention_mask, let model compute internally
            # This avoids mismatch between mask size and sliding window cache size
            expanded_mask = None
        elif attention_mask is not None:
            # For regular caches: use 4D mask for SDPA robustness
            expanded_mask = (
                attention_mask.unsqueeze(1)
                .expand(bs, top_k, past_len)
                .reshape(bs * top_k, past_len)
            )
            expanded_mask = torch.cat(
                [
                    expanded_mask,
                    torch.ones(
                        (bs * top_k, 1),
                        dtype=expanded_mask.dtype,
                        device=expanded_mask.device,
                    ),
                ],
                dim=1,
            )
            # Convert to 4D for SDPA: [batch, 1, 1, kv_len] broadcasts across heads
            # SDPA expects additive float mask: 0.0 = attend, -inf = don't attend
            dtype = self.backbone.dtype
            expanded_mask = expanded_mask.view(bs * top_k, 1, 1, past_len + 1)
            expanded_mask = expanded_mask.to(dtype=dtype)
            expanded_mask = (1.0 - expanded_mask) * torch.finfo(dtype).min
        else:
            expanded_mask = None

        # Run candidates through model with explicit positions
        cand_out = self.backbone(
            input_ids=candidate_ids,
            attention_mask=expanded_mask,
            position_ids=candidate_positions,
            use_cache=False,
            past_key_values=expanded_pkv,
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden = cand_out.hidden_states[-1][:, -1]
        logits = self.score(last_hidden)

        if self.loss_type == "mle":
            logits = logits.view(bs, top_k, self.num_atoms)
        else:
            logits = logits.view(bs, top_k)

        return SimpleNamespace(logits=logits, past_key_values=prefix_pkv)

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
