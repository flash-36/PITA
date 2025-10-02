from __future__ import annotations

from typing import Any, Dict, List, Optional, Iterator

import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader, Sampler, BatchSampler, DistributedSampler
from transformers import PreTrainedTokenizerBase
from pita.core.prompts import build_instruction_prompt
import random


class PairwiseTrainerBase:
    """Reusable utilities for pairwise preference training.

    Handles tokenization, collation, log-prob computation over response tokens,
    and a basic training loop skeleton to be used by concrete trainers.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        use_chat_template: bool,
        dtype: str,
        batch_size: int,
        num_workers: int,
    ) -> None:
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.tokenizer.padding_side = "left"

        self.use_chat_template = bool(use_chat_template)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.dtype_str = str(dtype)

    class DynamicBatchSampler(BatchSampler):
        def __init__(
            self,
            dataset: TorchDataset,
            tokenizer: PreTrainedTokenizerBase,
            use_chat_template: bool,
            max_batch_num_tokens: int,
            *,
            shuffle: bool = True,
        ) -> None:
            self.dataset = dataset
            self.tokenizer = tokenizer
            self.use_chat_template = bool(use_chat_template)
            self.max_tokens = int(max_batch_num_tokens)
            self.shuffle = bool(shuffle)
            self.epoch = 0

        def __len__(self) -> int:
            # Rough estimate: conservative lower bound assuming each example at least 1 token
            return max(1, len(self.dataset))

        def set_epoch(self, epoch: int) -> None:
            self.epoch = int(epoch)

        def _estimate_tokens(self, idx: int) -> int:
            ex = self.dataset[int(idx)]
            prompt = build_instruction_prompt(
                ex.prompt, tokenizer=self.tokenizer, use_chat_template=self.use_chat_template
            )
            # Estimate by token length of prompt+max(chosen,rejected)
            p_ids = self.tokenizer(prompt, add_special_tokens=False).get("input_ids", [])
            c_ids = self.tokenizer(ex.chosen, add_special_tokens=False).get("input_ids", [])
            r_ids = self.tokenizer(ex.rejected, add_special_tokens=False).get("input_ids", [])
            resp_len = max(len(c_ids), len(r_ids))
            return len(p_ids) + resp_len

        def __iter__(self) -> Iterator[List[int]]:
            n = len(self.dataset)
            indices = list(range(n))
            if self.shuffle:
                rng = random.Random(self.epoch)
                rng.shuffle(indices)
            # Shard indices across distributed ranks if initialized
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
                indices = indices[rank::world_size]

            batch: List[int] = []
            cur_tokens = 0
            for idx in indices:
                est = self._estimate_tokens(idx)
                if batch and cur_tokens + est > self.max_tokens:
                    yield batch
                    batch = []
                    cur_tokens = 0
                batch.append(idx)
                cur_tokens += est
            if batch:
                yield batch

    def _tokenize_batch(
        self, batch: List[Any], max_length: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        # Build prompt+response sequences for chosen and rejected
        prompts = [
            build_instruction_prompt(
                ex.prompt,
                tokenizer=self.tokenizer,
                use_chat_template=self.use_chat_template,
            )
            for ex in batch
        ]
        chosen_texts = [ex.chosen for ex in batch]
        rejected_texts = [ex.rejected for ex in batch]

        # Tokenize separately to compute response masks
        tok_prompts = self.tokenizer(
            prompts,
            add_special_tokens=False,
            return_tensors=None,
        )
        tok_chosen = self.tokenizer(
            chosen_texts, add_special_tokens=False, return_tensors=None
        )
        tok_rejected = self.tokenizer(
            rejected_texts, add_special_tokens=False, return_tensors=None
        )

        chosen_inputs: List[List[int]] = []
        rejected_inputs: List[List[int]] = []
        chosen_resp_mask: List[List[int]] = []
        rejected_resp_mask: List[List[int]] = []

        for i in range(len(batch)):
            p_ids = tok_prompts["input_ids"][i]
            c_ids = tok_chosen["input_ids"][i]
            r_ids = tok_rejected["input_ids"][i]

            ch = p_ids + c_ids
            rj = p_ids + r_ids

            ch_mask = [0] * len(p_ids) + [1] * len(c_ids)
            rj_mask = [0] * len(p_ids) + [1] * len(r_ids)

            if max_length is not None:
                ch = ch[-max_length:]
                rj = rj[-max_length:]
                # Align masks after truncation from the left
                ch_mask = ch_mask[-max_length:]
                rj_mask = rj_mask[-max_length:]

            chosen_inputs.append(ch)
            rejected_inputs.append(rj)
            chosen_resp_mask.append(ch_mask)
            rejected_resp_mask.append(rj_mask)

        batch_chosen = self.tokenizer.pad(
            {"input_ids": chosen_inputs},
            padding=True,
            return_tensors="pt",
        )
        batch_rejected = self.tokenizer.pad(
            {"input_ids": rejected_inputs},
            padding=True,
            return_tensors="pt",
        )

        # Use tokenizer-provided attention masks
        chosen_attn = batch_chosen["attention_mask"].long()
        rejected_attn = batch_rejected["attention_mask"].long()

        # Pad response masks to the same length as input_ids
        def pad_mask(masks: List[List[int]], target_len: int) -> torch.Tensor:
            padded: List[List[int]] = []
            for m in masks:
                pad_amt = target_len - len(m)
                if pad_amt > 0:
                    padded.append([0] * pad_amt + m)
                else:
                    padded.append(m)
            return torch.tensor(padded, dtype=torch.long)

        ch_resp_mask = pad_mask(chosen_resp_mask, batch_chosen["input_ids"].shape[1])
        rj_resp_mask = pad_mask(
            rejected_resp_mask, batch_rejected["input_ids"].shape[1]
        )

        return {
            "chosen_input_ids": batch_chosen["input_ids"],
            "chosen_attention_mask": chosen_attn,
            "chosen_response_mask": ch_resp_mask,
            "rejected_input_ids": batch_rejected["input_ids"],
            "rejected_attention_mask": rejected_attn,
            "rejected_response_mask": rj_resp_mask,
        }

    @staticmethod
    def _compute_logps(
        logits: torch.Tensor, input_ids: torch.Tensor, response_mask: torch.Tensor
    ) -> torch.Tensor:
        # logits: [B, T, V], input_ids: [B, T], response_mask: [B, T]
        logprobs = torch.log_softmax(logits.float()[:, :-1, :], dim=-1)  # [B, T-1, V]
        target = input_ids[:, 1:]  # predict token t given t-1
        mask = response_mask[:, 1:].float()

        gathered = logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
        masked = gathered * mask
        seq_logps = masked.sum(dim=-1)  # [B]
        return seq_logps

    def create_loader(self, ds: TorchDataset, shuffle: bool = True) -> DataLoader:
        max_tokens = int(getattr(self, "max_batch_num_tokens", -1) or -1)
        if max_tokens > 0:
            sampler = self.DynamicBatchSampler(
                ds,
                tokenizer=self.tokenizer,
                use_chat_template=self.use_chat_template,
                max_batch_num_tokens=max_tokens,
                shuffle=shuffle,
            )
            return DataLoader(
                ds,
                batch_sampler=sampler,
                num_workers=self.num_workers,
                collate_fn=lambda batch: self._tokenize_batch(batch),
                pin_memory=True,
                persistent_workers=(self.num_workers > 0),
            )
        else:
            sampler = None
            use_shuffle = shuffle
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                sampler = DistributedSampler(ds, shuffle=shuffle)
                use_shuffle = False
            return DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=use_shuffle,
                sampler=sampler,
                num_workers=self.num_workers,
                collate_fn=lambda batch: self._tokenize_batch(batch),
                pin_memory=True,
                persistent_workers=(self.num_workers > 0),
            )
