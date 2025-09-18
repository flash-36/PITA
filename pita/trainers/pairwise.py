from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import PreTrainedTokenizerBase
from datasets import load_from_disk, Dataset


@dataclass
class PairExample:
    prompt: str
    chosen: str
    rejected: str


class PreferencePairDataset(TorchDataset):
    """Loads the HF dataset saved by PostTrainingAlgorithms.generate_data and
    exposes (prompt, chosen, rejected) pairs based on the 'preferred' field.
    """

    def __init__(self, hf_dir: Path):
        self.hf_dir = Path(hf_dir)
        self.ds: Dataset = load_from_disk(str(self.hf_dir))

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> PairExample:
        ex = self.ds[int(idx)]
        prompt = str(ex["prompt"])  # strict
        y_a = str(ex["y_a"])  # strict
        y_b = str(ex["y_b"])  # strict
        preferred = int(ex["preferred"])  # strict
        if preferred == 0:
            chosen, rejected = y_a, y_b
        else:
            chosen, rejected = y_b, y_a
        return PairExample(prompt=prompt, chosen=chosen, rejected=rejected)


class PairwiseTrainerBase:
    """Reusable utilities for pairwise preference training.

    Handles tokenization, collation, log-prob computation over response tokens,
    and a basic training loop skeleton to be used by concrete trainers.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        use_chat_template: bool = True,
        dtype: str = None,
        batch_size: int = 2,
        num_workers: int = 2,
    ) -> None:
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.tokenizer.padding_side = "left"  # more memory efficient for causal LM

        self.use_chat_template = bool(use_chat_template)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.dtype_str = str(dtype)

    def build_prompt(self, instruction: str) -> str:
        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": instruction}]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return instruction

    def _tokenize_batch(
        self, batch: List[PairExample], max_length: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        # Build prompt+response sequences for chosen and rejected
        prompts = [self.build_prompt(ex.prompt) for ex in batch]
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
        logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)  # [B, T-1, V]
        target = input_ids[:, 1:]  # predict token t given t-1
        mask = response_mask[:, 1:].float()

        gathered = logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
        masked = gathered * mask
        seq_logps = masked.sum(dim=-1)  # [B]
        return seq_logps

    def create_loader(self, ds: TorchDataset, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=lambda batch: self._tokenize_batch(batch),
            pin_memory=True,
        )
