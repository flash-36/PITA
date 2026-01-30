from __future__ import annotations

from typing import Dict, Any, List, Iterator
import os
import time
import random

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, BatchSampler
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from transformers import PreTrainedTokenizerBase
from loguru import logger

from pita.models.hf import HFModel
from pita.core.prompts import build_instruction_prompt


class GRPOTrainer:
    def __init__(
        self,
        policy: HFModel,
        reference: HFModel,
        grpo_cfg: Any = None,
        use_chat_template: bool = True,
        micro_batch_size: int = 0,
        amp_dtype: str | None = None,
        clear_cache_interval: int = 0,
        max_batch_num_tokens: int = -1,
    ) -> None:
        self.cfg = grpo_cfg
        self.tokenizer: PreTrainedTokenizerBase = policy.tokenizer
        self.tokenizer.padding_side = "left"
        self.use_chat_template = bool(use_chat_template)

        self.policy = policy.model
        self.reference = reference.model
        for p in self.reference.parameters():
            p.requires_grad_(False)

        self.optimizer = AdamW(
            (p for p in self.policy.parameters() if p.requires_grad),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
        )

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        self.amp_dtype = dtype_map[str(amp_dtype or policy.gen_cfg.dtype)]
        self.autocast_enabled = torch.cuda.is_available() and self.amp_dtype in (
            torch.bfloat16,
            torch.float16,
        )
        self.scaler = GradScaler(
            device="cuda",
            enabled=self.autocast_enabled and self.amp_dtype == torch.float16,
        )
        self.micro_batch_size = int(micro_batch_size)
        self.clear_cache_interval = int(clear_cache_interval)
        self.batch_size = int(grpo_cfg.batch_size)
        self.num_workers = int(grpo_cfg.num_workers)
        self.max_batch_num_tokens = int(max_batch_num_tokens)

    def _tokenize_batch(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        prompts = [
            build_instruction_prompt(
                ex.prompt,
                tokenizer=self.tokenizer,
                use_chat_template=self.use_chat_template,
            )
            for ex in batch
        ]

        responses = [ex.response for ex in batch]
        rewards = torch.tensor([ex.reward for ex in batch], dtype=torch.float32)

        max_length = getattr(self.policy.config, "max_position_embeddings", 2048)

        tok_prompts = self.tokenizer(
            prompts,
            add_special_tokens=False,
            return_tensors=None,
        )
        tok_responses = self.tokenizer(
            responses,
            add_special_tokens=False,
            return_tensors=None,
        )

        full_inputs: List[List[int]] = []
        response_masks: List[List[int]] = []

        for i in range(len(batch)):
            p_ids = tok_prompts["input_ids"][i]
            r_ids = tok_responses["input_ids"][i]

            full = p_ids + r_ids
            mask = [0] * len(p_ids) + [1] * len(r_ids)

            if len(full) > max_length:
                full = full[-max_length:]
                mask = mask[-max_length:]

            # Use tokenizer vocab_size instead of config (more portable across model types)
            vocab_size = len(self.tokenizer)
            unk_id = self.tokenizer.unk_token_id or 0
            full = [tid if tid < vocab_size else unk_id for tid in full]

            full_inputs.append(full)
            response_masks.append(mask)

        batch_tensors = self.tokenizer.pad(
            {"input_ids": full_inputs},
            padding=True,
            return_tensors="pt",
        )

        attention_mask = batch_tensors["attention_mask"].long()

        def pad_mask(masks: List[List[int]], target_len: int) -> torch.Tensor:
            padded: List[List[int]] = []
            for m in masks:
                pad_amt = target_len - len(m)
                if pad_amt > 0:
                    padded.append([0] * pad_amt + m)
                else:
                    padded.append(m)
            return torch.tensor(padded, dtype=torch.long)

        resp_mask = pad_mask(response_masks, batch_tensors["input_ids"].shape[1])

        return {
            "input_ids": batch_tensors["input_ids"],
            "attention_mask": attention_mask,
            "response_mask": resp_mask,
            "rewards": rewards,
            "group_ids": torch.tensor([ex.group_id for ex in batch], dtype=torch.long),
        }

    @staticmethod
    def _compute_logps(
        logits: torch.Tensor, input_ids: torch.Tensor, response_mask: torch.Tensor
    ) -> torch.Tensor:
        logprobs = torch.log_softmax(logits.float()[:, :-1, :], dim=-1)
        target = input_ids[:, 1:]
        mask = response_mask[:, 1:].float()

        gathered = logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        masked = gathered * mask
        seq_logps = masked.sum(dim=-1)
        return seq_logps

    def grpo_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        device = self.policy.device
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            with autocast(
                device_type="cuda", dtype=self.amp_dtype, enabled=self.autocast_enabled
            ):
                ref_logits = self.reference(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                ).logits
            ref_logps = self._compute_logps(
                ref_logits, batch["input_ids"], batch["response_mask"]
            )

        with autocast(
            device_type="cuda", dtype=self.amp_dtype, enabled=self.autocast_enabled
        ):
            pol_logits = self.policy(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            ).logits

        pol_logps = self._compute_logps(
            pol_logits, batch["input_ids"], batch["response_mask"]
        )

        rewards = batch["rewards"].float()
        group_ids = batch["group_ids"]

        unique_groups = group_ids.unique()
        advantages = torch.zeros_like(rewards)

        for gid in unique_groups:
            mask = group_ids == gid
            group_rewards = rewards[mask]
            mean_r = group_rewards.mean()
            std_r = group_rewards.std(unbiased=False).clamp_min(1e-8)
            advantages[mask] = (group_rewards - mean_r) / std_r

        kl_penalty = float(self.cfg.kl_coef) * (pol_logps - ref_logps)
        policy_loss = -(pol_logps * advantages.detach() - kl_penalty)
        loss = policy_loss.mean()

        with torch.no_grad():
            stats = {
                "loss": float(loss.item()),
                "avg_reward": float(rewards.mean().item()),
                "avg_advantage": float(advantages.mean().item()),
                "avg_kl": float((pol_logps - ref_logps).mean().item()),
                "avg_pol_logp": float(pol_logps.mean().item()),
            }

        return {"loss": loss, "stats": stats}

    def _collate_wrapper(self, batch):
        """Wrapper method for collate_fn that can be pickled for multiprocessing."""
        result = self._tokenize_batch(batch)
        # Ensure tensors are on CPU to avoid CUDA IPC issues with multiprocessing
        if self.num_workers > 0:
            result = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in result.items()
            }
        return result

    class DynamicBatchSampler(BatchSampler):
        """Dynamic batch sampler that adjusts batch size based on sequence length."""

        def __init__(
            self,
            dataset,
            tokenizer: PreTrainedTokenizerBase,
            use_chat_template: bool,
            max_batch_size: int,
            max_batch_num_tokens: int,
            *,
            shuffle: bool = True,
        ) -> None:
            self.dataset = dataset
            self.tokenizer = tokenizer
            self.use_chat_template = bool(use_chat_template)
            self.max_batch_size = int(max_batch_size)
            self.max_tokens = int(max_batch_num_tokens)
            self.shuffle = bool(shuffle)
            self.epoch = 0

        def __len__(self) -> int:
            # Rough estimate: conservative lower bound
            return max(1, len(self.dataset) // self.max_batch_size)

        def set_epoch(self, epoch: int) -> None:
            self.epoch = int(epoch)

        def _estimate_tokens(self, idx: int) -> int:
            """Estimate tokens for a single example."""
            try:
                ex = self.dataset[int(idx)]
                # Estimate: prompt + response
                prompt = build_instruction_prompt(
                    ex.prompt,
                    tokenizer=self.tokenizer,
                    use_chat_template=self.use_chat_template,
                )
                p_result = self.tokenizer(prompt, add_special_tokens=False)
                r_result = self.tokenizer(ex.response, add_special_tokens=False)
                p_ids = p_result["input_ids"] if "input_ids" in p_result else []
                r_ids = r_result["input_ids"] if "input_ids" in r_result else []
                total = len(p_ids) + len(r_ids)
                # Safety: return at least 1 to avoid division by zero issues
                return max(total, 1)
            except Exception as e:
                # If estimation fails, return a conservative estimate
                logger.warning(
                    f"Failed to estimate tokens for idx {idx}: {e}, using default 512"
                )
                return 512

        def __iter__(self) -> Iterator[List[int]]:
            n = len(self.dataset)
            indices = list(range(n))
            if self.shuffle:
                rng = random.Random(self.epoch)
                rng.shuffle(indices)

            batch: List[int] = []
            cur_tokens = 0
            for idx in indices:
                est = self._estimate_tokens(idx)
                # If adding this example would exceed limits, yield current batch
                if batch and (
                    cur_tokens + est > self.max_tokens
                    or len(batch) >= self.max_batch_size
                ):
                    yield batch
                    batch = []
                    cur_tokens = 0
                batch.append(idx)
                cur_tokens += est
            # Yield remaining batch
            if batch:
                yield batch

    def create_loader(self, ds, shuffle: bool = True) -> DataLoader:
        # In parallel mode, disable workers to avoid CUDA IPC issues
        # Each job already has dedicated GPU resources
        num_workers = (
            0 if os.environ.get("PITA_PARALLEL_MODE") == "1" else self.num_workers
        )

        if self.max_batch_num_tokens > 0:
            # Use dynamic batching based on token count
            logger.info(
                f"GRPOTrainer: Using dynamic batching with max_batch_num_tokens={self.max_batch_num_tokens}, "
                f"max_batch_size={self.batch_size}"
            )
            sampler = self.DynamicBatchSampler(
                ds,
                tokenizer=self.tokenizer,
                use_chat_template=self.use_chat_template,
                max_batch_size=self.batch_size,
                max_batch_num_tokens=self.max_batch_num_tokens,
                shuffle=shuffle,
            )
            return DataLoader(
                ds,
                batch_sampler=sampler,
                num_workers=num_workers,
                collate_fn=self._collate_wrapper,
                pin_memory=True,
                persistent_workers=False,
            )
        else:
            # Use fixed batch size (original behavior)
            logger.info(f"GRPOTrainer: Using fixed batch size={self.batch_size}")
            return DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=self._collate_wrapper,
                pin_memory=True,
                persistent_workers=False,
            )

    def train(self, loader: DataLoader, epochs: int = 1) -> Dict[str, Any]:
        self.policy.train()
        steps = 0
        accum = {"loss": 0.0, "avg_reward": 0.0, "avg_advantage": 0.0, "avg_kl": 0.0}
        total_epochs = int(epochs)
        start_time = time.perf_counter()
        epoch_bar = tqdm(range(total_epochs), desc="GRPO:epochs")
        for e in epoch_bar:
            total_batches = len(loader)
            batch_bar = tqdm(
                loader, desc=f"GRPO:epoch {e + 1}/{total_epochs}", leave=False
            )
            for i, batch in enumerate(batch_bar):
                remaining = total_batches - (i + 1)
                if i % 10 == 0 or remaining == 0:
                    logger.info(f"⏳ Epoch {e+1}/{total_epochs} | Batch {i+1}/{total_batches} | {remaining} batches remaining")
                
                if (
                    self.micro_batch_size
                    and batch["input_ids"].size(0) > self.micro_batch_size
                ):
                    n = batch["input_ids"].size(0)
                    last_stats = None
                    for s in range(0, n, self.micro_batch_size):
                        e2 = min(s + self.micro_batch_size, n)
                        sub = {k: v[s:e2] for k, v in batch.items()}
                        out = self.grpo_step(sub)
                        loss: torch.Tensor = out["loss"] * ((e2 - s) / n)
                        if self.scaler.is_enabled():
                            self.scaler.scale(loss).backward()
                        else:
                            loss.backward()
                        last_stats = out["stats"]
                    stats = last_stats or {
                        "loss": float(loss.detach().item()),
                        "avg_reward": 0.0,
                    }
                else:
                    out = self.grpo_step(batch)
                    loss: torch.Tensor = out["loss"]
                    stats = out["stats"]
                    if self.scaler.is_enabled():
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.cfg.grad_clip
                    )

                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                steps += 1
                for k in accum:
                    accum[k] += stats.get(k, 0.0)

                batch_bar.set_postfix(
                    loss=stats["loss"],
                    reward=stats["avg_reward"],
                    kl=stats["avg_kl"],
                )

                if self.clear_cache_interval > 0 and (
                    steps % self.clear_cache_interval == 0
                ):
                    torch.cuda.empty_cache()

            avg_metrics = {k: v / max(1, steps) for k, v in accum.items()}
            epoch_bar.set_postfix(
                avg_loss=f"{avg_metrics['loss']:.4f}",
                avg_r=f"{avg_metrics['avg_reward']:.4f}",
            )

        elapsed = time.perf_counter() - start_time
        n = max(1, steps)
        return {k: v / n for k, v in accum.items()} | {
            "steps": steps,
            "train_time_seconds": round(elapsed, 2),
        }
