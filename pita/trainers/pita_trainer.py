from __future__ import annotations

from typing import Dict, Any, List, Iterator
import os
import time
import random

import torch
from torch.utils.data import DataLoader, BatchSampler
from torch.optim import AdamW
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from loguru import logger


class PITATrainer:
    def __init__(
        self,
        classifier,
        tokenizer,
        *,
        batch_size: int,
        num_workers: int,
        lr: float,
        weight_decay: float,
        grad_clip: float | int | None,
        pad_token_id: int,
        micro_batch_size: int,
        amp_dtype: str,
        clear_cache_interval: int,
        max_batch_num_tokens: int = -1,
        max_length: int = -1,
    ) -> None:
        self.classifier = classifier
        self.tokenizer = tokenizer
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.optimizer = AdamW(
            (p for p in self.classifier.parameters() if p.requires_grad),
            lr=float(lr),
            weight_decay=float(weight_decay),
        )
        self.grad_clip = float(grad_clip) if grad_clip else 0.0
        self.pad_token_id = int(pad_token_id)
        self.micro_batch_size = int(micro_batch_size)
        self.max_batch_num_tokens = int(max_batch_num_tokens)
        self.max_length = int(max_length)
        logger.info(
            f"PITATrainer initialized: batch_size={self.batch_size}, "
            f"micro_batch_size={self.micro_batch_size}, "
            f"max_batch_num_tokens={self.max_batch_num_tokens}, max_length={self.max_length}"
        )
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        self.amp_dtype = dtype_map[str(amp_dtype)]
        self.autocast_enabled = torch.cuda.is_available() and self.amp_dtype in (
            torch.bfloat16,
            torch.float16,
        )
        self.scaler = GradScaler(
            device="cuda",
            enabled=self.autocast_enabled and self.amp_dtype == torch.float16,
        )
        self.clear_cache_interval = int(clear_cache_interval)

    class DynamicBatchSampler(BatchSampler):
        """Dynamic batch sampler that adjusts batch size based on sequence length."""

        def __init__(
            self,
            dataset,
            max_batch_size: int,
            max_batch_num_tokens: int,
            *,
            shuffle: bool = True,
        ) -> None:
            self.dataset = dataset
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
                # Estimate: prompt + max(chosen, rejected)
                # Use dictionary access for HuggingFace datasets
                prompt_len = len(ex["input_ids"]) if "input_ids" in ex else 0
                chosen_len = (
                    len(ex["chosen_target_ids"]) if "chosen_target_ids" in ex else 0
                )
                rejected_len = (
                    len(ex["rejected_target_ids"]) if "rejected_target_ids" in ex else 0
                )
                total = prompt_len + max(chosen_len, rejected_len)
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
            if n == 0:
                logger.warning("DynamicBatchSampler: Empty dataset!")
                return

            # Test token estimation on first example
            try:
                test_est = self._estimate_tokens(0)
                logger.info(
                    f"DynamicBatchSampler: First example has ~{test_est} tokens "
                    f"(max_tokens={self.max_tokens}, max_batch_size={self.max_batch_size})"
                )
                if test_est == 0:
                    logger.error(
                        "DynamicBatchSampler: Token estimation returning 0! Falling back to fixed batching."
                    )
                    # Yield fixed-size batches as fallback
                    for i in range(0, n, self.max_batch_size):
                        yield list(range(i, min(i + self.max_batch_size, n)))
                    return
            except Exception as e:
                logger.error(f"DynamicBatchSampler: Token estimation failed: {e}")
                # Yield fixed-size batches as fallback
                for i in range(0, n, self.max_batch_size):
                    yield list(range(i, min(i + self.max_batch_size, n)))
                return

            indices = list(range(n))
            if self.shuffle:
                rng = random.Random(self.epoch)
                rng.shuffle(indices)

            batch: List[int] = []
            cur_tokens = 0
            batch_count = 0

            for idx in indices:
                est = self._estimate_tokens(idx)
                # If adding this example would exceed limits, yield current batch
                if batch and (
                    cur_tokens + est > self.max_tokens
                    or len(batch) >= self.max_batch_size
                ):
                    batch_count += 1
                    if batch_count <= 3:  # Log first 3 batches
                        logger.info(
                            f"DynamicBatch #{batch_count}: size={len(batch)}, "
                            f"est_tokens={cur_tokens}, avg_tokens_per_ex={cur_tokens//len(batch)}"
                        )
                    yield batch
                    batch = []
                    cur_tokens = 0
                batch.append(idx)
                cur_tokens += est
            # Yield remaining batch
            if batch:
                batch_count += 1
                if batch_count <= 3:
                    logger.info(
                        f"DynamicBatch #{batch_count}: size={len(batch)}, "
                        f"est_tokens={cur_tokens}, avg_tokens_per_ex={cur_tokens//len(batch)}"
                    )
                yield batch

            logger.info(f"DynamicBatchSampler created {batch_count} batches total")

    def create_loader(self, ds) -> DataLoader:
        # In parallel mode, disable workers to avoid CUDA IPC issues
        # Each job already has dedicated GPU resources
        num_workers = (
            0 if os.environ.get("PITA_PARALLEL_MODE") == "1" else self.num_workers
        )

        if self.max_batch_num_tokens > 0:
            # Use dynamic batching based on token count
            logger.info(
                f"PITATrainer: Using dynamic batching with max_batch_num_tokens={self.max_batch_num_tokens}, "
                f"max_batch_size={self.batch_size}"
            )
            sampler = self.DynamicBatchSampler(
                ds,
                max_batch_size=self.batch_size,
                max_batch_num_tokens=self.max_batch_num_tokens,
                shuffle=True,
            )
            return DataLoader(
                ds,
                batch_sampler=sampler,
                num_workers=num_workers,
                collate_fn=self._collate,
                pin_memory=True,
                persistent_workers=False,
            )
        else:
            # Use fixed batch size (original behavior)
            logger.info(f"PITATrainer: Using fixed batch size={self.batch_size}")
            return DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=self._collate,
                pin_memory=True,
                persistent_workers=False,
            )

    def _collate(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Optionally truncate sequences that exceed max_length (only if max_length > 0)
        max_before_truncation = 0
        truncation_count = 0
        for x in batch:
            total_chosen = len(x["input_ids"]) + len(x["chosen_target_ids"])
            total_rejected = len(x["input_ids"]) + len(x["rejected_target_ids"])
            max_before_truncation = max(
                max_before_truncation, total_chosen, total_rejected
            )

            if self.max_length > 0 and total_chosen > self.max_length:
                truncation_count += 1
                # Truncate from the end of target (keep prompt intact)
                overflow = total_chosen - self.max_length
                x["chosen_target_ids"] = (
                    x["chosen_target_ids"][:-overflow]
                    if overflow < len(x["chosen_target_ids"])
                    else x["chosen_target_ids"][:1]
                )

            if self.max_length > 0 and total_rejected > self.max_length:
                truncation_count += 1
                # Truncate from the end of target (keep prompt intact)
                overflow = total_rejected - self.max_length
                x["rejected_target_ids"] = (
                    x["rejected_target_ids"][:-overflow]
                    if overflow < len(x["rejected_target_ids"])
                    else x["rejected_target_ids"][:1]
                )

        # Log truncation statistics (only once per trainer to avoid spam)
        if truncation_count > 0 and not hasattr(self, "_truncation_logged"):
            logger.warning(
                f"PITA Truncation: {truncation_count}/{len(batch)*2} sequences truncated, "
                f"max_seq_len_before={max_before_truncation}, max_length={self.max_length}"
            )
            self._truncation_logged = True

        # Calculate max_len, optionally capped at max_length
        max_len_in_batch = max(
            len(x["input_ids"])
            + max(len(x["chosen_target_ids"]), len(x["rejected_target_ids"]))
            for x in batch
        )

        # Always log max sequence length in batch for debugging
        if not hasattr(self, "_last_logged_batch") or self._last_logged_batch != len(
            batch
        ):
            effective_len = (
                min(max_len_in_batch, self.max_length)
                if self.max_length > 0
                else max_len_in_batch
            )
            logger.debug(
                f"PITA Batch Info: batch_size={len(batch)}, max_seq_len={max_len_in_batch}, "
                f"padded_to={effective_len}"
            )
            self._last_logged_batch = len(batch)

        max_len = (
            min(max_len_in_batch, self.max_length)
            if self.max_length > 0
            else max_len_in_batch
        )
        chosen_input_ids = []
        rejected_input_ids = []
        chosen_attention_mask = []
        rejected_attention_mask = []
        chosen_response_mask = []
        rejected_response_mask = []
        for x in batch:
            pad_len_chosen = max_len - len(x["input_ids"]) - len(x["chosen_target_ids"])
            pad_len_rejected = (
                max_len - len(x["input_ids"]) - len(x["rejected_target_ids"])
            )

            chosen_input_ids.append(
                torch.cat(
                    [
                        torch.full(
                            (pad_len_chosen,), self.pad_token_id, dtype=torch.long
                        ),
                        torch.tensor(x["input_ids"], dtype=torch.long),
                        torch.tensor(x["chosen_target_ids"], dtype=torch.long),
                    ]
                )
            )
            rejected_input_ids.append(
                torch.cat(
                    [
                        torch.full(
                            (pad_len_rejected,), self.pad_token_id, dtype=torch.long
                        ),
                        torch.tensor(x["input_ids"], dtype=torch.long),
                        torch.tensor(x["rejected_target_ids"], dtype=torch.long),
                    ]
                )
            )

            chosen_attention_mask.append(
                torch.cat(
                    [
                        torch.zeros(pad_len_chosen, dtype=torch.long),
                        torch.ones(
                            len(x["input_ids"]) + len(x["chosen_target_ids"]),
                            dtype=torch.long,
                        ),
                    ]
                )
            )
            rejected_attention_mask.append(
                torch.cat(
                    [
                        torch.zeros(pad_len_rejected, dtype=torch.long),
                        torch.ones(
                            len(x["input_ids"]) + len(x["rejected_target_ids"]),
                            dtype=torch.long,
                        ),
                    ]
                )
            )

            chosen_response_mask.append(
                torch.cat(
                    [
                        torch.zeros(
                            pad_len_chosen + len(x["input_ids"]), dtype=torch.long
                        ),
                        torch.ones(len(x["chosen_target_ids"]), dtype=torch.long),
                    ]
                )
            )
            rejected_response_mask.append(
                torch.cat(
                    [
                        torch.zeros(
                            pad_len_rejected + len(x["input_ids"]), dtype=torch.long
                        ),
                        torch.ones(len(x["rejected_target_ids"]), dtype=torch.long),
                    ]
                )
            )

        return {
            "chosen_input_ids": torch.stack(chosen_input_ids),
            "rejected_input_ids": torch.stack(rejected_input_ids),
            "chosen_attention_mask": torch.stack(chosen_attention_mask),
            "rejected_attention_mask": torch.stack(rejected_attention_mask),
            "chosen_response_mask": torch.stack(chosen_response_mask),
            "rejected_response_mask": torch.stack(rejected_response_mask),
        }

    def train(self, loader: DataLoader, *, num_epochs: int) -> Dict[str, Any]:
        self.classifier.train()
        steps = 0
        accum_loss = 0.0
        start_time = time.perf_counter()
        epoch_bar = tqdm(range(int(num_epochs)), desc="PITA:epochs")
        for e in epoch_bar:
            total_batches = len(loader)
            batch_bar = tqdm(
                loader, desc=f"PITA:epoch {e + 1}/{int(num_epochs)}", leave=False
            )
            for i, batch in enumerate(batch_bar):
                remaining = total_batches - (i + 1)
                if i % 10 == 0 or remaining == 0:
                    logger.info(f"⏳ Epoch {e+1}/{num_epochs} | Batch {i+1}/{total_batches} | {remaining} batches remaining")
                
                device = self.classifier.device
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                input_ids = torch.cat(
                    [batch["chosen_input_ids"], batch["rejected_input_ids"]], dim=0
                )
                attention_mask = torch.cat(
                    [batch["chosen_attention_mask"], batch["rejected_attention_mask"]],
                    dim=0,
                )
                loss_mask = torch.cat(
                    [batch["chosen_response_mask"], batch["rejected_response_mask"]],
                    dim=0,
                ).float()
                num_chosen = batch["chosen_input_ids"].size(0)
                num_rejected = batch["rejected_input_ids"].size(0)
                labels = torch.cat(
                    [
                        torch.ones(num_chosen, device=device),
                        torch.zeros(num_rejected, device=device),
                    ],
                    dim=0,
                )
                loss_weights = torch.ones_like(labels)
                total_bs = input_ids.size(0)
                micro_bs = (
                    self.micro_batch_size if self.micro_batch_size > 0 else total_bs
                )
                last_loss_val = 0.0
                for start in range(0, total_bs, micro_bs):
                    end = min(start + micro_bs, total_bs)
                    mb_slice = slice(start, end)
                    scale = (end - start) / max(1, total_bs)
                    with autocast(
                        device_type="cuda",
                        dtype=self.amp_dtype,
                        enabled=self.autocast_enabled,
                    ):
                        outputs = self.classifier(
                            input_ids=input_ids[mb_slice],
                            attention_mask=attention_mask[mb_slice],
                            labels=labels[mb_slice],
                            loss_mask=loss_mask[mb_slice],
                            loss_weights=loss_weights[mb_slice],
                            return_dict=True,
                            use_cache=False,
                        )
                        loss = outputs.loss * scale
                    if self.scaler.is_enabled():
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    last_loss_val = float(loss.detach().item()) / max(scale, 1e-8)
                if self.grad_clip and self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.classifier.parameters(), self.grad_clip
                    )
                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                steps += 1
                accum_loss += float(last_loss_val)
                batch_bar.set_postfix(loss=float(last_loss_val))
                if self.clear_cache_interval > 0 and (
                    steps % self.clear_cache_interval == 0
                ):
                    torch.cuda.empty_cache()
            avg_loss = accum_loss / max(1, steps)
            epoch_bar.set_postfix(avg_loss=f"{avg_loss:.4f}")
        elapsed = time.perf_counter() - start_time
        return {
            "loss": accum_loss / max(1, steps),
            "steps": steps,
            "train_time_seconds": round(elapsed, 2),
        }
