from __future__ import annotations

from typing import Dict, Any, List
import os

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from torch.amp import autocast, GradScaler


class QSharpTrainer:
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

    def create_loader(self, ds) -> DataLoader:
        # In parallel mode, disable workers to avoid CUDA IPC issues
        # Each job already has dedicated GPU resources
        num_workers = (
            0 if os.environ.get("PITA_PARALLEL_MODE") == "1" else self.num_workers
        )

        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self._collate,
            pin_memory=True,
            persistent_workers=False,  # Avoid CUDA tensor sharing issues
        )

    def _collate(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) + len(x["target_ids"]) for x in batch)
        padded_seq = []
        attention_mask = []
        loss_mask = []
        for x in batch:
            pad_len = max_len - len(x["input_ids"]) - len(x["target_ids"])
            padded_seq.append(
                torch.cat(
                    [
                        torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
                        torch.tensor(x["input_ids"], dtype=torch.long),
                        torch.tensor(x["target_ids"], dtype=torch.long),
                    ]
                )
            )
            attention_mask.append(
                torch.cat(
                    [
                        torch.zeros(pad_len, dtype=torch.long),
                        torch.ones(
                            len(x["input_ids"]) + len(x["target_ids"]), dtype=torch.long
                        ),
                    ]
                )
            )
            loss_mask.append(
                torch.cat(
                    [
                        torch.zeros(pad_len + len(x["input_ids"]), dtype=torch.long),
                        torch.ones(len(x["target_ids"]), dtype=torch.long),
                    ]
                )
            )
        return {
            "input_ids": torch.stack(padded_seq),
            "attention_mask": torch.stack(attention_mask),
            "loss_mask": torch.stack(loss_mask),
            "rewards": torch.tensor([x["rewards"] for x in batch]).float(),
            "loss_weights": torch.tensor([x["loss_weights"] for x in batch]).float(),
        }

    def train(self, loader: DataLoader, *, num_epochs: int) -> Dict[str, Any]:
        self.classifier.train()
        steps = 0
        accum_loss = 0.0
        epoch_bar = tqdm(range(int(num_epochs)), desc="Q#:epochs")
        for e in epoch_bar:
            batch_bar = tqdm(
                loader, desc=f"Q#:epoch {e + 1}/{int(num_epochs)}", leave=False
            )
            for batch in batch_bar:
                batch = {
                    k: (
                        v.to(self.classifier.device, non_blocking=True)
                        if isinstance(v, torch.Tensor)
                        else v
                    )
                    for k, v in batch.items()
                }
                total_bs = batch["input_ids"].size(0)
                micro_bs = (
                    self.micro_batch_size if self.micro_batch_size > 0 else total_bs
                )
                last_loss_val = 0.0
                for start in range(0, total_bs, micro_bs):
                    end = min(start + micro_bs, total_bs)
                    mb = slice(start, end)
                    scale = (end - start) / max(1, total_bs)
                    with autocast(
                        device_type="cuda",
                        dtype=self.amp_dtype,
                        enabled=self.autocast_enabled,
                    ):
                        outputs = self.classifier(
                            input_ids=batch["input_ids"][mb],
                            attention_mask=batch["attention_mask"][mb],
                            labels=batch["rewards"][mb],
                            loss_weights=batch["loss_weights"][mb],
                            loss_mask=batch["loss_mask"][mb],
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
        return {"loss": accum_loss / max(1, steps), "steps": steps}
