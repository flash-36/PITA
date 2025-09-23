from __future__ import annotations

from typing import Dict, Any, List

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from torch.amp import autocast, GradScaler


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
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )

    def _collate(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(
            len(x["input_ids"])
            + max(len(x["chosen_target_ids"]), len(x["rejected_target_ids"]))
            for x in batch
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
        epoch_bar = tqdm(range(int(num_epochs)), desc="PITA:epochs")
        for e in epoch_bar:
            batch_bar = tqdm(
                loader, desc=f"PITA:epoch {e + 1}/{int(num_epochs)}", leave=False
            )
            for batch in batch_bar:
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
        return {"loss": accum_loss / max(1, steps), "steps": steps}
