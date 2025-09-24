from __future__ import annotations

from typing import Any, Dict

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import GradScaler, autocast

from .pairwise import PairwiseTrainerBase


class RewardTrainer(PairwiseTrainerBase):
    def __init__(
        self,
        classifier,
        *,
        use_chat_template: bool,
        batch_size: int,
        num_workers: int,
        lr: float,
        weight_decay: float,
        grad_clip: float | int | None,
        micro_batch_size: int,
        amp_dtype: str,
        clear_cache_interval: int,
    ) -> None:
        super().__init__(
            tokenizer=classifier.tokenizer,
            use_chat_template=use_chat_template,
            dtype=str(getattr(classifier.backbone.config, "torch_dtype", "float32")),
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.classifier = classifier
        self.optimizer = AdamW(
            (p for p in self.classifier.parameters() if p.requires_grad),
            lr=float(lr),
            weight_decay=float(weight_decay),
        )
        self.grad_clip = float(grad_clip) if grad_clip else 0.0
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

    def create_loader(self, ds, shuffle: bool = True) -> DataLoader:
        return super().create_loader(ds, shuffle=shuffle)

    def _reward_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        device = self.classifier.device
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast(
            device_type="cuda", dtype=self.amp_dtype, enabled=self.autocast_enabled
        ):
            loss_pos = self.classifier.compute_loss(
                input_ids=batch["chosen_input_ids"],
                attention_mask=batch["chosen_attention_mask"],
                labels=torch.ones(
                    (batch["chosen_input_ids"].size(0),),
                    dtype=torch.float32,
                    device=device,
                ),
                loss_weights=torch.ones(
                    (batch["chosen_input_ids"].size(0),),
                    dtype=torch.float32,
                    device=device,
                ),
                loss_mask=batch["chosen_response_mask"],
                use_cache=False,
            )
            loss_neg = self.classifier.compute_loss(
                input_ids=batch["rejected_input_ids"],
                attention_mask=batch["rejected_attention_mask"],
                labels=torch.zeros(
                    (batch["rejected_input_ids"].size(0),),
                    dtype=torch.float32,
                    device=device,
                ),
                loss_weights=torch.ones(
                    (batch["rejected_input_ids"].size(0),),
                    dtype=torch.float32,
                    device=device,
                ),
                loss_mask=batch["rejected_response_mask"],
                use_cache=False,
            )
            loss = 0.5 * (loss_pos + loss_neg)
        return {"loss": loss, "stats": {"loss": float(loss.detach().item())}}

    def train(self, loader: DataLoader, epochs: int = 1) -> Dict[str, Any]:
        self.classifier.train()
        steps = 0
        accum = {"loss": 0.0}
        total_epochs = int(epochs)
        epoch_bar = tqdm(range(total_epochs), desc="RM:epochs")
        for e in epoch_bar:
            batch_bar = tqdm(
                loader, desc=f"RM:epoch {e + 1}/{total_epochs}", leave=False
            )
            for batch in batch_bar:
                if (
                    self.micro_batch_size
                    and batch["chosen_input_ids"].size(0) > self.micro_batch_size
                ):
                    n = batch["chosen_input_ids"].size(0)
                    for s in range(0, n, self.micro_batch_size):
                        e2 = min(s + self.micro_batch_size, n)
                        sub = {k: v[s:e2] for k, v in batch.items()}
                        out = self._reward_step(sub)
                        loss: torch.Tensor = out["loss"] * ((e2 - s) / n)
                        if self.scaler.is_enabled():
                            self.scaler.scale(loss).backward()
                        else:
                            loss.backward()
                        last_loss = float(loss.detach().item())
                    stats = {"loss": last_loss}
                else:
                    out = self._reward_step(batch)
                    loss: torch.Tensor = out["loss"]
                    stats = out["stats"]
                    if self.scaler.is_enabled():
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

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
                accum["loss"] += stats["loss"]
                batch_bar.set_postfix(loss=stats["loss"])
                if self.clear_cache_interval > 0 and (
                    steps % self.clear_cache_interval == 0
                ):
                    torch.cuda.empty_cache()

            avg_loss = accum["loss"] / max(1, steps)
            epoch_bar.set_postfix(avg_loss=f"{avg_loss:.4f}")

        n = max(1, steps)
        return {"loss": accum["loss"] / n, "steps": steps}
