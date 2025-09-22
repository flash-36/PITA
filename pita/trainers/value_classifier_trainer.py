from __future__ import annotations

from typing import Dict, Any, List

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm


class ValueClassifierTrainer:
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
                outputs = self.classifier(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["rewards"],
                    loss_weights=batch["loss_weights"],
                    loss_mask=batch["loss_mask"],
                    return_dict=True,
                )
                loss = outputs.loss
                loss.backward()
                if self.grad_clip and self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.classifier.parameters(), self.grad_clip
                    )
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                steps += 1
                accum_loss += float(loss.item())
                batch_bar.set_postfix(loss=float(loss.item()))
            avg_loss = accum_loss / max(1, steps)
            epoch_bar.set_postfix(avg_loss=f"{avg_loss:.4f}")
        return {"loss": accum_loss / max(1, steps), "steps": steps}
