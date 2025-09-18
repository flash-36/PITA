from __future__ import annotations

from typing import Dict, Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .pairwise import PairwiseTrainerBase
from pita.models.hf import HFModel


class DPOTrainer(PairwiseTrainerBase):
    def __init__(
        self,
        policy: HFModel,
        reference: HFModel,
        dpo_cfg: Any = None,
        use_chat_template: bool = True,
    ) -> None:
        self.cfg = dpo_cfg
        super().__init__(
            tokenizer=policy.tokenizer,
            use_chat_template=use_chat_template,
            batch_size=dpo_cfg.batch_size,
            num_workers=dpo_cfg.num_workers,
            dtype=policy.gen_cfg.dtype,
        )

        self.policy = policy.model
        self.reference = reference.model
        for p in self.reference.parameters():
            p.requires_grad_(False)

        self.optimizer = AdamW(
            (p for p in self.policy.parameters() if p.requires_grad),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
        )

    def dpo_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        device = self.policy.device
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward passes for chosen and rejected
        with torch.no_grad():
            ref_chosen = self.reference(
                input_ids=batch["chosen_input_ids"],
                attention_mask=batch["chosen_attention_mask"],
            ).logits
            ref_rejected = self.reference(
                input_ids=batch["rejected_input_ids"],
                attention_mask=batch["rejected_attention_mask"],
            ).logits

        pol_chosen = self.policy(
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
        ).logits
        pol_rejected = self.policy(
            input_ids=batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"],
        ).logits

        # Compute log-probs over response tokens
        ref_lp_ch = self._compute_logps(
            ref_chosen, batch["chosen_input_ids"], batch["chosen_response_mask"]
        )
        ref_lp_rj = self._compute_logps(
            ref_rejected, batch["rejected_input_ids"], batch["rejected_response_mask"]
        )
        pol_lp_ch = self._compute_logps(
            pol_chosen, batch["chosen_input_ids"], batch["chosen_response_mask"]
        )
        pol_lp_rj = self._compute_logps(
            pol_rejected, batch["rejected_input_ids"], batch["rejected_response_mask"]
        )

        # DPO loss: -log sigma(beta * (Δπ - Δref))
        delta_ref = ref_lp_ch - ref_lp_rj
        delta_pol = pol_lp_ch - pol_lp_rj
        beta = self.cfg.beta
        loss_vec = -torch.log(torch.sigmoid(beta * (delta_pol - delta_ref)))
        loss = loss_vec.mean()

        # Stats
        with torch.no_grad():
            acc = (delta_pol > delta_ref).float().mean()
            stats = {
                "loss": float(loss.item()),
                "acc": float(acc.item()),
                "pol_lp_ch": float(pol_lp_ch.mean().item()),
                "pol_lp_rj": float(pol_lp_rj.mean().item()),
                "ref_lp_ch": float(ref_lp_ch.mean().item()),
                "ref_lp_rj": float(ref_lp_rj.mean().item()),
            }
        return {"loss": loss, "stats": stats}

    def train(self, loader: DataLoader, epochs: int = 1) -> Dict[str, Any]:
        self.policy.train()
        steps = 0
        accum = {"loss": 0.0, "acc": 0.0}
        total_epochs = int(epochs)
        epoch_bar = tqdm(range(total_epochs), desc="DPO:epochs")
        for e in epoch_bar:
            batch_bar = tqdm(
                loader, desc=f"DPO:epoch {e + 1}/{total_epochs}", leave=False
            )
            for batch in batch_bar:
                out = self.dpo_step(batch)
                loss: torch.Tensor = out["loss"]
                stats = out["stats"]

                loss.backward()
                if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.cfg.grad_clip
                    )
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                steps += 1
                accum["loss"] += stats["loss"]
                accum["acc"] += stats["acc"]
                batch_bar.set_postfix(loss=stats["loss"], acc=stats["acc"])

            avg_loss = accum["loss"] / max(1, steps)
            avg_acc = accum["acc"] / max(1, steps)
            epoch_bar.set_postfix(avg_loss=f"{avg_loss:.4f}", avg_acc=f"{avg_acc:.4f}")

        n = max(1, steps)
        return {"loss": accum["loss"] / n, "acc": accum["acc"] / n, "steps": steps}
