from __future__ import annotations

from typing import Dict, Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler

from .pairwise import PairwiseTrainerBase
from pita.models.hf import HFModel


class DPOTrainer(PairwiseTrainerBase):
    def __init__(
        self,
        policy: HFModel,
        reference: HFModel,
        dpo_cfg: Any = None,
        use_chat_template: bool = True,
        micro_batch_size: int = 0,
        amp_dtype: str | None = None,
        clear_cache_interval: int = 0,
        grad_accumulation_steps: int | None = None,
        warmup_steps: int | None = None,
        save_dir: str | None = None,
        ckpt_freq: int | None = None,
        eval_freq: int | None = None,
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
        # Scheduler: linear warmup then constant
        warmup = int(warmup_steps or getattr(self.cfg, "warmup_steps", 0) or 0)
        if warmup > 0:
            self.scheduler = LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: min(1.0, (step + 1) / max(1, warmup)),
            )
        else:
            self.scheduler = None
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
        self.grad_accumulation_steps = int(
            grad_accumulation_steps
            or getattr(self.cfg, "gradient_accumulation_steps", 1)
            or 1
        )
        self.save_dir = str(save_dir) if save_dir else None
        self.ckpt_freq = int(ckpt_freq or getattr(self.cfg, "ckpt_freq", -1) or -1)
        self.eval_freq = int(eval_freq or getattr(self.cfg, "eval_freq", -1) or -1)
        self.max_batch_num_tokens = int(
            getattr(self.cfg, "max_batch_num_tokens", -1) or -1
        )

    def dpo_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        device = self.policy.device
        # Move batch to device
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        # Forward passes for chosen and rejected
        with torch.no_grad():
            with autocast(
                device_type="cuda", dtype=self.amp_dtype, enabled=self.autocast_enabled
            ):
                ref_chosen = self.reference(
                    input_ids=batch["chosen_input_ids"],
                    attention_mask=batch["chosen_attention_mask"],
                    use_cache=False,
                ).logits
                ref_rejected = self.reference(
                    input_ids=batch["rejected_input_ids"],
                    attention_mask=batch["rejected_attention_mask"],
                    use_cache=False,
                ).logits

        with autocast(
            device_type="cuda", dtype=self.amp_dtype, enabled=self.autocast_enabled
        ):
            pol_chosen = self.policy(
                input_ids=batch["chosen_input_ids"],
                attention_mask=batch["chosen_attention_mask"],
                use_cache=False,
            ).logits
            pol_rejected = self.policy(
                input_ids=batch["rejected_input_ids"],
                attention_mask=batch["rejected_attention_mask"],
                use_cache=False,
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

        # DPO loss (float32): softplus for numerical stability
        delta_ref = (ref_lp_ch - ref_lp_rj).float()
        delta_pol = (pol_lp_ch - pol_lp_rj).float()
        beta = float(self.cfg.beta)
        loss_vec = torch.nn.functional.softplus(-beta * (delta_pol - delta_ref))
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
        # Explicitly free larger intermediates before returning
        del ref_chosen, ref_rejected, pol_chosen, pol_rejected
        torch.cuda.empty_cache()
        return {"loss": loss, "stats": stats}

    def train(self, loader: DataLoader, epochs: int = 1) -> Dict[str, Any]:
        self.policy.train()
        steps = 0
        opt_steps = 0
        accum = {"loss": 0.0, "acc": 0.0}
        total_epochs = int(epochs)
        epoch_bar = tqdm(range(total_epochs), desc="DPO:epochs")
        for e in epoch_bar:
            bs = getattr(loader, "batch_sampler", None)
            if hasattr(bs, "set_epoch"):
                bs.set_epoch(e)
            smp = getattr(loader, "sampler", None)
            if hasattr(smp, "set_epoch"):
                smp.set_epoch(e)
            batch_bar = tqdm(
                loader,
                desc=f"DPO:epoch {e + 1}/{total_epochs}",
                leave=False,
            )
            for batch in batch_bar:
                # Optional micro-batching at DPO step granularity
                if (
                    self.micro_batch_size
                    and batch["chosen_input_ids"].size(0) > self.micro_batch_size
                ):
                    n = batch["chosen_input_ids"].size(0)
                    last_stats = None
                    for s in range(0, n, self.micro_batch_size):
                        e2 = min(s + self.micro_batch_size, n)
                        sub = {k: v[s:e2] for k, v in batch.items()}
                        out = self.dpo_step(sub)
                        loss: torch.Tensor = out["loss"] * ((e2 - s) / n)
                        self.accelerator.backward(loss)
                        last_stats = out["stats"]
                    stats = last_stats or {
                        "loss": float(loss.detach().item()),
                        "acc": 0.0,
                    }
                else:
                    out = self.dpo_step(batch)
                    loss: torch.Tensor = out["loss"]
                    stats = out["stats"]

                    # Backward with gradient scaling
                    if self.scaler.is_enabled():
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                # Gradient accumulation and step
                if self.grad_accumulation_steps > 1:
                    should_step = (steps + 1) % self.grad_accumulation_steps == 0
                else:
                    should_step = True

                if should_step:
                    if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                        if self.scaler.is_enabled():
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.policy.parameters(), self.cfg.grad_clip
                        )

                    if self.scaler.is_enabled():
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    opt_steps += 1

                steps += 1
                accum["loss"] += stats["loss"]
                accum["acc"] += stats["acc"]
                batch_bar.set_postfix(loss=stats["loss"], acc=stats["acc"])
                if self.clear_cache_interval > 0 and (
                    steps % self.clear_cache_interval == 0
                ):
                    torch.cuda.empty_cache()
                # Periodic checkpoint
                if (
                    self.save_dir
                    and self.ckpt_freq > 0
                    and (opt_steps > 0)
                    and (opt_steps % self.ckpt_freq == 0)
                ):
                    from pathlib import Path

                    ckpt_dir = Path(self.save_dir) / f"ckpt_{opt_steps}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    self.policy.save_pretrained(str(ckpt_dir))
                    self.tokenizer.save_pretrained(str(ckpt_dir))

            avg_loss = accum["loss"] / max(1, steps)
            avg_acc = accum["acc"] / max(1, steps)
            epoch_bar.set_postfix(avg_loss=f"{avg_loss:.4f}", avg_acc=f"{avg_acc:.4f}")

        n = max(1, steps)
        return {"loss": accum["loss"] / n, "acc": accum["acc"] / n, "steps": steps}
