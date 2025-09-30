from __future__ import annotations

from typing import Dict, Any, List

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from transformers import PreTrainedTokenizerBase

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

            vocab_size = self.policy.config.vocab_size
            full = [
                tid if tid < vocab_size else self.tokenizer.unk_token_id for tid in full
            ]

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
            std_r = group_rewards.std() + 1e-8
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

    def create_loader(self, ds, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=lambda batch: self._tokenize_batch(batch),
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )

    def train(self, loader: DataLoader, epochs: int = 1) -> Dict[str, Any]:
        self.policy.train()
        steps = 0
        accum = {"loss": 0.0, "avg_reward": 0.0, "avg_advantage": 0.0, "avg_kl": 0.0}
        total_epochs = int(epochs)
        epoch_bar = tqdm(range(total_epochs), desc="GRPO:epochs")

        for e in epoch_bar:
            batch_bar = tqdm(
                loader, desc=f"GRPO:epoch {e + 1}/{total_epochs}", leave=False
            )
            for batch in batch_bar:
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

        n = max(1, steps)
        return {k: v / n for k, v in accum.items()} | {"steps": steps}
