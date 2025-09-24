from __future__ import annotations

from typing import Any, Dict, Iterable, List

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import GradScaler, autocast

from pita.models.hf import HFModel
from pita.core.prompts import build_instruction_prompt


class GRPOTrainer:
    def __init__(
        self,
        policy: HFModel,
        reward_model,  # ValueClassifier
        *,
        batch_size: int,
        num_workers: int,
        lr: float,
        weight_decay: float,
        grad_clip: float | int | None,
        micro_batch_size: int,
        amp_dtype: str,
        clear_cache_interval: int,
        use_chat_template: bool,
        rollout_per_prompt: int,
        max_new_tokens: int | None = None,
    ) -> None:
        self.policy = policy.model
        self.tokenizer = policy.tokenizer
        self.use_chat_template = bool(use_chat_template)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.rollout_per_prompt = int(rollout_per_prompt)
        self.max_new_tokens = (
            int(max_new_tokens)
            if max_new_tokens is not None
            else policy.gen_cfg.max_new_tokens
        )

        self.reward_model = reward_model
        for p in self.reward_model.parameters():
            p.requires_grad_(False)

        self.optimizer = AdamW(
            (p for p in self.policy.parameters() if p.requires_grad),
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

    def _make_prompt_batch(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        built = [
            build_instruction_prompt(
                p, tokenizer=self.tokenizer, use_chat_template=self.use_chat_template
            )
            for p in prompts
        ]
        tok = self.tokenizer(
            built, add_special_tokens=False, padding=True, return_tensors="pt"
        )
        return {"input_ids": tok["input_ids"], "attention_mask": tok["attention_mask"]}

    @torch.no_grad()
    def _rollout(self, prompts: List[str]) -> List[str]:
        texts: List[str] = []
        for p in prompts:
            for _ in range(self.rollout_per_prompt):
                texts.append(self._generate_one(p))
        return texts

    @torch.inference_mode()
    def _generate_one(self, prompt: str) -> str:
        built = build_instruction_prompt(
            prompt, tokenizer=self.tokenizer, use_chat_template=self.use_chat_template
        )
        ids = self.tokenizer(built, return_tensors="pt").to(self.policy.device)
        out = self.policy.generate(
            **ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = out[0][ids["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def create_loader(self, ds: Iterable[Any]) -> DataLoader:
        return DataLoader(
            list(ds),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def _policy_logprobs(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        logits = self.policy(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).logits
        logprobs = torch.log_softmax(logits.float()[:, :-1, :], dim=-1)
        target = input_ids[:, 1:]
        token_lp = logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        seq_lp = (token_lp * attention_mask[:, 1:].float()).sum(dim=-1)
        return seq_lp

    def _reward_scores(self, prompts: List[str], texts: List[str]) -> torch.Tensor:
        device = self.policy.device
        scores: List[float] = []
        for i, t in enumerate(texts):
            p = prompts[i // self.rollout_per_prompt]
            built = build_instruction_prompt(
                p, tokenizer=self.tokenizer, use_chat_template=self.use_chat_template
            )
            ids = self.tokenizer([built], add_special_tokens=False, return_tensors="pt")
            tgt = self.tokenizer([t], add_special_tokens=False, return_tensors="pt")
            full_ids = torch.cat([ids["input_ids"], tgt["input_ids"]], dim=1).to(device)
            attn = full_ids.ne(self.tokenizer.pad_token_id).long()
            with torch.no_grad():
                out = self.reward_model(
                    input_ids=full_ids, attention_mask=attn, use_cache=False
                )
                last = out.logits[:, -1]
                if last.dim() == 2:
                    prob = torch.softmax(last.float(), dim=-1).mean(dim=-1)
                    scores.append(float(prob.item()))
                else:
                    prob = torch.sigmoid(last.float()).mean(dim=-1)
                    scores.append(float(prob.item()))
        return torch.tensor(scores, dtype=torch.float32, device=device)

    def train(self, prompts_iter: Iterable[Any], epochs: int = 1) -> Dict[str, Any]:
        self.policy.train()
        steps = 0
        accum = {"loss": 0.0}
        total_epochs = int(epochs)
        epoch_bar = tqdm(range(total_epochs), desc="GRPO:epochs")
        for e in epoch_bar:
            prompts = []
            for ex in prompts_iter:
                prompts.append(getattr(ex, "question", str(ex)))
            if not prompts:
                break
            batch_bar = tqdm(
                range(0, len(prompts), self.batch_size),
                desc=f"GRPO:epoch {e + 1}/{total_epochs}",
                leave=False,
            )
            for s in batch_bar:
                e2 = min(s + self.batch_size, len(prompts))
                p_batch = prompts[s:e2]
                texts = self._rollout(p_batch)
                rewards = self._reward_scores(p_batch, texts)

                inputs = [
                    build_instruction_prompt(
                        pb,
                        tokenizer=self.tokenizer,
                        use_chat_template=self.use_chat_template,
                    )
                    + tx
                    for i, tx in enumerate(texts)
                    for pb in [p_batch[i // self.rollout_per_prompt]]
                ]
                tok = self.tokenizer(
                    inputs, add_special_tokens=False, padding=True, return_tensors="pt"
                ).to(self.policy.device)

                with autocast(
                    device_type="cuda",
                    dtype=self.amp_dtype,
                    enabled=self.autocast_enabled,
                ):
                    logps = self._policy_logprobs(
                        tok["input_ids"], tok["attention_mask"]
                    )  # [B]
                    base = logps.detach()
                    adv = rewards - rewards.mean()
                    adv = (adv - adv.mean()) / (adv.std().clamp_min(1e-6))
                    loss = -(adv * logps).mean()

                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if self.grad_clip and self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.grad_clip
                    )
                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                steps += 1
                l = float(loss.detach().item())
                accum["loss"] += l
                batch_bar.set_postfix(loss=l)
                if self.clear_cache_interval > 0 and (
                    steps % self.clear_cache_interval == 0
                ):
                    torch.cuda.empty_cache()

            avg_loss = accum["loss"] / max(1, steps)
            epoch_bar.set_postfix(avg_loss=f"{avg_loss:.4f}")

        n = max(1, steps)
        return {"loss": accum["loss"] / n, "steps": steps}
