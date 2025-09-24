from __future__ import annotations

from typing import Any, Dict

from pita.core.registry import register_algorithm
from .base import PostTrainingAlgorithms
from pita.core.io import get_run_root, get_snapshot_paths
from pita.datasets import PreferencePairDataset
from pita.trainers import RewardTrainer, GRPOTrainer
from pita.models.value_classifier import ValueClassifier


@register_algorithm("RLHF")
class RLHFAlgorithm(PostTrainingAlgorithms):
    ALGO_KEY = "RLHF"

    def run(
        self,
        cfg,
        ref_model: str,
        cls_model: str,
        dataset: str,
        family: str,
        output_dir,
        round_idx: int,
    ) -> Dict[str, Any]:
        run_root = get_run_root()
        _, hf_dir, _ = get_snapshot_paths(self.algo_key, dataset, family, round_idx)
        if not hf_dir.exists():
            raise FileNotFoundError(f"Missing RLHF dataset: {hf_dir}")

        pair_ds = PreferencePairDataset(hf_dir)

        ref = self._build_model(cfg, ref_model)

        reward_cls = ValueClassifier(
            cls_model,
            tokenizer=ref.tokenizer,
            device=ref.model.device,
            loss_type=str(self.cfg.loss_type),
            num_atoms=int(self.cfg.num_atoms),
            V_min=float(self.cfg.V_min),
            V_max=float(self.cfg.V_max),
            attn_impl=str(cfg.common.attn_impl),
            dtype=str(cfg.common.amp_dtype),
            gradient_checkpointing=bool(cfg.common.gradient_checkpointing),
        )

        rm_trainer = RewardTrainer(
            reward_cls,
            use_chat_template=bool(cfg.common.use_chat_template),
            batch_size=int(self.cfg.batch_size),
            num_workers=int(self.cfg.num_workers),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
            grad_clip=float(self.cfg.grad_clip),
            micro_batch_size=int(cfg.common.micro_batch_size),
            amp_dtype=str(cfg.common.amp_dtype),
            clear_cache_interval=int(cfg.common.clear_cache_interval),
        )
        rm_loader = rm_trainer.create_loader(pair_ds, shuffle=True)
        rm_stats = rm_trainer.train(rm_loader, epochs=int(self.cfg.rm_epochs))

        grpo = GRPOTrainer(
            policy=ref,
            reward_model=reward_cls,
            batch_size=int(self.cfg.batch_size),
            num_workers=int(self.cfg.num_workers),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
            grad_clip=float(self.cfg.grad_clip),
            micro_batch_size=int(cfg.common.micro_batch_size),
            amp_dtype=str(cfg.common.amp_dtype),
            clear_cache_interval=int(cfg.common.clear_cache_interval),
            use_chat_template=bool(cfg.common.use_chat_template),
            rollout_per_prompt=int(self.cfg.rollout_per_prompt),
            max_new_tokens=int(cfg.common.max_new_tokens),
        )
        ds_iter = self._build_dataset(cfg, dataset)
        grpo_stats = grpo.train(ds_iter.iter(), epochs=int(self.cfg.epochs))

        ckpt_dir = self.get_ckpt_dir(run_root, dataset, family, round_idx)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ref.model.save_pretrained(str(ckpt_dir))
        ref.tokenizer.save_pretrained(str(ckpt_dir))

        # Save reward classifier for potential reuse next round
        torch_path = ckpt_dir / "classifier.pt"
        import torch as _torch

        _torch.save(reward_cls.state_dict(), str(torch_path))

        return {
            "algo": "RLHF",
            "dataset": dataset,
            "rm_loss": float(rm_stats.get("loss", 0.0)),
            "grpo_loss": float(grpo_stats.get("loss", 0.0)),
        }
