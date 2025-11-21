from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
import random
from loguru import logger

from omegaconf import DictConfig
from datasets import Dataset
from tqdm import tqdm
from itertools import islice
import torch
from pita.core.prompts import build_instruction_prompt

from pita.models.hf import HFModel, GenerationConfig
from pita.models.guided import GuidedHFModel, GuidanceConfig
from pita.models.value_classifier import ValueClassifier
from pita.datasets import build_train_dataset, build_test_dataset
from pita.core.io import get_run_root, get_snapshot_paths, merge_and_save_hf
from pita.core.gpu_manager import get_gpu_manager
from pita.core.compute_tracker import get_compute_tracker
from pita.models import RewardScorer


class AlgorithmBase:
    def __init__(self, cfg):
        self.cfg = cfg

    @property
    def algo_key(self) -> str:
        return self.ALGO_KEY

    def family_cap(self, family: str) -> str:
        return str(family).capitalize()

    def get_prev_ckpt_dir(
        self, run_root: Path, dataset: str, family: str, round_idx: int
    ) -> Path:
        family_cap = self.family_cap(family)
        return (
            run_root
            / "models"
            / self.algo_key
            / f"{dataset}_{family_cap}_r{int(round_idx)}"
        )

    def resolve_ref_for_round(
        self,
        run_root: Path,
        dataset: str,
        family: str,
        ref_model_alias: str,
        round_idx: int,
    ) -> str:
        if int(round_idx) > 0:
            prev_ckpt_dir = self.get_prev_ckpt_dir(run_root, dataset, family, round_idx)
            if prev_ckpt_dir.exists():
                return str(prev_ckpt_dir)
        return ref_model_alias

    def _build_dataset(self, cfg: DictConfig, dataset_name: str):
        return build_train_dataset(cfg, dataset_name)

    def _build_test_dataset(self, cfg: DictConfig, dataset_name: str):
        return build_test_dataset(cfg, dataset_name)

    def generate_data(
        self,
        cfg,
        ref_model: str,
        dataset: str,
        family: str,
        round_idx: int,
        cls_model: Optional[str] = None,
        run_root: Optional[Path] = None,
    ) -> None:
        return None

    def run(
        self,
        cfg,
        ref_model: str,
        cls_model: str,
        dataset: str,
        family: str,
        output_dir,
        round_idx: int,
        run_root: Optional[Any] = None,
    ):
        raise NotImplementedError

    def _build_model(self, cfg: DictConfig, model_alias: str) -> HFModel:
        gen_cfg = GenerationConfig(
            max_new_tokens=int(cfg.generation.max_new_tokens),
            temperature=float(cfg.generation.temperature),
            top_p=float(cfg.generation.top_p),
            use_chat_template=bool(cfg.generation.use_chat_template),
            dtype=str(cfg.system.dtype),
            attn_impl=str(cfg.system.attn_impl),
            gradient_checkpointing=bool(cfg.training.gradient_checkpointing),
        )
        return HFModel(model_alias, gen_cfg)

    def get_ckpt_dir(
        self, run_root: Path, dataset: str, family: str, round_idx: int
    ) -> Path:
        family_cap = self.family_cap(family)
        return (
            run_root
            / "models"
            / self.algo_key
            / f"{dataset}_{family_cap}_r{int(round_idx)+1}"
        )


class ValueGuidedAlgorithms(AlgorithmBase):
    """Value-guided algorithms with parallel multi-GPU data generation support."""

    def maybe_load_classifier_from_prev_round(
        self,
        classifier: ValueClassifier,
        *,
        run_root: Path,
        dataset: str,
        family: str,
        round_idx: int,
        device: torch.device,
    ) -> bool:
        if int(round_idx) <= 0:
            return False
        prev_ckpt_dir = self.get_prev_ckpt_dir(run_root, dataset, family, round_idx)
        ckpt = prev_ckpt_dir / "classifier.pt"
        if ckpt.exists():
            state = torch.load(str(ckpt), map_location=device)
            classifier.load_state_dict(state)
            return True
        return False

    def generate_data(
        self,
        cfg: DictConfig,
        ref_model: str,
        dataset: str,
        family: str,
        round_idx: int,
        cls_model: Optional[str] = None,
        run_root: Optional[Path] = None,
    ) -> None:
        """Generate data using parallel multi-GPU processing when available."""
        from pita.core.io import (
            get_checkpoint_dir,
            load_all_checkpoints,
            clear_checkpoints,
        )

        if run_root is None:
            run_root = get_run_root()

        snap_hf_prev, snap_hf, snap_csv = get_snapshot_paths(
            self.algo_key, dataset, family, round_idx, run_root=run_root
        )

        checkpoint_dir = get_checkpoint_dir(
            self.algo_key, dataset, family, round_idx, run_root
        )

        existing_records = load_all_checkpoints(checkpoint_dir)
        if existing_records:
            logger.info(
                f"💾 Found {len(existing_records)} records from previous checkpoints"
            )

        gpu_manager = get_gpu_manager()
        parallel_enabled = (
            getattr(cfg, "parallel_generation", True)
            if hasattr(cfg, "parallel_generation")
            else True
        )
        use_parallel = parallel_enabled

        tracker = get_compute_tracker()
        with tracker.track_phase(f"data_generation_{dataset}"):
            if use_parallel:
                logger.info(
                    f"🔧 Using parallel generation with {gpu_manager.num_gpus} GPUs"
                )
                records = self._generate_data_parallel(
                    cfg, ref_model, dataset, family, round_idx, cls_model, run_root
                )
            else:
                logger.info("🔧 Using sequential generation")
                records = self._generate_data_sequential(
                    cfg, ref_model, dataset, family, round_idx, cls_model, run_root
                )

        all_records = existing_records + records

        if not all_records:
            return

        logger.info(f"💾 Saving {len(all_records)} total records")
        new_ds = Dataset.from_list(all_records)
        merge_and_save_hf(snap_hf_prev, new_ds, snap_hf, snap_csv)

        clear_checkpoints(checkpoint_dir)
        logger.info(f"🧹 Cleared checkpoints")

    def _generate_data_parallel(
        self,
        cfg: DictConfig,
        ref_model: str,
        dataset: str,
        family: str,
        round_idx: int,
        cls_model: Optional[str] = None,
        run_root: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """Parallel data generation across multiple GPUs."""
        import torch.multiprocessing as mp

        random.seed(int(cfg.data_collection.seed))

        # Prepare dataset
        ds = self._build_dataset(cfg, dataset)
        self._dataset = ds
        samples_per_example = int(cfg.data_collection.samples_per_example)
        max_examples = int(cfg.data_collection.max_examples or 0)
        limit = max_examples if max_examples > 0 else len(ds)

        # Collect examples
        examples = []
        for ex in islice(ds.iter(), limit):
            prompt = ds.hydrate_prompt(ex.question)
            examples.append(
                {
                    "question": ex.question,
                    "answer": ex.answer,
                    "prompt": prompt,
                }
            )

        if not examples:
            return []

        gpu_manager = get_gpu_manager()
        num_workers = gpu_manager.num_gpus
        jobs_per_worker = gpu_manager.distribute_work(len(examples))

        # Use spawn for CUDA compatibility
        mp_context = mp.get_context("spawn")
        result_queue = mp_context.Queue()
        processes = []

        try:
            for worker_id in range(num_workers):
                job_indices = jobs_per_worker[worker_id]
                if not job_indices:
                    continue

                worker_examples = [examples[i] for i in job_indices]
                gpu_id = gpu_manager.available_gpus[worker_id]

                p = mp_context.Process(
                    target=self._worker_generate_data,
                    args=(
                        worker_id,
                        gpu_id,
                        worker_examples,
                        cfg,
                        ref_model,
                        dataset,
                        family,
                        round_idx,
                        cls_model,
                        samples_per_example,
                        int(cfg.data_collection.seed) + worker_id,
                        result_queue,
                        run_root,
                    ),
                )
                p.start()
                processes.append(p)

            # Collect results
            all_records = []
            expected_batches = sum(1 for jobs in jobs_per_worker if jobs)

            for _ in range(expected_batches):
                worker_records = result_queue.get()
                if isinstance(worker_records, Exception):
                    raise worker_records
                all_records.extend(worker_records)
                logger.info(f"Collected {len(all_records)} records so far")

            # Wait for all processes
            for p in processes:
                p.join()

            return all_records

        except Exception as e:
            logger.error(f"Error in parallel generation: {e}")
            for p in processes:
                if p.is_alive():
                    p.terminate()
                    p.join()
            raise

    def _worker_generate_data(
        self,
        worker_id: int,
        gpu_id: int,
        examples: List[Dict[str, Any]],
        cfg: DictConfig,
        ref_model: str,
        dataset: str,
        family: str,
        round_idx: int,
        cls_model: Optional[str],
        samples_per_example: int,
        seed: int,
        result_queue,
        run_root: Optional[Path] = None,
    ):
        """Worker process for data generation."""
        try:
            # Set device
            import os

            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            torch.cuda.set_device(0)

            random.seed(seed)
            torch.manual_seed(seed)

            # Build model
            ref_hf = self._build_model_for_worker(cfg, ref_model, device_id=0)

            # Load classifier if needed
            model = ref_hf
            if int(round_idx) > 0 and cls_model is not None:
                prev_classifier = ValueClassifier(
                    cls_model,
                    tokenizer=ref_hf.tokenizer,
                    device=ref_hf.model.device,
                    loss_type=str(self.cfg.loss_type),
                    num_atoms=int(self.cfg.num_atoms),
                    V_min=float(self.cfg.V_min),
                    V_max=float(self.cfg.V_max),
                    attn_impl=str(cfg.system.attn_impl),
                    dtype=str(cfg.system.amp_dtype),
                    gradient_checkpointing=bool(cfg.training.gradient_checkpointing),
                )
                if run_root is None:
                    run_root = get_run_root()
                ckpt_loaded = self.maybe_load_classifier_from_prev_round(
                    prev_classifier,
                    run_root=run_root,
                    dataset=dataset,
                    family=family,
                    round_idx=round_idx,
                    device=ref_hf.model.device,
                )
                if ckpt_loaded:
                    model = self.build_guided_with(
                        cfg, ref=ref_hf, classifier=prev_classifier
                    )

            # Build reward scorer
            ds_cfg = cfg.datasets[dataset]
            rm_model = str(ds_cfg.reward_model)
            reward_batch_size = int(cfg.data_collection.reward_batch_size)
            reward_scorer = RewardScorer(
                rm_model,
                bt_sampling=bool(cfg.data_collection.bradley_terry_sampling),
                bt_beta=float(cfg.data_collection.bradley_terry_beta),
                device=0,
                dtype=str(cfg.system.dtype),
                batch_size=reward_batch_size,
            )

            gen_cfg = model.gen_cfg
            batch_size = int(cfg.algos[self.algo_key].batch_size)
            logger.info(
                f"🔄 Worker {worker_id} processing {len(examples)} examples with batch_size={batch_size}"
            )

            # PHASE 1: Batch all roll_in calls
            prompts = [ex["prompt"] for ex in examples]
            rollouts = model.roll_in_batch(
                prompts, gen_cfg.max_new_tokens, batch_size=batch_size
            )

            # PHASE 2: Prepare contexts for continuation
            contexts_to_continue = []
            for ex, rollout in tqdm(
                zip(examples, rollouts),
                total=len(examples),
                desc=f"Worker {worker_id}: preparing contexts",
            ):
                built_prompt = rollout["prompt"]
                prompt_token_count = len(model.tokenizer(built_prompt)["input_ids"])
                context_token_ids = rollout["context_ids"].tolist()
                full_token_count = len(context_token_ids)
                rollout_token_count = full_token_count - prompt_token_count

                if rollout_token_count < 2:
                    continue

                for _ in range(samples_per_example):
                    cutoff_tokens = random.randint(1, rollout_token_count - 1)

                    context_prefix_ids = context_token_ids[
                        : prompt_token_count + cutoff_tokens
                    ]
                    context_text = model.tokenizer.decode(
                        context_prefix_ids, skip_special_tokens=True
                    )
                    solution_prefix_ids = context_prefix_ids[prompt_token_count:]
                    solution_prefix_text = model.tokenizer.decode(
                        solution_prefix_ids, skip_special_tokens=True
                    )

                    remaining_token_budget = gen_cfg.max_new_tokens - cutoff_tokens
                    contexts_to_continue.append(
                        {
                            "ex": ex,
                            "context_text": context_text,
                            "solution_prefix_text": solution_prefix_text,
                            "cutoff_tokens": cutoff_tokens,
                            "remaining_token_budget": remaining_token_budget,
                        }
                    )

            if not contexts_to_continue:
                logger.info(
                    f"⚠️ Worker {worker_id} has no valid contexts (all rollouts too short)"
                )
                result_queue.put([])
                return

            # Filter out contexts with non-positive remaining tokens
            contexts_to_continue = [
                ctx
                for ctx in contexts_to_continue
                if ctx["remaining_token_budget"] >= 1
            ]

            if not contexts_to_continue:
                logger.info(
                    f"⚠️ Worker {worker_id} has no valid contexts (all have insufficient remaining tokens)"
                )
                result_queue.put([])
                return

            logger.info(
                f"🔄 Worker {worker_id} prepared {len(contexts_to_continue)} contexts, now batching continuations..."
            )

            # PHASE 3: Batch all continue_from_context calls
            all_contexts = [item["context_text"] for item in contexts_to_continue]
            min_remaining_tokens = min(
                item["remaining_token_budget"] for item in contexts_to_continue
            )

            greedy_continuations = model.continue_from_context_batch(
                all_contexts, min_remaining_tokens, greedy=True, batch_size=batch_size
            )
            sampled_continuations = model.continue_from_context_batch(
                all_contexts, min_remaining_tokens, greedy=False, batch_size=batch_size
            )

            # PHASE 4: Assemble generated samples
            generated_samples = []
            for ctx, y_ref, y_sample in zip(
                contexts_to_continue, greedy_continuations, sampled_continuations
            ):
                y_a = ctx["solution_prefix_text"] + y_ref
                y_b = ctx["solution_prefix_text"] + y_sample
                generated_samples.append(
                    {
                        "ex": ctx["ex"],
                        "y_a": y_a,
                        "y_b": y_b,
                        "cutoff_tokens": ctx["cutoff_tokens"],
                        "solution_prefix_text": ctx["solution_prefix_text"],
                    }
                )

            logger.info(
                f"✅ Worker {worker_id} generated {len(generated_samples)} samples, now batch scoring..."
            )

            # PHASE 5: Batch score all samples at once
            score_pairs = [
                (s["ex"]["question"], s["y_a"], s["y_b"]) for s in generated_samples
            ]
            scores = reward_scorer.score_batch(score_pairs)

            # PHASE 6: Create records with scores
            records = []
            for sample, (score_a, score_b, preferred) in zip(generated_samples, scores):
                records.append(
                    {
                        "question": sample["ex"]["question"],
                        "answer": sample["ex"]["answer"],
                        "prompt": sample["ex"]["prompt"],
                        "t": sample["cutoff_tokens"],
                        "context": sample["ex"]["prompt"]
                        + sample["solution_prefix_text"],
                        "y_a": sample["y_a"],
                        "y_b": sample["y_b"],
                        "score_a": score_a,
                        "score_b": score_b,
                        "preferred": preferred,
                    }
                )

            logger.info(f"✅ Worker {worker_id} generated {len(records)} records")

            from pita.core.io import get_checkpoint_dir, save_checkpoint

            if run_root is None:
                run_root = get_run_root()
            checkpoint_dir = get_checkpoint_dir(
                self.algo_key, dataset, family, round_idx, run_root
            )
            save_checkpoint(checkpoint_dir, worker_id, records)
            logger.info(f"💾 Worker {worker_id} saved checkpoint")

            result_queue.put(records)

        except Exception as e:
            logger.error(f"Worker {worker_id} failed: {e}")
            import traceback

            traceback.print_exc()
            result_queue.put(e)
        finally:
            if "reward_scorer" in locals() and reward_scorer is not None:
                reward_scorer.cleanup()
            if "model" in locals():
                del model
            if "ref_hf" in locals():
                del ref_hf
            if "prev_classifier" in locals():
                del prev_classifier
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _build_model_for_worker(
        self, cfg: DictConfig, model_alias: str, device_id: int
    ) -> HFModel:
        """Build model specifically for a worker process."""
        from pita.models.catalog import resolve_model_id
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_id = resolve_model_id(model_alias)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            trust_remote_code=True,
        )
        if getattr(tokenizer, "pad_token_id", None) is None:
            if getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                raise ValueError("Pad token not found")

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }[str(cfg.system.dtype)]

        # Load directly to specific device (not using device_map since we set CUDA_VISIBLE_DEVICES)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation=str(cfg.system.attn_impl),
            low_cpu_mem_usage=True,
        )

        if getattr(cfg.training, "gradient_checkpointing", False):
            model.gradient_checkpointing_enable()

        model.eval()

        gen_cfg = GenerationConfig(
            max_new_tokens=int(cfg.generation.max_new_tokens),
            temperature=float(cfg.generation.temperature),
            top_p=float(cfg.generation.top_p),
            use_chat_template=bool(cfg.generation.use_chat_template),
            dtype=str(cfg.system.dtype),
            attn_impl=str(cfg.system.attn_impl),
            gradient_checkpointing=bool(cfg.training.gradient_checkpointing),
        )

        hf_model = HFModel.__new__(HFModel)
        hf_model.tokenizer = tokenizer
        hf_model.model = model
        hf_model.gen_cfg = gen_cfg
        hf_model.eos_token_ids = hf_model._compute_eos_token_ids()
        hf_model.pad_token_id = hf_model._compute_pad_token_id()

        return hf_model

    def _generate_data_sequential(
        self,
        cfg: DictConfig,
        ref_model: str,
        dataset: str,
        family: str,
        round_idx: int,
        cls_model: Optional[str] = None,
        run_root: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """Sequential data generation (original implementation with optimizations)."""
        random.seed(int(cfg.data_collection.seed))

        ref_hf = self._build_model(cfg, ref_model)
        if int(round_idx) > 0 and cls_model is not None:
            prev_classifier = ValueClassifier(
                cls_model,
                tokenizer=ref_hf.tokenizer,
                device=ref_hf.model.device,
                loss_type=str(self.cfg.loss_type),
                num_atoms=int(self.cfg.num_atoms),
                V_min=float(self.cfg.V_min),
                V_max=float(self.cfg.V_max),
                attn_impl=str(cfg.system.attn_impl),
                dtype=str(cfg.system.amp_dtype),
                gradient_checkpointing=bool(cfg.training.gradient_checkpointing),
            )
            if run_root is None:
                run_root = get_run_root()
            ckpt_loaded = self.maybe_load_classifier_from_prev_round(
                prev_classifier,
                run_root=run_root,
                dataset=dataset,
                family=family,
                round_idx=round_idx,
                device=ref_hf.model.device,
            )
            if ckpt_loaded:
                model = self.build_guided_with(
                    cfg, ref=ref_hf, classifier=prev_classifier
                )
            else:
                model = ref_hf
        else:
            model = ref_hf

        ds = self._build_dataset(cfg, dataset)
        self._dataset = ds
        gen_cfg = model.gen_cfg

        ds_cfg = cfg.datasets[dataset]
        rm_model = str(ds_cfg.reward_model)
        reward_batch_size = int(cfg.data_collection.reward_batch_size)
        reward_scorer = RewardScorer(
            rm_model,
            bt_sampling=bool(cfg.data_collection.bradley_terry_sampling),
            bt_beta=float(cfg.data_collection.bradley_terry_beta),
            device=0 if torch.cuda.is_available() else -1,
            dtype=str(cfg.system.dtype),
            batch_size=reward_batch_size,
        )

        samples_per_example = int(cfg.data_collection.samples_per_example)
        batch_size = int(cfg.algos[self.algo_key].batch_size)
        chunk_size = int(getattr(cfg.data_collection, "chunk_size", 100))

        max_examples = int(cfg.data_collection.max_examples or 0)
        limit = max_examples if max_examples > 0 else len(ds)

        # Collect all examples (lightweight - just metadata)
        examples = []
        for ex in islice(ds.iter(), limit):
            prompt = ds.hydrate_prompt(ex.question)
            examples.append({"ex": ex, "prompt": prompt})

        logger.info(
            f"🔄 Processing {len(examples)} examples in chunks of {chunk_size} (batch_size={batch_size})"
        )

        all_records: List[Dict[str, Any]] = []

        # Process examples in chunks to reduce peak memory usage
        for chunk_start in range(0, len(examples), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(examples))
            chunk_examples = examples[chunk_start:chunk_end]

            logger.info(
                f"📦 Processing chunk {chunk_start//chunk_size + 1}/{(len(examples) + chunk_size - 1)//chunk_size} "
                f"(examples {chunk_start+1}-{chunk_end}/{len(examples)})"
            )

            # PHASE 1: Batch roll_in calls for this chunk
            prompts = [item["prompt"] for item in chunk_examples]
            rollouts = model.roll_in_batch(
                prompts, gen_cfg.max_new_tokens, batch_size=batch_size
            )

            # PHASE 2: Prepare contexts for continuation
            contexts_to_continue = []
            for item, rollout in zip(chunk_examples, rollouts):
                built_prompt = rollout["prompt"]
                prompt_token_count = len(model.tokenizer(built_prompt)["input_ids"])
                context_token_ids = rollout["context_ids"].tolist()
                full_token_count = len(context_token_ids)
                rollout_token_count = full_token_count - prompt_token_count

                if rollout_token_count < 2:
                    continue

                for _ in range(samples_per_example):
                    cutoff_tokens = random.randint(1, rollout_token_count - 1)
                    context_prefix_ids = context_token_ids[
                        : prompt_token_count + cutoff_tokens
                    ]
                    context_text = model.tokenizer.decode(
                        context_prefix_ids, skip_special_tokens=True
                    )
                    solution_prefix_ids = context_prefix_ids[prompt_token_count:]
                    solution_prefix_text = model.tokenizer.decode(
                        solution_prefix_ids, skip_special_tokens=True
                    )
                    remaining_token_budget = gen_cfg.max_new_tokens - cutoff_tokens

                    if remaining_token_budget >= 1:
                        contexts_to_continue.append(
                            {
                                "ex": item["ex"],
                                "prompt": item["prompt"],
                                "context_text": context_text,
                                "solution_prefix_text": solution_prefix_text,
                                "cutoff_tokens": cutoff_tokens,
                                "remaining_token_budget": remaining_token_budget,
                            }
                        )

            if not contexts_to_continue:
                logger.info(
                    f"⏭️  Chunk {chunk_start//chunk_size + 1}: No valid contexts, skipping"
                )
                del rollouts
                continue

            # PHASE 3: Batch continue_from_context calls
            all_contexts = [item["context_text"] for item in contexts_to_continue]
            min_remaining_tokens = min(
                item["remaining_token_budget"] for item in contexts_to_continue
            )

            greedy_continuations = model.continue_from_context_batch(
                all_contexts, min_remaining_tokens, greedy=True, batch_size=batch_size
            )
            sampled_continuations = model.continue_from_context_batch(
                all_contexts, min_remaining_tokens, greedy=False, batch_size=batch_size
            )

            # PHASE 4: Assemble generated samples
            generated_samples = []
            for ctx, y_ref, y_sample in zip(
                contexts_to_continue, greedy_continuations, sampled_continuations
            ):
                y_a = ctx["solution_prefix_text"] + y_ref
                y_b = ctx["solution_prefix_text"] + y_sample
                generated_samples.append(
                    {
                        "ex": ctx["ex"],
                        "prompt": ctx["prompt"],
                        "y_a": y_a,
                        "y_b": y_b,
                        "cutoff_tokens": ctx["cutoff_tokens"],
                    }
                )

            # PHASE 5: Batch score samples
            score_pairs = [
                (s["ex"].question, s["y_a"], s["y_b"]) for s in generated_samples
            ]
            scores = reward_scorer.score_batch(score_pairs)

            # PHASE 6: Create records
            for sample, (score_a, score_b, preferred) in zip(generated_samples, scores):
                all_records.append(
                    {
                        "question": sample["ex"].question,
                        "answer": sample["ex"].answer,
                        "prompt": sample["prompt"],
                        "t": sample["cutoff_tokens"],
                        "context": sample["prompt"]
                        + sample["y_a"][: sample["cutoff_tokens"]],
                        "y_a": sample["y_a"],
                        "y_b": sample["y_b"],
                        "score_a": score_a,
                        "score_b": score_b,
                        "preferred": preferred,
                    }
                )

            # Clear chunk data to free memory
            del rollouts, contexts_to_continue, all_contexts
            del greedy_continuations, sampled_continuations, generated_samples
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(
                f"✅ Chunk {chunk_start//chunk_size + 1} complete: {len(all_records)} total records so far"
            )

        reward_scorer.cleanup()
        logger.info(f"🎉 All chunks processed: {len(all_records)} total records")
        return all_records

    def score_samples(self, ex, y_a: str, y_b: str) -> Tuple[float, float, int]:
        raise NotImplementedError

    def _build_guided_model(
        self,
        cfg: DictConfig,
        ref_model_alias: str,
        cls_model_alias: str,
    ) -> GuidedHFModel:
        ref_model = self._build_model(cfg, ref_model_alias)
        cls = ValueClassifier(
            cls_model_alias,
            tokenizer=ref_model.tokenizer,
            device=ref_model.model.device,
            loss_type=str(self.cfg.loss_type),
            num_atoms=int(self.cfg.num_atoms),
            V_min=float(self.cfg.V_min),
            V_max=float(self.cfg.V_max),
            attn_impl=str(cfg.system.attn_impl),
            dtype=str(cfg.system.amp_dtype),
            gradient_checkpointing=bool(cfg.training.gradient_checkpointing),
        )
        g = GuidanceConfig(
            eta=float(self.cfg.guidance.eta),
            mode=str(self.cfg.guidance.mode),
            top_k=int(self.cfg.guidance.top_k),
            use_cache=bool(self.cfg.guidance.use_cache),
        )
        return GuidedHFModel(ref_model, cls, g)

    def build_guided_with(
        self,
        cfg: DictConfig,
        *,
        ref: HFModel,
        classifier: ValueClassifier,
    ) -> GuidedHFModel:
        g = GuidanceConfig(
            eta=float(self.cfg.guidance.eta),
            mode=str(self.cfg.guidance.mode),
            top_k=int(self.cfg.guidance.top_k),
            use_cache=bool(self.cfg.guidance.use_cache),
        )
        return GuidedHFModel(ref, classifier, g)


class PostTrainingAlgorithms(AlgorithmBase):
    """Post-training algorithms with parallel multi-GPU data generation support."""

    def generate_data(
        self,
        cfg: DictConfig,
        ref_model: str,
        dataset: str,
        family: str,
        round_idx: int,
        cls_model: Optional[str] = None,
        run_root: Optional[Path] = None,
    ) -> None:
        """Generate data using parallel multi-GPU processing when available."""
        from pita.core.io import (
            get_checkpoint_dir,
            load_all_checkpoints,
            clear_checkpoints,
        )

        if run_root is None:
            run_root = get_run_root()

        snap_hf_prev, snap_hf, snap_csv = get_snapshot_paths(
            self.algo_key, dataset, family, round_idx, run_root=run_root
        )

        checkpoint_dir = get_checkpoint_dir(
            self.algo_key, dataset, family, round_idx, run_root
        )

        existing_records = load_all_checkpoints(checkpoint_dir)
        if existing_records:
            logger.info(
                f"💾 Found {len(existing_records)} records from previous checkpoints"
            )

        gpu_manager = get_gpu_manager()
        parallel_enabled = (
            getattr(cfg, "parallel_generation", True)
            if hasattr(cfg, "parallel_generation")
            else True
        )
        use_parallel = parallel_enabled

        tracker = get_compute_tracker()
        with tracker.track_phase(f"data_generation_{dataset}"):
            if use_parallel:
                logger.info(
                    f"🔧 Using parallel generation with {gpu_manager.num_gpus} GPUs"
                )
                records = self._generate_data_parallel(
                    cfg, ref_model, dataset, family, round_idx, run_root
                )
            else:
                logger.info("🔧 Using sequential generation with optimized batching")
                records = self._generate_data_sequential(
                    cfg, ref_model, dataset, family, round_idx, run_root
                )

        all_records = existing_records + records

        if not all_records:
            return

        logger.info(f"💾 Saving {len(all_records)} total records")
        new_ds = Dataset.from_list(all_records)
        merge_and_save_hf(snap_hf_prev, new_ds, snap_hf, snap_csv)

        clear_checkpoints(checkpoint_dir)
        logger.info(f"🧹 Cleared checkpoints")

    def _generate_data_sequential(
        self,
        cfg: DictConfig,
        ref_model: str,
        dataset: str,
        family: str,
        round_idx: int,
        run_root: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """Sequential generation with optimized reward batching."""
        random.seed(int(cfg.data_collection.seed))
        model = self._build_model(cfg, ref_model)
        ds = self._build_dataset(cfg, dataset)
        gen_cfg = model.gen_cfg

        # Optimized reward model setup
        ds_cfg = cfg.datasets[dataset]
        rm_model = str(ds_cfg.reward_model)
        device = 0 if torch.cuda.is_available() else -1
        reward_batch_size = int(cfg.data_collection.reward_batch_size)
        self._reward = RewardScorer(
            rm_model,
            bt_sampling=bool(cfg.data_collection.bradley_terry_sampling),
            bt_beta=float(cfg.data_collection.bradley_terry_beta),
            device=device,
            dtype=str(cfg.system.dtype),
            batch_size=reward_batch_size,
        )

        samples_per_example = int(cfg.data_collection.samples_per_example)
        batch_size = int(cfg.algos[self.algo_key].batch_size)
        chunk_size = int(getattr(cfg.data_collection, "chunk_size", 100))

        max_examples = int(cfg.data_collection.max_examples or 0)
        limit = max_examples if max_examples > 0 else len(ds)

        # Collect all examples (lightweight - just metadata)
        examples = []
        for ex in islice(ds.iter(), limit):
            prompt = ds.hydrate_prompt(ex.question)
            built = build_instruction_prompt(
                prompt,
                tokenizer=model.tokenizer,
                use_chat_template=model.gen_cfg.use_chat_template,
            )
            examples.append({"ex": ex, "prompt": prompt, "built": built})

        logger.info(
            f"🔄 Processing {len(examples)} examples in chunks of {chunk_size} (batch_size={batch_size})"
        )

        all_records: List[Dict[str, Any]] = []

        # Process examples in chunks to reduce peak memory usage
        for chunk_start in range(0, len(examples), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(examples))
            chunk_examples = examples[chunk_start:chunk_end]

            logger.info(
                f"📦 Processing chunk {chunk_start//chunk_size + 1}/{(len(examples) + chunk_size - 1)//chunk_size} "
                f"(examples {chunk_start+1}-{chunk_end}/{len(examples)})"
            )

            # PHASE 1: Build prompts for this chunk
            prompts_to_generate = []
            for item in chunk_examples:
                for _ in range(samples_per_example):
                    prompts_to_generate.append(item)

            # PHASE 2: Batch generate y_a (sampled)
            all_prompts = [item["built"] for item in prompts_to_generate]
            y_a_list = model.continue_from_context_batch(
                all_prompts, gen_cfg.max_new_tokens, greedy=False, batch_size=batch_size
            )

            # PHASE 3: Batch generate y_b (sampled)
            y_b_list = model.continue_from_context_batch(
                all_prompts, gen_cfg.max_new_tokens, greedy=False, batch_size=batch_size
            )

            # PHASE 4: Assemble generated samples
            generated_samples = []
            for item, y_a, y_b in zip(prompts_to_generate, y_a_list, y_b_list):
                generated_samples.append(
                    {
                        "ex": item["ex"],
                        "prompt": item["prompt"],
                        "y_a": y_a,
                        "y_b": y_b,
                    }
                )

            # PHASE 5: Batch score samples
            score_pairs = [
                (s["ex"].question, s["y_a"], s["y_b"]) for s in generated_samples
            ]
            scores = self._reward.score_batch(score_pairs)

            # PHASE 6: Create records
            for sample, (score_a, score_b, preferred) in zip(generated_samples, scores):
                all_records.append(
                    {
                        "question": sample["ex"].question,
                        "answer": sample["ex"].answer,
                        "prompt": sample["prompt"],
                        "y_a": sample["y_a"],
                        "y_b": sample["y_b"],
                        "score_a": score_a,
                        "score_b": score_b,
                        "preferred": preferred,
                    }
                )

            # Clear chunk data to free memory
            del prompts_to_generate, all_prompts, y_a_list, y_b_list, generated_samples
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(
                f"✅ Chunk {chunk_start//chunk_size + 1} complete: {len(all_records)} total records so far"
            )

        self._reward.cleanup()
        logger.info(f"🎉 All chunks processed: {len(all_records)} total records")
        return all_records

    def _generate_data_parallel(
        self,
        cfg: DictConfig,
        ref_model: str,
        dataset: str,
        family: str,
        round_idx: int,
        run_root: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """Parallel data generation across multiple GPUs."""
        import torch.multiprocessing as mp

        random.seed(int(cfg.data_collection.seed))

        ds = self._build_dataset(cfg, dataset)
        samples_per_example = int(cfg.data_collection.samples_per_example)
        max_examples = int(cfg.data_collection.max_examples or 0)
        limit = max_examples if max_examples > 0 else len(ds)

        examples = []
        for ex in islice(ds.iter(), limit):
            prompt = ds.hydrate_prompt(ex.question)
            examples.append(
                {
                    "question": ex.question,
                    "answer": ex.answer,
                    "prompt": prompt,
                }
            )

        if not examples:
            return []

        gpu_manager = get_gpu_manager()
        num_workers = gpu_manager.num_gpus
        jobs_per_worker = gpu_manager.distribute_work(len(examples))

        mp_context = mp.get_context("spawn")
        result_queue = mp_context.Queue()
        processes = []

        try:
            for worker_id in range(num_workers):
                job_indices = jobs_per_worker[worker_id]
                if not job_indices:
                    continue

                worker_examples = [examples[i] for i in job_indices]
                gpu_id = gpu_manager.available_gpus[worker_id]

                p = mp_context.Process(
                    target=self._worker_generate_data_dpo,
                    args=(
                        worker_id,
                        gpu_id,
                        worker_examples,
                        cfg,
                        ref_model,
                        dataset,
                        family,
                        round_idx,
                        samples_per_example,
                        int(cfg.data_collection.seed) + worker_id,
                        result_queue,
                        run_root,
                    ),
                )
                p.start()
                processes.append(p)

            all_records = []
            expected_batches = sum(1 for jobs in jobs_per_worker if jobs)

            for _ in range(expected_batches):
                worker_records = result_queue.get()
                if isinstance(worker_records, Exception):
                    raise worker_records
                all_records.extend(worker_records)
                logger.info(f"Collected {len(all_records)} records so far")

            for p in processes:
                p.join()

            return all_records

        except Exception as e:
            logger.error(f"Error in parallel generation: {e}")
            for p in processes:
                if p.is_alive():
                    p.terminate()
                    p.join()
            raise

    def _worker_generate_data_dpo(
        self,
        worker_id: int,
        gpu_id: int,
        examples: List[Dict[str, Any]],
        cfg: DictConfig,
        ref_model: str,
        dataset: str,
        family: str,
        round_idx: int,
        samples_per_example: int,
        seed: int,
        result_queue,
        run_root: Optional[Path] = None,
    ):
        """Worker for DPO-style generation."""
        try:
            import os

            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            torch.cuda.set_device(0)

            random.seed(seed)
            torch.manual_seed(seed)

            # Build model
            model = self._build_model(cfg, ref_model)

            # Build reward scorer
            ds_cfg = cfg.datasets[dataset]
            rm_model = str(ds_cfg.reward_model)
            reward_batch_size = int(cfg.data_collection.reward_batch_size)
            reward_scorer = RewardScorer(
                rm_model,
                bt_sampling=bool(cfg.data_collection.bradley_terry_sampling),
                bt_beta=float(cfg.data_collection.bradley_terry_beta),
                device=0,
                dtype=str(cfg.system.dtype),
                batch_size=reward_batch_size,
            )

            gen_cfg = model.gen_cfg
            batch_size = int(cfg.algos[self.algo_key].batch_size)
            logger.info(
                f"🔄 Worker {worker_id} processing {len(examples)} examples with batch_size={batch_size}"
            )

            # PHASE 1: Build all prompts
            prompts_to_generate = []
            for ex in examples:
                built = build_instruction_prompt(
                    ex["prompt"],
                    tokenizer=model.tokenizer,
                    use_chat_template=model.gen_cfg.use_chat_template,
                )
                for _ in range(samples_per_example):
                    prompts_to_generate.append({"ex": ex, "built": built})

            logger.info(
                f"🔄 Worker {worker_id} prepared {len(prompts_to_generate)} prompts, now batching generations..."
            )

            # PHASE 2: Batch generate y_a (sampled)
            all_prompts = [item["built"] for item in prompts_to_generate]
            y_a_list = model.continue_from_context_batch(
                all_prompts, gen_cfg.max_new_tokens, greedy=False, batch_size=batch_size
            )

            # PHASE 3: Batch generate y_b (sampled)
            y_b_list = model.continue_from_context_batch(
                all_prompts, gen_cfg.max_new_tokens, greedy=False, batch_size=batch_size
            )

            # PHASE 4: Assemble generated samples
            generated_samples = []
            for item, y_a, y_b in zip(prompts_to_generate, y_a_list, y_b_list):
                generated_samples.append(
                    {
                        "ex": item["ex"],
                        "y_a": y_a,
                        "y_b": y_b,
                    }
                )

            logger.info(
                f"✅ Worker {worker_id} generated {len(generated_samples)} samples, now batch scoring..."
            )

            # PHASE 5: Batch score all samples at once
            score_pairs = [
                (s["ex"]["question"], s["y_a"], s["y_b"]) for s in generated_samples
            ]
            scores = reward_scorer.score_batch(score_pairs)

            # PHASE 6: Create records with scores
            records = []
            for sample, (score_a, score_b, preferred) in zip(generated_samples, scores):
                records.append(
                    {
                        "question": sample["ex"]["question"],
                        "answer": sample["ex"]["answer"],
                        "prompt": sample["ex"]["prompt"],
                        "y_a": sample["y_a"],
                        "y_b": sample["y_b"],
                        "score_a": score_a,
                        "score_b": score_b,
                        "preferred": preferred,
                    }
                )

            logger.info(f"✅ Worker {worker_id} generated {len(records)} records")

            from pita.core.io import get_checkpoint_dir, save_checkpoint

            if run_root is None:
                run_root = get_run_root()
            checkpoint_dir = get_checkpoint_dir(
                self.algo_key, dataset, family, round_idx, run_root
            )
            save_checkpoint(checkpoint_dir, worker_id, records)
            logger.info(f"💾 Worker {worker_id} saved checkpoint")

            result_queue.put(records)

        except Exception as e:
            logger.error(f"Worker {worker_id} failed: {e}")
            import traceback

            traceback.print_exc()
            result_queue.put(e)
        finally:
            if "reward_scorer" in locals() and reward_scorer is not None:
                reward_scorer.cleanup()
            if "model" in locals():
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def score_samples(self, ex, y_a: str, y_b: str) -> Tuple[float, float, int]:
        r_a, r_b, preferred = self._reward.score_pair(ex.question, y_a, y_b)
        return r_a, r_b, preferred

    # Evaluation handled directly in algorithms
