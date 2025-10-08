"""Parallel data generation across multiple GPUs."""

from __future__ import annotations

import os
import random
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import torch
import torch.multiprocessing as mp
from loguru import logger
from tqdm import tqdm

from pita.core.gpu_manager import get_gpu_manager, GPUManager


@dataclass
class GenerationJob:
    """A single data generation job."""

    example_idx: int
    question: str
    answer: str
    prompt: str
    samples_per_example: int


@dataclass
class GenerationResult:
    """Result from a data generation job."""

    question: str
    answer: str
    prompt: str
    t: int
    context: str
    y_a: str
    y_b: str
    score_a: float
    score_b: float
    preferred: int


class ParallelDataGenerator:
    """Manages parallel data generation across multiple GPUs."""

    def __init__(
        self,
        gpu_manager: GPUManager,
        num_workers: Optional[int] = None,
    ):
        self.gpu_manager = gpu_manager
        self.num_workers = num_workers or max(1, gpu_manager.num_gpus)

    def generate_data_parallel(
        self,
        examples: List[Dict[str, Any]],
        model_factory,
        reward_scorer_factory,
        samples_per_example: int,
        max_new_tokens: int,
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """Generate training data in parallel across GPUs.

        Args:
            examples: List of examples to process
            model_factory: Function that creates a model on a specific GPU
            reward_scorer_factory: Function that creates a reward scorer
            samples_per_example: Number of samples per example
            max_new_tokens: Max tokens for generation
            seed: Random seed

        Returns:
            List of generated training records
        """
        if self.gpu_manager.num_gpus <= 1:
            # Fall back to single-GPU processing
            logger.info("Single GPU mode - using sequential processing")
            return self._generate_sequential(
                examples,
                model_factory,
                reward_scorer_factory,
                samples_per_example,
                max_new_tokens,
                seed,
            )

        # Split examples across workers
        num_examples = len(examples)
        jobs_per_worker = self.gpu_manager.distribute_work(num_examples)

        logger.info(
            f"Distributing {num_examples} examples across {self.num_workers} workers"
        )

        # Use spawn method for CUDA compatibility
        mp_context = mp.get_context("spawn")
        result_queue = mp_context.Queue()
        processes = []

        try:
            for worker_id in range(self.num_workers):
                job_indices = jobs_per_worker[worker_id]
                if not job_indices:
                    continue

                worker_examples = [examples[i] for i in job_indices]
                gpu_id = self.gpu_manager.available_gpus[
                    worker_id % self.gpu_manager.num_gpus
                ]

                p = mp_context.Process(
                    target=self._worker_process,
                    args=(
                        worker_id,
                        gpu_id,
                        worker_examples,
                        model_factory,
                        reward_scorer_factory,
                        samples_per_example,
                        max_new_tokens,
                        seed + worker_id,
                        result_queue,
                    ),
                )
                p.start()
                processes.append(p)

            # Collect results
            all_records = []
            expected_results = sum(len(jobs) for jobs in jobs_per_worker if jobs)

            with tqdm(total=expected_results, desc="Generating data") as pbar:
                for _ in range(expected_results):
                    worker_records = result_queue.get()
                    if isinstance(worker_records, Exception):
                        raise worker_records
                    all_records.extend(worker_records)
                    pbar.update(1)

            # Wait for all processes
            for p in processes:
                p.join()

            return all_records

        except Exception as e:
            logger.error(f"Error in parallel generation: {e}")
            # Cleanup
            for p in processes:
                if p.is_alive():
                    p.terminate()
                    p.join()
            raise

    @staticmethod
    def _worker_process(
        worker_id: int,
        gpu_id: int,
        examples: List[Dict[str, Any]],
        model_factory,
        reward_scorer_factory,
        samples_per_example: int,
        max_new_tokens: int,
        seed: int,
        result_queue,
    ):
        """Worker process for parallel generation."""
        try:
            # Set device for this worker
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            torch.cuda.set_device(0)  # After setting CUDA_VISIBLE_DEVICES, use device 0

            random.seed(seed)
            torch.manual_seed(seed)

            # Create model and reward scorer on this GPU
            model = model_factory(
                device_id=0
            )  # Use 0 after setting CUDA_VISIBLE_DEVICES
            reward_scorer = reward_scorer_factory(device_id=0)

            # Process examples
            for ex in examples:
                records = ParallelDataGenerator._process_single_example(
                    ex,
                    model,
                    reward_scorer,
                    samples_per_example,
                    max_new_tokens,
                )
                # Send results back one example at a time to show progress
                result_queue.put(records)

        except Exception as e:
            logger.error(f"Worker {worker_id} failed: {e}")
            result_queue.put(e)

    @staticmethod
    def _process_single_example(
        ex: Dict[str, Any],
        model,
        reward_scorer,
        samples_per_example: int,
        max_new_tokens: int,
    ) -> List[Dict[str, Any]]:
        """Process a single example to generate training data."""
        records = []

        # Generate rollout
        rollout = model.roll_in(ex["prompt"], max_roll_tokens=max_new_tokens)

        built_prompt = rollout["prompt"]
        prompt_token_count = len(model.tokenizer(built_prompt)["input_ids"])
        context_token_ids = rollout["context_ids"].tolist()
        full_token_count = len(context_token_ids)
        rollout_token_count = full_token_count - prompt_token_count

        if rollout_token_count < 2:
            return records  # Skip short rollouts

        # Generate multiple samples
        for _ in range(samples_per_example):
            cutoff_tokens = random.randint(1, rollout_token_count - 1)

            context_prefix_ids = context_token_ids[: prompt_token_count + cutoff_tokens]
            context_text = model.tokenizer.decode(
                context_prefix_ids, skip_special_tokens=True
            )
            solution_prefix_ids = context_prefix_ids[prompt_token_count:]
            solution_prefix_text = model.tokenizer.decode(
                solution_prefix_ids, skip_special_tokens=True
            )

            remaining_token_budget = max_new_tokens - cutoff_tokens
            y_ref = model.continue_from_context(
                context_text, max_new_tokens=remaining_token_budget, greedy=True
            )
            y_sample = model.continue_from_context(
                context_text, max_new_tokens=remaining_token_budget, greedy=False
            )

            y_a = solution_prefix_text + y_ref
            y_b = solution_prefix_text + y_sample

            # Score samples
            score_a, score_b, preferred = reward_scorer.score_pair(
                ex["question"], y_a, y_b
            )

            records.append(
                {
                    "question": ex["question"],
                    "answer": ex["answer"],
                    "prompt": ex["prompt"],
                    "t": cutoff_tokens,
                    "context": ex["prompt"] + solution_prefix_text,
                    "y_a": y_a,
                    "y_b": y_b,
                    "score_a": score_a,
                    "score_b": score_b,
                    "preferred": preferred,
                }
            )

        return records

    def _generate_sequential(
        self,
        examples: List[Dict[str, Any]],
        model_factory,
        reward_scorer_factory,
        samples_per_example: int,
        max_new_tokens: int,
        seed: int,
    ) -> List[Dict[str, Any]]:
        """Fallback sequential generation for single GPU."""
        random.seed(seed)
        torch.manual_seed(seed)

        device_id = 0 if self.gpu_manager.num_gpus > 0 else -1
        model = model_factory(device_id=device_id)
        reward_scorer = reward_scorer_factory(device_id=device_id)

        all_records = []
        for ex in tqdm(examples, desc="Generating data (sequential)"):
            records = self._process_single_example(
                ex,
                model,
                reward_scorer,
                samples_per_example,
                max_new_tokens,
            )
            all_records.extend(records)

        return all_records


class BatchedRewardScorer:
    """Wrapper for reward scoring with batching and GPU optimization."""

    def __init__(
        self,
        base_scorer,
        batch_size: int = 32,
        device: Optional[int] = None,
    ):
        self.base_scorer = base_scorer
        self.batch_size = batch_size
        self.device = device

        # Update the base scorer's batch size
        if hasattr(self.base_scorer, "_batch_size"):
            self.base_scorer._batch_size = self.batch_size

    def score_pair(self, question: str, y_a: str, y_b: str) -> Tuple[float, float, int]:
        """Score a single pair (delegates to base scorer)."""
        return self.base_scorer.score_pair(question, y_a, y_b)

    def score_pairs_batch(
        self,
        questions: List[str],
        y_a_list: List[str],
        y_b_list: List[str],
    ) -> List[Tuple[float, float, int]]:
        """Score multiple pairs in batch."""
        results = []
        for q, ya, yb in zip(questions, y_a_list, y_b_list):
            results.append(self.base_scorer.score_pair(q, ya, yb))
        return results
