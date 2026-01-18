"""Parallel execution of training jobs across multiple GPUs."""

from __future__ import annotations

import os
import sys
from typing import List, Dict, Any, Callable, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.multiprocessing as mp
from queue import Empty
from loguru import logger
import traceback

from pita.core.gpu_manager import get_gpu_manager
from pita.core.job_state import JobStateManager, JobState


def _is_device_assert(e: BaseException) -> bool:
    """Check if exception is a CUDA device-side assert."""
    msg = str(e).lower()
    return (
        "device-side assert triggered" in msg
        or "device side assert triggered" in msg
        or "cuda error: device-side assert triggered" in msg
        or "_assert_async_cuda_kernel" in msg
    )


@dataclass
class TrainingJob:
    """A single training job specification."""

    job_id: str
    algo_key: str
    family_name: str
    ref_model_alias: str
    value_model_alias: str
    dataset_name: str
    round_idx: int

    def __str__(self) -> str:
        return f"{self.algo_key}:{self.dataset_name}:{self.family_name}:r{self.round_idx+1}"


@dataclass
class JobResult:
    """Result from a training job."""

    job: TrainingJob
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ParallelJobExecutor:
    """Executes training jobs in parallel across multiple GPUs."""

    def __init__(
        self,
        max_concurrent_jobs: Optional[int] = None,
        gpu_memory_buffer_gb: float = 5.0,
    ):
        """Initialize parallel executor.

        Args:
            max_concurrent_jobs: Maximum number of concurrent jobs (default: num GPUs)
            gpu_memory_buffer_gb: Memory buffer to keep free on each GPU
        """
        self.gpu_manager = get_gpu_manager()
        self.max_concurrent = max_concurrent_jobs or max(1, self.gpu_manager.num_gpus)
        self.memory_buffer = gpu_memory_buffer_gb * (1024**3)  # Convert to bytes

        logger.info(
            f"Initialized ParallelJobExecutor with {self.max_concurrent} max concurrent jobs"
        )

    def execute_jobs_sequential(
        self,
        jobs: List[TrainingJob],
        job_executor: Callable[[TrainingJob, int, Any], JobResult],
        executor_args: Any = None,
    ) -> List[JobResult]:
        """Execute jobs sequentially (fallback for single GPU or debugging).

        Args:
            jobs: List of jobs to execute
            job_executor: Function that executes a single job

        Returns:
            List of job results
        """
        logger.info(f"Executing {len(jobs)} jobs sequentially")

        results = []
        for job in jobs:
            logger.info(f"Starting job: {job}")
            # Use GPU 0 if available, otherwise CPU
            gpu_id = 0 if self.gpu_manager.num_gpus > 0 else -1
            result = job_executor(job, gpu_id, executor_args)
            results.append(result)

            if result.success:
                logger.info(f"✓ Job completed: {job}")
            else:
                logger.error(f"✗ Job failed: {job} - {result.error}")

        return results

    def execute_jobs_parallel(
        self,
        jobs: List[TrainingJob],
        job_executor: Callable[[TrainingJob, int, Any], JobResult],
        executor_args: Any = None,
    ) -> List[JobResult]:
        """Execute jobs in parallel across multiple GPUs.

        Args:
            jobs: List of jobs to execute
            job_executor: Function that executes a single job

        Returns:
            List of job results
        """
        if self.gpu_manager.num_gpus <= 1:
            return self.execute_jobs_sequential(jobs, job_executor, executor_args)

        logger.info(
            f"Executing {len(jobs)} jobs in parallel across {self.gpu_manager.num_gpus} GPUs"
        )

        # Use spawn for CUDA compatibility
        mp_context = mp.get_context("spawn")

        # Create job and result queues
        job_queue = mp_context.Queue()
        result_queue = mp_context.Queue()

        try:
            # Fill job queue
            for job in jobs:
                job_queue.put(job)

            # Add sentinel values to signal workers to stop
            for _ in range(self.max_concurrent):
                job_queue.put(None)

            # Start worker processes
            workers = []
            for worker_id in range(self.max_concurrent):
                gpu_id = self.gpu_manager.available_gpus[
                    worker_id % self.gpu_manager.num_gpus
                ]

                p = mp_context.Process(
                    target=self._worker_loop,
                    args=(
                        worker_id,
                        gpu_id,
                        job_queue,
                        result_queue,
                        job_executor,
                        executor_args,
                    ),
                )
                p.start()
                workers.append(p)

            # Collect results
            results = []
            completed = 0
            total = len(jobs)

            from tqdm import tqdm

            with tqdm(total=total, desc="Training jobs") as pbar:
                while completed < total:
                    try:
                        result = result_queue.get(timeout=1)
                        results.append(result)
                        completed += 1
                        pbar.update(1)

                        if result.success:
                            logger.info(f"✓ Completed: {result.job}")
                        else:
                            logger.error(f"✗ Failed: {result.job} - {result.error}")

                    except Empty:
                        continue

            # Wait for all workers to finish
            for p in workers:
                p.join()
        finally:
            job_queue.close()
            result_queue.close()
            job_queue.join_thread()
            result_queue.join_thread()

        return results

    @staticmethod
    def _worker_loop(
        worker_id: int,
        gpu_id: int,
        job_queue: mp.Queue,
        result_queue: mp.Queue,
        job_executor: Callable[[TrainingJob, int, Any], JobResult],
        executor_args: Any = None,
    ):
        """Worker process that executes jobs from the queue.

        Args:
            worker_id: Worker ID
            gpu_id: GPU device ID for this worker
            job_queue: Queue of jobs to execute
            result_queue: Queue to put results
            job_executor: Function to execute a job
        """
        try:
            # Set GPU for this worker
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            torch.cuda.set_device(0)

            logger.info(f"Worker {worker_id} started on GPU {gpu_id}")

            while True:
                try:
                    job = job_queue.get(timeout=1)

                    # Sentinel value to stop
                    if job is None:
                        break

                    logger.info(f"Worker {worker_id} executing: {job}")

                    # Execute job with device_id=0 since we set CUDA_VISIBLE_DEVICES
                    result = job_executor(job, 0, executor_args)
                    result_queue.put(result)

                    # Clean up GPU memory
                    torch.cuda.empty_cache()

                except Empty:
                    continue
                except Exception as e:
                    logger.error(f"Worker {worker_id} error: {e}")
                    traceback.print_exc()

                    if _is_device_assert(e):
                        logger.error(
                            f"🔥 Worker {worker_id} encountered CUDA device-side assert; terminating worker to reset CUDA context."
                        )
                        if "job" in locals():
                            result_queue.put(
                                JobResult(
                                    job=job,
                                    success=False,
                                    error=f"CUDA device-side assert: {str(e)}",
                                )
                            )
                        os._exit(1)

                    if "job" in locals():
                        result_queue.put(
                            JobResult(
                                job=job,
                                success=False,
                                error=str(e),
                            )
                        )

                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

            logger.info(f"Worker {worker_id} finished")

        except Exception as e:
            logger.error(f"Worker {worker_id} fatal error: {e}")
            traceback.print_exc()

    def execute_jobs(
        self,
        jobs: List[TrainingJob],
        job_executor: Callable[[TrainingJob, int, Any], JobResult],
        executor_args: Any = None,
        force_sequential: bool = False,
    ) -> List[JobResult]:
        """Execute jobs with automatic parallel/sequential selection.

        Args:
            jobs: List of jobs to execute
            job_executor: Function that executes a single job
            force_sequential: Force sequential execution

        Returns:
            List of job results
        """
        if force_sequential or self.gpu_manager.num_gpus <= 1:
            return self.execute_jobs_sequential(jobs, job_executor, executor_args)
        else:
            return self.execute_jobs_parallel(jobs, job_executor, executor_args)

    def execute_jobs_with_priority(
        self,
        jobs: List[TrainingJob],
        job_executor: Callable[[TrainingJob, int, Any], JobResult],
        executor_args: Any = None,
        state_manager: Optional[JobStateManager] = None,
    ) -> List[JobResult]:
        """Execute jobs in parallel with priority-based scheduling.

        Jobs closer to completion get higher priority so eval results come faster.

        Priority order (highest to lowest):
        1. MODEL_TRAINED (just needs eval)
        2. DATA_GENERATED (needs training + eval)
        3. NOT_STARTED (needs data gen + training + eval)

        Args:
            jobs: List of jobs to execute
            job_executor: Function that executes a single job
            executor_args: Arguments to pass to job executor
            state_manager: Job state manager for priority calculation

        Returns:
            List of job results
        """
        if self.gpu_manager.num_gpus <= 1:
            return self.execute_jobs_sequential(jobs, job_executor, executor_args)

        logger.info(
            f"🚀 Starting priority-based parallel execution across {self.gpu_manager.num_gpus} GPUs"
        )

        # Use spawn for CUDA compatibility
        mp_context = mp.get_context("spawn")

        # Create priority job queue and result queue
        with mp_context.Manager() as manager:
            job_queue = manager.Queue()
            result_queue = mp_context.Queue()

            # Calculate priorities and add jobs to queue
            def get_job_priority(job: TrainingJob) -> int:
                """Calculate priority for a job (higher = more urgent)."""
                if state_manager is None:
                    base_priority = 0
                else:
                    state = state_manager.get_state(job.job_id)
                    if state == JobState.EVAL_COMPLETED:
                        return 1000  # Already done, will be skipped
                    elif state == JobState.MODEL_TRAINED:
                        base_priority = 900  # Just needs eval
                    elif state == JobState.DATA_GENERATED:
                        base_priority = 500  # Needs training + eval
                    else:
                        base_priority = 100  # Needs everything

                # Within same state, prioritize earlier rounds
                round_priority = -job.round_idx * 10

                return base_priority + round_priority

            # Sort jobs by priority (highest first)
            prioritized_jobs = sorted(jobs, key=get_job_priority, reverse=True)

            logger.info("📊 Job priority breakdown:")
            priority_counts = {}
            for job in prioritized_jobs:
                priority = get_job_priority(job)
                stage = (
                    "COMPLETED"
                    if priority >= 1000
                    else (
                        "MODEL_TRAINED"
                        if priority >= 900
                        else "DATA_GENERATED" if priority >= 500 else "NOT_STARTED"
                    )
                )
                priority_counts[stage] = priority_counts.get(stage, 0) + 1
            for stage, count in sorted(priority_counts.items()):
                logger.info(f"  {stage}: {count} jobs")

            # Fill job queue with prioritized jobs
            for job in prioritized_jobs:
                job_queue.put(job)

            # Add sentinel values to signal workers to stop
            for _ in range(self.max_concurrent):
                job_queue.put(None)

            # Start worker processes
            workers = []
            for worker_id in range(self.max_concurrent):
                gpu_id = self.gpu_manager.available_gpus[
                    worker_id % self.gpu_manager.num_gpus
                ]

                p = mp_context.Process(
                    target=self._worker_loop,
                    args=(
                        worker_id,
                        gpu_id,
                        job_queue,
                        result_queue,
                        job_executor,
                        executor_args,
                    ),
                )
                p.start()
                workers.append(p)

            # Collect results
            results = []
            completed = 0
            total = len(jobs)

            from tqdm import tqdm

            with tqdm(total=total, desc="⚙️  Jobs", unit="job") as pbar:
                while completed < total:
                    try:
                        result = result_queue.get(timeout=1)
                        results.append(result)
                        completed += 1
                        pbar.update(1)

                        if result.success:
                            logger.info(f"✅ Completed: {result.job}")
                        else:
                            logger.error(f"❌ Failed: {result.job} - {result.error}")

                    except Empty:
                        continue

            # Wait for all workers to finish
            for p in workers:
                p.join()

            result_queue.close()
            result_queue.join_thread()

        logger.info(f"🎉 All jobs complete!")
        return results


def create_job_list(cfg) -> List[TrainingJob]:
    """Create a list of training jobs from configuration.

    Args:
        cfg: Hydra configuration

    Returns:
        List of training jobs
    """
    from pita.models.catalog import resolve_family_pair

    jobs = []

    model_pairs = []
    for family in cfg.model_pairs:
        ref_model_alias, value_model_alias = resolve_family_pair(str(family))
        model_pairs.append((str(family), ref_model_alias, value_model_alias))

    datasets = list(cfg.training.datasets)
    rounds = int(cfg.rounds_of_training)

    for algo_key in cfg.algos.keys():
        for family_name, ref_model_alias, value_model_alias in model_pairs:
            for dataset_name in datasets:
                for round_idx in range(rounds):
                    job_id = f"{algo_key}_{family_name}_{dataset_name}_r{round_idx+1}"
                    job = TrainingJob(
                        job_id=job_id,
                        algo_key=algo_key,
                        family_name=family_name,
                        ref_model_alias=ref_model_alias,
                        value_model_alias=value_model_alias,
                        dataset_name=dataset_name,
                        round_idx=round_idx,
                    )
                    jobs.append(job)

    return jobs


def estimate_job_priority(job: TrainingJob, cfg) -> int:
    """Estimate priority for a job (higher = more important).

    Can be used to schedule important jobs first.

    Args:
        job: Training job
        cfg: Configuration

    Returns:
        Priority value
    """
    # Later rounds have higher priority (need results from earlier rounds)
    round_priority = job.round_idx * 100

    # Could add other factors:
    # - Model size (smaller models first to get quick results)
    # - Dataset size (smaller datasets first)

    return round_priority


def group_jobs_by_dependencies(jobs: List[TrainingJob]) -> List[List[TrainingJob]]:
    """Group jobs into batches based on round dependencies.

    Jobs in the same batch can run in parallel. Each batch must complete
    before the next batch starts.

    Args:
        jobs: List of all jobs

    Returns:
        List of job batches
    """
    # Group by round
    rounds_dict = {}
    for job in jobs:
        if job.round_idx not in rounds_dict:
            rounds_dict[job.round_idx] = []
        rounds_dict[job.round_idx].append(job)

    # Return as ordered list of batches
    batches = []
    for round_idx in sorted(rounds_dict.keys()):
        batches.append(rounds_dict[round_idx])

    return batches
