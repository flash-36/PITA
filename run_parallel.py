"""Parallel multi-GPU version of the main training script."""

from __future__ import annotations

from typing import Dict, Any
import os

# CRITICAL: Set CUDA allocator config BEFORE importing torch
# This reduces memory fragmentation by allowing PyTorch to expand memory segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import random
import numpy as np
import torch

import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger

from pita.core.io import create_subdir, save_json, get_run_root
from pita.core.registry import get_algorithm_registry
from pita.core.parallel_executor import (
    ParallelJobExecutor,
    TrainingJob,
    JobResult,
    create_job_list,
    group_jobs_by_dependencies,
)
from pita.core.gpu_manager import get_gpu_manager
import pita.algos  # trigger registration imports
import pita.datasets  # trigger dataset registration
from pita.plotting.hooks import plot_after_run


def execute_single_job(job: TrainingJob, device_id: int, args: tuple) -> JobResult:
    """Execute a single training job.

    Args:
        job: Job to execute
        device_id: GPU device ID (or -1 for CPU)
        args: Tuple of (cfg_dict, run_root_str) passed from main

    Returns:
        Job result
    """
    from omegaconf import OmegaConf
    from pathlib import Path

    # Import algorithms to trigger registration in worker process
    import pita.algos
    import pita.datasets

    # Unpack arguments
    cfg_dict, run_root_str = args

    # Reconstruct OmegaConf object from dict
    cfg = OmegaConf.create(cfg_dict)
    run_root = Path(run_root_str)

    try:
        logger.info(f"🔧 Executing job: {job} on device {device_id}")

        # Restrict this job to only use its assigned GPU
        # This prevents conflicts when multiple jobs run in parallel
        if device_id >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
            logger.info(f"🔒 Restricted job to GPU {device_id}")

        # Disable DataLoader workers in parallel mode to avoid CUDA IPC issues
        # Each job already has dedicated GPU resources
        os.environ["PITA_PARALLEL_MODE"] = "1"

        algo_registry = get_algorithm_registry()
        algo_cfg = cfg.algos[job.algo_key]
        algo_cls = algo_registry.get(job.algo_key)

        if algo_cls is None:
            raise KeyError(f"Algorithm '{job.algo_key}' is not registered")

        algo = algo_cls(algo_cfg)

        # Resolve reference model for this round
        ref_for_train = algo.resolve_ref_for_round(
            run_root=run_root,
            dataset=job.dataset_name,
            family=job.family_name,
            ref_model_alias=job.ref_model_alias,
            round_idx=job.round_idx,
        )

        # Generate data
        logger.info(f"🧪 Generating data: {job}")
        algo.generate_data(
            cfg=cfg,
            ref_model=ref_for_train,
            cls_model=job.value_model_alias,
            dataset=job.dataset_name,
            family=job.family_name,
            round_idx=job.round_idx,
            run_root=run_root,
        )

        # Train
        logger.info(f"🏋️ Training: {job}")
        out_dir = create_subdir(
            run_root,
            [
                "results",
                job.algo_key,
                f"{job.family_name}",
                job.dataset_name,
                f"r{job.round_idx + 1}",
            ],
        )

        # Temporarily set a minimal Hydra-like context or pass run_root explicitly
        # Check if algorithm's run() accepts run_root parameter
        import inspect

        sig = inspect.signature(algo.run)
        if "run_root" in sig.parameters:
            result = algo.run(
                cfg=cfg,
                ref_model=ref_for_train,
                cls_model=job.value_model_alias,
                dataset=job.dataset_name,
                family=job.family_name,
                output_dir=out_dir,
                round_idx=job.round_idx,
                run_root=run_root,
            )
        else:
            result = algo.run(
                cfg=cfg,
                ref_model=ref_for_train,
                cls_model=job.value_model_alias,
                dataset=job.dataset_name,
                family=job.family_name,
                output_dir=out_dir,
                round_idx=job.round_idx,
            )

        save_json(out_dir / "result.json", result or {})

        logger.info(f"✅ Completed job: {job}")

        # Explicit cleanup to free memory before returning
        del algo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        return JobResult(
            job=job,
            success=True,
            result=result,
        )

    except Exception as e:
        logger.error(f"❌ Job failed: {job} - {e}")
        import traceback

        traceback.print_exc()

        # Cleanup on failure too
        if "algo" in locals():
            del algo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return JobResult(
            job=job,
            success=False,
            error=str(e),
        )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training loop with parallel execution."""
    run_root = get_run_root()

    logger.add(
        str(run_root / "run.log"),
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
        level="INFO",
        enqueue=True,
    )
    logger.info("🧾 Loaded config:\n{}", OmegaConf.to_yaml(cfg, resolve=True))

    exp_name = cfg.experiment.name
    if exp_name in (None, "", "???"):
        raise ValueError("experiment.name must be set (non-empty)")

    # Global seeding for reproducibility
    seed = int(cfg.experiment.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    logger.info("📂 Run directory: {}", run_root)

    # Initialize GPU manager
    gpu_manager = get_gpu_manager()
    gpu_manager.print_memory_summary()

    # Determine execution mode
    parallel_exec_enabled = (
        getattr(cfg, "use_parallel_execution", True)
        if hasattr(cfg, "use_parallel_execution")
        else True
    )
    use_parallel = parallel_exec_enabled

    if use_parallel:
        logger.info(f"🚀 Using PARALLEL execution with {gpu_manager.num_gpus} GPUs")
    else:
        logger.info("🐌 Using SEQUENTIAL execution")

    # Create job list
    all_jobs = create_job_list(cfg)
    logger.info(f"📋 Created {len(all_jobs)} training jobs")

    # Group jobs by round (for dependency handling)
    job_batches = group_jobs_by_dependencies(all_jobs)
    logger.info(f"📦 Grouped into {len(job_batches)} batches by round")

    # Execute jobs
    executor = ParallelJobExecutor(max_concurrent_jobs=gpu_manager.num_gpus)
    all_results: Dict[str, Any] = {}

    # Convert config to plain dict for pickling (OmegaConf objects don't pickle well with spawn)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    run_root_str = str(run_root)
    executor_args = (cfg_dict, run_root_str)

    for batch_idx, batch in enumerate(job_batches):
        logger.info(
            f"🔄 Executing batch {batch_idx + 1}/{len(job_batches)} ({len(batch)} jobs)"
        )

        # Execute batch with args
        if use_parallel:
            batch_results = executor.execute_jobs_parallel(
                batch, execute_single_job, executor_args
            )
        else:
            batch_results = executor.execute_jobs_sequential(
                batch, execute_single_job, executor_args
            )

        # Process results
        for job_result in batch_results:
            if job_result.success and job_result.result:
                job = job_result.job
                all_results.setdefault(job.algo_key, {}).setdefault(
                    f"{job.family_name}", {}
                ).setdefault(job.dataset_name, {})[
                    f"r{job.round_idx + 1}"
                ] = job_result.result

        # Print batch summary
        success_count = sum(1 for r in batch_results if r.success)
        logger.info(
            f"📊 Batch {batch_idx + 1} complete: {success_count}/{len(batch)} succeeded"
        )

        # Clear cache between batches
        gpu_manager.clear_cache()
        gpu_manager.synchronize_all()

    # Generate plots
    figs_dir = create_subdir(run_root, ["figures"])
    try:
        plot_after_run(cfg=cfg, results=all_results, output_dir=figs_dir)
    except Exception as e:
        logger.warning(f"Failed to generate plots: {e}")

    # Final summary
    logger.info("✅ All batches complete!")
    logger.info(f"📊 Final summary: Check {run_root} for results")

    gpu_manager.print_memory_summary()
    logger.info("🎉 Done.")


if __name__ == "__main__":
    # Set multiprocessing start method for CUDA
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)

    main()
