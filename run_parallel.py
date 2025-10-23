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
        args: Tuple of (cfg_dict, run_root_str, state_manager_dict) passed from main

    Returns:
        Job result
    """
    from omegaconf import OmegaConf
    from pathlib import Path
    from pita.core.job_state import JobStateManager, JobState
    from pita.core.io import create_subdir
    import json

    # Import algorithms to trigger registration in worker process
    import pita.algos
    import pita.datasets

    # Unpack arguments
    cfg_dict, run_root_str, state_manager_dict = args

    # Reconstruct OmegaConf object from dict
    cfg = OmegaConf.create(cfg_dict)
    run_root = Path(run_root_str)

    # Reconstruct state manager
    state_manager = JobStateManager(run_root)
    if state_manager_dict:
        state_manager.states = state_manager_dict

    try:
        logger.info(f"🔧 Executing job: {job} on device {device_id}")

        # Check current state
        current_state = state_manager.get_state(job.job_id)

        # Skip if already completed
        if current_state == JobState.EVAL_COMPLETED:
            logger.info(f"⏭️  Skipping completed job: {job}")
            # Load existing result
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
            result_file = out_dir / "result.json"
            result = {}
            if result_file.exists():
                with open(result_file, "r") as f:
                    result = json.load(f)
            return JobResult(job=job, success=True, result=result)

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

        # Generate data (skip if already done)
        if current_state in (
            JobState.DATA_GENERATED,
            JobState.MODEL_TRAINED,
            JobState.EVAL_COMPLETED,
        ):
            logger.info(f"⏭️  Skipping data generation (already done): {job}")
        else:
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
            state_manager.update_state(job.job_id, JobState.DATA_GENERATED)

        # Train and evaluate
        if current_state == JobState.EVAL_COMPLETED:
            logger.info(f"⏭️  Job already completed: {job}")
            result = None
        else:
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

            if current_state in (JobState.MODEL_TRAINED, JobState.EVAL_COMPLETED):
                logger.info(f"⏭️  Skipping training (model already trained): {job}")
            else:
                logger.info(f"🏋️ Training: {job}")

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

            state_manager.update_state(job.job_id, JobState.MODEL_TRAINED)

            save_json(out_dir / "result.json", result or {})
            state_manager.update_state(job.job_id, JobState.EVAL_COMPLETED)

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

        state_manager.update_state(job.job_id, JobState.FAILED)

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
    from pathlib import Path
    from pita.core.job_state import JobStateManager
    from datetime import datetime

    # Check if resuming from a previous run
    resume_from = cfg.get("resume_from", None)
    if resume_from:
        resume_path = Path(resume_from)
        if not resume_path.exists():
            raise ValueError(f"Resume path does not exist: {resume_from}")

        logger.info(f"🔄 Resuming from: {resume_from}")

        # Load config from the resume directory
        original_config_path = resume_path / ".hydra" / "config.yaml"
        if not original_config_path.exists():
            raise ValueError(f"No config found at: {original_config_path}")

        original_cfg = OmegaConf.load(original_config_path)

        # Merge current config overrides into original config
        # Remove resume_from from overrides to avoid recursion
        current_overrides = OmegaConf.to_container(cfg, resolve=False)
        current_overrides.pop("resume_from", None)

        # Merge overrides into original config
        cfg = OmegaConf.merge(original_cfg, current_overrides)

        # Save the updated config with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        resume_config_path = resume_path / ".hydra" / f"config_resume_{timestamp}.yaml"
        resume_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(resume_config_path, "w") as f:
            OmegaConf.save(cfg, f)
        logger.info(f"💾 Saved resumed config to: {resume_config_path}")

        run_root = resume_path
    else:
        run_root = get_run_root()

    logger.add(
        str(run_root / "run.log"),
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
        level="INFO",
        enqueue=True,
    )

    if resume_from:
        logger.info("=" * 80)
        logger.info(f"🔄 RESUMING RUN FROM: {resume_from}")
        logger.info(
            f"⏰ Resume timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        logger.info("=" * 80)

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

    # Initialize state manager
    state_manager = JobStateManager(run_root)
    state_manager.initialize_from_jobs(all_jobs)

    # Group jobs by round (for dependency handling)
    job_batches = group_jobs_by_dependencies(all_jobs)
    logger.info(f"📦 Grouped into {len(job_batches)} batches by round")

    # Execute jobs
    executor = ParallelJobExecutor(max_concurrent_jobs=gpu_manager.num_gpus)
    all_results: Dict[str, Any] = {}

    # Convert config to plain dict for pickling (OmegaConf objects don't pickle well with spawn)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    run_root_str = str(run_root)
    state_manager_dict = state_manager.states
    executor_args = (cfg_dict, run_root_str, state_manager_dict)

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
