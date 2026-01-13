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
from pita.core.job_state import JobStateManager, JobState
from pita.core.registry import get_algorithm_registry
from pita.core.parallel_executor import (
    ParallelJobExecutor,
    TrainingJob,
    JobResult,
    create_job_list,
    group_jobs_by_dependencies,
)
from pita.core.gpu_manager import get_gpu_manager
from pita.core.logging_context import setup_context_logging
import pita.algos  # trigger registration imports
import pita.datasets  # trigger dataset registration
from pita.plotting.hooks import plot_after_run


def evaluate_base_models(cfg: DictConfig, run_root, state_manager: JobStateManager) -> Dict[str, Any]:
    """Evaluate base reference models on all datasets.
    
    Uses job state tracking for resume capability.
    Returns dict of {family: {dataset: metrics}}.
    """
    from pathlib import Path
    from pita.models.hf import HFModel, GenerationConfig
    from pita.models.catalog import resolve_family_pair
    from pita.eval.evaluate import evaluate_pass1_maj8, evaluate_avg_reward
    from pita.core.io import ensure_dir, check_base_eval_completed
    
    results_dir = Path(run_root) / "results" / "base_model"
    
    logger.info("=" * 80)
    logger.info("📊 EVALUATING BASE REFERENCE MODEL")
    logger.info("=" * 80)
    
    results: Dict[str, Any] = {}
    model_pairs = list(cfg.model_pairs)
    
    # Collect all unique eval datasets from datasets_by_train
    eval_datasets = set()
    for train_ds in cfg.training.datasets:
        eval_datasets.update(cfg.evaluation.datasets_by_train.get(train_ds, [train_ds]))
    datasets = sorted(eval_datasets)
    
    # Count completed vs pending
    pending_evals = []
    for family in model_pairs:
        for dataset in datasets:
            job_id = f"base_model_{family}_{dataset}"
            if state_manager.get_state(job_id) != JobState.EVAL_COMPLETED:
                if not check_base_eval_completed(family, dataset, Path(run_root)):
                    pending_evals.append((family, dataset, job_id))
                else:
                    state_manager.update_state(job_id, JobState.EVAL_COMPLETED)
    
    if not pending_evals:
        logger.info("📂 All base model evaluations already completed")
        # Load cached results
        for family in model_pairs:
            results[family] = {}
            for dataset in datasets:
                result_file = results_dir / family / dataset / "results.json"
                if result_file.exists():
                    import json
                    with open(result_file) as f:
                        results[family][dataset] = json.load(f)
        return results
    
    logger.info(f"📋 {len(pending_evals)} base model evaluations pending")
    
    # Group pending evals by family to minimize model reloading
    from collections import defaultdict
    pending_by_family = defaultdict(list)
    for family, dataset, job_id in pending_evals:
        pending_by_family[family].append((dataset, job_id))
    
    for family in model_pairs:
        results[family] = {}
        
        # Load cached results for this family
        for dataset in datasets:
            result_file = results_dir / family / dataset / "results.json"
            if result_file.exists():
                import json
                with open(result_file) as f:
                    results[family][dataset] = json.load(f)
        
        # Skip if no pending evals for this family
        if family not in pending_by_family:
            continue
        
        ref_alias, _ = resolve_family_pair(family)
        logger.info(f"\n🔧 Loading base model: {family} ({ref_alias})")
        
        gen_cfg = GenerationConfig(
            max_new_tokens=int(cfg.generation.max_new_tokens),
            temperature=float(cfg.generation.temperature),
            top_p=float(cfg.generation.top_p),
            use_chat_template=bool(cfg.generation.use_chat_template),
            dtype=str(cfg.system.dtype),
            attn_impl=str(cfg.system.attn_impl),
            gradient_checkpointing=bool(cfg.training.gradient_checkpointing),
        )
        ref = HFModel(ref_alias, gen_cfg)
        
        for dataset, job_id in pending_by_family[family]:
            save_dir = results_dir / family / dataset
            ensure_dir(save_dir)
            
            logger.info(f"   📊 Evaluating on {dataset}...")
            if dataset in {"TLDR", "IMDBGen"}:
                metrics = evaluate_avg_reward(cfg, ref, dataset, ref_model=None, save_dir=save_dir)
            else:
                metrics = evaluate_pass1_maj8(cfg, ref, dataset, ref_model=None, save_dir=save_dir)
            
            # Remove KL for base model (always 0 against itself)
            metrics.pop("avg_kl", None)
            results[family][dataset] = metrics
            
            # Update state
            state_manager.update_state(job_id, JobState.EVAL_COMPLETED)
            logger.info(f"   ✅ {dataset}: {metrics}")
            
            torch.cuda.empty_cache()
        
        del ref
        torch.cuda.empty_cache()
    
    return results


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
    from pita.core.logging_context import logging_context, setup_context_logging
    import json
    import time

    # Import algorithms to trigger registration in worker process
    import pita.algos
    import pita.datasets

    # Setup context-aware logging in worker process
    setup_context_logging()

    # Unpack arguments
    cfg_dict, run_root_str, state_manager_dict = args

    # Reconstruct OmegaConf object from dict
    cfg = OmegaConf.create(cfg_dict)
    run_root = Path(run_root_str)

    # Reconstruct state manager
    state_manager = JobStateManager(run_root)
    if state_manager_dict:
        state_manager.states = state_manager_dict

    # Set up logging context for this job
    with logging_context(
        algo=job.algo_key,
        model=job.family_name,
        dataset=job.dataset_name,
        round_num=job.round_idx + 1,
    ):
        try:
            # Print job header with all context
            logger.info("\n" + "=" * 100)
            logger.info(f"🎯 JOB START")
            logger.info(f"   GPU: {device_id} | Job ID: {job.job_id}")
            logger.info("=" * 100)

            # Check current state
            current_state = state_manager.get_state(job.job_id)
            logger.info(f"📊 Current state: {current_state.value}")

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

            start_time = time.time()

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
                logger.info(f"⏭️  STAGE 1/3: DATA GENERATION - Already done")
            else:
                with logging_context(stage="DATA_GEN"):
                    logger.info(f"\n{'─'*100}")
                    logger.info(f"🧪 STAGE 1/3: DATA GENERATION - Starting")
                    logger.info(f"{'─'*100}")

                    data_gen_start = time.time()

                    algo.generate_data(
                        cfg=cfg,
                        ref_model=ref_for_train,
                        cls_model=job.value_model_alias,
                        dataset=job.dataset_name,
                        family=job.family_name,
                        round_idx=job.round_idx,
                        run_root=run_root,
                    )

                    elapsed = time.time() - data_gen_start
                    state_manager.update_state(job.job_id, JobState.DATA_GENERATED)
                    logger.info(
                        f"✅ STAGE 1/3: DATA GENERATION - Complete ({elapsed:.1f}s)"
                    )

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

                with logging_context(stage="TRAINING"):
                    logger.info(f"\n{'─'*100}")
                    logger.info(f"🏋️  STAGE 2/3: MODEL TRAINING - Starting")
                    logger.info(f"{'─'*100}")

                    train_start = time.time()

                    import inspect

                    # algo.run() handles phase skipping internally via check_phase_complete
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

                    train_elapsed = time.time() - train_start
                    state_manager.update_state(job.job_id, JobState.MODEL_TRAINED)
                    logger.info(
                        f"✅ STAGE 2/3: MODEL TRAINING - Complete ({train_elapsed:.1f}s)"
                    )

                with logging_context(stage="EVAL"):
                    logger.info(f"\n{'─'*100}")
                    logger.info(f"📊 STAGE 3/3: EVALUATION - Starting")
                    logger.info(f"{'─'*100}")

                    eval_start = time.time()

                    save_json(out_dir / "result.json", result or {})

                    eval_elapsed = time.time() - eval_start
                    state_manager.update_state(job.job_id, JobState.EVAL_COMPLETED)
                    logger.info(
                        f"✅ STAGE 3/3: EVALUATION - Complete ({eval_elapsed:.1f}s)"
                    )

                    total_elapsed = time.time() - start_time
                    logger.info(f"\n{'='*100}")
                    logger.info(f"🎉 JOB COMPLETE")
                    logger.info(f"   Total time: {total_elapsed:.1f}s")
                    logger.info(f"{'='*100}\n")

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

    setup_context_logging()

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

    # Execute jobs with priority-based scheduling
    all_results: Dict[str, Any] = {}

    # Convert config to plain dict for pickling
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    run_root_str = str(run_root)
    state_manager_dict = state_manager.states
    executor_args = (cfg_dict, run_root_str, state_manager_dict)

    # Flatten batches
    all_jobs = [job for batch in job_batches for job in batch]
    total_jobs = len(all_jobs)

    logger.info(
        f"📋 Executing {total_jobs} jobs with priority-based parallel scheduling"
    )
    logger.info(f"🔧 Using {gpu_manager.num_gpus} GPUs in parallel")

    # Execute with priority-based parallel scheduler
    executor = ParallelJobExecutor(max_concurrent_jobs=gpu_manager.num_gpus)

    if use_parallel and gpu_manager.num_gpus > 1:
        batch_results = executor.execute_jobs_with_priority(
            all_jobs, execute_single_job, executor_args, state_manager
        )
    else:
        batch_results = executor.execute_jobs_sequential(
            all_jobs, execute_single_job, executor_args
        )

    # Process results
    completed_jobs = 0
    failed_jobs = 0

    for job_result in batch_results:
        if job_result.success:
            completed_jobs += 1
            if job_result.result:
                job = job_result.job
                all_results.setdefault(job.algo_key, {}).setdefault(
                    f"{job.family_name}", {}
                ).setdefault(job.dataset_name, {})[
                    f"r{job.round_idx + 1}"
                ] = job_result.result
        else:
            failed_jobs += 1

    logger.info(
        f"📊 Final: {completed_jobs} completed, {failed_jobs} failed out of {total_jobs} total"
    )

    gpu_manager.clear_cache()
    gpu_manager.synchronize_all()

    # Evaluate base models (after algorithm jobs, so it doesn't block parallel execution)
    base_model_results = evaluate_base_models(cfg, run_root, state_manager)
    logger.info(f"📊 Base model evaluation complete")

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
