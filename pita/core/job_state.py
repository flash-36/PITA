"""Job state tracking for resume functionality."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Dict
from loguru import logger

from pita.core.io import check_data_generated, check_model_trained, check_eval_completed


class JobState(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    DATA_GENERATED = "DATA_GENERATED"
    MODEL_TRAINED = "MODEL_TRAINED"
    EVAL_COMPLETED = "EVAL_COMPLETED"
    FAILED = "FAILED"


class JobStateManager:
    """Manages job state persistence for resume functionality."""

    def __init__(self, run_root: Path):
        self.run_root = Path(run_root)
        self.state_file = self.run_root / "job_states.json"
        self.states: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        """Load existing state from file or initialize from file system."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    loaded_states = data.get("jobs", {})
                
                # Check if job IDs look like old format (will be corrected by filesystem scan)
                if loaded_states and any("_r0" in job_id for job_id in loaded_states.keys()):
                    logger.info(f"⚠️  Detected old job ID format, will reconstruct from filesystem")
                    self.states = {}  # Force reconstruction
                else:
                    self.states = loaded_states
                    logger.info(f"💾 Loaded {len(self.states)} job states from {self.state_file}")
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"⚠️  Corrupted state file detected: {e}")
                logger.warning(f"⚠️  Backing up corrupted file and rescanning filesystem")
                
                # Backup corrupted file
                import shutil
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = self.state_file.with_suffix(f".json.corrupted_{timestamp}")
                shutil.copy2(self.state_file, backup_path)
                logger.info(f"💾 Backed up corrupted state to: {backup_path}")
                
                # Start fresh - will be reconstructed by initialize_from_jobs
                self.states = {}
        else:
            logger.info("ℹ️  No existing state file found, will create new state tracking")

    def _save(self) -> None:
        """Save current state to file atomically."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temporary file first, then atomic rename
        temp_file = self.state_file.with_suffix(".json.tmp")
        try:
            with open(temp_file, "w") as f:
                json.dump({"jobs": self.states}, f, indent=2)
            
            # Atomic rename (overwrites existing file on Unix)
            temp_file.replace(self.state_file)
        except Exception as e:
            # Clean up temp file if something went wrong
            if temp_file.exists():
                temp_file.unlink()
            raise e

    def get_state(self, job_id: str) -> JobState:
        """Get current state of a job."""
        state_str = self.states.get(job_id, JobState.NOT_STARTED.value)
        return JobState(state_str)

    def update_state(self, job_id: str, state: JobState) -> None:
        """Update state of a job and persist to disk."""
        # Reload from disk to get any concurrent updates before saving
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    self.states = data.get("jobs", {})
            except (json.JSONDecodeError, ValueError):
                pass  # Keep current state if file is corrupted
        
        self.states[job_id] = state.value
        try:
            self._save()
            logger.debug(f"Updated job {job_id} to state {state.value}")
        except Exception as e:
            logger.error(f"⚠️  Failed to save job state for {job_id}: {e}")
            logger.warning(f"⚠️  Continuing execution, but state may not be persisted")

    def scan_file_system(
        self, algo_key: str, dataset: str, family: str, round_idx: int
    ) -> JobState:
        """Determine actual state by checking file system."""
        if check_eval_completed(algo_key, dataset, family, round_idx, self.run_root):
            return JobState.EVAL_COMPLETED
        elif check_model_trained(algo_key, dataset, family, round_idx, self.run_root):
            return JobState.MODEL_TRAINED
        elif check_data_generated(algo_key, dataset, family, round_idx, self.run_root):
            return JobState.DATA_GENERATED
        else:
            return JobState.NOT_STARTED

    def initialize_from_jobs(self, jobs: list) -> None:
        """Scan file system for all jobs and initialize states."""
        had_states = len(self.states) > 0
        
        if had_states:
            logger.info(f"🔍 Verifying {len(self.states)} existing job states against filesystem...")
        else:
            logger.info("🔍 Scanning file system to determine job states...")
        
        updates = 0
        for job in jobs:
            actual_state = self.scan_file_system(
                job.algo_key, job.dataset_name, job.family_name, job.round_idx
            )
            current_state = self.get_state(job.job_id)

            if current_state != actual_state:
                updates += 1
                if had_states:
                    logger.debug(f"  Updated {job.job_id}: {current_state.value} → {actual_state.value}")
            
            if (
                current_state == JobState.NOT_STARTED
                or actual_state != JobState.NOT_STARTED
            ):
                self.states[job.job_id] = actual_state.value

        self._save()
        
        if had_states and updates > 0:
            logger.info(f"✅ Updated {updates} job states based on filesystem")

        eval_completed = sum(
            1 for s in self.states.values() if s == JobState.EVAL_COMPLETED.value
        )
        model_trained = sum(
            1 for s in self.states.values() if s == JobState.MODEL_TRAINED.value
        )
        data_only = sum(
            1 for s in self.states.values() if s == JobState.DATA_GENERATED.value
        )
        not_started = sum(
            1 for s in self.states.values() if s == JobState.NOT_STARTED.value
        )
        failed = sum(1 for s in self.states.values() if s == JobState.FAILED.value)

        logger.info(
            f"📊 Job state summary: {eval_completed} completed, {model_trained} trained, "
            f"{data_only} data-only, {not_started} not started, {failed} failed"
        )
