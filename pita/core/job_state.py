"""Job state tracking for resume functionality."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Dict
from loguru import logger

from pita.core.io import check_data_generated, check_training_completed


class JobState(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    DATA_GENERATED = "DATA_GENERATED"
    TRAINING_COMPLETED = "TRAINING_COMPLETED"
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
            with open(self.state_file, "r") as f:
                data = json.load(f)
                self.states = data.get("jobs", {})
            logger.info(f"Loaded {len(self.states)} job states from {self.state_file}")
        else:
            logger.info("No existing state file found, will create new state tracking")

    def _save(self) -> None:
        """Save current state to file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump({"jobs": self.states}, f, indent=2)

    def get_state(self, job_id: str) -> JobState:
        """Get current state of a job."""
        state_str = self.states.get(job_id, JobState.NOT_STARTED.value)
        return JobState(state_str)

    def update_state(self, job_id: str, state: JobState) -> None:
        """Update state of a job and persist to disk."""
        self.states[job_id] = state.value
        self._save()
        logger.debug(f"Updated job {job_id} to state {state.value}")

    def scan_file_system(
        self, algo_key: str, dataset: str, family: str, round_idx: int
    ) -> JobState:
        """Determine actual state by checking file system."""
        if check_training_completed(
            algo_key, dataset, family, round_idx, self.run_root
        ):
            return JobState.TRAINING_COMPLETED
        elif check_data_generated(algo_key, dataset, family, round_idx, self.run_root):
            return JobState.DATA_GENERATED
        else:
            return JobState.NOT_STARTED

    def initialize_from_jobs(self, jobs: list) -> None:
        """Scan file system for all jobs and initialize states."""
        logger.info("Scanning file system to determine job states...")
        for job in jobs:
            actual_state = self.scan_file_system(
                job.algo_key, job.dataset_name, job.family_name, job.round_idx
            )
            current_state = self.get_state(job.job_id)

            if (
                current_state == JobState.NOT_STARTED
                or actual_state != JobState.NOT_STARTED
            ):
                self.states[job.job_id] = actual_state.value

        self._save()

        completed = sum(
            1 for s in self.states.values() if s == JobState.TRAINING_COMPLETED.value
        )
        data_only = sum(
            1 for s in self.states.values() if s == JobState.DATA_GENERATED.value
        )
        not_started = sum(
            1 for s in self.states.values() if s == JobState.NOT_STARTED.value
        )
        failed = sum(1 for s in self.states.values() if s == JobState.FAILED.value)

        logger.info(
            f"Job state summary: {completed} completed, {data_only} data-only, "
            f"{not_started} not started, {failed} failed"
        )
