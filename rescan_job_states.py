"""Rescan file system and update job_states.json for an existing run."""

from pathlib import Path
import sys
from omegaconf import OmegaConf
from pita.core.job_state import JobStateManager, JobState
from pita.core.parallel_executor import create_job_list


def rescan_job_states(run_path: str):
    """Rescan and update job states for a run directory."""
    run_root = Path(run_path)

    if not run_root.exists():
        print(f"❌ Error: Run directory does not exist: {run_path}")
        return

    config_path = run_root / ".hydra" / "config.yaml"
    if not config_path.exists():
        print(f"❌ Error: Config not found: {config_path}")
        return

    print(f"📂 Info: Loading config from: {config_path}")
    cfg = OmegaConf.load(config_path)

    print(f"🔍 Info: Creating job list...")
    jobs = create_job_list(cfg)
    print(f"📋 Info: Found {len(jobs)} jobs to check")

    print(f"🔄 Info: Initializing state manager...")
    state_manager = JobStateManager(run_root)

    print(f"🔍 Info: Scanning file system for job states...")
    state_manager.initialize_from_jobs(jobs)

    eval_completed = sum(
        1 for s in state_manager.states.values() if s == JobState.EVAL_COMPLETED.value
    )
    model_trained = sum(
        1 for s in state_manager.states.values() if s == JobState.MODEL_TRAINED.value
    )
    data_only = sum(
        1 for s in state_manager.states.values() if s == JobState.DATA_GENERATED.value
    )
    not_started = sum(
        1 for s in state_manager.states.values() if s == JobState.NOT_STARTED.value
    )
    failed = sum(1 for s in state_manager.states.values() if s == JobState.FAILED.value)

    print(f"\n✅ Success: Job states updated successfully!")
    print(f"📄 Info: Updated file: {run_root / 'job_states.json'}")
    print(f"\n📊 Summary:")
    print(f"   ✅ Eval completed:  {eval_completed}")
    print(f"   🏋️  Model trained:   {model_trained}")
    print(f"   🧪 Data generated:  {data_only}")
    print(f"   ⏸️  Not started:     {not_started}")
    print(f"   ❌ Failed:          {failed}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rescan_job_states.py <run_directory>")
        sys.exit(1)

    rescan_job_states(sys.argv[1])
