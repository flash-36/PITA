"""Rescan file system and update job_states.json for an existing run."""

from pathlib import Path
import sys
from omegaconf import OmegaConf
from pita.core.job_state import JobStateManager
from pita.core.parallel_executor import create_job_list


def rescan_job_states(run_path: str):
    """Rescan and update job states for a run directory."""
    run_root = Path(run_path)

    if not run_root.exists():
        print(f"❌ Run directory does not exist: {run_path}")
        return

    config_path = run_root / ".hydra" / "config.yaml"
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return

    print(f"📂 Loading config from: {config_path}")
    cfg = OmegaConf.load(config_path)

    print(f"🔍 Creating job list...")
    jobs = create_job_list(cfg)
    print(f"📋 Found {len(jobs)} jobs to check")

    print(f"🔄 Initializing state manager...")
    state_manager = JobStateManager(run_root)

    print(f"🔍 Scanning file system...")
    state_manager.initialize_from_jobs(jobs)

    print(f"✅ Job states updated successfully!")
    print(f"📄 Updated file: {run_root / 'job_states.json'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rescan_job_states.py <run_directory>")
        sys.exit(1)

    rescan_job_states(sys.argv[1])
