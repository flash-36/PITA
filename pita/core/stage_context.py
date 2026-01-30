"""Stage context tracking for logging."""

import threading
from typing import Optional
from contextlib import contextmanager

_stage_context = threading.local()


def get_current_stage() -> Optional[str]:
    """Get the current execution stage."""
    return getattr(_stage_context, "stage", None)


def set_stage(stage: Optional[str]) -> None:
    """Set the current execution stage."""
    _stage_context.stage = stage


@contextmanager
def stage_context(stage: str):
    """Context manager for setting the execution stage."""
    old_stage = get_current_stage()
    set_stage(stage)
    try:
        yield
    finally:
        set_stage(old_stage)


def format_log_message(message: str) -> str:
    """Format a log message with the current stage prefix."""
    stage = get_current_stage()
    if stage:
        return f"[{stage}] {message}"
    return message

