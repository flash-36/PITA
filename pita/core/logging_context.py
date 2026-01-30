"""Context-aware logging for job execution tracking."""

from __future__ import annotations
from contextlib import contextmanager
from typing import Optional
from loguru import logger
import sys

# Global context storage
_current_context: Optional[dict] = None


def get_context() -> Optional[dict]:
    """Get current logging context."""
    return _current_context


def format_context(ctx: Optional[dict] = None) -> str:
    """Format context as a prefix string."""
    if ctx is None:
        ctx = _current_context
    
    if ctx is None:
        return ""
    
    base_parts = []
    if ctx.get("algo"):
        base_parts.append(ctx["algo"])
    if ctx.get("dataset"):
        base_parts.append(ctx["dataset"])
    if ctx.get("model"):
        base_parts.append(ctx["model"])
    
    round_stage_parts = []
    if ctx.get("round"):
        round_stage_parts.append(f"r{ctx['round']}")
    if ctx.get("stage"):
        round_stage_parts.append(ctx["stage"])
    
    prefix = ">".join(base_parts)
    if round_stage_parts:
        if prefix:
            prefix += ">>" + ">".join(round_stage_parts)
        else:
            prefix = ">".join(round_stage_parts)
    
    return prefix + " │ " if prefix else ""


class ContextFilter:
    """Filter that adds context prefix to log messages."""
    
    def __call__(self, record):
        ctx = get_context()
        if ctx:
            record["extra"]["context_prefix"] = format_context(ctx)
        else:
            record["extra"]["context_prefix"] = ""
        return True


@contextmanager
def logging_context(algo: str = None, model: str = None, dataset: str = None, 
                     round_num: int = None, stage: str = None):
    """Context manager for structured logging.
    
    Usage:
        with logging_context(algo="PITA", model="phi", dataset="GSM8K", round_num=1, stage="data_gen"):
            logger.info("Generating data")  # Will show: PITA > phi > GSM8K > r1 > [data_gen] │ Generating data
    """
    global _current_context
    
    # Save previous context
    prev_context = _current_context
    
    # Create new context (inherit from previous if exists)
    new_context = prev_context.copy() if prev_context else {}
    
    if algo is not None:
        new_context["algo"] = algo
    if model is not None:
        new_context["model"] = model
    if dataset is not None:
        new_context["dataset"] = dataset
    if round_num is not None:
        new_context["round"] = round_num
    if stage is not None:
        new_context["stage"] = stage
    
    _current_context = new_context
    
    try:
        yield
    finally:
        # Restore previous context
        _current_context = prev_context


def setup_context_logging():
    """Setup loguru to use context-aware formatting."""
    # Remove default handler
    logger.remove()
    
    # Add handler with context prefix
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level:<8}</level> | <cyan>{extra[context_prefix]}</cyan><level>{message}</level>",
        filter=ContextFilter(),
        colorize=True,
    )

