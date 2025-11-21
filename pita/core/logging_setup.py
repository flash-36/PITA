"""Logging setup with stage-aware formatting."""

import sys
from loguru import logger
from pita.core.stage_context import format_log_message


class StageFilter:
    """Loguru filter that adds stage prefixes to messages."""
    
    def __call__(self, record):
        record["message"] = format_log_message(record["message"])
        return True


def setup_stage_logging():
    """Configure loguru to use stage-aware formatting."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {message}",
        level="INFO",
        filter=StageFilter(),
    )


def add_file_logger(log_path: str):
    """Add a file logger with stage-aware formatting."""
    logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {message}",
        level="INFO",
        filter=StageFilter(),
        enqueue=True,
    )

