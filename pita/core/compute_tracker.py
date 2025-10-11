from __future__ import annotations

import time
from typing import Optional, Dict, Any
from contextlib import contextmanager
import torch
from loguru import logger


class ComputeTracker:
    """Tracks compute requirements (runtime and optionally FLOPs) for algorithm phases."""

    def __init__(self):
        self.timings: Dict[str, float] = {}
        self.flops: Dict[str, int] = {}
        self._has_fvcore = self._check_fvcore()

    def _check_fvcore(self) -> bool:
        """Check if fvcore is available for FLOPs counting."""
        try:
            import fvcore

            return True
        except ImportError:
            return False

    @contextmanager
    def track_phase(self, phase_name: str, model: Optional[Any] = None):
        """Context manager to track a computation phase."""
        start_time = time.perf_counter()
        start_cuda = (
            torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        )
        end_cuda = (
            torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        )

        if start_cuda:
            start_cuda.record()

        try:
            yield
        finally:
            if end_cuda:
                end_cuda.record()
                torch.cuda.synchronize()
                cuda_time = start_cuda.elapsed_time(end_cuda) / 1000.0
                self.timings[phase_name] = cuda_time
                logger.info(f"⏱️ {phase_name}: {cuda_time:.2f}s")
            else:
                elapsed = time.perf_counter() - start_time
                self.timings[phase_name] = elapsed
                logger.info(f"⏱️ {phase_name}: {elapsed:.2f}s")

    def get_metrics(self) -> Dict[str, Any]:
        """Get all tracked compute metrics."""
        total_time = sum(self.timings.values())
        result = {
            "total_time_seconds": round(total_time, 2),
            "phase_times_seconds": {k: round(v, 2) for k, v in self.timings.items()},
        }
        if self.flops:
            result["phase_flops"] = self.flops
        return result

    def reset(self):
        """Reset all tracked metrics."""
        self.timings.clear()
        self.flops.clear()


_global_tracker: Optional[ComputeTracker] = None


def get_compute_tracker() -> ComputeTracker:
    """Get or create global compute tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = ComputeTracker()
    return _global_tracker


def reset_compute_tracker():
    """Reset the global compute tracker."""
    global _global_tracker
    _global_tracker = ComputeTracker()
