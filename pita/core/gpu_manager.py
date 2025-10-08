"""GPU management and allocation utilities for multi-GPU parallelization."""

from __future__ import annotations

import os
from typing import List, Optional, Dict, Any
import torch
from loguru import logger
from dataclasses import dataclass


@dataclass
class GPUInfo:
    """Information about a single GPU."""

    device_id: int
    name: str
    total_memory: int  # in bytes
    is_available: bool


class GPUManager:
    """Manages GPU allocation and provides utilities for multi-GPU operations."""

    def __init__(self):
        self._available_gpus: List[int] = []
        self._gpu_info: Dict[int, GPUInfo] = {}
        self._detect_gpus()

    def _detect_gpus(self) -> None:
        """Detect all available GPUs and their properties."""
        if not torch.cuda.is_available():
            logger.warning("No CUDA GPUs available")
            return

        num_gpus = torch.cuda.device_count()
        logger.info(f"Detected {num_gpus} GPU(s)")

        for i in range(num_gpus):
            try:
                props = torch.cuda.get_device_properties(i)
                info = GPUInfo(
                    device_id=i,
                    name=props.name,
                    total_memory=props.total_memory,
                    is_available=True,
                )
                self._gpu_info[i] = info
                self._available_gpus.append(i)

                memory_gb = info.total_memory / (1024**3)
                logger.info(f"  GPU {i}: {info.name} ({memory_gb:.1f} GB)")
            except Exception as e:
                logger.warning(f"Could not query GPU {i}: {e}")

    @property
    def num_gpus(self) -> int:
        """Number of available GPUs."""
        return len(self._available_gpus)

    @property
    def available_gpus(self) -> List[int]:
        """List of available GPU device IDs."""
        return self._available_gpus.copy()

    def get_gpu_memory(self, device_id: int) -> int:
        """Get total memory for a GPU in bytes."""
        if device_id in self._gpu_info:
            return self._gpu_info[device_id].total_memory
        return 0

    def get_free_memory(self, device_id: int) -> int:
        """Get free memory for a GPU in bytes."""
        if device_id not in self._available_gpus:
            return 0

        torch.cuda.synchronize(device_id)
        return torch.cuda.mem_get_info(device_id)[0]

    def estimate_model_size_gb(self, model_size: str) -> float:
        """Estimate model memory footprint in GB."""
        # Rough estimates for bfloat16 models with overhead
        size_map = {
            "3B": 10,
            "7B": 20,
            "8B": 22,
            "13B": 35,
            "15B": 40,
        }

        for key, gb in size_map.items():
            if key in model_size or key.lower() in model_size.lower():
                return gb

        # Default conservative estimate
        return 30

    def can_fit_model(self, device_id: int, model_size: str) -> bool:
        """Check if a model can fit on a specific GPU."""
        required_gb = self.estimate_model_size_gb(model_size)
        available_gb = self.get_free_memory(device_id) / (1024**3)
        return available_gb >= required_gb

    def get_optimal_batch_size(
        self, device_id: int, model_size: str, base_batch_size: int = 16
    ) -> int:
        """Estimate optimal batch size for a GPU."""
        memory_gb = self.get_gpu_memory(device_id) / (1024**3)

        # Scale batch size based on available memory
        if memory_gb >= 80:
            multiplier = 2.0
        elif memory_gb >= 40:
            multiplier = 1.0
        else:
            multiplier = 0.5

        return max(1, int(base_batch_size * multiplier))

    def distribute_work(self, num_jobs: int) -> List[List[int]]:
        """Distribute jobs across available GPUs.

        Returns a list of lists, where each inner list contains job indices
        assigned to each GPU.
        """
        if not self._available_gpus:
            return [[i for i in range(num_jobs)]]

        # Distribute jobs evenly across GPUs
        jobs_per_gpu = [[] for _ in range(self.num_gpus)]
        for job_idx in range(num_jobs):
            gpu_idx = job_idx % self.num_gpus
            jobs_per_gpu[gpu_idx].append(job_idx)

        return jobs_per_gpu

    def get_device_map(self, prefer_gpu: Optional[int] = None) -> str | Dict:
        """Get device map for model loading.

        Args:
            prefer_gpu: Preferred GPU ID, or None for auto

        Returns:
            "auto" or "cpu" or specific device map
        """
        if not self._available_gpus:
            return "cpu"

        if prefer_gpu is not None and prefer_gpu in self._available_gpus:
            return {"": f"cuda:{prefer_gpu}"}

        return "auto"

    def clear_cache(self, device_id: Optional[int] = None) -> None:
        """Clear CUDA cache for specific device or all devices."""
        if device_id is not None:
            with torch.cuda.device(device_id):
                torch.cuda.empty_cache()
        else:
            torch.cuda.empty_cache()

    def synchronize_all(self) -> None:
        """Synchronize all GPUs."""
        for gpu_id in self._available_gpus:
            torch.cuda.synchronize(gpu_id)

    def print_memory_summary(self) -> None:
        """Print memory usage summary for all GPUs."""
        logger.info("GPU Memory Summary:")
        for gpu_id in self._available_gpus:
            allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
            total = self.get_gpu_memory(gpu_id) / (1024**3)
            logger.info(
                f"  GPU {gpu_id}: {allocated:.2f} GB allocated, "
                f"{reserved:.2f} GB reserved, {total:.2f} GB total"
            )


# Global singleton instance
_gpu_manager: Optional[GPUManager] = None


def get_gpu_manager() -> GPUManager:
    """Get the global GPUManager instance."""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager()
    return _gpu_manager


def set_device_for_worker(worker_id: int, num_workers: int) -> int:
    """Set CUDA device for a worker process.

    Args:
        worker_id: Worker ID (0-indexed)
        num_workers: Total number of workers

    Returns:
        Assigned GPU device ID
    """
    manager = get_gpu_manager()
    if not manager.available_gpus:
        return -1

    gpu_id = manager.available_gpus[worker_id % manager.num_gpus]
    torch.cuda.set_device(gpu_id)
    return gpu_id
