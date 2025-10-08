"""Distributed training wrappers for multi-GPU training."""

from __future__ import annotations

import os
from typing import Optional, Any, Dict
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from loguru import logger


class DistributedTrainingWrapper:
    """Wrapper for distributed training with automatic setup and cleanup."""

    def __init__(self, use_ddp: bool = True):
        self.use_ddp = use_ddp
        self.is_initialized = False
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0

        if self.use_ddp and torch.cuda.is_available():
            self._init_distributed()

    def _init_distributed(self):
        """Initialize distributed training."""
        # Check if already initialized
        if dist.is_available() and dist.is_initialized():
            self.is_initialized = True
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            logger.info(
                f"Distributed already initialized: rank={self.rank}, "
                f"world_size={self.world_size}, local_rank={self.local_rank}"
            )
            return

        # Initialize if environment variables are set
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

            # Initialize process group
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=self.world_size,
                rank=self.rank,
            )
            self.is_initialized = True

            if self.is_master:
                logger.info(
                    f"Initialized distributed training: world_size={self.world_size}"
                )
        else:
            logger.info("DDP environment variables not set, using single GPU")
            self.use_ddp = False

    @property
    def is_master(self) -> bool:
        """Check if this is the master process."""
        return self.rank == 0

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Wrap a model for distributed training.

        Args:
            model: Model to wrap

        Returns:
            Wrapped model (DDP if distributed, otherwise original)
        """
        if not self.use_ddp or not self.is_initialized:
            return model

        # Move model to correct device
        device = torch.device(f"cuda:{self.local_rank}")
        model = model.to(device)

        # Wrap with DDP
        model = DDP(
            model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=True,  # More flexible but slower
        )

        if self.is_master:
            logger.info(f"Wrapped model with DDP on device {self.local_rank}")

        return model

    def create_dataloader(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        **kwargs,
    ) -> DataLoader:
        """Create a dataloader with distributed sampling if needed.

        Args:
            dataset: Dataset to load
            batch_size: Batch size per GPU
            shuffle: Whether to shuffle
            num_workers: Number of dataloader workers
            **kwargs: Additional DataLoader arguments

        Returns:
            DataLoader with appropriate sampler
        """
        if self.use_ddp and self.is_initialized:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle,
            )
            # Don't shuffle in DataLoader when using DistributedSampler
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True,
                **kwargs,
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
                **kwargs,
            )

        return loader

    def barrier(self):
        """Synchronize all processes."""
        if self.use_ddp and self.is_initialized:
            dist.barrier()

    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
        """All-reduce a tensor across processes.

        Args:
            tensor: Tensor to reduce
            op: Reduction operation

        Returns:
            Reduced tensor
        """
        if self.use_ddp and self.is_initialized:
            dist.all_reduce(tensor, op=op)
        return tensor

    def gather_dict(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Gather dictionary data from all processes to master.

        Args:
            data: Dictionary to gather

        Returns:
            Merged dictionary on master, None on workers
        """
        if not self.use_ddp or not self.is_initialized:
            return data

        # Convert to list for gathering
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, data)

        if self.is_master:
            # Merge dictionaries
            merged = {}
            for d in gathered:
                if d is not None:
                    for k, v in d.items():
                        if k not in merged:
                            merged[k] = []
                        merged[k].append(v)

            # Average numeric values
            result = {}
            for k, v_list in merged.items():
                if isinstance(v_list[0], (int, float)):
                    result[k] = sum(v_list) / len(v_list)
                else:
                    result[k] = v_list[0]

            return result
        else:
            return None

    def cleanup(self):
        """Cleanup distributed training."""
        if self.use_ddp and self.is_initialized:
            dist.destroy_process_group()
            self.is_initialized = False
            if self.is_master:
                logger.info("Cleaned up distributed training")


def get_optimal_num_workers(num_gpus: int) -> int:
    """Get optimal number of dataloader workers based on GPU count.

    Args:
        num_gpus: Number of GPUs

    Returns:
        Optimal number of workers
    """
    import multiprocessing as mp

    cpu_count = mp.cpu_count()

    # Use 2-4 workers per GPU, but don't exceed CPU count
    workers_per_gpu = 3
    total_workers = min(num_gpus * workers_per_gpu, cpu_count // 2)

    return max(0, total_workers)


def should_use_ddp() -> bool:
    """Check if DDP should be used based on environment and GPU count.

    Returns:
        True if DDP should be used
    """
    if not torch.cuda.is_available():
        return False

    # Check if in distributed environment
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return True

    # Check if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        # Could use DDP but not required
        return False

    return False
