"""
Distributed Training Utilities
DDP setup, SyncBatchNorm, multi-GPU support
"""

import os
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from app.core.logging import logger


def setup_distributed(
    backend: str = "nccl",
    init_method: str = "env://",
):
    """
    Initialize distributed training
    
    Args:
        backend: nccl (GPU) or gloo (CPU)
        init_method: Initialization method
    """
    if not dist.is_available():
        logger.warning("Distributed training not available")
        return False
    
    if dist.is_initialized():
        logger.info("Distributed already initialized")
        return True
    
    try:
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
        )
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        logger.info(f"Distributed initialized: rank={rank}, world_size={world_size}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize distributed: {e}")
        return False


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed cleaned up")


def get_rank() -> int:
    """Get current process rank"""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes"""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if current process is main (rank 0)"""
    return get_rank() == 0


def convert_syncbatchnorm(model: torch.nn.Module) -> torch.nn.Module:
    """
    Convert BatchNorm to SyncBatchNorm for distributed training
    
    Args:
        model: PyTorch model
        
    Returns:
        Model with SyncBatchNorm
    """
    if dist.is_initialized():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logger.info("Converted to SyncBatchNorm")
    
    return model


def wrap_ddp(
    model: torch.nn.Module,
    device_ids: Optional[list] = None,
    find_unused_parameters: bool = False,
) -> torch.nn.Module:
    """
    Wrap model with DistributedDataParallel
    
    Args:
        model: PyTorch model
        device_ids: GPU device IDs
        find_unused_parameters: Handle unused parameters
        
    Returns:
        DDP-wrapped model
    """
    if not dist.is_initialized():
        logger.warning("Distributed not initialized, returning unwrapped model")
        return model
    
    rank = get_rank()
    
    if device_ids is None:
        device_ids = [rank]
    
    model = DDP(
        model,
        device_ids=device_ids,
        find_unused_parameters=find_unused_parameters,
    )
    
    logger.info(f"Model wrapped with DDP: rank={rank}, devices={device_ids}")
    
    return model


def reduce_tensor(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    """
    Reduce tensor across all processes
    
    Args:
        tensor: Tensor to reduce
        average: If True, compute average; otherwise sum
        
    Returns:
        Reduced tensor
    """
    if not dist.is_initialized():
        return tensor
    
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    if average:
        tensor /= get_world_size()
    
    return tensor


def barrier():
    """Synchronization barrier"""
    if dist.is_initialized():
        dist.barrier()


def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """
    Broadcast tensor from source to all processes
    
    Args:
        tensor: Tensor to broadcast
        src: Source rank
        
    Returns:
        Broadcasted tensor
    """
    if dist.is_initialized():
        dist.broadcast(tensor, src=src)
    
    return tensor
