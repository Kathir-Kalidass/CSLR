"""
Tensor Utilities
Helper functions for tensor operations
"""

import torch
import numpy as np
from typing import List, Optional, Sequence, Tuple, Union


def to_tensor(
    data: Union[np.ndarray, list],
    device: str = "cpu",
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Convert numpy array or list to tensor
    
    Args:
        data: Input data
        device: Target device
        dtype: Target dtype
    
    Returns:
        PyTorch tensor
    """
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=dtype)
    
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    else:
        tensor = torch.tensor(data)
    
    return tensor.to(device=device, dtype=dtype)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor to numpy array
    
    Args:
        tensor: Input tensor
    
    Returns:
        Numpy array
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    
    return tensor.detach().cpu().numpy()


def ensure_batch_dim(tensor: torch.Tensor) -> torch.Tensor:
    """
    Ensure tensor has batch dimension
    Adds dimension if missing
    
    Args:
        tensor: Input tensor
    
    Returns:
        Tensor with batch dimension
    """
    if tensor.dim() == 3:  # (C, H, W) -> (1, C, H, W)
        return tensor.unsqueeze(0)
    return tensor


def remove_batch_dim(tensor: torch.Tensor) -> torch.Tensor:
    """Remove batch dimension if batch size is 1"""
    if tensor.size(0) == 1:
        return tensor.squeeze(0)
    return tensor


def pad_sequence(
    sequences: List[torch.Tensor],
    batch_first: bool = True,
    padding_value: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad variable-length sequences
    
    Args:
        sequences: List of tensors with varying lengths
        batch_first: If True, output shape is (B, T, ...)
        padding_value: Value for padding
    
    Returns:
        Tuple of (padded sequences, lengths)
    """
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded = torch.nn.utils.rnn.pad_sequence(
        sequences,
        batch_first=batch_first,
        padding_value=padding_value
    )
    
    return padded, lengths


def create_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """
    Create attention mask from lengths
    
    Args:
        lengths: Tensor of sequence lengths (B,)
        max_len: Maximum length (optional)
    
    Returns:
        Binary mask (B, max_len)
    """
    if max_len is None:
        max_len = int(lengths.max().item())
    
    batch_size = lengths.size(0)
    mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len)
    mask = mask < lengths.unsqueeze(1)
    
    return mask.float()


def normalize_tensor(
    tensor: torch.Tensor,
    mean: Union[float, Sequence[float]] = 0.0,
    std: Union[float, Sequence[float]] = 1.0
) -> torch.Tensor:
    """
    Normalize tensor with mean and std
    
    Args:
        tensor: Input tensor
        mean: Mean value(s)
        std: Std value(s)
    
    Returns:
        Normalized tensor
    """
    if isinstance(mean, (int, float)):
        mean_tensor: Union[float, torch.Tensor] = float(mean)
    else:
        mean_tensor = torch.tensor(list(mean), device=tensor.device, dtype=tensor.dtype)
        mean_tensor = mean_tensor.view(-1, 1, 1)  # For (C, H, W)
    
    if isinstance(std, (int, float)):
        std_tensor: Union[float, torch.Tensor] = float(std)
    else:
        std_tensor = torch.tensor(list(std), device=tensor.device, dtype=tensor.dtype)
        std_tensor = std_tensor.view(-1, 1, 1)
    
    return (tensor - mean_tensor) / std_tensor


def denormalize_tensor(
    tensor: torch.Tensor,
    mean: Union[float, Sequence[float]] = 0.0,
    std: Union[float, Sequence[float]] = 1.0
) -> torch.Tensor:
    """Reverse normalization"""
    if isinstance(mean, (int, float)):
        mean_tensor: Union[float, torch.Tensor] = float(mean)
    else:
        mean_tensor = torch.tensor(list(mean), device=tensor.device, dtype=tensor.dtype)
        mean_tensor = mean_tensor.view(-1, 1, 1)
    
    if isinstance(std, (int, float)):
        std_tensor: Union[float, torch.Tensor] = float(std)
    else:
        std_tensor = torch.tensor(list(std), device=tensor.device, dtype=tensor.dtype)
        std_tensor = std_tensor.view(-1, 1, 1)
    
    return tensor * std_tensor + mean_tensor
