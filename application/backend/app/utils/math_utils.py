"""
Math Utilities
Mathematical helper functions
"""

import numpy as np
import torch
from typing import List, Optional


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numpy softmax
    
    Args:
        x: Input array
        axis: Axis to apply softmax
    
    Returns:
        Softmax output
    """
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity
    
    Args:
        a: Vector a
        b: Vector b
    
    Returns:
        Cosine similarity
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Euclidean distance
    
    Args:
        a: Vector a
        b: Vector b
    
    Returns:
        Euclidean distance
    """
    return float(np.linalg.norm(a - b))


def moving_average(values: List[float], window_size: int = 5) -> List[float]:
    """
    Compute moving average
    
    Args:
        values: Input values
        window_size: Window size
    
    Returns:
        Smoothed values
    """
    if len(values) < window_size:
        return values
    
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window_size // 2)
        end = min(len(values), i + window_size // 2 + 1)
        smoothed.append(np.mean(values[start:end]))
    
    return smoothed


def normalize_array(arr: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Z-score normalization
    
    Args:
        arr: Input array
        axis: Axis for normalization
    
    Returns:
        Normalized array
    """
    mean = np.mean(arr, axis=axis, keepdims=True)
    std = np.std(arr, axis=axis, keepdims=True) + 1e-8
    return (arr - mean) / std


def interpolate_sequence(
    sequence: np.ndarray,
    target_length: int
) -> np.ndarray:
    """
    Interpolate sequence to target length
    
    Args:
        sequence: Input sequence (T, ...)
        target_length: Target length
    
    Returns:
        Interpolated sequence
    """
    current_length = len(sequence)
    if current_length == target_length:
        return sequence
    
    # Create interpolation indices
    indices = np.linspace(0, current_length - 1, target_length)
    
    # Interpolate
    interpolated = np.array([
        sequence[int(idx)] for idx in indices
    ])
    
    return interpolated
