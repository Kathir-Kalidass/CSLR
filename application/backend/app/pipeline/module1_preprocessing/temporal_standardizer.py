"""
Temporal Standardizer
Handles variable-length sequences and temporal alignment
"""

import numpy as np
import torch
from typing import List, Tuple


class TemporalStandardizer:
    """
    Standardizes temporal dimension
    Pads/truncates to fixed length
    """
    
    def __init__(self, clip_length: int = 32):
        self.clip_length = clip_length
    
    def standardize(
        self, 
        sequence: np.ndarray, 
        pad_value: float = 0.0
    ) -> Tuple[np.ndarray, int]:
        """
        Standardize sequence to fixed length
        
        Args:
            sequence: Input sequence (T, ...)
            pad_value: Value for padding
        
        Returns:
            Tuple of (standardized sequence, original length)
        """
        original_length = len(sequence)
        
        if original_length == self.clip_length:
            return sequence, original_length
        
        elif original_length < self.clip_length:
            # Pad
            pad_length = self.clip_length - original_length
            pad_shape = (pad_length,) + sequence.shape[1:]
            padding = np.full(pad_shape, pad_value, dtype=sequence.dtype)
            padded = np.concatenate([sequence, padding], axis=0)
            return padded, original_length
        
        else:
            # Truncate (sample evenly)
            indices = np.linspace(0, original_length - 1, self.clip_length, dtype=int)
            truncated = sequence[indices]
            return truncated, self.clip_length
    
    def create_attention_mask(self, original_length: int) -> np.ndarray:
        """
        Create attention mask for padded sequence
        
        Args:
            original_length: Original sequence length
        
        Returns:
            Binary mask (1 for real, 0 for padded)
        """
        mask = np.zeros(self.clip_length, dtype=np.float32)
        mask[:min(original_length, self.clip_length)] = 1.0
        return mask
    
    def batch_standardize(
        self, 
        sequences: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Standardize batch of sequences
        
        Args:
            sequences: List of variable-length sequences
        
        Returns:
            Tuple of (batched sequences, attention masks)
        """
        standardized = []
        masks = []
        
        for seq in sequences:
            std_seq, orig_len = self.standardize(seq)
            mask = self.create_attention_mask(orig_len)
            
            standardized.append(std_seq)
            masks.append(mask)
        
        return np.stack(standardized), np.stack(masks)
