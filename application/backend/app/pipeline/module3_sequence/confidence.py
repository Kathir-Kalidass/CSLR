"""
Confidence Scoring
Calculates confidence scores for predictions
"""

import torch
import numpy as np
from typing import List, Tuple


class ConfidenceScorer:
    """
    Computes confidence scores for CTC predictions
    """
    
    def __init__(self):
        pass
    
    def compute_frame_confidence(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute per-frame confidence
        
        Args:
            logits: (B, T, vocab_size) - Model logits
        
        Returns:
            (B, T) - Frame-level confidence scores
        """
        probs = torch.softmax(logits, dim=-1)
        max_probs, _ = torch.max(probs, dim=-1)
        return max_probs
    
    def compute_sequence_confidence(
        self,
        logits: torch.Tensor,
        decoded_tokens: List[int]
    ) -> float:
        """
        Compute overall sequence confidence
        
        Args:
            logits: (T, vocab_size) - Logits for single sequence
            decoded_tokens: Decoded token sequence
        
        Returns:
            Average confidence score
        """
        if not decoded_tokens:
            return 0.0
        
        probs = torch.softmax(logits, dim=-1)
        
        # Get probabilities of predicted tokens
        token_probs = []
        for token in decoded_tokens:
            # Find frames where this token was predicted
            max_tokens = torch.argmax(probs, dim=-1)
            mask = (max_tokens == token)
            
            if mask.any():
                token_prob = probs[mask, token].max().item()
                token_probs.append(token_prob)
        
        if not token_probs:
            return 0.0
        
        return float(np.mean(token_probs))
    
    def compute_batch_confidence(
        self,
        logits: torch.Tensor,
        decoded_batch: List[List[int]]
    ) -> List[float]:
        """
        Compute confidence for batch
        
        Args:
            logits: (B, T, vocab_size)
            decoded_batch: Batch of decoded sequences
        
        Returns:
            List of confidence scores
        """
        confidences = []
        
        for i, decoded in enumerate(decoded_batch):
            conf = self.compute_sequence_confidence(logits[i], decoded)
            confidences.append(conf)
        
        return confidences

    def compute_confidence(
        self,
        logits: torch.Tensor,
        predictions: List[List[str]]
    ) -> List[float]:
        """
        Backward-compatible batch confidence API.
        """
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)

        frame_conf = self.compute_frame_confidence(logits)  # (B, T)
        scores = frame_conf.mean(dim=1).detach().cpu().tolist()
        return [float(min(1.0, max(0.0, s))) for s in scores]
    
    def apply_threshold(
        self,
        decoded: List[int],
        confidence: float,
        threshold: float = 0.7
    ) -> Tuple[List[int], bool]:
        """
        Apply confidence threshold
        
        Args:
            decoded: Decoded sequence
            confidence: Confidence score
            threshold: Minimum confidence
        
        Returns:
            (filtered sequence, is_confident)
        """
        if confidence >= threshold:
            return decoded, True
        else:
            return [], False
