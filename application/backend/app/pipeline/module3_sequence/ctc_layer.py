"""
CTC Layer
Connectionist Temporal Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CTCLayer(nn.Module):
    """
    CTC Loss and Inference Layer
    """
    
    def __init__(self, blank_idx: int = 0, vocab_size: Optional[int] = None):
        super().__init__()
        self.blank_idx = blank_idx
        self.vocab_size = vocab_size
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, reduction='mean', zero_infinity=True)
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CTC loss
        
        Args:
            logits: (B, T, vocab_size+1) - Model outputs
            targets: (B, S) - Target sequences
            input_lengths: (B,) - Actual sequence lengths
            target_lengths: (B,) - Target lengths
        
        Returns:
            CTC loss value
        """
        # CTC expects (T, B, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.transpose(0, 1)  # (T, B, vocab_size+1)
        
        loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        
        return loss

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward alias to compute_loss() for nn.Module-style calls.
        """
        return self.compute_loss(logits, targets, input_lengths, target_lengths)
    
    def decode_greedy(self, logits: torch.Tensor) -> list:
        """
        Greedy CTC decoding
        
        Args:
            logits: (B, T, vocab_size+1)
        
        Returns:
            List of decoded sequences (one per batch)
        """
        # Get most likely tokens
        predictions = torch.argmax(logits, dim=-1)  # (B, T)
        
        decoded = []
        for pred in predictions:
            # Remove blanks and consecutive duplicates
            pred = pred.tolist()
            decoded_seq = []
            prev = None
            
            for token in pred:
                if token != self.blank_idx and token != prev:
                    decoded_seq.append(token)
                prev = token
            
            decoded.append(decoded_seq)
        
        return decoded
    
    def decode_beam_search(
        self,
        logits: torch.Tensor,
        beam_width: int = 5
    ) -> list:
        """
        Beam search CTC decoding
        
        Args:
            logits: (B, T, vocab_size+1)
            beam_width: Beam width
        
        Returns:
            List of decoded sequences
        """
        # TODO: Implement proper beam search
        # For now, use greedy decoding
        return self.decode_greedy(logits)


def ctc_greedy_decode(logits: torch.Tensor, blank_idx: int = 0) -> list:
    """
    Standalone greedy CTC decoder
    
    Args:
        logits: (B, T, vocab_size)
        blank_idx: Blank token index
    
    Returns:
        Decoded sequences
    """
    predictions = torch.argmax(logits, dim=-1)
    
    decoded = []
    for pred in predictions:
        pred = pred.tolist()
        decoded_seq = []
        prev = None
        
        for token in pred:
            if token != blank_idx and token != prev:
                decoded_seq.append(token)
            prev = token
        
        decoded.append(decoded_seq)
    
    return decoded
