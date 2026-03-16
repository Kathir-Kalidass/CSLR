"""
Advanced CTC Decoder (Production-Ready)
Greedy and beam search decoding with confidence scoring
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class DecodeResult:
    """CTC decoding result with confidence"""
    gloss_tokens: List[str]
    confidence: float
    frame_confidences: Optional[List[float]] = None


class CTCDecoder:
    """
    Enhanced CTC Decoder with greedy and beam search.
    
    Supports:
    - Greedy decoding (fast)
    - Beam search decoding (accurate)
    - Frame-level confidence scores
    - Heuristic fallback for untrained models
    """

    def __init__(
        self, 
        labels: List[str],
        blank_idx: int = 0,
        beam_width: int = 5
    ) -> None:
        """
        Args:
            labels: Vocabulary list (excluding blank token)
            blank_idx: Index of CTC blank token (usually 0)
            beam_width: Beam size for beam search
        """
        self.labels = labels
        self.blank_idx = blank_idx
        self.beam_width = beam_width

    def ctc_greedy_decode(
        self, 
        logits: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> DecodeResult:
        """
        Greedy CTC decoding

        Args:
            logits: (T, num_classes) or (B, T, num_classes)
            lengths: Optional sequence lengths (B,)

        Returns:
            DecodeResult with tokens and confidence
        """
        # Handle batch dimension
        if logits.dim() == 3:
            assert logits.shape[0] == 1, "Batch size must be 1 for greedy decode"
            logits = logits.squeeze(0)
        
        # Compute probabilities
        probs = F.softmax(logits, dim=-1)  # (T, num_classes)
        
        # Frame-level confidence (max probability per frame)
        frame_confidences = torch.max(probs, dim=-1).values.detach().cpu().numpy().tolist()
        avg_confidence = float(np.mean(frame_confidences))
        
        # Get predictions
        pred_indices = torch.argmax(probs, dim=-1).detach().cpu().numpy().tolist()  # (T,)
        
        # CTC collapse: remove blanks and consecutive duplicates
        tokens: List[int] = []
        prev_idx = None
        for idx in pred_indices:
            if idx != self.blank_idx and idx != prev_idx:
                tokens.append(idx)
            prev_idx = idx
        
        # Convert indices to labels (assuming label indices start from 1)
        decoded_tokens = []
        for idx in tokens:
            if 1 <= idx <= len(self.labels):
                decoded_tokens.append(self.labels[idx - 1])
        
        return DecodeResult(
            gloss_tokens=decoded_tokens,
            confidence=avg_confidence,
            frame_confidences=frame_confidences
        )

    def beam_search_decode(
        self, 
        logits: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> DecodeResult:
        """
        Beam search CTC decoding
        
        Args:
            logits: (T, num_classes) or (B, T, num_classes)
            lengths: Optional sequence lengths (B,)
        
        Returns:
            DecodeResult with best beam path
        """
        # Handle batch dimension
        if logits.dim() == 3:
            assert logits.shape[0] == 1, "Batch size must be 1 for beam search"
            logits = logits.squeeze(0)
        
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # (T, num_classes)
        T, num_classes = log_probs.shape
        
        # Initialize beams: (prefix, score)
        beams = [("", 0.0)]  # Empty prefix with score 0
        
        for t in range(T):
            candidates = []
            
            for prefix, score in beams:
                for c in range(num_classes):
                    # Update score
                    new_score = score + log_probs[t, c].item()
                    
                    if c == self.blank_idx:
                        # Blank doesn't extend prefix
                        candidates.append((prefix, new_score))
                    else:
                        # Add character to prefix
                        label = self.labels[c - 1] if 1 <= c <= len(self.labels) else ""
                        if label:
                            # Avoid consecutive duplicates
                            if not prefix or prefix.split()[-1] != label:
                                new_prefix = f"{prefix} {label}".strip()
                            else:
                                new_prefix = prefix
                            candidates.append((new_prefix, new_score))
            
            # Keep top beam_width candidates
            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:self.beam_width]
        
        # Best beam
        best_prefix, best_score = beams[0]
        tokens = best_prefix.split() if best_prefix else []
        
        # Approximate confidence from score
        confidence = min(1.0, max(0.0, np.exp(best_score / max(T, 1))))
        
        return DecodeResult(
            gloss_tokens=tokens,
            confidence=confidence
        )

    def heuristic_fallback(
        self, 
        pose_window: torch.Tensor
    ) -> DecodeResult:
        """
        Heuristic decoding for demo/untrained models
        
        Args:
            pose_window: (T, N, D) pose keypoints
        
        Returns:
            DecodeResult with heuristic predictions
        """
        arr = pose_window.detach().cpu().numpy()
        
        # Compute hand and torso energy
        hand_energy = float(np.mean(np.abs(arr[:, 33:, :])))  # Hands (landmarks 33-74)
        torso_energy = float(np.mean(np.abs(arr[:, :33, :])))  # Pose (landmarks 0-32)
        motion_ratio = hand_energy / (torso_energy + 1e-6)
        
        # Select token based on motion characteristics
        idx = int((motion_ratio * 10 + hand_energy * 20) % len(self.labels))
        token = self.labels[idx]
        
        # Generate sequence based on motion ratio
        if motion_ratio > 1.6:
            seq = [token, "PLEASE"] if token != "PLEASE" else [token]
        elif motion_ratio < 0.9:
            seq = ["HELLO", token] if token != "HELLO" else [token]
        else:
            seq = [token]
        
        confidence = min(0.95, 0.55 + hand_energy)
        
        return DecodeResult(
            gloss_tokens=seq,
            confidence=confidence
        )


class CaptionPostProcessor:
    """
    Post-process decoded glosses for display.
    
    Features:
    - Deduplication
    - Confidence filtering
    - Temporal smoothing
    """

    def __init__(
        self, 
        max_history: int = 20,
        min_confidence: float = 0.45
    ) -> None:
        """
        Args:
            max_history: Maximum history size for smoothing
            min_confidence: Minimum confidence threshold
        """
        self.min_confidence = min_confidence
        self.history: deque[str] = deque(maxlen=max_history)

    @staticmethod
    def _dedupe(tokens: Iterable[str]) -> List[str]:
        """Remove consecutive duplicates"""
        out = []
        prev = None
        for tok in tokens:
            if tok != prev:
                out.append(tok)
            prev = tok
        return out

    def update(
        self, 
        tokens: List[str], 
        confidence: float
    ) -> List[str]:
        """
        Update history and return smoothed output

        Args:
            tokens: Decoded gloss tokens
            confidence: Prediction confidence

        Returns:
            Smoothed token sequence
        """
        if confidence < self.min_confidence or not tokens:
            return []

        # Deduplicate and add to history
        deduped = self._dedupe(tokens)
        self.history.extend(deduped)
        
        # Return deduplicated history
        return self._dedupe(self.history)

    def reset(self) -> None:
        """Clear history"""
        self.history.clear()
