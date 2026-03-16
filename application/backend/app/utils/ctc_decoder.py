"""
Advanced CTC Decoder (Production-Ready)
Greedy and beam search decoding with confidence scoring
"""

from __future__ import annotations

import math
from collections import Counter, deque
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
        beam_width: int = 5,
        min_token_run: int = 2,
        min_token_margin: float = 0.04,
        length_norm_alpha: float = 0.35,
        repetition_penalty: float = 0.15,
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
        self.min_token_run = max(1, int(min_token_run))
        self.min_token_margin = max(0.0, float(min_token_margin))
        self.length_norm_alpha = max(0.0, float(length_norm_alpha))
        self.repetition_penalty = max(0.0, float(repetition_penalty))

    def _idx_to_label(self, idx: int) -> str:
        """Map class index to label while respecting blank index."""
        if idx == self.blank_idx:
            return ""
        if 0 <= idx < len(self.labels):
            return self.labels[idx]
        return ""

    @staticmethod
    def _sequence_repetition_count(tokens: List[str]) -> int:
        """Count repeated tokens in a sequence (global duplicates)."""
        if not tokens:
            return 0
        counts = Counter(tokens)
        return int(sum(max(0, v - 1) for v in counts.values()))

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
        
        # Confidence features from per-frame distribution
        top2_vals = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
        top1 = top2_vals[:, 0]
        top2 = top2_vals[:, 1] if top2_vals.shape[1] > 1 else torch.zeros_like(top1)
        margins = (top1 - top2).detach().cpu().numpy().tolist()
        frame_confidences = top1.detach().cpu().numpy().tolist()

        entropy = -torch.sum(probs * torch.log(probs.clamp_min(1e-8)), dim=-1)
        entropy_norm = entropy / math.log(max(2, probs.shape[-1]))
        entropy_penalty = float(entropy_norm.mean().item())
        margin_gain = float(np.mean(margins))
        raw_conf = float(np.mean(frame_confidences))
        avg_confidence = float(np.clip(raw_conf * (1.0 - 0.35 * entropy_penalty) * (0.7 + 0.3 * margin_gain), 0.0, 1.0))
        
        # Get predictions
        pred_indices = torch.argmax(probs, dim=-1).detach().cpu().numpy().tolist()  # (T,)
        
        # CTC collapse with run-length and margin-based filtering
        tokens: List[int] = []
        prev_idx = None
        run_len = 0
        run_margin_sum = 0.0

        def flush(prev: Optional[int], length: int, margin_sum: float) -> None:
            if prev is None or prev == self.blank_idx:
                return
            avg_margin = margin_sum / max(length, 1)
            if length >= self.min_token_run and avg_margin >= self.min_token_margin:
                tokens.append(prev)

        for idx, margin in zip(pred_indices, margins):
            if prev_idx is None or idx != prev_idx:
                flush(prev_idx, run_len, run_margin_sum)
                prev_idx = idx
                run_len = 1
                run_margin_sum = float(margin)
            else:
                run_len += 1
                run_margin_sum += float(margin)
        flush(prev_idx, run_len, run_margin_sum)
        
        # Convert indices to labels (assuming label indices start from 1)
        decoded_tokens = []
        for idx in tokens:
            label = self._idx_to_label(int(idx))
            if label:
                decoded_tokens.append(label)
        
        return DecodeResult(
            gloss_tokens=decoded_tokens,
            confidence=avg_confidence,
            frame_confidences=frame_confidences
        )

    def beam_search_decode(
        self, 
        logits: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        beam_width: Optional[int] = None,
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
        
        beam_k = beam_width or self.beam_width

        probs = F.softmax(logits, dim=-1)
        frame_confidences = torch.max(probs, dim=-1).values.detach().cpu().numpy().tolist()

        # Initialize beams: (tokens, score)
        beams: List[tuple[List[str], float]] = [([], 0.0)]
        
        for t in range(T):
            candidates = []
            
            for prefix_tokens, score in beams:
                for c in range(num_classes):
                    # Update score
                    new_score = score + log_probs[t, c].item()
                    
                    if c == self.blank_idx:
                        # Blank doesn't extend prefix
                        candidates.append((prefix_tokens, new_score))
                    else:
                        # Add character to prefix
                        label = self._idx_to_label(c)
                        if label:
                            # Avoid consecutive duplicates
                            if not prefix_tokens or prefix_tokens[-1] != label:
                                new_prefix = prefix_tokens + [label]
                            else:
                                new_prefix = prefix_tokens
                            candidates.append((new_prefix, new_score))
            
            # Rank by length-normalized score and repetition penalty.
            ranked = []
            for toks, score in candidates:
                norm = score / (max(1, len(toks)) ** self.length_norm_alpha)
                rep_penalty = self.repetition_penalty * self._sequence_repetition_count(toks)
                ranked.append((toks, score, norm - rep_penalty))

            beams = [(toks, score) for toks, score, _ in sorted(ranked, key=lambda x: x[2], reverse=True)[:beam_k]]
        
        # Best beam
        tokens, best_score = beams[0]
        second_score = beams[1][1] if len(beams) > 1 else best_score
        beam_margin = max(0.0, best_score - second_score)
        
        # Confidence from score and beam margin
        base_conf = float(np.exp(best_score / max(T, 1)))
        margin_boost = float(np.clip(beam_margin / max(T, 1), 0.0, 0.2))
        confidence = float(np.clip(base_conf + margin_boost, 0.0, 1.0))
        
        return DecodeResult(
            gloss_tokens=tokens,
            confidence=confidence,
            frame_confidences=frame_confidences,
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
        min_confidence: float = 0.45,
        vote_window: int = 5,
        min_votes: int = 2,
    ) -> None:
        """
        Args:
            max_history: Maximum history size for smoothing
            min_confidence: Minimum confidence threshold
        """
        self.min_confidence = min_confidence
        self.history: deque[str] = deque(maxlen=max_history)
        self.sequence_history: deque[List[str]] = deque(maxlen=max(vote_window, 1))
        self.vote_window = max(1, vote_window)
        self.min_votes = max(1, min_votes)

    def _consensus_sequence(self, min_votes_override: Optional[int] = None) -> List[str]:
        """Build token-wise consensus across recent decoded sequences."""
        if not self.sequence_history:
            return []

        window = list(self.sequence_history)[-self.vote_window :]
        max_len = max(len(seq) for seq in window)
        consensus: List[str] = []
        required_votes = max(1, int(min_votes_override if min_votes_override is not None else self.min_votes))

        for i in range(max_len):
            counts: Counter = Counter(seq[i] for seq in window if i < len(seq))
            if not counts:
                continue
            tok, votes = counts.most_common(1)[0]
            if votes >= required_votes:
                consensus.append(tok)

        return self._dedupe(consensus)

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
        confidence: float,
        min_votes_override: Optional[int] = None,
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
        self.sequence_history.append(deduped)
        self.history.extend(deduped)

        consensus = self._consensus_sequence(min_votes_override=min_votes_override)
        return consensus if consensus else self._dedupe(self.history)

    def reset(self) -> None:
        """Clear history"""
        self.history.clear()
        self.sequence_history.clear()
