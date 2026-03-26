"""
Advanced CTC Decoder (Production-Ready)
Greedy and beam search decoding with confidence scoring
"""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class DecodeResult:
    """CTC decoding result with confidence"""
    gloss_tokens: List[str]
    confidence: float
    frame_confidences: Optional[List[float]] = None
    acoustic_score: Optional[float] = None
    lm_score: Optional[float] = None


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
        lm_path: Optional[str] = None,
        lm_weight: float = 0.0,
        lm_token_bonus: float = 0.0,
        lm_candidates: int = 20,
        language_model: Optional[Any] = None,
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
        self.lm_weight = max(0.0, float(lm_weight))
        self.lm_token_bonus = float(lm_token_bonus)
        self.lm_candidates = max(self.beam_width, int(lm_candidates))
        self.lm_path = lm_path
        self._language_model = language_model or self._load_language_model(lm_path)

    @property
    def has_language_model(self) -> bool:
        return self._language_model is not None and self.lm_weight > 0.0

    @property
    def language_model_loaded(self) -> bool:
        return self._language_model is not None

    def _load_language_model(self, lm_path: Optional[str]) -> Optional[Any]:
        if not lm_path:
            return None

        path = Path(lm_path)
        if not path.exists():
            return None

        try:
            import kenlm  # type: ignore
        except ImportError:
            return None

        try:
            return kenlm.Model(str(path))
        except Exception:
            return None

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

    def _score_language_model(self, tokens: List[str]) -> float:
        if not self.has_language_model or not tokens:
            return 0.0
        try:
            return float(self._language_model.score(" ".join(tokens), bos=True, eos=True))
        except Exception:
            return 0.0

    def _acoustic_rank_score(self, tokens: List[str], score: float) -> float:
        norm = score / (max(1, len(tokens)) ** self.length_norm_alpha)
        rep_penalty = self.repetition_penalty * self._sequence_repetition_count(tokens)
        return float(norm - rep_penalty)

    def _combined_rank_score(self, tokens: List[str], score: float) -> Tuple[float, float, float]:
        acoustic = self._acoustic_rank_score(tokens, score)
        lm_score = self._score_language_model(tokens)
        total = acoustic + (self.lm_weight * lm_score) + (self.lm_token_bonus * len(tokens))
        return total, acoustic, lm_score

    @staticmethod
    def _dedupe_candidates(candidates: List[Tuple[List[str], float]]) -> List[Tuple[List[str], float]]:
        best_by_tokens: Dict[Tuple[str, ...], float] = {}
        for tokens, score in candidates:
            key = tuple(tokens)
            current = best_by_tokens.get(key)
            if current is None or score > current:
                best_by_tokens[key] = score
        return [(list(tokens), score) for tokens, score in best_by_tokens.items()]

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
            frame_confidences=frame_confidences,
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
        candidate_limit = max(beam_k, self.lm_candidates if self.has_language_model else beam_k)

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
            deduped = self._dedupe_candidates(candidates)
            ranked = []
            for toks, score in deduped:
                ranked.append((toks, score, self._acoustic_rank_score(toks, score)))

            beams = [(toks, score) for toks, score, _ in sorted(ranked, key=lambda x: x[2], reverse=True)[:candidate_limit]]

        ranked_final = []
        for toks, score in self._dedupe_candidates(beams):
            total, acoustic_score, lm_score = self._combined_rank_score(toks, score)
            ranked_final.append((toks, score, total, acoustic_score, lm_score))
        ranked_final.sort(key=lambda x: x[2], reverse=True)

        # Best beam
        tokens, best_score, best_total, best_acoustic, best_lm = ranked_final[0]
        second_total = ranked_final[1][2] if len(ranked_final) > 1 else best_total
        beam_margin = max(0.0, best_total - second_total)
        
        # Confidence from score and beam margin
        base_conf = float(np.exp(best_score / max(T, 1)))
        margin_boost = float(np.clip(beam_margin / max(T, 1), 0.0, 0.2))
        confidence = float(np.clip(base_conf + margin_boost, 0.0, 1.0))
        
        return DecodeResult(
            gloss_tokens=tokens,
            confidence=confidence,
            frame_confidences=frame_confidences,
            acoustic_score=best_acoustic,
            lm_score=best_lm,
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
