"""
CTC Layer
Connectionist Temporal Classification
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List, Optional


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
        beam_width: int = 5,
        lm_scorer: Optional[Callable[[List[int]], float]] = None,
        lm_weight: float = 0.3,
    ) -> list:
        """
        CTC Prefix Beam Search decoding (Graves et al. 2012).

        Args:
            logits:     (B, T, vocab_size+1) – raw model outputs (pre-softmax)
            beam_width: number of active beams per timestep
            lm_scorer:  optional callable (token_id_list -> float log-prob) for LM re-scoring
            lm_weight:  interpolation weight for LM score (0 = CTC only)

        Returns:
            List of decoded token-id sequences, one per batch item.
        """
        NEG_INF = float("-inf")

        def _log_add(a: float, b: float) -> float:
            """Numerically stable log(exp(a) + exp(b))."""
            if a == NEG_INF:
                return b
            if b == NEG_INF:
                return a
            if a > b:
                return a + math.log1p(math.exp(b - a))
            return b + math.log1p(math.exp(a - b))

        log_probs = F.log_softmax(logits.float(), dim=-1)  # (B, T, V)
        B, T, V = log_probs.shape

        results: List[List[int]] = []

        for b in range(B):
            lp = log_probs[b].cpu().tolist()  # list[list[float]]  T × V

            # beam: prefix (tuple[int]) -> [log_p_blank, log_p_nonblank]
            beams: dict = {(): [0.0, NEG_INF]}

            for t in range(T):
                new_beams: dict = {}

                # Prune input beams to top-beam_width before expansion
                active = sorted(
                    beams.items(),
                    key=lambda x: _log_add(x[1][0], x[1][1]),
                    reverse=True,
                )[:beam_width]

                for prefix, (log_pb, log_pnb) in active:
                    log_p_total = _log_add(log_pb, log_pnb)
                    last_c = prefix[-1] if prefix else None

                    # ── 1. Emit blank → same prefix ──────────────────────────
                    new_log_pb = log_p_total + lp[t][self.blank_idx]
                    if prefix not in new_beams:
                        new_beams[prefix] = [NEG_INF, NEG_INF]
                    new_beams[prefix][0] = _log_add(new_beams[prefix][0], new_log_pb)

                    # ── 2-3. Emit non-blank token c ──────────────────────────
                    for c in range(V):
                        if c == self.blank_idx:
                            continue

                        if c == last_c:
                            # Same as last symbol:
                            # (a) non-blank path → prefix STAYS the same
                            nb_stay = log_pnb + lp[t][c]
                            new_beams[prefix][1] = _log_add(new_beams[prefix][1], nb_stay)

                            # (b) blank path → new symbol appended (prefix + c)
                            new_prefix = prefix + (c,)
                            if new_prefix not in new_beams:
                                new_beams[new_prefix] = [NEG_INF, NEG_INF]
                            nb_new = log_pb + lp[t][c]
                            new_beams[new_prefix][1] = _log_add(new_beams[new_prefix][1], nb_new)
                        else:
                            # Different symbol → new prefix via both paths
                            new_prefix = prefix + (c,)
                            if new_prefix not in new_beams:
                                new_beams[new_prefix] = [NEG_INF, NEG_INF]
                            nb_new = log_p_total + lp[t][c]
                            new_beams[new_prefix][1] = _log_add(new_beams[new_prefix][1], nb_new)

                # Keep top beams for next step (2× beam_width so LM has room)
                beams = dict(
                    sorted(
                        new_beams.items(),
                        key=lambda x: _log_add(x[1][0], x[1][1]),
                        reverse=True,
                    )[: beam_width * 2]
                )

            # ── Final selection: CTC score + optional LM re-scoring ─────────
            if lm_scorer is not None and lm_weight > 0.0:
                scored: List[tuple] = []
                for prefix, (log_pb, log_pnb) in beams.items():
                    ctc_score = _log_add(log_pb, log_pnb)
                    lm_score = lm_scorer(list(prefix))
                    total = ctc_score + lm_weight * lm_score
                    scored.append((list(prefix), total))
                scored.sort(key=lambda x: x[1], reverse=True)
                results.append(scored[0][0] if scored else [])
            else:
                best = max(beams.items(), key=lambda x: _log_add(x[1][0], x[1][1]))
                results.append(list(best[0]))

        return results


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
