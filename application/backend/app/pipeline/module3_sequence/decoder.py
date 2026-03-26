"""
Decoder
Converts token indices to gloss strings
"""

from typing import Dict, List, Optional
import json
import torch

from app.utils.ctc_decoder import CTCDecoder as _AdvancedCTCDecoder


class GlossDecoder:
    """
    Decodes token indices to gloss strings
    Manages vocabulary mapping
    """
    
    def __init__(self, vocab: Optional[Dict[int, str]] = None):
        """
        Args:
            vocab: Dictionary mapping index -> gloss string
        """
        self.vocab = vocab or self._default_vocab()
        self.idx2gloss = self.vocab
        self.gloss2idx = {v: k for k, v in self.vocab.items()}
    
    def _default_vocab(self) -> Dict[int, str]:
        """Create default vocabulary"""
        # TODO: Load from file
        # For now, create dummy vocab
        return {
            0: "<blank>",
            1: "HELLO",
            2: "WORLD",
            3: "HOW",
            4: "ARE",
            5: "YOU",
            6: "THANK",
            7: "PLEASE",
            8: "YES",
            9: "NO",
        }
    
    def decode(self, token_ids: List[int]) -> List[str]:
        """
        Decode token IDs to gloss strings
        
        Args:
            token_ids: List of token indices
        
        Returns:
            List of gloss strings
        """
        glosses = []
        for token_id in token_ids:
            if token_id in self.idx2gloss:
                gloss = self.idx2gloss[token_id]
                if gloss != "<blank>":
                    glosses.append(gloss)
        
        return glosses
    
    def decode_batch(self, batch_token_ids: List[List[int]]) -> List[List[str]]:
        """
        Decode batch of token sequences
        
        Args:
            batch_token_ids: Batch of token ID lists
        
        Returns:
            Batch of gloss string lists
        """
        return [self.decode(token_ids) for token_ids in batch_token_ids]
    
    def encode(self, glosses: List[str]) -> List[int]:
        """
        Encode gloss strings to token IDs
        
        Args:
            glosses: List of gloss strings
        
        Returns:
            List of token IDs
        """
        token_ids = []
        for gloss in glosses:
            if gloss in self.gloss2idx:
                token_ids.append(self.gloss2idx[gloss])
            else:
                # Unknown token
                token_ids.append(0)
        
        return token_ids
    
    @classmethod
    def from_file(cls, vocab_path: str) -> 'GlossDecoder':
        """
        Load vocabulary from JSON file
        
        Args:
            vocab_path: Path to vocabulary JSON file
        
        Returns:
            GlossDecoder instance
        """
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        
        # Convert string keys to int
        vocab = {int(k): v for k, v in vocab.items()}
        
        return cls(vocab=vocab)
    
    def save_vocab(self, save_path: str):
        """Save vocabulary to JSON file"""
        with open(save_path, 'w') as f:
            json.dump(self.vocab, f, indent=2)


class Decoder(GlossDecoder):
    """
    Backward-compatible alias for GlossDecoder.
    Supports initialization with a label list.
    """

    def __init__(
        self,
        labels: Optional[List[str]] = None,
        vocab: Optional[Dict[int, str]] = None,
    ):
        if vocab is None and labels is not None:
            vocab = {i: label for i, label in enumerate(labels)}
        super().__init__(vocab=vocab)


class CTCDecoder:
    """
    Batch-friendly compatibility wrapper around the advanced CTC decoder.
    """

    def __init__(
        self,
        labels: List[str],
        blank_id: int = 0,
        blank_idx: Optional[int] = None,
        beam_width: int = 5,
        lm_path: Optional[str] = None,
        lm_weight: float = 0.0,
        lm_token_bonus: float = 0.0,
        lm_candidates: int = 20,
    ):
        effective_blank = blank_idx if blank_idx is not None else blank_id
        self._decoder = _AdvancedCTCDecoder(
            labels=labels,
            blank_idx=effective_blank,
            beam_width=beam_width,
            lm_path=lm_path,
            lm_weight=lm_weight,
            lm_token_bonus=lm_token_bonus,
            lm_candidates=lm_candidates,
        )

    @property
    def has_language_model(self) -> bool:
        return self._decoder.has_language_model

    @property
    def language_model_loaded(self) -> bool:
        return self._decoder.language_model_loaded

    def greedy_decode(self, logits: torch.Tensor) -> List[List[str]]:
        """
        Decode batched logits using greedy CTC collapse.
        """
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)

        decoded: List[List[str]] = []
        for seq_logits in logits:
            result = self._decoder.ctc_greedy_decode(seq_logits)
            decoded.append(result.gloss_tokens)
        return decoded

    def beam_search_decode(
        self,
        logits: torch.Tensor,
        beam_width: Optional[int] = None,
    ) -> List[List[str]]:
        """
        Decode batched logits using beam search.
        """
        if beam_width is not None:
            self._decoder.beam_width = beam_width

        if logits.dim() == 2:
            logits = logits.unsqueeze(0)

        decoded: List[List[str]] = []
        for seq_logits in logits:
            result = self._decoder.beam_search_decode(seq_logits)
            decoded.append(result.gloss_tokens)
        return decoded
