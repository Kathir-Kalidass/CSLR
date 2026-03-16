"""
Gloss Buffer
Manages gloss sequence buffering for streaming
"""

from typing import List, Deque, Optional
from collections import deque


class GlossBuffer:
    """
    Buffers gloss tokens for streaming inference
    Handles temporal smoothing and deduplication
    """
    
    def __init__(
        self,
        buffer_size: int = 50,
        min_frequency: int = 2,
        smoothing_window: int = 5
    ):
        """
        Args:
            buffer_size: Maximum buffer size
            min_frequency: Minimum occurrences to accept token
            smoothing_window: Window for temporal smoothing
        """
        self.buffer_size = buffer_size
        self.min_frequency = min_frequency
        self.smoothing_window = smoothing_window
        
        self.buffer: Deque[str] = deque(maxlen=buffer_size)
        self.token_counts: dict = {}
        self.confirmed_glosses: List[str] = []
    
    def add_token(self, token: str) -> Optional[str]:
        """
        Add token to buffer
        
        Args:
            token: Gloss token
        
        Returns:
            Confirmed token if threshold met, else None
        """
        self.buffer.append(token)
        
        # Update counts
        self.token_counts[token] = self.token_counts.get(token, 0) + 1
        
        # Check if token should be confirmed
        if self.token_counts[token] >= self.min_frequency:
            if token not in self.confirmed_glosses[-1:]:  # Avoid duplicates
                self.confirmed_glosses.append(token)
                return token
        
        return None
    
    def add_sequence(self, tokens: List[str]) -> List[str]:
        """
        Add sequence of tokens
        
        Args:
            tokens: List of gloss tokens
        
        Returns:
            List of newly confirmed tokens
        """
        confirmed = []
        for token in tokens:
            result = self.add_token(token)
            if result:
                confirmed.append(result)
        
        return confirmed
    
    def get_confirmed(self) -> List[str]:
        """Get all confirmed glosses"""
        return self.confirmed_glosses.copy()
    
    def get_recent(self, n: int = 10) -> List[str]:
        """Get n most recent confirmed glosses"""
        return self.confirmed_glosses[-n:]
    
    def smooth_sequence(self, tokens: List[str]) -> List[str]:
        """
        Apply temporal smoothing
        Remove noisy tokens
        """
        if len(tokens) < self.smoothing_window:
            return tokens
        
        smoothed = []
        for i in range(len(tokens)):
            # Get window around current token
            start = max(0, i - self.smoothing_window // 2)
            end = min(len(tokens), i + self.smoothing_window // 2 + 1)
            window = tokens[start:end]
            
            # Keep token if it appears multiple times in window
            if window.count(tokens[i]) >= 2:
                smoothed.append(tokens[i])
        
        return smoothed
    
    def reset(self):
        """Reset buffer"""
        self.buffer.clear()
        self.token_counts.clear()
        self.confirmed_glosses.clear()
    
    def __len__(self):
        return len(self.buffer)
    
    def __repr__(self):
        return f"GlossBuffer(size={len(self.buffer)}, confirmed={len(self.confirmed_glosses)})"


class CaptionBuffer:
    """
    Backward-compatible caption buffer API used by legacy tests/services.
    """

    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self._captions: Deque[str] = deque(maxlen=max_size)

    def add(self, caption: str) -> None:
        self._captions.append(caption)

    def get_all(self) -> List[str]:
        return list(self._captions)

    def clear(self) -> None:
        self._captions.clear()

    def __len__(self) -> int:
        return len(self._captions)
