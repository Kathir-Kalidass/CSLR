"""
Frame Sampler
Temporal sampling and FPS normalization
"""

import numpy as np
from typing import List, Optional
from app.core.logging import logger


class FrameSampler:
    """
    Samples frames to target FPS
    Handles temporal normalization
    """
    
    def __init__(
        self,
        target_fps: int = 25,
        clip_length: int = 32,
        target_length: Optional[int] = None,
    ):
        self.target_fps = target_fps
        self.clip_length = target_length if target_length is not None else clip_length
    
    def sample_frames(
        self, 
        frames: List[np.ndarray], 
        original_fps: Optional[float] = None,
        target_fps: Optional[float] = None,
    ) -> List[np.ndarray]:
        """
        Sample frames to target FPS
        
        Args:
            frames: Input frame list
            original_fps: Original video FPS
        
        Returns:
            Sampled frames at target FPS
        """
        if original_fps is None:
            original_fps = target_fps
        if original_fps is None or original_fps <= 0:
            return frames

        if abs(original_fps - self.target_fps) < 0.1:
            return frames
        
        num_frames = len(frames)
        sample_rate = original_fps / self.target_fps
        
        indices = [int(i * sample_rate) for i in range(int(num_frames / sample_rate))]
        sampled = [frames[i] for i in indices if i < num_frames]
        
        logger.info(f"Sampled {len(sampled)} frames from {num_frames} ({original_fps} -> {self.target_fps} FPS)")
        return sampled

    def sample(self, frames: List[np.ndarray], original_fps: float) -> List[np.ndarray]:
        """
        Backward-compatible alias for sample_frames().
        """
        return self.sample_frames(frames, original_fps)

    def uniform_sample(
        self,
        frames: List[np.ndarray],
        target_length: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        Uniformly sample to a fixed length.
        """
        if not frames:
            return []

        length = target_length if target_length is not None else self.clip_length
        if length <= 0:
            return []

        total = len(frames)
        if total >= length:
            indices = np.linspace(0, total - 1, length, dtype=int)
            return [frames[i] for i in indices]

        indices = list(range(total)) + [total - 1] * (length - total)
        return [frames[i] for i in indices]
    
    def extract_clips(
        self, 
        frames: List[np.ndarray], 
        stride: Optional[int] = None
    ) -> List[List[np.ndarray]]:
        """
        Extract fixed-length clips with sliding window
        
        Args:
            frames: Input frames
            stride: Sliding window stride (default: clip_length //2)
        
        Returns:
            List of clips
        """
        if stride is None:
            stride = self.clip_length // 2
        
        clips = []
        for i in range(0, len(frames) - self.clip_length + 1, stride):
            clip = frames[i:i + self.clip_length]
            clips.append(clip)
        
        logger.info(f"Extracted {len(clips)} clips of length {self.clip_length}")
        return clips
