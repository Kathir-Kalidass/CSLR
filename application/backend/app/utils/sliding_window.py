"""
Sliding Window Buffer (Production-Ready)
Maintains rolling buffers for temporal modeling
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple

import torch


class SlidingWindowBuffer:
    """
    Maintains rolling buffers and emits windows with configurable stride.
    
    Optimized for real-time inference with minimal memory overhead.
    """

    def __init__(self, window_size: int = 64, stride: int = 32) -> None:
        """
        Args:
            window_size: Number of frames in each window
            stride: Number of frames to advance before emitting next window
        """
        if stride > window_size:
            raise ValueError("stride must be <= window_size")
        
        self.window_size = window_size
        self.stride = stride
        self.rgb_buffer: Deque[torch.Tensor] = deque(maxlen=window_size)
        self.pose_buffer: Deque[torch.Tensor] = deque(maxlen=window_size)
        self._frames_since_emit = 0

    def add(
        self, 
        rgb_frame: torch.Tensor, 
        pose_frame: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Add a frame pair to the buffer.

        Args:
            rgb_frame: (3, H, W) or (C, H, W)
            pose_frame: (N, D) keypoints

        Returns:
            If window is ready: (rgb_window, pose_window)
            - rgb_window: (window_size, C, H, W)
            - pose_window: (window_size, N, D)
            Otherwise: None
        """
        self.rgb_buffer.append(rgb_frame)
        self.pose_buffer.append(pose_frame)

        # Wait until we have enough frames
        if len(self.rgb_buffer) < self.window_size:
            return None

        # Emit window based on stride
        self._frames_since_emit += 1
        if self._frames_since_emit < self.stride:
            return None

        self._frames_since_emit = 0
        
        # Stack frames into windows
        rgb = torch.stack(list(self.rgb_buffer), dim=0)
        pose = torch.stack(list(self.pose_buffer), dim=0)
        
        return rgb, pose

    def counts(self) -> int:
        """Return current buffer size"""
        return len(self.rgb_buffer)

    def reset(self) -> None:
        """Clear all buffers"""
        self.rgb_buffer.clear()
        self.pose_buffer.clear()
        self._frames_since_emit = 0

    def is_ready(self) -> bool:
        """Check if buffer has enough frames for a window"""
        return len(self.rgb_buffer) >= self.window_size
