"""
Normalization
Spatial and intensity normalization for frames and poses
"""

import numpy as np
import cv2
from typing import List, Tuple


class Normalizer:
    """
    Normalizes RGB frames and pose coordinates
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Normalize RGB frame (ImageNet stats)
        
        Args:
            frame: RGB frame (H, W, 3)
        
        Returns:
            Normalized frame (3, H, W)
        """
        # Resize
        resized = cv2.resize(frame, self.target_size)
        
        # Convert to float [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Normalize with mean/std
        normalized = (normalized - self.mean) / self.std
        
        # Transpose to (C, H, W)
        normalized = np.transpose(normalized, (2, 0, 1))
        
        return normalized

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Backward-compatible alias for normalize_frame().
        """
        return self.normalize_frame(image)

    def normalize_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Normalize a batch of images and stack to (B, C, H, W).
        """
        if not images:
            return np.zeros((0, 3, self.target_size[1], self.target_size[0]), dtype=np.float32)
        normalized = [self.normalize_frame(img) for img in images]
        return np.stack(normalized, axis=0).astype(np.float32)
    
    def normalize_pose(self, pose_sequence: np.ndarray) -> np.ndarray:
        """
        Normalize pose keypoints
        
        Args:
            pose_sequence: (T, num_keypoints)
        
        Returns:
            Normalized pose sequence
        """
        # Z-score normalization
        mean = np.mean(pose_sequence, axis=0, keepdims=True)
        std = np.std(pose_sequence, axis=0, keepdims=True) + 1e-6
        
        normalized = (pose_sequence - mean) / std
        return normalized.astype(np.float32)
    
    def denormalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Reverse normalization for visualization"""
        # Transpose back to (H, W, C)
        frame = np.transpose(frame, (1, 2, 0))
        
        # Denormalize
        frame = frame * self.std + self.mean
        
        # Clip and convert to uint8
        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
        
        return frame


class ImageNormalizer(Normalizer):
    """
    Backward-compatible alias for Normalizer.
    """
    pass
