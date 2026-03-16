"""
Pose Extractor
Extracts human pose keypoints using MediaPipe
"""

import cv2
import numpy as np
from typing import Any, List, Optional, cast
import mediapipe as mp
from app.core.logging import logger


class PoseExtractor:
    """
    Extracts pose keypoints from frames
    Uses MediaPipe Holistic (face, pose, hands)
    """
    
    def __init__(self):
        mp_solutions = cast(Any, mp).solutions
        self.holistic = mp_solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract_pose(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract pose keypoints from single frame
        
        Args:
            frame: RGB frame
        
        Returns:
            Flattened keypoint array or None
        """
        results = self.holistic.process(frame)
        
        # Collect all keypoints
        keypoints = []
        
        # Pose landmarks (33 points)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            keypoints.extend([0.0] * (33 * 4))
        
        # Left hand (21 points)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * (21 * 3))
        
        # Right hand (21 points)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * (21 * 3))
        
        return np.array(keypoints, dtype=np.float32)

    def extract(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Backward-compatible alias for extract_pose().
        """
        return self.extract_pose(frame)
    
    def extract_sequence(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Extract pose sequence from frame list
        
        Args:
            frames: List of RGB frames
        
        Returns:
            Array of shape (T, num_keypoints)
        """
        pose_sequence = []
        
        for frame in frames:
            pose = self.extract_pose(frame)
            if pose is not None:
                pose_sequence.append(pose)
        
        if not pose_sequence:
            logger.warning("No pose detected in sequence")
            return np.zeros((len(frames), 258), dtype=np.float32)  # 33*4 + 21*3 + 21*3
        
        return np.array(pose_sequence, dtype=np.float32)
    
    def __del__(self):
        """Cleanup MediaPipe resources"""
        if hasattr(self, 'holistic'):
            self.holistic.close()
