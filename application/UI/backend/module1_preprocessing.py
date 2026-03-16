"""
MODULE 1: Video Acquisition & Preprocessing (Optimized for 4GB GPU)

Real-time webcam preprocessing pipeline:
- Motion-based frame filtering
- Temporal subsampling
- ROI (Region of Interest) cropping
- MediaPipe Holistic pose extraction
- RGB tensor normalization (224x224)
- Pose keypoint normalization (75 landmarks)
- Dual buffer management (RGB + Pose)

Performance Target: 15-20 FPS on 4GB GPU
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import torch
import torchvision.transforms as transforms


@dataclass
class ProcessingStats:
    """Real-time statistics for Module 1"""
    fps: float = 0.0
    motion_score: float = 0.0
    frames_kept: int = 0
    frames_discarded: int = 0
    buffer_fill: int = 0
    buffer_capacity: int = 64
    roi_detected: bool = False
    pose_detected: bool = False
    processing_time_ms: float = 0.0


@dataclass
class ProcessedFrame:
    """Single frame output from preprocessing pipeline"""
    rgb_tensor: Optional[torch.Tensor]  # (3, 224, 224)
    pose_tensor: Optional[torch.Tensor]  # (75, 2)
    display_frame: np.ndarray  # For visualization
    kept: bool  # Whether frame was kept or discarded
    motion_score: float
    stats: ProcessingStats


class Module1PreprocessingEngine:
    """
    Optimized real-time preprocessing engine for 4GB GPU environments.
    
    Pipeline stages:
    1. Frame capture
    2. Motion-based filtering
    3. Temporal subsampling
    4. MediaPipe Holistic processing
    5. ROI extraction and cropping
    6. RGB normalization
    7. Pose keypoint normalization
    8. Buffer management
    """
    
    def __init__(
        self,
        frame_width: int = 640,
        frame_height: int = 480,
        target_fps: int = 20,
        process_every_n_frame: int = 2,
        motion_threshold: float = 5.0,
        buffer_size: int = 64,
        enable_adaptive_motion: bool = True,
        draw_landmarks: bool = True,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.target_fps = target_fps
        self.process_every_n_frame = process_every_n_frame
        self.motion_threshold = motion_threshold
        self.buffer_size = buffer_size
        self.enable_adaptive_motion = enable_adaptive_motion
        self.draw_landmarks = draw_landmarks
        
        # Internal state
        self._prev_gray: Optional[np.ndarray] = None
        self._frame_count = 0
        self._frames_kept = 0
        self._frames_discarded = 0
        self._recent_motion_scores: list[float] = []
        self._fps_tracker: deque = deque(maxlen=30)
        
        # Buffers for Module 2
        self.rgb_buffer: deque = deque(maxlen=buffer_size)
        self.pose_buffer: deque = deque(maxlen=buffer_size)
        
        # MediaPipe Holistic (LIGHTWEIGHT configuration)
        self.mp_holistic = mp.solutions.holistic  # type: ignore
        self.mp_drawing = mp.solutions.drawing_utils  # type: ignore
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,  # NOT 2 - saves ~40% compute
            smooth_landmarks=True,
            enable_segmentation=False,  # Disabled for speed
            refine_face_landmarks=False  # Disabled - we don't need face
        )
        
        # RGB preprocessing transform (ImageNet normalization)
        self.rgb_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _should_keep_frame(self, frame: np.ndarray) -> tuple[bool, float]:
        """
        Motion-based frame filtering using optical flow approximation.
        
        Returns:
            (keep: bool, motion_score: float)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self._prev_gray is None:
            self._prev_gray = gray
            return True, 255.0  # Always keep first frame
        
        # Compute frame difference
        diff = cv2.absdiff(self._prev_gray, gray)
        motion_score = float(np.mean(diff))
        
        self._prev_gray = gray
        
        # Adaptive threshold adjustment
        threshold = self.motion_threshold
        if self.enable_adaptive_motion and self._recent_motion_scores:
            # Adjust threshold based on recent motion history
            mean_motion = float(np.mean(self._recent_motion_scores[-30:]))
            threshold = max(3.0, min(12.0, 0.6 * mean_motion + 2.0))
        
        # Track motion history
        self._recent_motion_scores.append(motion_score)
        if len(self._recent_motion_scores) > 200:
            self._recent_motion_scores.pop(0)
        
        return motion_score > threshold, motion_score
    
    def _extract_landmarks(self, results, landmark_type: str, expected_count: int) -> np.ndarray:
        """Extract x,y coordinates from MediaPipe landmarks."""
        if landmark_type == "pose":
            landmarks = results.pose_landmarks
        elif landmark_type == "left_hand":
            landmarks = results.left_hand_landmarks
        elif landmark_type == "right_hand":
            landmarks = results.right_hand_landmarks
        else:
            landmarks = None
        
        if landmarks is None:
            return np.zeros((expected_count, 2), dtype=np.float32)
        
        points = np.array(
            [[lm.x, lm.y] for lm in landmarks.landmark],
            dtype=np.float32
        )
        
        if points.shape[0] != expected_count:
            # Pad or truncate
            padded = np.zeros((expected_count, 2), dtype=np.float32)
            n = min(expected_count, points.shape[0])
            padded[:n] = points[:n]
            return padded
        
        return points
    
    def _normalize_keypoints(self, points_75: np.ndarray) -> np.ndarray:
        """
        Normalize 75 keypoints relative to shoulder center.
        
        Makes pose invariant to:
        - Camera position
        - Person distance from camera
        - Different body sizes
        """
        # Get shoulder landmarks (indices 11, 12 in pose landmarks)
        left_shoulder = points_75[11]
        right_shoulder = points_75[12]
        
        # Compute center point
        center = (left_shoulder + right_shoulder) / 2.0
        
        # Compute torso width for scaling
        torso_width = float(np.linalg.norm(left_shoulder - right_shoulder))
        if torso_width < 1e-4:
            torso_width = 0.2  # Fallback to prevent division by zero
        
        # Normalize: center at origin, scale by torso width
        normalized = (points_75 - center) / torso_width
        
        # Clip to reasonable range
        normalized = np.clip(normalized, -1.5, 1.5)
        
        return normalized.astype(np.float32)
    
    def _extract_roi(self, frame: np.ndarray, pose_landmarks: np.ndarray) -> np.ndarray:
        """
        Extract Region of Interest (upper body + hands).
        
        Reduces background noise and focuses on signing area.
        Reduces processing by ~60%.
        """
        h, w, _ = frame.shape
        
        # Check if pose detected
        left_shoulder = pose_landmarks[11]
        right_shoulder = pose_landmarks[12]
        
        if np.all(left_shoulder == 0) or np.all(right_shoulder == 0):
            # No pose detected, return full frame
            return frame
        
        # Convert normalized coords to pixel coords
        x1 = int(left_shoulder[0] * w)
        x2 = int(right_shoulder[0] * w)
        y = int(left_shoulder[1] * h)
        
        # Define ROI box
        center_x = (x1 + x2) // 2
        shoulder_width = max(40, abs(x2 - x1))
        
        # Expand box to include hands and upper body
        roi_width = int(2.5 * shoulder_width)
        roi_height = int(2.8 * shoulder_width)
        
        # Adjust vertical center slightly downward
        y_center = int(y + 0.5 * shoulder_width)
        
        # Calculate bounds with clipping
        x_min = max(center_x - roi_width // 2, 0)
        x_max = min(center_x + roi_width // 2, w)
        y_min = max(y_center - roi_height // 2, 0)
        y_max = min(y_center + roi_height // 2, h)
        
        # Extract ROI
        if x_max <= x_min or y_max <= y_min:
            return frame
        
        return frame[y_min:y_max, x_min:x_max]
    
    def _draw_landmarks_on_frame(self, frame: np.ndarray, results) -> np.ndarray:
        """Draw MediaPipe landmarks on display frame."""
        if not self.draw_landmarks:
            return frame
        
        # Draw pose
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
        
        # Draw left hand
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
        
        # Draw right hand
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
        
        return frame
    
    def process_frame(self, frame: np.ndarray) -> ProcessedFrame:
        """
        Main preprocessing pipeline for a single frame.
        
        Returns:
            ProcessedFrame with tensors, display frame, and statistics
        """
        start_time = time.time()
        
        self._frame_count += 1
        display_frame = frame.copy()
        
        # Stage 1: Temporal subsampling (frame skipping)
        if self._frame_count % self.process_every_n_frame != 0:
            self._frames_discarded += 1
            stats = ProcessingStats(
                frames_discarded=self._frames_discarded,
                buffer_fill=len(self.rgb_buffer),
                buffer_capacity=self.buffer_size
            )
            return ProcessedFrame(
                rgb_tensor=None,
                pose_tensor=None,
                display_frame=display_frame,
                kept=False,
                motion_score=0.0,
                stats=stats
            )
        
        # Stage 2: Motion-based filtering
        keep_frame, motion_score = self._should_keep_frame(frame)
        
        if not keep_frame:
            self._frames_discarded += 1
            stats = ProcessingStats(
                motion_score=motion_score,
                frames_discarded=self._frames_discarded,
                buffer_fill=len(self.rgb_buffer),
                buffer_capacity=self.buffer_size
            )
            return ProcessedFrame(
                rgb_tensor=None,
                pose_tensor=None,
                display_frame=display_frame,
                kept=False,
                motion_score=motion_score,
                stats=stats
            )
        
        # Stage 3: MediaPipe Holistic processing (CPU)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb_frame)
        
        # Stage 4: Extract landmarks
        pose_xy = self._extract_landmarks(results, "pose", 33)
        left_hand_xy = self._extract_landmarks(results, "left_hand", 21)
        right_hand_xy = self._extract_landmarks(results, "right_hand", 21)
        
        # Concatenate to 75 landmarks
        points_75 = np.concatenate([pose_xy, left_hand_xy, right_hand_xy], axis=0)
        
        # Stage 5: ROI extraction
        roi_frame = self._extract_roi(frame, pose_xy)
        
        # Stage 6: RGB preprocessing
        try:
            transformed = self.rgb_transform(cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB))
            if not isinstance(transformed, torch.Tensor):
                raise TypeError("RGB transform did not return a torch.Tensor")
            rgb_tensor = transformed
        except Exception:
            # If ROI extraction failed, skip this frame
            self._frames_discarded += 1
            stats = ProcessingStats(
                motion_score=motion_score,
                frames_discarded=self._frames_discarded,
                buffer_fill=len(self.rgb_buffer),
                buffer_capacity=self.buffer_size
            )
            return ProcessedFrame(
                rgb_tensor=None,
                pose_tensor=None,
                display_frame=display_frame,
                kept=False,
                motion_score=motion_score,
                stats=stats
            )
        
        # Stage 7: Pose normalization
        normalized_pose = self._normalize_keypoints(points_75)
        pose_tensor = torch.from_numpy(normalized_pose)
        
        # Stage 8: Buffer management
        self.rgb_buffer.append(rgb_tensor)
        self.pose_buffer.append(pose_tensor)
        
        # Draw landmarks on display frame
        display_frame = self._draw_landmarks_on_frame(display_frame, results)
        
        # Update statistics
        self._frames_kept += 1
        processing_time = (time.time() - start_time) * 1000  # ms
        self._fps_tracker.append(processing_time)
        
        avg_processing_time = np.mean(self._fps_tracker) if self._fps_tracker else processing_time
        fps = 1000.0 / avg_processing_time if avg_processing_time > 0 else 0.0
        
        stats = ProcessingStats(
            fps=float(fps),
            motion_score=motion_score,
            frames_kept=self._frames_kept,
            frames_discarded=self._frames_discarded,
            buffer_fill=len(self.rgb_buffer),
            buffer_capacity=self.buffer_size,
            roi_detected=results.pose_landmarks is not None,
            pose_detected=results.pose_landmarks is not None,
            processing_time_ms=processing_time
        )
        
        return ProcessedFrame(
            rgb_tensor=rgb_tensor,
            pose_tensor=pose_tensor,
            display_frame=display_frame,
            kept=True,
            motion_score=motion_score,
            stats=stats
        )
    
    def is_buffer_ready(self) -> bool:
        """Check if buffer is full and ready for Module 2."""
        return len(self.rgb_buffer) >= self.buffer_size
    
    def get_buffer_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get stacked tensors from buffers for Module 2.
        
        Returns:
            rgb_tensor: (T, 3, 224, 224)
            pose_tensor: (T, 75, 2)
        """
        rgb_stacked = torch.stack(list(self.rgb_buffer))
        pose_stacked = torch.stack(list(self.pose_buffer))
        return rgb_stacked, pose_stacked
    
    def clear_buffers(self):
        """Clear buffers after processing."""
        self.rgb_buffer.clear()
        self.pose_buffer.clear()
    
    def reset(self):
        """Reset all state."""
        self._prev_gray = None
        self._frame_count = 0
        self._frames_kept = 0
        self._frames_discarded = 0
        self._recent_motion_scores.clear()
        self._fps_tracker.clear()
        self.rgb_buffer.clear()
        self.pose_buffer.clear()
    
    def close(self):
        """Clean up resources."""
        self.holistic.close()
