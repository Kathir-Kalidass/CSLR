"""
Advanced Video Preprocessing (Production-Ready)
Optimized webcam preprocessing pipeline for low-memory GPUs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import torch
import torchvision.transforms as transforms

from app.core.logging import logger


@dataclass
class ProcessedFrame:
    """Processed frame data container"""
    display_frame: np.ndarray
    rgb_tensor: Optional[torch.Tensor]
    pose_tensor: Optional[torch.Tensor]
    kept: bool
    motion_score: float
    pose_landmarks: list[list[float]] = field(default_factory=list)
    hand_landmarks: list[dict] = field(default_factory=list)


class VideoPreprocessor:
    """
    Real-time webcam preprocessing optimized for low-memory GPUs.

    Pipeline: capture -> frame skip -> motion filter -> holistic -> ROI -> normalize
    Output tensors:
    - RGB: (3, 224, 224) normalized with ImageNet stats
    - Pose: (75, 2) normalized around shoulders (33 pose + 21*2 hands)
    """

    def __init__(
        self,
        frame_width: int = 640,
        frame_height: int = 480,
        capture_fps: int = 20,
        process_every_n_frame: int = 2,
        motion_threshold: float = 5.0,
        adaptive_motion: bool = True,
        draw_landmarks: bool = True,
    ) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.capture_fps = capture_fps
        self.process_every_n_frame = process_every_n_frame
        self.motion_threshold = motion_threshold
        self.adaptive_motion = adaptive_motion
        self.draw_landmarks = draw_landmarks

        self._prev_gray: Optional[np.ndarray] = None
        self._frame_count = 0
        self._recent_motion: list[float] = []

        # MediaPipe Holistic
        self.mp_holistic = None
        self.mp_drawing = None
        self.holistic = None
        if hasattr(mp, "solutions"):
            self.mp_holistic = mp.solutions.holistic  # type: ignore[attr-defined]
            self.mp_drawing = mp.solutions.drawing_utils  # type: ignore[attr-defined]
            self.holistic = self.mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                refine_face_landmarks=False,
            )
        else:
            logger.warning("MediaPipe solutions are unavailable; using frame-only preprocessing fallback")

        # ImageNet normalization
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _should_keep_frame(self, frame: np.ndarray) -> tuple[bool, float]:
        """Motion-based frame filtering with adaptive threshold"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            self._prev_gray = gray
            return True, 255.0

        diff = cv2.absdiff(self._prev_gray, gray)
        motion_score = float(np.mean(diff))
        self._prev_gray = gray

        threshold = self.motion_threshold
        if self.adaptive_motion and self._recent_motion:
            mean_motion = float(np.mean(self._recent_motion[-30:]))
            threshold = max(3.0, min(12.0, 0.6 * mean_motion + 2.0))

        self._recent_motion.append(motion_score)
        if len(self._recent_motion) > 200:
            self._recent_motion.pop(0)

        return motion_score > threshold, motion_score

    @staticmethod
    def _extract_xy(landmarks_obj, expected_len: int) -> np.ndarray:
        """Extract (x, y) coordinates from MediaPipe landmarks"""
        if landmarks_obj is None:
            return np.zeros((expected_len, 2), dtype=np.float32)
        points = np.array([[lm.x, lm.y] for lm in landmarks_obj.landmark], dtype=np.float32)
        if points.shape[0] != expected_len:
            padded = np.zeros((expected_len, 2), dtype=np.float32)
            n = min(expected_len, points.shape[0])
            padded[:n] = points[:n]
            return padded
        return points

    @staticmethod
    def _extract_hand_landmarks(results) -> list[dict]:
        hands: list[dict] = []
        if results is None:
            return hands

        if getattr(results, "left_hand_landmarks", None):
            hands.append(
                {
                    "label": "Left",
                    "landmarks": [[lm.x, lm.y] for lm in results.left_hand_landmarks.landmark],
                }
            )

        if getattr(results, "right_hand_landmarks", None):
            hands.append(
                {
                    "label": "Right",
                    "landmarks": [[lm.x, lm.y] for lm in results.right_hand_landmarks.landmark],
                }
            )

        return hands

    @staticmethod
    def _to_coco17_pose(results) -> list[list[float]]:
        if results is None or getattr(results, "pose_landmarks", None) is None:
            return []

        # COCO-17 order mapped from MediaPipe pose landmarks.
        mp_to_coco = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        src = results.pose_landmarks.landmark
        out: list[list[float]] = []
        for idx in mp_to_coco:
            if idx < len(src):
                lm = src[idx]
                out.append([float(lm.x), float(lm.y)])
            else:
                out.append([0.0, 0.0])
        return out

    @staticmethod
    def _normalize_keypoints(points75: np.ndarray) -> np.ndarray:
        """Normalize keypoints around shoulder center with torso scale"""
        left_shoulder = points75[11]
        right_shoulder = points75[12]
        center = (left_shoulder + right_shoulder) / 2.0
        torso = float(np.linalg.norm(left_shoulder - right_shoulder))
        if torso < 1e-4:
            torso = 0.2

        normed = (points75 - center) / torso
        normed = np.clip(normed, -1.5, 1.5)
        return normed.astype(np.float32)

    @staticmethod
    def _safe_roi(frame: np.ndarray, pose_xy: np.ndarray) -> np.ndarray:
        """Extract region of interest around upper body"""
        h, w, _ = frame.shape
        left_shoulder = pose_xy[11]
        right_shoulder = pose_xy[12]

        if np.all(left_shoulder == 0) or np.all(right_shoulder == 0):
            return frame

        x1, y1 = int(left_shoulder[0] * w), int(left_shoulder[1] * h)
        x2, y2 = int(right_shoulder[0] * w), int(right_shoulder[1] * h)

        center_x = (x1 + x2) // 2
        shoulder_width = max(40, abs(x2 - x1))

        roi_w = int(2.5 * shoulder_width)
        roi_h = int(2.8 * shoulder_width)
        y_center = int((y1 + y2) * 0.5 + 0.5 * shoulder_width)

        x_min = max(center_x - roi_w // 2, 0)
        x_max = min(center_x + roi_w // 2, w)
        y_min = max(y_center - roi_h // 2, 0)
        y_max = min(y_center + roi_h // 2, h)

        if x_max <= x_min or y_max <= y_min:
            return frame
        return frame[y_min:y_max, x_min:x_max]

    def process(self, frame: np.ndarray) -> ProcessedFrame:
        """
        Process a single frame

        Args:
            frame: BGR frame from webcam (H, W, 3)

        Returns:
            ProcessedFrame with RGB tensor, pose tensor, and metadata
        """
        self._frame_count += 1

        # Frame skipping
        if self._frame_count % self.process_every_n_frame != 0:
            return ProcessedFrame(
                display_frame=frame,
                rgb_tensor=None,
                pose_tensor=None,
                kept=False,
                motion_score=0.0,
            )

        # Motion filtering
        should_keep, motion_score = self._should_keep_frame(frame)
        if not should_keep:
            return ProcessedFrame(
                display_frame=frame,
                rgb_tensor=None,
                pose_tensor=None,
                kept=False,
                motion_score=motion_score,
            )

        # MediaPipe Holistic
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb_frame) if self.holistic is not None else None

        # Extract keypoints: 33 pose + 21 left hand + 21 right hand = 75
        pose_xy = self._extract_xy(results.pose_landmarks if results else None, 33)
        left_hand_xy = self._extract_xy(results.left_hand_landmarks if results else None, 21)
        right_hand_xy = self._extract_xy(results.right_hand_landmarks if results else None, 21)

        points75 = np.vstack([pose_xy, left_hand_xy, right_hand_xy])  # (75, 2)
        points75_normed = self._normalize_keypoints(points75)
        pose_landmarks = self._to_coco17_pose(results)
        hand_landmarks = self._extract_hand_landmarks(results)

        # ROI extraction
        roi_frame = self._safe_roi(frame, pose_xy)

        # RGB tensor
        transformed = self.transform(roi_frame)
        if isinstance(transformed, torch.Tensor):
            rgb_tensor = transformed
        else:
            rgb_tensor = torch.as_tensor(transformed)

        # Pose tensor
        pose_tensor = torch.from_numpy(points75_normed)  # (75, 2)

        # Draw landmarks on display frame
        if self.draw_landmarks and results is not None and results.pose_landmarks and self.mp_drawing and self.mp_holistic:
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS
            )
            if results.left_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS
                )
            if results.right_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS
                )

        return ProcessedFrame(
            display_frame=frame,
            rgb_tensor=rgb_tensor,
            pose_tensor=pose_tensor,
            kept=True,
            motion_score=motion_score,
            pose_landmarks=pose_landmarks,
            hand_landmarks=hand_landmarks,
        )

    def __del__(self):
        """Cleanup MediaPipe resources"""
        if hasattr(self, "holistic") and self.holistic is not None:
            self.holistic.close()
