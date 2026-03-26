from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any

from .config import BEST_MODEL_PATH
from .deps import cv2, np, torch
from .runtime_helpers import ctc_decode, gloss_to_sentence
from .runtime_loader import initialize_runtime
from .runtime_pose import extract_hand_landmarks, extract_pose_vector


class RealtimeModelRuntime:
    """
    Browser-frame real-time inference runtime backed by best.pt checkpoint.

    This runtime is optional. If initialization fails, the UI falls back to
    simulation mode and still remains fully functional.
    """

    def __init__(self, checkpoint_path: Path) -> None:
        self.available = False
        self.reason = "not_initialized"
        self.error_detail = ""
        self.device = "cpu"
        self.window_size = 64
        self.stride = 32
        self.frames_since_emit = 0
        self.pose_input_dim = 34
        self.id_to_token: dict[int, str] = {}
        self.last_timestamps: deque[float] = deque(maxlen=30)

        self.rgb_buffer: deque[Any] = deque(maxlen=self.window_size)
        self.pose_buffer: deque[Any] = deque(maxlen=self.window_size)

        self._pose_model = None
        self._mp_pose = None
        self._mp_hands = None
        self._rgb_transform = None
        self._latest_landmarks: list = []
        self._latest_hand_landmarks: list = []

        self.rgb_stream = None
        self.pose_stream = None
        self.fusion = None
        self.temporal = None

        self.available, self.reason, self.error_detail = initialize_runtime(self, checkpoint_path)

    def reset(self) -> None:
        self.rgb_buffer.clear()
        self.pose_buffer.clear()
        self.frames_since_emit = 0

    def process_frame(self, frame_bgr: Any) -> dict[str, Any]:
        if not self.available:
            return {"ready": False, "reason": self.reason}
        if self._rgb_transform is None:
            return {"ready": False, "reason": "transform_unavailable"}

        start = time.time()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb_tensor = self._rgb_transform(frame_rgb)
        pose_vec, raw_landmarks = extract_pose_vector(
            frame_bgr=frame_bgr,
            pose_model=self._pose_model,
            mp_pose=self._mp_pose,
            pose_input_dim=self.pose_input_dim,
        )
        pose_tensor = torch.from_numpy(pose_vec).float()
        self._latest_landmarks = raw_landmarks

        # Extract hand/finger landmarks (for visualisation, not model input)
        self._latest_hand_landmarks = extract_hand_landmarks(
            frame_bgr=frame_bgr,
            mp_hands=self._mp_hands,
        )

        self.rgb_buffer.append(rgb_tensor)
        self.pose_buffer.append(pose_tensor)
        self.frames_since_emit += 1

        self.last_timestamps.append(time.time())
        fps = 0.0
        if len(self.last_timestamps) >= 2:
            dt = (self.last_timestamps[-1] - self.last_timestamps[0]) / max(1, (len(self.last_timestamps) - 1))
            if dt > 1e-6:
                fps = 1.0 / dt

        if len(self.rgb_buffer) < self.window_size or self.frames_since_emit < self.stride:
            return {
                "ready": False,
                "fps": round(fps, 2),
                "buffer_fill": len(self.rgb_buffer),
                "latency_ms": round((time.time() - start) * 1000, 2),
                "pose_landmarks": self._latest_landmarks,
                "hand_landmarks": self._latest_hand_landmarks,
            }
        self.frames_since_emit = 0

        rgb_window = torch.stack(list(self.rgb_buffer), dim=0).unsqueeze(0).to(self.device)
        pose_window = torch.stack(list(self.pose_buffer), dim=0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            rgb_feat = self.rgb_stream(rgb_window)
            pose_feat = self.pose_stream(pose_window)
            fused_feat, alpha, beta = self.fusion(rgb_feat, pose_feat)
            logits = self.temporal(fused_feat)
            gloss_tokens, confidence = ctc_decode(logits.squeeze(0), id_to_token=self.id_to_token)

        sentence = gloss_to_sentence(gloss_tokens)
        return {
            "ready": True,
            "gloss_tokens": gloss_tokens,
            "gloss_text": " ".join(gloss_tokens) if gloss_tokens else "--",
            "sentence": sentence if sentence else "Processing sign language...",
            "confidence": round(float(confidence), 3),
            "attn_rgb": round(float(alpha.mean().item()), 3) if alpha is not None else 0.5,
            "attn_pose": round(float(beta.mean().item()), 3) if beta is not None else 0.5,
            "fps": round(float(fps), 2),
            "buffer_fill": len(self.rgb_buffer),
            "latency_ms": round((time.time() - start) * 1000, 2),
            "pose_landmarks": self._latest_landmarks,
            "hand_landmarks": self._latest_hand_landmarks,
        }


REAL_RUNTIME = RealtimeModelRuntime(BEST_MODEL_PATH)
