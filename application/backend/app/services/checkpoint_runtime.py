from __future__ import annotations

import importlib.util
import json
from collections import deque
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch

from app.core.config import settings
from app.core.logging import logger
from app.utils.grammar_correction import GrammarCorrector

try:
    import mediapipe as mp
except Exception:  # pragma: no cover - optional runtime dependency
    mp = None


BACKEND_ROOT = Path(__file__).resolve().parents[2]
TRAIN_ISIGN_PATH = BACKEND_ROOT / "scripts" / "train_isign.py"


def _load_train_isign_symbols() -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("backend_train_isign", str(TRAIN_ISIGN_PATH))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load train_isign.py from {TRAIN_ISIGN_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {
        "ISignCSLRModel": getattr(module, "ISignCSLRModel"),
        "load_checkpoint": getattr(module, "load_checkpoint"),
        "streaming_predict_with_early_exit": getattr(module, "streaming_predict_with_early_exit"),
    }


def _resolve_backend_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return BACKEND_ROOT / path


def _load_vocab(config: dict[str, Any], vocab_size: int) -> list[str]:
    candidates: list[Path] = []
    for raw in (
        settings.ISIGN_VOCAB_FILE,
        str(config.get("vocab") or ""),
        "checkpoints/isl_cslrt_experiment/vocab_tokens.json",
        "checkpoints/isl_cslrt_v2_improved/vocab_tokens.json",
    ):
        if raw:
            candidates.append(_resolve_backend_path(raw))

    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, list) and payload:
            if payload[0] != "<blank>":
                return ["<blank>", *[str(item) for item in payload]][:vocab_size]
            return [str(item) for item in payload][:vocab_size]
        if isinstance(payload, dict):
            token_to_id = payload
            ordered = ["<blank>"] * vocab_size
            for token, idx in token_to_id.items():
                try:
                    j = int(idx)
                except Exception:
                    continue
                if 0 <= j < vocab_size:
                    ordered[j] = str(token)
            if any(tok != "<blank>" for tok in ordered[1:]):
                return ordered

    return ["<blank>", *[f"TOKEN_{idx}" for idx in range(1, vocab_size)]]


def _extract_pose_vector(
    frame_bgr: np.ndarray,
    pose_model: Any,
    mp_pose: Any,
    mp_hands: Any,
    pose_input_dim: int,
) -> tuple[np.ndarray, list[list[float]], list[dict[str, Any]]]:
    empty_pose = np.zeros((pose_input_dim,), dtype=np.float32)
    arr: Optional[np.ndarray] = None
    raw_landmarks: list[list[float]] = []
    hand_landmarks: list[dict[str, Any]] = []

    h, w = frame_bgr.shape[:2] if frame_bgr is not None else (1, 1)

    if pose_model is not None:
        try:
            results = pose_model.predict(source=frame_bgr, verbose=False, max_det=1)
            if results and getattr(results[0], "keypoints", None) is not None:
                keypoints_xy = results[0].keypoints.xy
                if keypoints_xy is not None and len(keypoints_xy) > 0:
                    points = keypoints_xy[0]
                    if torch.is_tensor(points):
                        points = points.detach().cpu().numpy()
                    points = np.asarray(points, dtype=np.float32)
                    padded = np.zeros((17, 2), dtype=np.float32)
                    valid = min(17, points.shape[0])
                    padded[:valid] = points[:valid]
                    arr = padded
                    raw_landmarks = [
                        [round(float(pt[0]) / max(w, 1), 4), round(float(pt[1]) / max(h, 1), 4)]
                        for pt in padded
                    ]
        except Exception:
            arr = None

    if arr is None and mp_pose is not None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = mp_pose.process(frame_rgb)
        if results.pose_landmarks:
            lms = results.pose_landmarks.landmark
            # OpenPose BODY_18 style ordering for the model:
            # nose, neck, R shoulder/elbow/wrist, L shoulder/elbow/wrist,
            # R hip/knee/ankle, L hip/knee/ankle, R eye, L eye, R ear, L ear
            right_shoulder = np.array([float(lms[12].x), float(lms[12].y)], dtype=np.float32)
            left_shoulder = np.array([float(lms[11].x), float(lms[11].y)], dtype=np.float32)
            neck = ((left_shoulder + right_shoulder) / 2.0).astype(np.float32)
            coords18 = np.array(
                [
                    [float(lms[0].x), float(lms[0].y)],
                    neck.tolist(),
                    [float(lms[12].x), float(lms[12].y)],
                    [float(lms[14].x), float(lms[14].y)],
                    [float(lms[16].x), float(lms[16].y)],
                    [float(lms[11].x), float(lms[11].y)],
                    [float(lms[13].x), float(lms[13].y)],
                    [float(lms[15].x), float(lms[15].y)],
                    [float(lms[24].x), float(lms[24].y)],
                    [float(lms[26].x), float(lms[26].y)],
                    [float(lms[28].x), float(lms[28].y)],
                    [float(lms[23].x), float(lms[23].y)],
                    [float(lms[25].x), float(lms[25].y)],
                    [float(lms[27].x), float(lms[27].y)],
                    [float(lms[5].x), float(lms[5].y)],
                    [float(lms[2].x), float(lms[2].y)],
                    [float(lms[8].x), float(lms[8].y)],
                    [float(lms[7].x), float(lms[7].y)],
                ],
                dtype=np.float32,
            )
            arr = coords18
            # UI uses COCO-17 style display ordering.
            display_ids = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
            raw_landmarks = [
                [round(float(lms[idx].x), 4), round(float(lms[idx].y), 4)]
                for idx in display_ids
            ]

        if mp_hands is not None:
            hand_results = mp_hands.process(frame_rgb)
            multi_hands = hand_results.multi_hand_landmarks or []
            handedness = hand_results.multi_handedness or []
            for idx, hand_lms in enumerate(multi_hands):
                label = "Unknown"
                if idx < len(handedness):
                    try:
                        label = str(handedness[idx].classification[0].label)
                    except Exception:
                        label = "Unknown"
                hand_landmarks.append(
                    {
                        "label": label,
                        "landmarks": [
                            [round(float(lm.x), 4), round(float(lm.y), 4)]
                            for lm in hand_lms.landmark
                        ],
                    }
                )

    if arr is None:
        return empty_pose, raw_landmarks, hand_landmarks

    if arr.shape[0] == 18:
        left_shoulder = arr[5]
        right_shoulder = arr[2]
    else:
        left_shoulder = arr[5]
        right_shoulder = arr[6]
    center = (left_shoulder + right_shoulder) / 2.0
    width = float(np.linalg.norm(left_shoulder - right_shoulder))
    if width < 1e-4:
        width = 0.2
    arr = np.clip((arr - center) / width, -1.5, 1.5)

    flat = arr.reshape(-1).astype(np.float32)
    if flat.shape[0] < pose_input_dim:
        padded = np.zeros((pose_input_dim,), dtype=np.float32)
        padded[: flat.shape[0]] = flat
        return padded, raw_landmarks, hand_landmarks
    return flat[:pose_input_dim], raw_landmarks, hand_landmarks


class CheckpointRuntime:
    def __init__(self) -> None:
        self.device = torch.device(settings.DEVICE)
        self.available = False
        self.error_detail = ""
        self.reason = "not_initialized"
        self.window_size = settings.CLIP_LENGTH
        self.stride = max(1, self.window_size // 2)
        self.pose_input_dim = 34
        self.pose_frame_dim = 34
        self.use_rgb = False
        self.vocab: list[str] = []
        self.grammar = GrammarCorrector()
        self._pose_model = None
        self._mp_pose = None
        self._mp_hands = None
        self.model = None
        self._stream_state_template: dict[str, Any] = {}
        self._initialize()

    def _initialize(self) -> None:
        ckpt_path = _resolve_backend_path(settings.ISIGN_CHECKPOINT_PATH)
        cfg_path = _resolve_backend_path(settings.ISIGN_CHECKPOINT_CONFIG)

        if not ckpt_path.exists():
            self.reason = "checkpoint_missing"
            self.error_detail = str(ckpt_path)
            logger.error("Checkpoint not found: %s", ckpt_path)
            return
        if not cfg_path.exists():
            self.reason = "config_missing"
            self.error_detail = str(cfg_path)
            logger.error("Checkpoint config not found: %s", cfg_path)
            return

        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            symbols = _load_train_isign_symbols()
            checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            model_state = checkpoint.get("model", {})
            if not isinstance(model_state, dict) or not model_state:
                raise ValueError("Checkpoint missing 'model' state")

            self.pose_input_dim = int(model_state["pose_enc.net.0.weight"].shape[1])
            hidden_dim = int(model_state["head.weight"].shape[1])
            vocab_size = int(model_state["head.weight"].shape[0])
            self.use_rgb = not bool(cfg.get("pose_only", False))
            self.window_size = int(cfg.get("num_frames", self.window_size))
            self.stride = int(cfg.get("stream_stride", max(1, self.window_size // 2)))
            self.pose_frame_dim = (
                self.pose_input_dim // self.window_size
                if self.pose_input_dim > 36 and self.pose_input_dim % self.window_size == 0
                else self.pose_input_dim
            )
            self.vocab = _load_vocab(cfg, vocab_size)

            model = symbols["ISignCSLRModel"](
                vocab_size=vocab_size,
                hidden_dim=hidden_dim,
                pose_input_dim=self.pose_input_dim,
                dropout=float(cfg.get("dropout", 0.3)),
                pretrained_cnn=False,
                pose_fusion_weight=float(cfg.get("pose_fusion_weight", 0.7)),
                attention_heads=int(cfg.get("attention_heads", 4)),
                freeze_rgb_stages=int(cfg.get("freeze_rgb_stages", 4)),
                use_rgb=self.use_rgb,
            ).to(self.device)
            symbols["load_checkpoint"](ckpt_path, model)
            model.eval()

            if mp is not None and getattr(mp, "solutions", None) is not None:
                self._mp_pose = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=2,
                    min_detection_confidence=0.2,
                    min_tracking_confidence=0.2,
                )
                self._mp_hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.2,
                    min_tracking_confidence=0.2,
                )

            self.model = model
            self.decode_fn = symbols["streaming_predict_with_early_exit"]
            self.available = True
            self.reason = "ready"
            logger.info(
                "Checkpoint runtime ready | checkpoint=%s | pose_only=%s | window=%d | stride=%d | pose_input_dim=%d | pose_frame_dim=%d",
                ckpt_path,
                not self.use_rgb,
                self.window_size,
                self.stride,
                self.pose_input_dim,
                self.pose_frame_dim,
            )
        except Exception as exc:
            self.reason = f"init_error:{type(exc).__name__}"
            self.error_detail = str(exc)
            logger.exception("Failed to initialize checkpoint runtime")

    def make_stream_state(self) -> dict[str, Any]:
        return {
            "pose_buffer": deque(maxlen=self.window_size),
            "base_pose_buffer": deque(maxlen=self.window_size),
            "rgb_buffer": deque(maxlen=self.window_size),
            "last_gloss": [],
            "last_sentence": "",
            "last_confidence": 0.0,
            "last_pose_landmarks": [],
            "last_hand_landmarks": [],
            "frames_since_emit": 0,
        }

    def _adapt_pose_features(self, base_pose_items: list[np.ndarray]) -> list[np.ndarray]:
        if not base_pose_items:
            return []

        if self.pose_input_dim <= self.pose_frame_dim:
            return [item[: self.pose_input_dim].astype(np.float32, copy=False) for item in base_pose_items]

        zero_frame = np.zeros((self.pose_frame_dim,), dtype=np.float32)
        adapted: list[np.ndarray] = []
        for idx in range(len(base_pose_items)):
            start = max(0, idx - self.window_size + 1)
            context = list(base_pose_items[start : idx + 1])
            if len(context) < self.window_size:
                context = [zero_frame] * (self.window_size - len(context)) + context
            adapted.append(np.stack(context[-self.window_size :], axis=0).reshape(-1).astype(np.float32, copy=False))
        return adapted

    def _predict_from_buffers(self, pose_items: list[np.ndarray], rgb_items: list[torch.Tensor] | None = None) -> dict[str, Any]:
        if not self.available or self.model is None:
            return {"gloss": [], "sentence": "", "confidence": 0.0}

        pose = torch.from_numpy(np.stack(pose_items)).float()
        rgb = None
        if self.use_rgb and rgb_items:
            rgb = torch.stack(rgb_items).permute(1, 0, 2, 3).float()

        result = self.decode_fn(
            self.model,
            rgb=rgb,
            pose=pose,
            blank_id=0,
            stride=1,
            confidence_threshold=float(settings.CONFIDENCE_THRESHOLD),
            vocab=self.vocab,
        )
        gloss_tokens = [str(token) for token in result.get("prediction", []) if str(token).strip()]
        if not gloss_tokens:
            return {
                "gloss": [],
                "sentence": "",
                "confidence": 0.0,
                "window_start": int(result.get("window_start", 0)),
                "window_end": int(result.get("window_end", max(0, len(pose_items) - 1))),
                "runtime_status": "blank_decode",
            }

        sentence = self.grammar.gloss_to_sentence(gloss_tokens)
        return {
            "gloss": gloss_tokens,
            "sentence": sentence,
            "confidence": float(result.get("confidence", 0.0)),
            "window_start": int(result.get("window_start", 0)),
            "window_end": int(result.get("window_end", max(0, len(pose_items) - 1))),
            "runtime_status": "ready",
        }

    def process_frames(self, frames: list[np.ndarray]) -> dict[str, Any]:
        base_pose_items: list[np.ndarray] = []
        pose_landmarks: list[list[float]] = []
        hand_landmarks: list[dict[str, Any]] = []
        for frame in frames:
            pose_vec, raw_landmarks, raw_hands = _extract_pose_vector(
                frame,
                self._pose_model,
                self._mp_pose,
                self._mp_hands,
                self.pose_frame_dim,
            )
            base_pose_items.append(pose_vec)
            if raw_landmarks:
                pose_landmarks = raw_landmarks
            if raw_hands:
                hand_landmarks = raw_hands

        if not base_pose_items:
            return {"gloss": [], "sentence": "", "confidence": 0.0, "pose_landmarks": [], "hand_landmarks": []}

        pose_items = self._adapt_pose_features(base_pose_items)
        result = self._predict_from_buffers(pose_items)
        result["pose_landmarks"] = pose_landmarks
        result["hand_landmarks"] = hand_landmarks
        if not pose_landmarks:
            result["runtime_status"] = "no_pose_detected"
        return result

    def process_stream_frame(self, frame: np.ndarray, state: Optional[dict[str, Any]]) -> dict[str, Any]:
        runtime_state = state or self.make_stream_state()
        pose_vec, raw_landmarks, raw_hands = _extract_pose_vector(
            frame,
            self._pose_model,
            self._mp_pose,
            self._mp_hands,
            self.pose_frame_dim,
        )
        runtime_state["base_pose_buffer"].append(pose_vec)
        adapted_pose = self._adapt_pose_features(list(runtime_state["base_pose_buffer"]))[-1]
        runtime_state["pose_buffer"].append(adapted_pose)
        runtime_state["last_pose_landmarks"] = raw_landmarks
        runtime_state["last_hand_landmarks"] = raw_hands
        runtime_state["frames_since_emit"] += 1

        if len(runtime_state["pose_buffer"]) < self.window_size:
            return {
                "gloss": runtime_state["last_gloss"],
                "sentence": runtime_state["last_sentence"],
                "confidence": runtime_state["last_confidence"],
                "pose_landmarks": runtime_state["last_pose_landmarks"],
                "hand_landmarks": runtime_state["last_hand_landmarks"],
                "partial": True,
                "buffer_fill": len(runtime_state["pose_buffer"]),
                "state": runtime_state,
                "runtime_status": "no_pose_detected" if not runtime_state["last_pose_landmarks"] else "buffering",
            }

        if runtime_state["frames_since_emit"] < self.stride:
            return {
                "gloss": runtime_state["last_gloss"],
                "sentence": runtime_state["last_sentence"],
                "confidence": runtime_state["last_confidence"],
                "pose_landmarks": runtime_state["last_pose_landmarks"],
                "hand_landmarks": runtime_state["last_hand_landmarks"],
                "partial": False,
                "buffer_fill": len(runtime_state["pose_buffer"]),
                "state": runtime_state,
                "runtime_status": "no_pose_detected" if not runtime_state["last_pose_landmarks"] else "waiting_stride",
            }

        runtime_state["frames_since_emit"] = 0
        pose_items = list(runtime_state["pose_buffer"])
        result = self._predict_from_buffers(pose_items)
        runtime_state["last_gloss"] = result["gloss"]
        runtime_state["last_sentence"] = result["sentence"]
        runtime_state["last_confidence"] = result["confidence"]

        result.update(
            {
                "pose_landmarks": runtime_state["last_pose_landmarks"],
                "hand_landmarks": runtime_state["last_hand_landmarks"],
                "partial": False,
                "buffer_fill": len(runtime_state["pose_buffer"]),
                "state": runtime_state,
                "runtime_status": "no_pose_detected" if not runtime_state["last_pose_landmarks"] else result.get("runtime_status", "ready"),
            }
        )
        return result
