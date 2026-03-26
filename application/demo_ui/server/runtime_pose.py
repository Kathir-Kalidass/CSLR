from __future__ import annotations

from typing import Any, Optional, Tuple

from .deps import cv2, mp, np, torch


def extract_hand_landmarks(
    frame_bgr: Any,
    mp_hands: Any,
) -> list:
    """Return list of hand landmark sets using MediaPipe HandLandmarker (Tasks API).

    Each hand is a list of 21 [x, y] pairs in 0-1 normalised space.
    Returns up to 2 hands: [{"landmarks": [[x,y],...], "label": "Left"|"Right"}, ...].
    Returns empty list when no hands detected.
    """
    if mp_hands is None or mp is None or frame_bgr is None:
        return []
    try:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = mp_hands.detect(mp_image)
        if not result.hand_landmarks:
            return []
        hands = []
        for i, hand_lms in enumerate(result.hand_landmarks):
            pts = []
            for lm in hand_lms:
                pts.append([round(float(lm.x), 4), round(float(lm.y), 4)])
            label = "Unknown"
            if result.handedness and i < len(result.handedness):
                label = result.handedness[i][0].category_name
            hands.append({"landmarks": pts, "label": label})
        return hands
    except Exception:
        return []


def extract_pose_vector(
    frame_bgr: Any,
    pose_model: Any,
    mp_pose: Any,
    pose_input_dim: int,
) -> Tuple[Any, list]:
    """Return (normalised_vector, raw_landmarks).

    raw_landmarks is a list of [x, y] pairs normalised to 0-1 relative to
    frame dimensions (suitable for drawing in the browser).  Empty list when
    no person is detected.
    """
    points34 = np.zeros((34,), dtype=np.float32)
    arr: Optional[Any] = None
    raw_landmarks: list = []

    h, w = frame_bgr.shape[:2] if frame_bgr is not None else (1, 1)

    # Preferred path: YOLOv8-pose extractor (matches training setup).
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
                    # Raw landmarks in 0-1 coords
                    raw_landmarks = [[round(float(pt[0]) / max(w, 1), 4),
                                      round(float(pt[1]) / max(h, 1), 4)] for pt in padded]
        except Exception:
            arr = None

    # Fallback path: MediaPipe pose subset.
    if arr is None and mp_pose is not None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = mp_pose.process(frame_rgb)
        if results.pose_landmarks:
            landmark_ids = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
            coords = []
            for idx in landmark_ids:
                lm = results.pose_landmarks.landmark[idx]
                coords.append([float(lm.x), float(lm.y)])
            arr = np.array(coords, dtype=np.float32)
            # MediaPipe already in 0-1 space
            raw_landmarks = [[round(c[0], 4), round(c[1], 4)] for c in coords]

    if arr is None:
        return points34, raw_landmarks

    # Normalize using shoulder center and shoulder width for scale invariance.
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
        return padded, raw_landmarks
    return flat[:pose_input_dim], raw_landmarks
