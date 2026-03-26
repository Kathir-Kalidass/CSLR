from __future__ import annotations

import base64
from typing import Any

from ..deps import cv2, np
from ..runtime import REAL_RUNTIME


def _decode_and_process(frame_b64: str, client_state: dict[str, Any], seq: int) -> None:
    """Decode one base64 JPEG and feed it to the runtime."""
    try:
        frame_bytes = base64.b64decode(frame_b64)
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        if frame_bgr is not None:
            result = REAL_RUNTIME.process_frame(frame_bgr)
            client_state["latest_inference"] = result
            client_state["processed_frame_seq"] = seq
    except Exception:
        client_state["latest_inference"] = {"ready": False, "reason": "frame_decode_error"}


def maybe_process_latest_client_frame(client_state: dict[str, Any]) -> None:
    if not (REAL_RUNTIME.available and cv2 is not None and np is not None):
        return

    if not bool(client_state.get("running", False)):
        return

    # Process ALL queued frames so the 64-frame buffer fills at real camera FPS
    frame_queue = client_state.get("frame_queue")
    if frame_queue and len(frame_queue) > 0:
        while len(frame_queue) > 0:
            b64, seq = frame_queue.popleft()
            _decode_and_process(b64, client_state, seq)
        return

    # Fallback: single-frame path (backwards-compatibility)
    frame_b64 = client_state.get("latest_frame_b64")
    frame_seq = int(client_state.get("latest_frame_seq", -1))
    processed_seq = int(client_state.get("processed_frame_seq", -1))
    if not frame_b64 or frame_seq <= processed_seq:
        return
    _decode_and_process(frame_b64, client_state, frame_seq)
