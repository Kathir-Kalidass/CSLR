from __future__ import annotations

from typing import Any

from ..runtime import REAL_RUNTIME


def apply_real_inference_override(payload: dict[str, Any], client_state: dict[str, Any]) -> dict[str, Any]:
    if not REAL_RUNTIME.available:
        return payload

    latest_inference = client_state.get("latest_inference", {})
    if not isinstance(latest_inference, dict):
        return payload

    # Always pass pose landmarks for live drawing
    payload["pose_landmarks"] = latest_inference.get("pose_landmarks", [])
    payload["hand_landmarks"] = latest_inference.get("hand_landmarks", [])

    payload["parser_console"].append("[runtime] best.pt real inference path active")
    if latest_inference.get("ready", False):
        payload["partial_gloss"] = str(latest_inference.get("gloss_text", payload["partial_gloss"]))
        payload["final_sentence"] = str(latest_inference.get("sentence", payload["final_sentence"]))
        payload["confidence"] = float(latest_inference.get("confidence", payload["confidence"]))
        payload["fps"] = int(round(float(latest_inference.get("fps", payload["fps"]))))
        payload["latency_ms"] = int(round(float(latest_inference.get("latency_ms", payload["latency_ms"]))))
        payload["attention"] = {
            "rgb": float(latest_inference.get("attn_rgb", payload["attention"]["rgb"])),
            "pose": float(latest_inference.get("attn_pose", payload["attention"]["pose"])),
        }
        payload["module3"]["output"] = f"partial_gloss={payload['partial_gloss']}"
        payload["module5"]["output"] = payload["final_sentence"]
        payload["module6"]["input"] = payload["final_sentence"]
        payload["module6"]["output"] = (
            "audio=playing" if client_state.get("tts_enabled", True) else "audio=muted"
        )
        payload["module1"]["note"] = (
            f"Real-frame inference. buffer_fill={latest_inference.get('buffer_fill', 0)}/{REAL_RUNTIME.window_size}"
        )
    else:
        buffer_fill = int(latest_inference.get("buffer_fill", 0))
        ingest_fps = latest_inference.get("fps", 0)
        payload["module1"]["note"] = f"Collecting frames {buffer_fill}/{REAL_RUNTIME.window_size}"
        payload["fps"] = int(round(float(ingest_fps))) if ingest_fps else payload.get("fps", 0)
        payload["latency_ms"] = int(round(float(latest_inference.get("latency_ms", 0))))
        payload["parser_console"].append(
            f"[runtime] buffer_fill={buffer_fill}/{REAL_RUNTIME.window_size} fps={ingest_fps}"
        )

    return payload
