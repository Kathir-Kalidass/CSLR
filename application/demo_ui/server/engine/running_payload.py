from __future__ import annotations

import math
import time
from typing import Any

from ..config import best_model_info
from ..runtime import REAL_RUNTIME
from ..tts_service import tts_engine_status
from .overrides import apply_real_inference_override
from .running_modules import build_running_module_data


def _compute_active_stage(
    pipeline_order: list[str],
    inference_ready: bool,
    camera_active: bool,
    latest_inference: dict,
) -> str:
    """Return the pipeline stage that is *actually* active right now."""
    if not camera_active:
        return pipeline_order[0]  # capture

    buffer_fill = int(latest_inference.get("buffer_fill", 0)) if isinstance(latest_inference, dict) else 0

    if not inference_ready:
        # Still collecting frames into the window
        if buffer_fill < 64:
            return pipeline_order[0]  # module1 – capture
        return pipeline_order[1]  # module2 – features (about to emit)

    # Inference produced a result — show the later stages
    confidence = float(latest_inference.get("confidence", 0.0)) if isinstance(latest_inference, dict) else 0.0
    gloss = str(latest_inference.get("gloss_text", "--")) if isinstance(latest_inference, dict) else "--"

    if gloss == "--" or confidence < 0.01:
        return pipeline_order[2]  # module3 – decode (low confidence)
    if confidence < 0.25:
        return pipeline_order[3]  # module4 – cleanup
    return pipeline_order[4]  # module5 – sentence construction


def build_running_payload(
    tick: int,
    client_state: dict[str, Any],
    init_sequence: list[str],
    pipeline_order: list[str],
) -> dict[str, Any]:
    camera_active = bool(client_state.get("camera_active", False))
    frame_hint = int(client_state.get("frame_hint", 0))
    client_resolution = str(client_state.get("resolution", "unknown"))
    latest_inference = client_state.get("latest_inference", {})
    inference_ready = isinstance(latest_inference, dict) and bool(latest_inference.get("ready", False))

    fps = int(round(float(latest_inference.get("fps", 0.0)))) if inference_ready else 0
    frame_count = 64
    confidence = round(float(latest_inference.get("confidence", 0.0)), 3) if inference_ready else 0.0
    latency_ms = int(round(float(latest_inference.get("latency_ms", 0.0)))) if inference_ready else 0
    partial = str(latest_inference.get("gloss_text", "--")) if inference_ready else "--"
    sentence = str(latest_inference.get("sentence", "Analyzing live sign window...")) if inference_ready else (
        "Analyzing live sign window..."
    )

    window_start = tick * 32
    window_end = window_start + frame_count - 1
    pose_detected = 75 if camera_active else 0
    hand_visibility = round(float(latest_inference.get("confidence", 0.0)), 2) if inference_ready else 0.0
    attn_rgb = round(float(latest_inference.get("attn_rgb", 0.5)), 3) if inference_ready else 0.5
    attn_pose = round(float(latest_inference.get("attn_pose", 0.5)), 3) if inference_ready else 0.5

    history = client_state.setdefault("transcript_history", [])
    if inference_ready and sentence and sentence != "Processing sign language..." and (not history or history[0] != sentence):
        history.insert(0, sentence)
    client_state["transcript_history"] = history[:12]

    module_data = build_running_module_data(
        camera_active=camera_active,
        frame_hint=frame_hint,
        client_resolution=client_resolution,
        window_start=window_start,
        window_end=window_end,
        pose_detected=pose_detected,
        hand_visibility=hand_visibility,
        attn_rgb=attn_rgb,
        attn_pose=attn_pose,
        partial=partial,
        sentence=sentence,
        confidence=confidence,
        tts_enabled=bool(client_state.get("tts_enabled", True)),
        inference_ready=inference_ready,
        fps=fps,
        latency_ms=latency_ms,
    )

    stage_labels = {
        "module1": "capture",
        "module2": "features",
        "module3": "decode",
        "module4": "cleanup",
        "module5": "sentence",
        "module6": "voice",
        "module7": "insights",
    }
    parser_console = []
    for module_key in pipeline_order:
        label = stage_labels.get(module_key, module_key)
        parser_console.extend([f"[{label}] {line}" for line in module_data[module_key]["parse"]])

    win0 = f"frames[{window_start}:{window_end}]"
    win1 = f"frames[{window_start + 32}:{window_end + 32}]"
    win2 = f"frames[{window_start + 64}:{window_end + 64}]"
    tts_enabled = bool(client_state.get("tts_enabled", True))
    wave_base = max(0.08, min(1.0, confidence if confidence > 0 else 0.2))
    audio_wave = [
        round((wave_base * (0.35 + 0.65 * abs(math.sin((tick + idx) / 2.8)))) if tts_enabled else 0.0, 3)
        for idx in range(24)
    ]

    payload = {
        "timestamp": time.time(),
        "tick": tick,
        "status": "active",
        "inference_mode": "real" if REAL_RUNTIME.available else "simulated",
        "runtime_status": REAL_RUNTIME.reason,
        "init_sequence": init_sequence,
        "latency_ms": latency_ms,
        "fps": fps,
        "confidence": confidence,
        "audio_state": "speaking" if (tts_enabled and inference_ready and sentence.strip() not in {"", "--"}) else "muted",
        "partial_gloss": partial,
        "final_sentence": sentence,
        "active_stage": _compute_active_stage(pipeline_order, inference_ready, camera_active, latest_inference),
        "pipeline_order": pipeline_order,
        "model_info": best_model_info(),
        "tts_engine": tts_engine_status(),
        "attention": {"rgb": attn_rgb, "pose": attn_pose},
        "timeline": {
            "active_window": win0,
            "windows": [win0, win1, win2],
        },
        "audio_wave": audio_wave,
        "metrics": {
            "wer_proxy": round(max(0.0, 1.0 - confidence), 3),
            "bleu_proxy": round(confidence, 3),
            "window_frames": frame_count,
            "stride": 32,
            "accuracy_proxy": round(confidence, 3),
        },
        "transcript_history": client_state.get("transcript_history", []),
        "control_state": {
            "running": True,
            "tts_enabled": client_state.get("tts_enabled", True),
            "camera_active": camera_active,
            "grand_mode": bool(client_state.get("grand_mode", False)),
        },
        "module1": module_data["module1"],
        "module2": module_data["module2"],
        "module3": module_data["module3"],
        "module4": module_data["module4"],
        "module5": module_data["module5"],
        "module6": module_data["module6"],
        "module7": module_data["module7"],
        "parser_console": parser_console,
    }

    return apply_real_inference_override(payload, client_state)
