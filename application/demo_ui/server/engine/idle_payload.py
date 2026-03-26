from __future__ import annotations

import time
from typing import Any

from ..config import best_model_info
from ..runtime import REAL_RUNTIME
from ..tts_service import tts_engine_status


def build_idle_payload(
    client_state: dict[str, Any],
    init_sequence: list[str],
    pipeline_order: list[str],
) -> dict[str, Any]:
    return {
        "timestamp": time.time(),
        "tick": client_state.get("tick", 0),
        "status": "idle",
        "inference_mode": "real" if REAL_RUNTIME.available else "simulated",
        "runtime_status": REAL_RUNTIME.reason,
        "init_sequence": init_sequence,
        "latency_ms": 0,
        "fps": 0,
        "confidence": 0.0,
        "audio_state": "disabled" if not client_state.get("tts_enabled", True) else "idle",
        "partial_gloss": "--",
        "final_sentence": "Open camera and start live recognition.",
        "active_stage": "module1",
        "pipeline_order": pipeline_order,
        "model_info": best_model_info(),
        "tts_engine": tts_engine_status(),
        "attention": {"rgb": 0.5, "pose": 0.5},
        "timeline": {
            "active_window": "frames[0:63]",
            "windows": ["frames[0:63]", "frames[32:95]", "frames[64:127]"],
        },
        "audio_wave": [0.0] * 24,
        "metrics": {
            "wer_proxy": 0.0,
            "bleu_proxy": 0.0,
            "window_frames": 64,
            "stride": 32,
            "accuracy_proxy": 0.0,
        },
        "transcript_history": client_state.get("transcript_history", []),
        "control_state": {
            "running": False,
            "tts_enabled": client_state.get("tts_enabled", True),
            "camera_active": bool(client_state.get("camera_active", False)),
            "grand_mode": bool(client_state.get("grand_mode", False)),
        },
        "module1": {
            "title": "Capture and Frame Window",
            "input": "Webcam stream",
            "process": "Capture -> sample -> frame selection -> normalize + pose extraction",
            "output": "RGB tensor + Pose tensor",
            "note": "Flow paused.",
            "parse": ["waiting_for_start=true"],
        },
        "module2": {
            "title": "RGB and Pose Feature Extraction",
            "input": "RGB + Pose tensors",
            "process": "ResNet18 RGB stream + MLP pose stream",
            "output": "Feature vectors",
            "note": "No active window.",
            "parse": ["window_ready=false"],
        },
        "module3": {
            "title": "Attention Fusion and Temporal Decode",
            "input": "RGB and pose features",
            "process": "Attention fusion + BiLSTM + CTC",
            "output": "Gloss tokens",
            "note": "Waiting for buffered frames.",
            "parse": ["stride=32, window_size=64"],
        },
        "module4": {
            "title": "Token Cleanup",
            "input": "Raw gloss sequence",
            "process": "Deduplicate + confidence filtering",
            "output": "Clean gloss string",
            "note": "No active prediction.",
            "parse": ["dedup_active=true"],
        },
        "module5": {
            "title": "Sentence Construction",
            "input": "Clean gloss",
            "process": "Rule-based grammar correction",
            "output": "Readable English sentence",
            "note": "Waiting for gloss input.",
            "parse": ["grammar_mode=rule_based"],
        },
        "module6": {
            "title": "AI Voice Generation",
            "input": "Corrected sentence",
            "process": "Queue and play speech",
            "output": "Audio output",
            "note": "TTS disabled" if not client_state.get("tts_enabled", True) else "Ready",
            "parse": [f"tts_enabled={client_state.get('tts_enabled', True)}"],
        },
        "module7": {
            "title": "Live Insights",
            "input": "Prediction stream",
            "process": "Compute FPS, latency, WER, BLEU, accuracy",
            "output": "Dashboard metrics",
            "note": "Metrics reset while idle.",
            "parse": ["stream=live_camera"],
        },
        "parser_console": ["[system] idle: waiting for Start"],
    }
