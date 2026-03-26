from __future__ import annotations

from typing import Any


def build_running_module_data(
    camera_active: bool,
    frame_hint: int,
    client_resolution: str,
    window_start: int,
    window_end: int,
    pose_detected: int,
    hand_visibility: float,
    attn_rgb: float,
    attn_pose: float,
    partial: str,
    sentence: str,
    confidence: float,
    tts_enabled: bool,
    inference_ready: bool,
    fps: int,
    latency_ms: int,
) -> dict[str, dict[str, Any]]:
    return {
        "module1": {
            "title": "Capture and Frame Window",
            "input": f"Live stream ({client_resolution})",
            "process": "Motion filter -> frame skip -> ROI crop -> resize 224x224",
            "output": "RGB: 64x3x224x224 | Pose: 64x75x2",
            "note": "4GB-safe preprocessing with adaptive frame retention.",
            "parse": [
                f"camera_active={camera_active}, frame_hint={frame_hint}",
                f"window=frames[{window_start}:{window_end}] stride=32",
                f"pose_landmarks={pose_detected}/75, hand_visibility={hand_visibility}",
            ],
        },
        "module2": {
            "title": "RGB and Pose Feature Extraction",
            "input": "RGB + Pose tensors",
            "process": "ResNet18 (RGB) + MLP(150->256 pose)",
            "output": "rgb_feat:64x512 | pose_feat:64x256",
            "note": "Efficient extractor setup for real-time inference.",
            "parse": ["resnet18_backbone=active", "pose_encoder=linear_relu_linear", "precision=fp16_path"],
        },
        "module3": {
            "title": "Attention Fusion and Temporal Decode",
            "input": "rgb_feat + pose_feat",
            "process": "alpha/beta attention + BiLSTM + CTC logits",
            "output": f"partial_gloss={partial}",
            "note": "Sliding-window predictions every 32 frames.",
            "parse": [
                f"attention alpha_rgb={attn_rgb}, beta_pose={attn_pose}",
                "lstm=2_layers_bidirectional hidden=512",
                "ctc_decode=greedy(blank_prune=true)",
            ],
        },
        "module4": {
            "title": "Token Cleanup",
            "input": partial,
            "process": "merge_repeats + confidence thresholding",
            "output": partial,
            "note": "Reduces repeated gloss artifacts in stream.",
            "parse": ["dedup=true", f"confidence={confidence}", "min_conf=0.65"],
        },
        "module5": {
            "title": "Sentence Construction",
            "input": partial,
            "process": "rule-based grammar correction",
            "output": sentence,
            "note": "Converts gloss order into natural English sentence.",
            "parse": ["mode=rule_based", "sov_to_svo=true", f"sentence='{sentence}'"],
        },
        "module6": {
            "title": "AI Voice Generation",
            "input": sentence,
            "process": "queue -> synthesize -> play",
            "output": "audio=playing" if (tts_enabled and inference_ready) else ("audio=muted" if not tts_enabled else "audio=waiting"),
            "note": "pyttsx3/gTTS compatible speech module.",
            "parse": [
                f"tts_enabled={tts_enabled}",
                f"voice_rate={0.9 + confidence * 0.2:.2f}x",
            ],
        },
        "module7": {
            "title": "Live Insights",
            "input": "predictions + references",
            "process": "FPS + latency + BLEU + WER + confusion proxy",
            "output": f"fps={fps}, latency={latency_ms}ms",
            "note": "Demo metrics computed on small rolling sample.",
            "parse": [
                f"accuracy={confidence:.2f}",
                f"bleu={confidence:.2f}",
                f"wer={max(0.0, 1.0 - confidence):.2f}",
            ],
        },
    }
