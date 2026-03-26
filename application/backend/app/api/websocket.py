"""
WebSocket Endpoints
Real-time streaming inference
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Dict

import cv2
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.monitoring.gpu_monitor import gpu_monitor

from app.core.config import settings
from app.core.logging import logger
from app.services.streaming_service import StreamingService
from app.utils.image_utils import decode_base64_image

router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected. Total: {len(self.active_connections)}")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected. Total: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, client_id: str):
        ws = self.active_connections.get(client_id)
        if ws is not None:
            await ws.send_json(message)


manager = ConnectionManager()


def build_demo_payload(
    *,
    tick: int,
    running: bool,
    camera_active: bool,
    tts_enabled: bool,
    grand_mode: bool,
    result: dict,
    transcript_history: list[str],
    checkpoint_path: str,
) -> dict:
    gloss_tokens = [str(token) for token in result.get("gloss", []) if str(token).strip()]
    gloss_text = " ".join(gloss_tokens) if gloss_tokens else "--"
    idle_sentence = "Open camera and start live recognition."
    processing_sentence = "Analyzing live sign window..."
    runtime_status = result.get("runtime_status", "ready")
    sentence = result.get("sentence") or (processing_sentence if running else idle_sentence)
    if running and runtime_status == "blank_decode":
        sentence = "No recognisable gloss decoded from the current live pose window."
    confidence = float(result.get("confidence", 0.0))
    buffer_fill = int(result.get("buffer_fill", 0))
    buffer_capacity = int(getattr(settings, "CLIP_LENGTH", 64))
    active_stage = "module1"
    if buffer_fill >= buffer_capacity:
        active_stage = "module5" if sentence and sentence != "Analyzing live sign window..." else "module3"
    elif buffer_fill > 0:
        active_stage = "module2"

    parser_console = [
        f"[capture] camera_active={camera_active} running={running}",
        f"[window] fill={buffer_fill}/{buffer_capacity}",
        f"[decode] gloss={gloss_text}",
        f"[sentence] {sentence}",
        f"[runtime] checkpoint={checkpoint_path}",
        f"[runtime_status] {runtime_status}",
    ]

    return {
        "timestamp": time.time(),
        "tick": tick,
        "status": "active" if running else "idle",
        "inference_mode": "real",
        "runtime_status": runtime_status,
        "latency_ms": 0 if not running else int(round(float(result.get("latency_ms", 0.0)))),
        "fps": float(result.get("fps", 0.0)) if running else 0.0,
        "confidence": confidence,
        "audio_state": "speaking" if tts_enabled and sentence and sentence not in {"", "--", processing_sentence, idle_sentence} else "idle",
        "partial_gloss": gloss_text,
        "final_sentence": sentence,
        "active_stage": active_stage,
        "pipeline_order": ["module1", "module2", "module3", "module4", "module5", "module6", "module7"],
        "attention": {"rgb": 0.0, "pose": 1.0},
        "timeline": {
            "active_window": f"frames[{max(0, tick * max(1, buffer_capacity // 2))}:{max(0, tick * max(1, buffer_capacity // 2) + max(buffer_fill - 1, 0))}]",
            "windows": [
                f"frames[{max(0, tick * max(1, buffer_capacity // 2))}:{max(0, tick * max(1, buffer_capacity // 2) + max(buffer_fill - 1, 0))}]",
            ],
        },
        "audio_wave": [round((0.2 + confidence * 0.8) if tts_enabled else 0.0, 3) for _ in range(24)],
        "metrics": {
            "wer_proxy": round(max(0.0, 1.0 - confidence), 3),
            "bleu_proxy": round(confidence, 3),
            "window_frames": buffer_capacity,
            "stride": max(1, buffer_capacity // 2),
            "accuracy_proxy": round(confidence, 3),
        },
        "transcript_history": transcript_history[:12],
        "control_state": {
            "running": running,
            "tts_enabled": tts_enabled,
            "camera_active": camera_active,
            "grand_mode": grand_mode,
        },
        "module1": {
            "title": "Capture and Frame Window",
            "input": "Webcam stream",
            "process": "Browser frame upload -> pose extraction",
            "output": "Pose tensor",
            "note": f"Buffer {buffer_fill}/{buffer_capacity}",
            "parse": [f"buffer_fill={buffer_fill}", f"camera_active={camera_active}"],
        },
        "module2": {
            "title": "Pose Feature Extraction",
            "input": "Pose tensor",
            "process": "Pose encoder",
            "output": "Pose features",
            "note": "Pose-only checkpoint active",
            "parse": ["pose_only=true"],
        },
        "module3": {
            "title": "Temporal Decode",
            "input": "Pose features",
            "process": "Causal TCN + attention + CTC",
            "output": gloss_text,
            "note": result.get("runtime_status", "ready"),
            "parse": [f"confidence={confidence:.3f}"],
        },
        "module4": {
            "title": "Gloss Cleanup",
            "input": gloss_text,
            "process": "CTC dedupe",
            "output": gloss_text,
            "note": "Running",
            "parse": [f"gloss={gloss_text}"],
        },
        "module5": {
            "title": "Sentence Construction",
            "input": gloss_text,
            "process": "Rule-based grammar correction",
            "output": sentence,
            "note": sentence,
            "parse": [sentence],
        },
        "module6": {
            "title": "Audio Output",
            "input": sentence,
            "process": "Backend TTS endpoint",
            "output": "audio",
            "note": "Ready" if tts_enabled else "TTS disabled",
            "parse": [f"tts_enabled={tts_enabled}"],
        },
        "module7": {
            "title": "Live Insights",
            "input": "prediction stream",
            "process": "client metrics update",
            "output": "dashboard",
            "note": "Streaming" if running else "Idle",
            "parse": parser_console[-2:],
        },
        "parser_console": parser_console,
        "pose_landmarks": result.get("pose_landmarks", []),
        "hand_landmarks": result.get("hand_landmarks", []),
        "model_info": {
            "active_model": checkpoint_path,
            "state": "loaded",
        },
        "tts_engine": settings.TTS_ENGINE,
        "init_sequence": [
            "Loading backend checkpoint runtime",
            "Initializing pose inference",
            "Connecting demo websocket compatibility",
            "Backend ready",
        ],
    }


@router.websocket("/inference")
async def websocket_inference(websocket: WebSocket):
    """
    Client sends: {"frame": "base64_encoded_image", "timestamp": 123456}
    Server sends: {"gloss": [...], "sentence": "...", "confidence": 0.9, ...}
    """
    client_id = f"client_{id(websocket)}"

    if len(manager.active_connections) >= settings.MAX_CLIENTS:
        await websocket.close(code=1008, reason="Maximum clients reached")
        return

    await manager.connect(websocket, client_id)

    app = websocket.app
    service = getattr(app.state, "inference_service", None)

    streaming_service = getattr(app.state, "streaming_service", None)
    if streaming_service is None:
        streaming_service = StreamingService()
    stream_state = streaming_service.get_stream(client_id)

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "")

            # Silently ack control / stats messages without processing as frames
            if msg_type in ("control", "client_video_stats"):
                continue

            timestamp = data.get("timestamp", time.time())
            # Accept both field names used by different frontend builds
            frame_b64 = data.get("frame") or data.get("image_jpeg_base64", "")

            if service is None:
                await manager.send_personal_message(
                    {
                        "gloss": ["SERVICE_UNAVAILABLE"],
                        "sentence": "Inference service not loaded",
                        "confidence": 0.0,
                        "fps": 0.0,
                        "timestamp": timestamp,
                    },
                    client_id,
                )
                continue

            if not frame_b64:
                continue

            try:
                t_recv = time.time()
                frame_rgb = decode_base64_image(frame_b64)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                stream_state.add_frame(frame_bgr)

                result = await service.process_frame_stream(
                    frame_bgr, stream_state.pipeline_state
                )
                stream_state.pipeline_state = result.get("state", None)
                latency_ms = round((time.time() - t_recv) * 1000, 1)

                # Compute a simple attention proxy from confidence
                conf = result.get("confidence", 0.0)
                rgb_w = round(0.5 + conf * 0.3, 3)
                pose_w = round(0.5 - conf * 0.3, 3)

                await manager.send_personal_message(
                    {
                        "status": "active",
                        "inference_mode": "live",
                        "gloss": result.get("gloss", []),
                        "sentence": result.get("sentence", ""),
                        "confidence": conf,
                        "fps": result.get("fps", 0.0),
                        "latency_ms": latency_ms,
                        "pose_landmarks": result.get("pose_landmarks", []),
                        "hand_landmarks": result.get("hand_landmarks", []),
                        "attention": {"rgb": rgb_w, "pose": pose_w},
                        "buffer_fill": len(stream_state.frame_buffer),
                        "buffer_capacity": stream_state.clip_length,
                        "metrics": {"wer_proxy": 0.0, "bleu_proxy": 0.0},
                        "timeline": {"windows": []},
                        "control_state": {"running": True, "tts_enabled": True, "grand_mode": False},
                        "timestamp": timestamp,
                    },
                    client_id,
                )
            except Exception as e:
                logger.error(f"Frame processing error: {e}")
                await manager.send_personal_message(
                    {"error": str(e), "timestamp": timestamp},
                    client_id,
                )

    except WebSocketDisconnect:
        manager.disconnect(client_id)
        streaming_service.close_stream(client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(client_id)
        streaming_service.close_stream(client_id)
        await websocket.close(code=1011, reason=str(e))


async def websocket_demo_compat(websocket: WebSocket):
    client_id = f"demo_{id(websocket)}"
    await manager.connect(websocket, client_id)

    app = websocket.app
    service = getattr(app.state, "inference_service", None)
    checkpoint_path = getattr(settings, "ISIGN_CHECKPOINT_PATH", "")

    client_state = {
        "running": False,
        "camera_active": False,
        "tts_enabled": True,
        "grand_mode": False,
        "tick": 0,
        "pipeline_state": None,
        "transcript_history": [],
        "latest_result": {
            "gloss": [],
            "sentence": "",
            "confidence": 0.0,
            "buffer_fill": 0,
            "runtime_status": "ready",
        },
    }

    async def send_state() -> None:
        payload = build_demo_payload(
            tick=client_state["tick"],
            running=client_state["running"],
            camera_active=client_state["camera_active"],
            tts_enabled=client_state["tts_enabled"],
            grand_mode=client_state["grand_mode"],
            result=client_state["latest_result"],
            transcript_history=client_state["transcript_history"],
            checkpoint_path=checkpoint_path,
        )
        await manager.send_personal_message(payload, client_id)
        client_state["tick"] += 1

    try:
        await send_state()
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "")

            if msg_type == "client_video_stats":
                client_state["camera_active"] = bool(data.get("camera_active", False))
                await send_state()
                continue

            if msg_type == "control":
                action = data.get("action", "")
                if action == "start":
                    client_state["running"] = True
                elif action == "stop":
                    client_state["running"] = False
                elif action == "clear":
                    client_state["transcript_history"] = []
                    client_state["pipeline_state"] = None
                    client_state["latest_result"] = {
                        "gloss": [],
                        "sentence": "",
                        "confidence": 0.0,
                        "buffer_fill": 0,
                        "runtime_status": "ready",
                    }
                elif action == "toggle_tts":
                    client_state["tts_enabled"] = not client_state["tts_enabled"]
                elif action == "set_grand_mode":
                    client_state["grand_mode"] = bool(data.get("value", False))
                await send_state()
                continue

            if msg_type != "client_video_frame":
                continue

            client_state["camera_active"] = True
            if not client_state["running"]:
                await send_state()
                continue

            frame_b64 = data.get("image_jpeg_base64", "")
            if not frame_b64 or service is None:
                await send_state()
                continue

            try:
                frame_rgb = decode_base64_image(frame_b64)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                result = await service.process_frame_stream(frame_bgr, client_state["pipeline_state"])
                client_state["pipeline_state"] = result.get("state")
                client_state["latest_result"] = result
            except Exception as exc:
                logger.error("Demo frame processing error: %s", exc)
                client_state["latest_result"] = {
                    **client_state["latest_result"],
                    "runtime_status": f"frame_error:{type(exc).__name__}",
                }
                await send_state()
                continue

            sentence = (result.get("sentence") or "").strip()
            if sentence and sentence not in {"Analyzing live sign window..."}:
                history = client_state["transcript_history"]
                if not history or history[0] != sentence:
                    history.insert(0, sentence)
                    client_state["transcript_history"] = history[:12]

            await send_state()

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as exc:
        logger.error("Demo websocket error: %s", exc)
        manager.disconnect(client_id)
        try:
            await websocket.close(code=1011, reason=str(exc))
        except Exception:
            pass
