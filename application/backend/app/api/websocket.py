"""
WebSocket Endpoints
Real-time streaming inference
"""

from __future__ import annotations

import time
from typing import Dict

import cv2
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

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
    audio_service = getattr(app.state, "audio_service", None)

    streaming_service = getattr(app.state, "streaming_service", None)
    if streaming_service is None:
        streaming_service = StreamingService()
    stream_state = streaming_service.get_stream(client_id)
    last_spoken_sentence = ""

    try:
        while True:
            data = await websocket.receive_json()
            timestamp = data.get("timestamp", time.time())
            frame_b64 = data.get("frame", "")

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
                await manager.send_personal_message(
                    {"error": "No frame data", "timestamp": timestamp},
                    client_id,
                )
                continue

            try:
                frame_rgb = decode_base64_image(frame_b64)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                stream_state.add_frame(frame_bgr)

                result = await service.process_frame_stream(
                    frame_bgr, stream_state.pipeline_state
                )
                stream_state.pipeline_state = result.get("state", None)

                sentence = result.get("sentence", "")
                if (
                    settings.AUTO_TTS_STREAM
                    and audio_service is not None
                    and sentence
                    and sentence != last_spoken_sentence
                ):
                    await audio_service.synthesize_streaming(sentence)
                    last_spoken_sentence = sentence

                await manager.send_personal_message(
                    {
                        "gloss": result.get("gloss", []),
                        "sentence": sentence,
                        "confidence": result.get("confidence", 0.0),
                        "fps": result.get("fps", 0.0),
                        "buffer_fill": len(stream_state.frame_buffer),
                        "buffer_capacity": stream_state.clip_length,
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
