from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .demo_engine import engine
from .runtime import REAL_RUNTIME

router = APIRouter()


@router.websocket("/ws/demo")
async def websocket_demo(websocket: WebSocket) -> None:
    await websocket.accept()
    from collections import deque as _deque
    client_state: dict[str, Any] = {
        "camera_active": False,
        "frame_hint": 0,
        "resolution": "unknown",
        "latest_frame_b64": "",
        "latest_frame_seq": -1,
        "processed_frame_seq": -1,
        "frame_queue": _deque(maxlen=120),
        "latest_inference": {"ready": False},
        "running": False,
        "tts_enabled": True,
        "grand_mode": False,
        "transcript_history": [],
    }

    async def recv_client() -> None:
        while True:
            text = await websocket.receive_text()
            try:
                message = json.loads(text)
            except json.JSONDecodeError:
                continue

            msg_type = message.get("type")
            if msg_type == "client_video_stats":
                client_state["camera_active"] = bool(message.get("camera_active", False))
                client_state["frame_hint"] = int(message.get("frame_hint", 0))
                client_state["resolution"] = str(message.get("resolution", "unknown"))
            elif msg_type == "client_video_frame":
                # Browser webcam frame (JPEG base64) for real checkpoint inference.
                b64 = str(message.get("image_jpeg_base64", ""))
                seq = int(message.get("frame_seq", 0))
                client_state["latest_frame_b64"] = b64
                client_state["latest_frame_seq"] = seq
                # Queue ALL frames so the buffer fills quickly
                if b64:
                    client_state["frame_queue"].append((b64, seq))
            elif msg_type == "control":
                action = message.get("action", "")
                if action == "start":
                    client_state["running"] = True
                    client_state["latest_inference"] = {"ready": False}
                    client_state["processed_frame_seq"] = -1
                    if REAL_RUNTIME.available:
                        REAL_RUNTIME.reset()
                elif action == "stop":
                    client_state["running"] = False
                elif action == "clear":
                    client_state["transcript_history"] = []
                    client_state["latest_inference"] = {"ready": False}
                    if REAL_RUNTIME.available:
                        REAL_RUNTIME.reset()
                elif action == "toggle_tts":
                    client_state["tts_enabled"] = not bool(client_state.get("tts_enabled", True))
                elif action == "set_grand_mode":
                    client_state["grand_mode"] = bool(message.get("value", False))

    try:
        sender = asyncio.create_task(engine.stream(websocket, client_state=client_state))
        receiver = asyncio.create_task(recv_client())
        done, pending = await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_EXCEPTION)
        for task in pending:
            task.cancel()
        for task in done:
            exc = task.exception()
            if exc:
                raise exc
    except WebSocketDisconnect:
        return
