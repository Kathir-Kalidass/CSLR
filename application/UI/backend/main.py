"""
FastAPI WebSocket Server for Real-Time CSLR
4-Module Pipeline: Preprocessing → Feature Extraction → Recognition → Translation
"""

from __future__ import annotations

import asyncio
import base64
import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from realtime_engine import RealtimeCSLREngine, PreprocessingStats

# Default CSV path: two levels up from this file → application/backend/checkpoints/…
_DEFAULT_CSV = str(
    Path(__file__).resolve().parent.parent.parent
    / "backend"
    / "checkpoints"
    / "isl_cslrt_experiment"
    / "training_metrics.csv"
)
TRAINING_METRICS_CSV = os.environ.get("TRAINING_METRICS_CSV", _DEFAULT_CSV)

app = FastAPI(title="CSLR Real-Time Server", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Vocabulary for ISL recognition
VOCABULARY = [
    "HELLO", "HI", "HOW", "YOU", "ME", "GO", "COME", "SCHOOL",
    "THANK-YOU", "THANKS", "PLEASE", "WATER", "NAME", "YES",
    "NO", "WANT", "NEED", "LIKE", "HAVE", "WHAT", "WHERE",
    "WHEN", "WHY", "FINE", "GOOD", "BAD", "SORRY", "WELCOME"
]


class ConnectionManager:
    """Manage WebSocket connections and broadcasting."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.engine: Optional[RealtimeCSLREngine] = None
        self.running = False
        self.camera_active = False
        self.tts_enabled = True
        self.selected_module = "module1"
    
    async def connect(self, websocket: WebSocket):
        """Accept new connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Initialize engine if not already done
        if self.engine is None:
            self.engine = RealtimeCSLREngine(
                vocabulary=VOCABULARY,
                demo_mode=True  # Use demo mode fallback for untrained weights
            )
    
    def disconnect(self, websocket: WebSocket):
        """Remove connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        """Send message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)


manager = ConnectionManager()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "engine_initialized": manager.engine is not None,
        "camera_active": manager.engine.camera is not None if manager.engine else False,
        "active_connections": len(manager.active_connections)
    }


@app.get("/api/training-metrics")
async def training_metrics():
    """Return per-epoch training metrics from the CSV written by train_isl_cslrt.py."""
    csv_path = Path(TRAINING_METRICS_CSV)
    if not csv_path.exists():
        return JSONResponse({"rows": [], "available": False})

    rows: List[Dict] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric strings to floats where possible
            cleaned: Dict = {}
            for k, v in row.items():
                try:
                    cleaned[k] = float(v)
                except (ValueError, TypeError):
                    cleaned[k] = v
            rows.append(cleaned)

    return JSONResponse({"rows": rows, "available": True, "csv_path": str(csv_path)})


async def realtime_processing_loop():
    """Main processing loop - runs when system is active."""
    tick = 0
    last_sentence = ""
    glosses_str = ""
    confidence = 0.0
    
    while manager.running:
        try:
            if manager.engine and manager.engine.camera:
                # Process frame through complete pipeline
                result_glosses, result_sentence, result_conf, stats, display_frame = manager.engine.process_frame()
                
                # Update if we got new results
                if result_sentence:
                    glosses_str = result_glosses or ""
                    if result_sentence != last_sentence:
                        last_sentence = result_sentence
                        confidence = result_conf
                    else:
                        confidence = result_conf
                
                # Encode frame for display
                _, buffer = cv2.imencode('.jpg', display_frame)
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                
                # Build payload
                payload = build_payload(
                    tick=tick,
                    running=manager.running,
                    camera_active=manager.camera_active,
                    glosses=glosses_str,
                    sentence=last_sentence,
                    confidence=confidence,
                    stats=stats,
                    transcripts=list(manager.engine.transcripts),
                    tts_enabled=manager.tts_enabled,
                    selected_module=manager.selected_module,
                    frame=frame_b64
                )
                
                await manager.broadcast(payload)
                tick += 1
                
            await asyncio.sleep(0.05)  # ~20 FPS
            
        except Exception as e:
            print(f"Processing error: {e}")
            await asyncio.sleep(0.1)


def build_payload(
    tick: int,
    running: bool,
    camera_active: bool,
    glosses: str,
    sentence: str,
    confidence: float,
    stats: PreprocessingStats,
    transcripts: List[str],
    tts_enabled: bool,
    selected_module: str,
    frame: Optional[str] = None
) -> dict:
    """Build WebSocket payload."""
    
    # Determine active stage based on buffer fill
    buffer_fill_pct = (stats.buffer_fill / stats.buffer_capacity) * 100
    
    if buffer_fill_pct < 25:
        active_stage = "module1"
    elif buffer_fill_pct < 60:
        active_stage = "module2"
    elif buffer_fill_pct < 90:
        active_stage = "module3"
    else:
        active_stage = "module4"
    
    # Build console messages
    console_messages = [
        f"[Module 1] Motion: {stats.motion_score:.1f}, Kept: {stats.frames_kept}, Discarded: {stats.frames_discarded}",
        f"[Module 1] ROI: {stats.roi_detected}, Pose: {stats.pose_detected}",
        f"[Module 2] Feature extraction active",
        f"[Module 3] BiLSTM processing: {buffer_fill_pct:.0f}%",
        f"[Module 4] Translation: {sentence if sentence else 'waiting...'}"
    ]
    
    return {
        "status": "running" if running else "idle",
        "tick": tick,
        "active_stage": active_stage,
        "selected_module": selected_module,
        "partial_gloss": glosses if glosses else "--",
        "final_sentence": sentence if sentence else "Processing sign language...",
        "audio_state": "playing" if tts_enabled and sentence else "idle",
        "confidence": round(confidence, 2),
        "fps": int(stats.fps),
        "latency_ms": int(stats.processing_time_ms),
        "metrics": {
            "accuracy": round(confidence, 2),
            "wer": 0.0,
            "bleu": 0.0,
            "window_size": 64,
            "stride": 32,
        },
        "transcript_history": transcripts[-10:],  # Last 10
        "control_state": {
            "running": running,
            "tts_enabled": tts_enabled,
            "camera_active": camera_active,
            "using_real_camera": True,
        },
        "parser_console": console_messages,
        "modules": {
            "module1": "Video Preprocessing",
            "module2": "Feature Extraction",
            "module3": "Temporal Recognition",
            "module4": "Translation & Output",
        },
        "module1_debug": {
            "buffer_fill": stats.buffer_fill,
            "buffer_capacity": stats.buffer_capacity,
            "frames_kept": stats.frames_kept,
            "frames_discarded": stats.frames_discarded,
            "motion_score": round(stats.motion_score, 2),
            "roi_detected": stats.roi_detected,
            "pose_detected": stats.pose_detected,
        },
        "video_frame": frame,  # Base64 encoded JPEG
    }


@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication."""
    await manager.connect(websocket)
    processing_task: Optional[asyncio.Task] = None
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            msg_type = message.get("type")
            
            if msg_type == "open_camera":
                if not manager.camera_active:
                    engine = manager.engine
                    if engine is None:
                        await websocket.send_json({"status": "error", "message": "Engine not initialized"})
                        continue
                    # Initialize camera
                    if engine.start_camera(camera_id=0):
                        manager.camera_active = True
                        await websocket.send_json({"status": "camera_opened", "message": "Camera initialized successfully"})
                    else:
                        await websocket.send_json({"status": "error", "message": "Failed to open camera - no camera detected"})
                else:
                    await websocket.send_json({"status": "camera_opened", "message": "Camera already active"})

            elif msg_type == "close_camera":
                if manager.running:
                    manager.running = False
                    if processing_task:
                        processing_task.cancel()
                        try:
                            await processing_task
                        except asyncio.CancelledError:
                            pass
                if manager.engine:
                    manager.engine.stop_camera()
                manager.camera_active = False
                await websocket.send_json({"status": "camera_closed", "message": "Camera closed"})
            
            elif msg_type == "start":
                if not manager.running and manager.camera_active:
                    manager.running = True
                    # Start processing loop
                    processing_task = asyncio.create_task(realtime_processing_loop())
                    await websocket.send_json({"status": "started", "message": "Processing started"})
                elif not manager.camera_active:
                    await websocket.send_json({"status": "error", "message": "Please open camera first"})
                else:
                    await websocket.send_json({"status": "started", "message": "Processing already running"})
            
            elif msg_type == "stop":
                if manager.running:
                    manager.running = False
                    if processing_task:
                        processing_task.cancel()
                        try:
                            await processing_task
                        except asyncio.CancelledError:
                            pass
                    await websocket.send_json({"status": "stopped", "message": "Processing stopped"})
                else:
                    await websocket.send_json({"status": "stopped", "message": "Processing already stopped"})

            elif msg_type == "clear":
                if manager.engine:
                    manager.engine.clear_runtime()
                await websocket.send_json({"status": "cleared", "message": "Transcripts and buffers cleared"})
            
            elif msg_type == "toggle_tts":
                manager.tts_enabled = not manager.tts_enabled
                await websocket.send_json({"status": "tts_toggled", "enabled": manager.tts_enabled})
            
            elif msg_type == "module_select":
                module_id = message.get("module", "module1")
                manager.selected_module = module_id
                await websocket.send_json({"status": "module_selected", "module": module_id})
            
            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        if manager.running:
            manager.running = False
            if processing_task:
                processing_task.cancel()
            if manager.engine:
                manager.engine.stop_camera()
        manager.camera_active = False
    
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@app.on_event("startup")
async def startup_event():
    """Initialize on server startup."""
    print("=" * 60)
    print("CSLR Real-Time Server Started")
    print("=" * 60)
    print(f"Vocabulary size: {len(VOCABULARY)}")
    print("WebSocket: ws://localhost:8080/ws/realtime")
    print("=" * 60)


@app.on_event("shutdown") 
async def shutdown_event():
    """Cleanup on server shutdown."""
    if manager.engine:
        manager.engine.stop_camera()
    print("Server shutdown complete")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
