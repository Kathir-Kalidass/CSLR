from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_FILE = BASE_DIR / "templates" / "index.html"
REPORT_DIAGRAMS_DIR = BASE_DIR.parent / "report_pages" / "architecture_diagram"

app = FastAPI(title="CSLR Pipeline Demo UI")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
if REPORT_DIAGRAMS_DIR.exists():
    app.mount("/report-diagrams", StaticFiles(directory=str(REPORT_DIAGRAMS_DIR)), name="report-diagrams")


class DemoEngine:
    def __init__(self) -> None:
        self.gloss_tokens = [
            "HELLO",
            "HOW",
            "YOU",
            "ME",
            "GO",
            "SCHOOL",
            "TOMORROW",
            "BOOK",
            "WHERE",
            "NAME",
            "THANK-YOU",
            "FINE",
        ]
        self.sentences = [
            "Hello, how are you?",
            "I am going to school tomorrow.",
            "Where is the red book?",
            "What is your name?",
            "I am fine, thank you.",
        ]
        self.pipeline_order = [
            "module1",
            "module2",
            "module3",
            "module4",
            "module5",
            "module6",
            "module7",
        ]

    def _idle_payload(self, client_state: dict[str, Any]) -> dict[str, Any]:
        return {
            "timestamp": time.time(),
            "tick": client_state.get("tick", 0),
            "status": "idle",
            "latency_ms": 0,
            "fps": 0,
            "confidence": 0.0,
            "audio_state": "disabled" if not client_state.get("tts_enabled", True) else "idle",
            "partial_gloss": "--",
            "final_sentence": "Press Start to run real-time pipeline",
            "active_stage": "module1",
            "pipeline_order": self.pipeline_order,
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
            },
            "module1": {
                "title": "Module 1: Video Acquisition + Preprocessing",
                "input": "Webcam stream",
                "process": "Capture -> sample -> crop ROI -> normalize + pose extraction",
                "output": "RGB tensor + Pose tensor",
                "note": "Pipeline paused.",
                "parse": ["waiting_for_start=true"],
            },
            "module2": {
                "title": "Module 2: Dual Feature Extraction",
                "input": "RGB + Pose tensors",
                "process": "ResNet18 RGB stream + MLP pose stream",
                "output": "Feature vectors",
                "note": "No active window.",
                "parse": ["window_ready=false"],
            },
            "module3": {
                "title": "Module 3: Attention Fusion + Temporal Model",
                "input": "RGB and pose features",
                "process": "Attention fusion + BiLSTM + CTC",
                "output": "Gloss tokens",
                "note": "Waiting for buffered frames.",
                "parse": ["stride=32, window_size=64"],
            },
            "module4": {
                "title": "Module 4: Gloss Post-Processing",
                "input": "Raw gloss sequence",
                "process": "Deduplicate + confidence filtering",
                "output": "Clean gloss string",
                "note": "No active prediction.",
                "parse": ["dedup_active=true"],
            },
            "module5": {
                "title": "Module 5: Gloss -> Sentence",
                "input": "Clean gloss",
                "process": "Rule-based grammar correction",
                "output": "Readable English sentence",
                "note": "Waiting for gloss input.",
                "parse": ["grammar_mode=rule_based"],
            },
            "module6": {
                "title": "Module 6: Text-to-Speech",
                "input": "Corrected sentence",
                "process": "Queue and play speech",
                "output": "Audio output",
                "note": "TTS disabled" if not client_state.get("tts_enabled", True) else "Ready",
                "parse": [f"tts_enabled={client_state.get('tts_enabled', True)}"],
            },
            "module7": {
                "title": "Module 7: Live Metrics",
                "input": "Prediction stream",
                "process": "Compute FPS, latency, WER, BLEU, accuracy",
                "output": "Dashboard metrics",
                "note": "Metrics reset while idle.",
                "parse": ["dataset=demo_subset(20)"],
            },
            "parser_console": ["[system] idle: waiting for Start"],
        }

    def _running_payload(self, tick: int, client_state: dict[str, Any]) -> dict[str, Any]:
        camera_active = bool(client_state.get("camera_active", False))
        frame_hint = int(client_state.get("frame_hint", 0))
        client_resolution = str(client_state.get("resolution", "unknown"))

        fps = random.randint(16, 27)
        frame_count = 64
        confidence = round(random.uniform(0.74, 0.95), 2)
        latency_ms = random.randint(240, 420)

        sample_gloss = random.sample(self.gloss_tokens, k=3)
        sentence = self.sentences[tick % len(self.sentences)]
        partial = " ".join(sample_gloss)

        window_start = tick * 32
        window_end = window_start + frame_count - 1
        pose_detected = random.randint(68, 75)
        hand_visibility = round(random.uniform(0.81, 0.99), 2)
        attn_rgb = round(random.uniform(0.42, 0.71), 2)
        attn_pose = round(1.0 - attn_rgb, 2)

        history = client_state.setdefault("transcript_history", [])
        if not history or history[0] != sentence:
            history.insert(0, sentence)
        client_state["transcript_history"] = history[:12]

        module_data = {
            "module1": {
                "title": "Module 1: Video Acquisition + Preprocessing",
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
                "title": "Module 2: Dual Feature Extraction",
                "input": "RGB + Pose tensors",
                "process": "ResNet18 (RGB) + MLP(150->256 pose)",
                "output": "rgb_feat:64x512 | pose_feat:64x256",
                "note": "Efficient extractor setup for real-time inference.",
                "parse": ["resnet18_backbone=active", "pose_encoder=linear_relu_linear", "precision=fp16_path"],
            },
            "module3": {
                "title": "Module 3: Attention Fusion + Temporal Model",
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
                "title": "Module 4: Caption Buffer & Post-Processing",
                "input": partial,
                "process": "merge_repeats + confidence thresholding",
                "output": partial,
                "note": "Reduces repeated gloss artifacts in stream.",
                "parse": ["dedup=true", f"confidence={confidence}", "min_conf=0.65"],
            },
            "module5": {
                "title": "Module 5: AI Sentence Correction",
                "input": partial,
                "process": "rule-based grammar correction",
                "output": sentence,
                "note": "Converts gloss order into natural English sentence.",
                "parse": ["mode=rule_based", "sov_to_svo=true", f"sentence='{sentence}'"],
            },
            "module6": {
                "title": "Module 6: Text-to-Speech",
                "input": sentence,
                "process": "queue -> synthesize -> play",
                "output": "audio=playing" if client_state.get("tts_enabled", True) else "audio=muted",
                "note": "pyttsx3/gTTS compatible speech module.",
                "parse": [
                    f"tts_enabled={client_state.get('tts_enabled', True)}",
                    f"voice_rate={random.choice(['0.95x', '1.0x', '1.05x'])}",
                ],
            },
            "module7": {
                "title": "Module 7: Performance Metrics",
                "input": "predictions + references",
                "process": "FPS + latency + BLEU + WER + confusion proxy",
                "output": f"fps={fps}, latency={latency_ms}ms",
                "note": "Demo metrics computed on small rolling sample.",
                "parse": [
                    f"accuracy={round(random.uniform(0.73, 0.9), 2)}",
                    f"bleu={round(random.uniform(0.31, 0.47), 2)}",
                    f"wer={round(random.uniform(0.12, 0.23), 2)}",
                ],
            },
        }

        parser_console = []
        for module_key in self.pipeline_order:
            parser_console.extend([f"[{module_key}] {line}" for line in module_data[module_key]["parse"]])

        return {
            "timestamp": time.time(),
            "tick": tick,
            "status": "active",
            "latency_ms": latency_ms,
            "fps": fps,
            "confidence": confidence,
            "audio_state": "speaking" if client_state.get("tts_enabled", True) else "muted",
            "partial_gloss": partial,
            "final_sentence": sentence,
            "active_stage": self.pipeline_order[tick % len(self.pipeline_order)],
            "pipeline_order": self.pipeline_order,
            "metrics": {
                "wer_proxy": round(random.uniform(0.12, 0.23), 3),
                "bleu_proxy": round(random.uniform(0.31, 0.47), 3),
                "window_frames": frame_count,
                "stride": 32,
                "accuracy_proxy": round(random.uniform(0.73, 0.9), 3),
            },
            "transcript_history": client_state.get("transcript_history", []),
            "control_state": {
                "running": True,
                "tts_enabled": client_state.get("tts_enabled", True),
                "camera_active": camera_active,
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

    def make_payload(self, tick: int, client_state: dict[str, Any]) -> dict[str, Any]:
        running = bool(client_state.get("running", False))
        if not running:
            return self._idle_payload(client_state)
        return self._running_payload(tick=tick, client_state=client_state)

    async def stream(self, websocket: WebSocket, client_state: dict[str, Any]) -> None:
        tick = 0
        while True:
            client_state["tick"] = tick
            payload = self.make_payload(tick=tick, client_state=client_state)
            await websocket.send_text(json.dumps(payload))
            tick += 1
            await asyncio.sleep(0.95)


engine = DemoEngine()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return TEMPLATE_FILE.read_text(encoding="utf-8")


@app.websocket("/ws/demo")
async def websocket_demo(websocket: WebSocket) -> None:
    await websocket.accept()
    client_state: dict[str, Any] = {
        "camera_active": False,
        "frame_hint": 0,
        "resolution": "unknown",
        "running": False,
        "tts_enabled": True,
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
            elif msg_type == "control":
                action = message.get("action", "")
                if action == "start":
                    client_state["running"] = True
                elif action == "stop":
                    client_state["running"] = False
                elif action == "clear":
                    client_state["transcript_history"] = []
                elif action == "toggle_tts":
                    client_state["tts_enabled"] = not bool(client_state.get("tts_enabled", True))

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
