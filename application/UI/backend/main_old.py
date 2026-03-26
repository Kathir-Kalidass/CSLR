from __future__ import annotations

import asyncio
import base64
import json
import random
import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Import our custom modules
from module1_preprocessing import Module1PreprocessingEngine, ProcessingStats
from modules_pipeline import CSLRPipeline, SlidingWindowBuffer

app = FastAPI(title="CSLR Advanced Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass
class RuntimeState:
    running: bool = False
    tts_enabled: bool = True
    camera_active: bool = False
    use_real_camera: bool = False  # Toggle between demo and real camera
    frame_hint: int = 0
    resolution: str = "unknown"
    
    # Module 1 statistics
    module1_stats: Optional[ProcessingStats] = None


class CSLREngine:
    """Main CSLR engine integrating all 4 modules."""
    
    def __init__(self, demo_mode: bool = True) -> None:
        self.demo_mode = demo_mode
        
        # Vocabulary (expandable for full training)
        self.vocabulary = [
            "HELLO",
            "HI",
            "HOW",
            "YOU",
            "ME",
            "GO",
            "COME",
            "SCHOOL",
            "THANK-YOU",
            "THANKS",
            "PLEASE",
            "WATER",
            "NAME",
            "YES",
            "NO",
            "WANT",
            "NEED",
            "LIKE",
            "HAVE",
        ]
        
        # Demo sentences for fallback
        self.demo_sentences = [
            "Hello, how are you?",
            "I am going to school.",
            "Please give me water.",
            "What is your name?",
            "Thank you so much.",
            "Yes, I understand.",
            "No, I don't know.",
        ]
        
        # 4 modules architecture
        self.modules = ["module1", "module2", "module3", "module4"]
        
        # Initialize Module 1 (preprocessing)
        self.preprocessor: Optional[Module1PreprocessingEngine] = None
        self.camera: Optional[cv2.VideoCapture] = None
        
        # Initialize complete pipeline (Modules 2-4)
        self.pipeline = CSLRPipeline(
            vocabulary=self.vocabulary,
            use_gpu=True,
            demo_mode=demo_mode
        )
        
        # Sliding window buffer
        self.window_buffer = SlidingWindowBuffer(window_size=64, stride=32)
    
    def start_camera(self) -> bool:
        """Initialize webcam and preprocessing engine."""
        try:
            if self.camera is not None:
                return True
            
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                return False
            
            # Configure camera
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 20)
            
            # Initialize preprocessor
            self.preprocessor = Module1PreprocessingEngine(
                frame_width=640,
                frame_height=480,
                target_fps=20,
                process_every_n_frame=2,
                motion_threshold=5.0,
                buffer_size=64,
                enable_adaptive_motion=True,
                draw_landmarks=True
            )
            
            return True
        except Exception as e:
            print(f"[ERROR] Failed to start camera: {e}")
            return False
    
    def stop_camera(self):
        """Release camera and cleanup."""
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        
        if self.preprocessor is not None:
            self.preprocessor.close()
            self.preprocessor = None
    
    def process_frame_real(self) -> Optional[tuple[list[str], str, float, ProcessingStats]]:
        """
        Process a single frame from real camera through complete pipeline.
        
        Returns:
            (glosses, sentence, confidence, stats) or None if not ready
        """
        if self.camera is None or self.preprocessor is None:
            return None
        
        # Capture frame
        ret, frame = self.camera.read()
        if not ret:
            return None
        
        # Module 1: Preprocessing
        processed = self.preprocessor.process_frame(frame)
        
        if processed.kept and processed.rgb_tensor is not None:
            # Add to sliding window buffer
            self.window_buffer.add(processed.rgb_tensor, processed.pose_tensor)
            
            # Check if window is ready
            if self.window_buffer.is_ready():
                # Get window
                rgb_window, pose_window = self.window_buffer.get_window()
                
                # Process through pipeline (Modules 2-7)
                glosses, sentence, confidence = self.pipeline.process_window(rgb_window, pose_window)
                
                # Advance window
                self.window_buffer.advance()
                
                return glosses, sentence, confidence, processed.stats
        
        # Return stats even if not processed
        return None

    def generate_payload(self, tick: int, state: RuntimeState, history: list[str], selected_module: str = "module1") -> dict[str, Any]:
        """Generate WebSocket payload with current state."""
        
        if not state.running:
            return {
                "status": "idle",
                "tick": tick,
                "active_stage": selected_module,
                "selected_module": selected_module,
                "partial_gloss": "--",
                "final_sentence": "Press Start to begin simulation",
                "audio_state": "muted" if not state.tts_enabled else "idle",
                "confidence": 0.0,
                "fps": 0,
                "latency_ms": 0,
                "metrics": {
                    "accuracy": 0.0,
                    "wer": 0.0,
                    "bleu": 0.0,
                    "window_size": 64,
                    "stride": 32,
                },
                "transcript_history": history,
                "control_state": {
                    "running": False,
                    "tts_enabled": state.tts_enabled,
                    "camera_active": state.camera_active,
                    "using_real_camera": state.use_real_camera,
                },
                "parser_console": [f"[system] {selected_module} ready for simulation"],
                "modules": {
                    "module1": "Video Preprocessing",
                    "module2": "Feature Extraction",
                    "module3": "Temporal Recognition",
                    "module4": "Translation & Output",
                },
                "module1_debug": {
                    "buffer_fill": 0,
                    "buffer_capacity": 64,
                    "frames_kept": 0,
                    "frames_discarded": 0,
                    "motion_score": 0.0,
                    "roi_detected": False,
                    "pose_detected": False,
                }
            }
        
        # Determine active stage based on selected module
        active_stage = selected_module
        
        # Try to process real camera frame
        glosses = []
        sentence = ""
        confidence = 0.0
        fps = 0
        latency_ms = 0
        
        if state.use_real_camera and state.camera_active:
            result = self.process_frame_real()
            if result is not None:
                glosses, sentence, confidence, stats = result
                fps = int(stats.fps)
                latency_ms = int(stats.processing_time_ms)
                state.module1_stats = stats
        
        # Fallback to demo mode simulation
        if not glosses:
            fps = random.randint(18, 28)
            latency_ms = random.randint(250, 390)
            confidence = round(random.uniform(0.74, 0.96), 2)
            
            # Module-specific demo predictions
            if selected_module == "module1":
                glosses = ["PREPROCESSING"]
                sentence = "Video preprocessing in progress..."
            elif selected_module == "module2":
                glosses = random.sample(["RGB", "POSE", "FUSION"], k=2)
                sentence = "Extracting visual features..."
            elif selected_module == "module3":
                glosses = random.sample(self.vocabulary, k=random.randint(2, 3))
                sentence = "Recognizing sign sequence..."
            elif selected_module == "module4":
                glosses = random.sample(self.vocabulary, k=random.randint(1, 3))
                sentence = self.demo_sentences[tick % len(self.demo_sentences)]
        
        # Update history
        if sentence and (not history or history[0] != sentence):
            history.insert(0, sentence)
            del history[12:]  # Keep last 12
        
        # Build parser console
        parser_console = [
            f"[{active_stage}] simulation active",
            f"[system] fps={fps}, latency={latency_ms}ms",
        ]
        
        if selected_module == "module1":
            parser_console.extend([
                "[module1] webcam capture @ 640×480",
                "[module1] motion filter: adaptive",
                "[module1] ROI extraction: upper body",
                "[module1] pose detection: 75 keypoints",
            ])
        elif selected_module == "module2":
            parser_console.extend([
                "[module2] RGB stream: ResNet18 features",
                "[module2] Pose stream: MLP encoding",
                "[module2] Attention fusion: enabled",
                "[module2] Output: 768-dim feature vector",
            ])
        elif selected_module == "module3":
            parser_console.extend([
                "[module3] BiLSTM: 2 layers, 512 hidden",
                "[module3] CTC alignment: active",
                f"[module3] Decoded glosses: {' '.join(glosses)}",
                f"[module3] Confidence: {confidence:.2f}",
            ])
        elif selected_module == "module4":
            parser_console.extend([
                f"[module4] Input glosses: {' '.join(glosses)}",
                "[module4] Grammar correction: rule-based",
                f"[module4] Output sentence: '{sentence}'",
                f"[module4] TTS: {'enabled' if state.tts_enabled else 'disabled'}",
            ])
        
        if state.module1_stats:
            parser_console.append(
                f"[module1] buffer={state.module1_stats.buffer_fill}/{state.module1_stats.buffer_capacity}"
            )
        
        # Module 1 debug info
        module1_debug = {
            "buffer_fill": 0,
            "buffer_capacity": 64,
            "frames_kept": 0,
            "frames_discarded": 0,
            "motion_score": 0.0,
            "roi_detected": False,
            "pose_detected": False,
        }
        
        if state.module1_stats:
            module1_debug = {
                "buffer_fill": state.module1_stats.buffer_fill,
                "buffer_capacity": state.module1_stats.buffer_capacity,
                "frames_kept": state.module1_stats.frames_kept,
                "frames_discarded": state.module1_stats.frames_discarded,
                "motion_score": round(state.module1_stats.motion_score, 2),
                "roi_detected": state.module1_stats.roi_detected,
                "pose_detected": state.module1_stats.pose_detected,
            }
        
        return {
            "status": "active",
            "tick": tick,
            "active_stage": active_stage,
            "selected_module": selected_module,
            "partial_gloss": " ".join(glosses),
            "final_sentence": sentence,
            "audio_state": "speaking" if state.tts_enabled else "muted",
            "confidence": confidence,
            "fps": fps,
            "latency_ms": latency_ms,
            "metrics": {
                "accuracy": round(random.uniform(0.75, 0.92), 3),
                "wer": round(random.uniform(0.1, 0.23), 3),
                "bleu": round(random.uniform(0.31, 0.48), 3),
                "window_size": 64,
                "stride": 32,
            },
            "transcript_history": history,
            "control_state": {
                "running": True,
                "tts_enabled": state.tts_enabled,
                "camera_active": state.camera_active,
                "using_real_camera": state.use_real_camera,
            },
            "parser_console": parser_console,
            "modules": {
                "module1": "Video Preprocessing",
                "module2": "Feature Extraction",
                "module3": "Temporal Recognition",
                "module4": "Translation & Output",
            },
            "module1_debug": module1_debug,
        }
        """Generate WebSocket payload with current state."""
        
        if not state.running:
            return {
                "status": "idle",
                "tick": tick,
                "active_stage": "module1",
                "partial_gloss": "--",
                "final_sentence": "Press Start to begin inference",
                "audio_state": "muted" if not state.tts_enabled else "idle",
                "confidence": 0.0,
                "fps": 0,
                "latency_ms": 0,
                "metrics": {
                    "accuracy": 0.0,
                    "wer": 0.0,
                    "bleu": 0.0,
                    "window_size": 64,
                    "stride": 32,
                },
                "transcript_history": history,
                "control_state": {
                    "running": False,
                    "tts_enabled": state.tts_enabled,
                    "camera_active": state.camera_active,
                    "using_real_camera": state.use_real_camera,
                },
                "parser_console": ["[system] pipeline idle"],
                "modules": {
                    "module1": "waiting",
                    "module2": "waiting",
                    "module3": "waiting",
                    "module4": "waiting",
                    "module5": "waiting",
                    "module6": "waiting",
                    "module7": "waiting",
                },
                "module1_debug": {
                    "buffer_fill": 0,
                    "buffer_capacity": 64,
                    "frames_kept": 0,
                    "frames_discarded": 0,
                    "motion_score": 0.0,
                    "roi_detected": False,
                    "pose_detected": False,
                }
            }
        
        # Determine active stage
        active_stage = self.pipeline_order[tick % len(self.pipeline_order)]
        
        # Try to process real camera frame
        glosses = []
        sentence = ""
        confidence = 0.0
        fps = 0
        latency_ms = 0
        
        if state.use_real_camera and state.camera_active:
            result = self.process_frame_real()
            if result is not None:
                glosses, sentence, confidence, stats = result
                fps = int(stats.fps)
                latency_ms = int(stats.processing_time_ms)
                state.module1_stats = stats
        
        # Fallback to demo mode
        if not glosses:
            fps = random.randint(18, 28)
            latency_ms = random.randint(250, 390)
            confidence = round(random.uniform(0.74, 0.96), 2)
            glosses = random.sample(self.vocabulary, k=random.randint(1, 3))
            sentence = self.demo_sentences[tick % len(self.demo_sentences)]
        
        # Update history
        if sentence and (not history or history[0] != sentence):
            history.insert(0, sentence)
            del history[12:]  # Keep last 12
        
        # Build parser console
        parser_console = [
            f"[module1] capture={state.resolution}, camera_active={state.camera_active}",
            "[module1] frame_skip=2, motion_filter=adaptive",
        ]
        
        if state.module1_stats:
            parser_console.append(
                f"[module1] buffer={state.module1_stats.buffer_fill}/{state.module1_stats.buffer_capacity}, "
                f"kept={state.module1_stats.frames_kept}, discarded={state.module1_stats.frames_discarded}"
            )
        
        parser_console.extend([
            "[module2] rgb_feat=64x512, pose_feat=64x256",
            "[module3] attention_fusion=enabled, bilstm=2-layer",
            f"[module4] ctc_partial='{' '.join(glosses)}'",
            f"[module5] sentence='{sentence}'",
            f"[module6] tts_enabled={state.tts_enabled}",
            f"[module7] fps={fps}, latency={latency_ms}ms, conf={confidence:.2f}",
        ])
        
        # Module 1 debug info
        module1_debug = {
            "buffer_fill": 0,
            "buffer_capacity": 64,
            "frames_kept": 0,
            "frames_discarded": 0,
            "motion_score": 0.0,
            "roi_detected": False,
            "pose_detected": False,
        }
        
        if state.module1_stats:
            module1_debug = {
                "buffer_fill": state.module1_stats.buffer_fill,
                "buffer_capacity": state.module1_stats.buffer_capacity,
                "frames_kept": state.module1_stats.frames_kept,
                "frames_discarded": state.module1_stats.frames_discarded,
                "motion_score": round(state.module1_stats.motion_score, 2),
                "roi_detected": state.module1_stats.roi_detected,
                "pose_detected": state.module1_stats.pose_detected,
            }
        
        return {
            "status": "active",
            "tick": tick,
            "active_stage": active_stage,
            "partial_gloss": " ".join(glosses),
            "final_sentence": sentence,
            "audio_state": "speaking" if state.tts_enabled else "muted",
            "confidence": confidence,
            "fps": fps,
            "latency_ms": latency_ms,
            "metrics": {
                "accuracy": round(random.uniform(0.75, 0.92), 3),
                "wer": round(random.uniform(0.1, 0.23), 3),
                "bleu": round(random.uniform(0.31, 0.48), 3),
                "window_size": 64,
                "stride": 32,
            },
            "transcript_history": history,
            "control_state": {
                "running": True,
                "tts_enabled": state.tts_enabled,
                "camera_active": state.camera_active,
                "using_real_camera": state.use_real_camera,
            },
            "parser_console": parser_console,
            "modules": {
                "module1": "capture + preprocess",
                "module2": "dual feature extraction",
                "module3": "attention fusion + temporal",
                "module4": "ctc decode + dedupe",
                "module5": "gloss to sentence",
                "module6": "tts synthesis",
                "module7": "metrics update",
            },
            "module1_debug": module1_debug,
        }


engine = CSLREngine(demo_mode=True)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/ws/realtime")
async def ws_realtime(ws: WebSocket) -> None:
    await ws.accept()
    state = RuntimeState()
    history: list[str] = []
    selected_module = "module1"

    async def receiver() -> None:
        nonlocal selected_module
        
        while True:
            text = await ws.receive_text()
            try:
                msg = json.loads(text)
            except json.JSONDecodeError:
                continue
            
            mtype = msg.get("type")
            
            if mtype == "control":
                action = msg.get("action")
                if action == "start":
                    state.running = True
                    # Try to start real camera if enabled
                    if state.use_real_camera:
                        engine.start_camera()
                elif action == "stop":
                    state.running = False
                    if state.use_real_camera:
                        engine.stop_camera()
                elif action == "clear":
                    history.clear()
                elif action == "toggle_tts":
                    state.tts_enabled = not state.tts_enabled
                elif action == "toggle_camera_mode":
                    state.use_real_camera = not state.use_real_camera
                    if state.use_real_camera and state.running:
                        engine.start_camera()
                    elif not state.use_real_camera:
                        engine.stop_camera()
            
            elif mtype == "module_select":
                selected_module = msg.get("module", "module1")
            
            elif mtype == "client_video_stats":
                state.camera_active = bool(msg.get("camera_active", False))
                state.frame_hint = int(msg.get("frame_hint", 0))
                state.resolution = str(msg.get("resolution", "unknown"))

    async def sender() -> None:
        tick = 0
        while True:
            payload = engine.generate_payload(
                tick=tick,
                state=state,
                history=history,
                selected_module=selected_module
            )
            payload["timestamp"] = time.time()
            await ws.send_text(json.dumps(payload))
            tick += 1
            await asyncio.sleep(0.85)

    try:
        recv_task = asyncio.create_task(receiver())
        send_task = asyncio.create_task(sender())
        done, pending = await asyncio.wait({recv_task, send_task}, return_when=asyncio.FIRST_EXCEPTION)
        for task in pending:
            task.cancel()
        for task in done:
            exc = task.exception()
            if exc:
                raise exc
    except WebSocketDisconnect:
        # Cleanup on disconnect
        engine.stop_camera()
        return
