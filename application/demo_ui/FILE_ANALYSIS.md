# Demo UI File Analysis and Modular Split

This document maps every major file in `application/demo_ui` to its responsibility and the new split boundaries.

## 1) Entrypoints

- `app.py`
  - Thin runtime entrypoint.
  - Imports `create_app()` from server package and runs uvicorn in local mode.

- `templates/index.html`
  - Single-page app shell.
  - Loads Tailwind, GSAP, modular JS entrypoint, and modular stylesheet loader.

## 2) Backend (`server/`) Analysis

### App composition

- `server/__init__.py`
  - Exposes app factory.

- `server/app_factory.py`
  - Creates FastAPI app.
  - Mounts static assets and report diagrams.
  - Registers HTTP routes and websocket routes.

- `server/config.py`
  - Central paths/constants (`BASE_DIR`, `PROJECT_ROOT`, `BEST_MODEL_PATH`, etc.).
  - Model metadata helper (`best_model_info`).

- `server/deps.py`
  - Optional dependency loading in one place (`gTTS`, `pyttsx3`, `cv2`, `torch`, `YOLO`, etc.).

### API + websocket routing

- `server/routes.py`
  - HTTP endpoints:
  - `/`
  - `/components`
  - `/api/system`
  - `/api/tts`

- `server/websocket.py`
  - WebSocket endpoint `/ws/demo`.
  - Maintains per-client state.
  - Handles control and webcam frame packets.

### TTS service

- `server/tts_service.py`
  - `TTSRequest` schema.
  - Runtime TTS engine status.
  - gTTS + pyttsx3 synthesis helpers.

### Real-time inference split

- `server/runtime.py`
  - Public runtime class `RealtimeModelRuntime`.
  - Sliding-window buffer + `process_frame` inference path.
  - Exposes singleton `REAL_RUNTIME`.

- `server/runtime_loader.py`
  - Checkpoint/model initialization logic.
  - Restores RGB stream, pose stream, fusion, temporal modules.

- `server/runtime_pose.py`
  - Pose extraction helper (YOLO pose, MediaPipe fallback, normalization).

- `server/runtime_helpers.py`
  - Reusable runtime helpers:
  - state slicing,
  - temporal layer inference,
  - vocab/token mapping,
  - CTC decode,
  - gloss-to-sentence conversion.

### Demo engine split

- `server/demo_engine.py`
  - Compatibility re-export.
  - Keeps old import path stable.

- `server/engine/__init__.py`
  - Engine exports.

- `server/engine/core.py`
  - `DemoEngine` orchestration.
  - Decides idle vs running payload generation.
  - Stream loop entry.

- `server/engine/constants.py`
  - Static tokens, sample sentences, stage order, boot sequence.

- `server/engine/idle_payload.py`
  - Idle payload builder.

- `server/engine/running_modules.py`
  - Running payload blocks for each live processing step in the flow.

- `server/engine/overrides.py`
  - Real inference override logic for payload fields.

- `server/engine/running_payload.py`
  - Running payload builder orchestration.

- `server/engine/payloads.py`
  - Thin export surface for idle/running payload builders.

- `server/engine/frame_ingest.py`
  - Decodes browser JPEG frames and triggers runtime inference.

- `server/engine/stream_loop.py`
  - Async websocket streaming loop.

## 3) Frontend (`static/`) Analysis

### Entry and lifecycle split

- `static/app.js`
  - Thin frontend entrypoint.
  - Builds app context and starts runtime lifecycle.

- `static/core/app-context.js`
  - Construction layer.
  - Creates state, shell/layout, refs, controllers, and action bridges.

- `static/core/app-runtime.js`
  - Runtime layer.
  - Payload handling, system info refresh, toggles, socket lifecycle, init/dispose.

### Core state + action modules

- `static/core/state.js`
  - Global UI runtime state.

- `static/core/constants.js`
  - Stage order/labels/narration text + boot defaults.

- `static/core/control-actions.js`
  - WS control packet senders (`start`, `stop`, `clear`, `toggle_tts`, etc.).

- `static/core/event-bindings.js`
  - UI event wiring for buttons, module popups, and demo helper actions.

- `static/core/auto-demo.js`
  - Automated presenter/demo sequence scheduler.

- `static/core/utils.js`
  - Shared helpers (`now`, `escapeHtml`).

### UI components

- `static/components/layout.js`
  - DOM shell and major layout rendering.
  - Stage cards, module cards, gauges, wave, timeline.

- `static/components/pipeline-ui.js`
  - Applies backend payload to UI.
  - Updates chips, controls, HUD, flow, modules, metrics, history, parser console.

- `static/components/camera.js`
  - Webcam lifecycle, overlay rendering, skeleton simulation, frame streaming to backend.

- `static/components/animations.js`
  - Flow motion, boot animation, particle background, wave animations.

- `static/components/popup.js`
  - Popup rendering and animation.

### Service modules

- `static/services/socket.js`
  - WebSocket connection/reconnect handling.

- `static/services/system.js`
  - Fetches `/api/system`.

- `static/services/tts.js`
  - Calls `/api/tts` and plays generated audio.

## 4) Styles Analysis

- `static/style.css`
  - CSS entrypoint and import index.
  - Loads split CSS modules instead of a single 900+ line file.

- `static/styles/*.css`
  - Split by concern (tokens/base, camera, pipeline, popup/effects, responsive).
  - Reduces single-file styling complexity and improves maintainability.

## 5) Net Split Outcome

The largest mixed-responsibility files were broken into dedicated modules:

- Backend:
  - `runtime.py` -> runtime core + loader + pose + helpers.
  - `demo_engine.py` -> `engine/` package (payloads, modules, overrides, stream).

- Frontend:
  - `app.js` -> context builder + runtime lifecycle modules.
  - `style.css` -> modular style imports.

This gives clearer ownership and safer future changes without repeatedly editing one giant file.
