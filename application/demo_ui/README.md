# CSLR Demo UI (React + Vite + FastAPI)

This demo provides a full React frontend with live websocket inference updates from FastAPI.

## What is included
- Multi-page UI: `Live`, `Flow`, `Insights`
- Live webcam capture + frame streaming to backend
- Real-time text output from model inference (no static preloaded answers)
- RGB and Pose preview panes in camera view
- Live flow cards showing `Input -> Process -> Output`
- Live metrics, transcript history, and parser console
- Optional TTS playback via backend `/api/tts`

## Stack
- Backend: FastAPI + WebSocket
- Frontend: React + Vite + Tailwind + Framer Motion + Zustand

## Backend Run

```bash
cd application/demo_ui
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

## Frontend Dev Run

```bash
cd application/demo_ui/frontend
# use a Linux npm/node inside WSL for best compatibility
npm install
npm run dev
```

Open `http://localhost:3000`.

## Frontend Build for FastAPI Serving

```bash
cd application/demo_ui/frontend
npm run build
```

This writes build assets to `application/demo_ui/static/dist`, and FastAPI routes (`/live`, `/flow`, `/insights`) will serve the built React app.
