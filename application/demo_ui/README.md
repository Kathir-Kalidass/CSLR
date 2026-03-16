# CSLR Demo UI (Reviewer-Ready Animated Showcase)

This UI is a FastAPI + WebSocket animated interface for showing your CSLR module updates in live demo format.

## What is included
- Live camera preview (`getUserMedia`)
- Animated pipeline bus for `M1..M7`
- Module cards with live `Input -> Process -> Output`
- Controls: `Start`, `Stop`, `Clear`, `TTS ON/OFF`
- Live metrics: `FPS`, `Latency`, `Accuracy`, `WER`, `BLEU`
- Transcript history panel
- Backend parser console stream

## Stack
- Backend: FastAPI + WebSocket
- Frontend: TailwindCSS + GSAP + Vanilla JS

## Run

```bash
cd application/demo_ui
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

Open `http://localhost:8080`.

## Demo flow
1. Start webcam in UI.
2. Click `Start` to run module updates.
3. Use `TTS` toggle and `Clear` transcript during demo.
4. Use `Stop` to pause pipeline without disconnecting camera.
