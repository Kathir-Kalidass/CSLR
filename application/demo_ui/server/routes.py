from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from .config import FRONTEND_DIST_INDEX, REPORT_DIAGRAMS_DIR, TEMPLATE_FILE, best_model_info
from .runtime import REAL_RUNTIME
from .tts_service import (
    TTSRequest,
    synthesize_with_gtts,
    synthesize_with_pyttsx3,
    tts_engine_status,
)

router = APIRouter()


def _render_page(page_mode: str) -> str:
    if FRONTEND_DIST_INDEX.exists():
        return FRONTEND_DIST_INDEX.read_text(encoding="utf-8")
    if TEMPLATE_FILE.exists():
        # Keep legacy template file in repo, but do not serve static scripted UI.
        return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>CSLR Demo UI</title>
    <style>
      body { font-family: Arial, sans-serif; background: #0b1120; color: #e2e8f0; padding: 2rem; }
      code { background: #0f172a; padding: .2rem .4rem; border-radius: 6px; }
    </style>
  </head>
  <body>
    <h2>Frontend build not found</h2>
    <p>Build the React frontend before opening this page:</p>
    <pre><code>cd application/demo_ui/frontend
npm install
npm run build</code></pre>
    <p>Then refresh this page.</p>
  </body>
</html>
""".strip()
    return "<h2>Frontend build missing</h2>"


@router.get("/api/system")
async def system_info() -> dict[str, Any]:
    return {
        "model_info": best_model_info(),
        "tts_engine": tts_engine_status(),
        "inference_mode": "real" if REAL_RUNTIME.available else "simulated",
        "runtime_status": REAL_RUNTIME.reason,
        "runtime_error": REAL_RUNTIME.error_detail,
        "runtime_window_size": REAL_RUNTIME.window_size,
        "runtime_stride": REAL_RUNTIME.stride,
        "report_diagrams_available": REPORT_DIAGRAMS_DIR.exists(),
    }


@router.post("/api/tts")
async def tts_endpoint(payload: TTSRequest) -> Response:
    sentence = payload.text.strip()
    if not sentence:
        raise HTTPException(status_code=400, detail="Text is required for TTS")

    audio_bytes = synthesize_with_gtts(sentence, payload.lang, payload.slow)
    media_type = "audio/mpeg"

    if audio_bytes is None:
        audio_bytes = synthesize_with_pyttsx3(sentence)
        media_type = "audio/wav"

    if audio_bytes is None:
        raise HTTPException(
            status_code=503,
            detail="No TTS engine available. Install 'gTTS' or 'pyttsx3' in demo_ui requirements.",
        )

    return Response(content=audio_bytes, media_type=media_type)


@router.get("/", response_class=HTMLResponse)
def index() -> RedirectResponse:
    return RedirectResponse(url="/live", status_code=307)


@router.get("/live", response_class=HTMLResponse)
def live_page() -> str:
    return _render_page("live")


@router.get("/flow", response_class=HTMLResponse)
def flow_page() -> str:
    return _render_page("flow")


@router.get("/insights", response_class=HTMLResponse)
def insights_page() -> str:
    return _render_page("insights")


@router.get("/components", response_class=HTMLResponse)
def components_page() -> str:
    return _render_page("flow")
