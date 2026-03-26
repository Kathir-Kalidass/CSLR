from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .config import REPORT_DIAGRAMS_DIR, STATIC_DIR
from .routes import router as http_router
from .websocket import router as ws_router


def create_app() -> FastAPI:
    app = FastAPI(title="CSLR Pipeline Demo UI")
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    if REPORT_DIAGRAMS_DIR.exists():
        app.mount("/report-diagrams", StaticFiles(directory=str(REPORT_DIAGRAMS_DIR)), name="report-diagrams")

    app.include_router(http_router)
    app.include_router(ws_router)
    return app
