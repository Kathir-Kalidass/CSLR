from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_FILE = BASE_DIR / "templates" / "index.html"

app = FastAPI(title="CSLR Demo UI")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(TEMPLATE_FILE.read_text(encoding="utf-8"))


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "ui": "demo_ui"}
