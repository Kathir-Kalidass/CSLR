from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent.parent
STATIC_DIR = BASE_DIR / "static"
FRONTEND_DIST_DIR = STATIC_DIR / "dist"
FRONTEND_DIST_INDEX = FRONTEND_DIST_DIR / "index.html"
TEMPLATE_FILE = BASE_DIR / "templates" / "index.html"
REPORT_DIAGRAMS_DIR = PROJECT_ROOT / "report_pages" / "architecture_diagram"
BEST_MODEL_PATH = Path(
    str(
        BASE_DIR.parent
        / "backend"
        / "checkpoints"
        / "isl_cslrt_experiment"
        / "checkpoints"
        / "best.pt"
    )
)


def best_model_info() -> dict[str, Any]:
    exists = BEST_MODEL_PATH.exists()
    info: dict[str, Any] = {
        "active_model": str(BEST_MODEL_PATH),
        "state": "loaded" if exists else "unavailable",
    }
    if exists:
        stat = BEST_MODEL_PATH.stat()
        info["size_mb"] = round(stat.st_size / (1024 * 1024), 2)
        info["modified_utc"] = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    return info
