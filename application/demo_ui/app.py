from __future__ import annotations

try:
    from .server import create_app  # type: ignore
except Exception:
    from server import create_app

app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
