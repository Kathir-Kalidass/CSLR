from __future__ import annotations

import io
import time

from pydantic import BaseModel

from .config import BASE_DIR
from .deps import gTTS, pyttsx3


class TTSRequest(BaseModel):
    text: str
    lang: str = "en"
    slow: bool = False


def tts_engine_status() -> str:
    if gTTS is not None:
        return "gtts"
    if pyttsx3 is not None:
        return "pyttsx3"
    return "unavailable"


def synthesize_with_gtts(text: str, lang: str, slow: bool) -> bytes | None:
    if gTTS is None:
        return None
    try:
        out = io.BytesIO()
        tts = gTTS(text=text, lang=lang, slow=slow)
        tts.write_to_fp(out)
        return out.getvalue()
    except Exception:
        return None


def synthesize_with_pyttsx3(text: str) -> bytes | None:
    if pyttsx3 is None:
        return None
    temp_path = BASE_DIR / "data" / f"tts_{int(time.time() * 1000)}.wav"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        engine_local = pyttsx3.init()
        engine_local.setProperty("rate", 168)
        engine_local.save_to_file(text, str(temp_path))
        engine_local.runAndWait()
        return temp_path.read_bytes() if temp_path.exists() else None
    except Exception:
        return None
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
