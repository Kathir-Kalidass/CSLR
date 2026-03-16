"""
Inference Endpoints
HTTP endpoints for batch inference
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import time

import cv2
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.core.config import settings
from app.core.logging import logger
from app.schemas.inference_schema import InferenceRequest, InferenceResponse
from app.utils.image_utils import decode_base64_image

router = APIRouter()


@router.post("/video", response_model=InferenceResponse)
async def infer_video(
    file: UploadFile = File(...),
    request: Request | None = None,
    speak: bool = False,
):
    """
    Process an uploaded video file and return recognition results.
    """
    start_time = time.time()
    temp_path = None

    try:
        service = getattr(request.app.state, "inference_service", None) if request else None
        if service is None:
            raise HTTPException(status_code=503, detail="InferenceService not initialized")

        logger.info(f"Processing video: {file.filename}")

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(file.filename or "upload.mp4")[1] or ".mp4",
        ) as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name

        result = await service.process_video(temp_path)
        audio_service = getattr(request.app.state, "audio_service", None) if request else None
        should_speak = settings.AUTO_TTS_INFERENCE or speak
        sentence = result.get("sentence", "")
        if should_speak and audio_service and sentence:
            await audio_service.synthesize(sentence)

        processing_time = time.time() - start_time

        return InferenceResponse(
            gloss=result.get("gloss", []),
            sentence=result.get("sentence", ""),
            confidence=result.get("confidence", 0.0),
            fps=result.get("fps", 0.0),
            processing_time=processing_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Video inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


@router.post("/frames", response_model=InferenceResponse)
async def infer_frames(
    inference_request: InferenceRequest,
    request: Request | None = None,
    speak: bool = False,
):
    """
    Process a sequence of base64-encoded frames.
    """
    start_time = time.time()

    try:
        service = getattr(request.app.state, "inference_service", None) if request else None
        if service is None:
            raise HTTPException(status_code=503, detail="InferenceService not initialized")

        cache_service = getattr(request.app.state, "cache_service", None) if request else None
        cache_key = None

        logger.info(f"Processing {len(inference_request.frames)} frames")

        if (
            cache_service
            and getattr(cache_service, "enabled", False)
            and settings.ENABLE_CACHE
        ):
            hasher = hashlib.sha256()
            hasher.update(str(inference_request.fps or 0.0).encode("utf-8"))
            for frame_b64 in inference_request.frames:
                hasher.update(frame_b64.encode("utf-8"))
            cache_key = f"frames:{hasher.hexdigest()}"
            cached = await cache_service.get(cache_key)
            if cached:
                processing_time = time.time() - start_time
                return InferenceResponse(
                    gloss=cached.get("gloss", []),
                    sentence=cached.get("sentence", ""),
                    confidence=cached.get("confidence", 0.0),
                    fps=cached.get("fps", 0.0),
                    processing_time=processing_time,
                )

        frames = []
        for frame_b64 in inference_request.frames:
            try:
                frame_rgb = decode_base64_image(frame_b64)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                frames.append(frame_bgr)
            except Exception:
                continue

        if not frames:
            raise HTTPException(status_code=400, detail="No valid frames decoded")

        result = await service.process_frames(frames)
        audio_service = getattr(request.app.state, "audio_service", None) if request else None
        should_speak = settings.AUTO_TTS_INFERENCE or speak
        sentence = result.get("sentence", "")
        if should_speak and audio_service and sentence:
            await audio_service.synthesize(sentence)

        processing_time = time.time() - start_time

        response = InferenceResponse(
            gloss=result.get("gloss", []),
            sentence=result.get("sentence", ""),
            confidence=result.get("confidence", 0.0),
            fps=len(frames) / processing_time if processing_time > 0 else 0.0,
            processing_time=processing_time,
        )

        if (
            cache_key
            and cache_service
            and getattr(cache_service, "enabled", False)
            and settings.ENABLE_CACHE
        ):
            await cache_service.set(
                cache_key,
                {
                    "gloss": response.gloss,
                    "sentence": response.sentence,
                    "confidence": response.confidence,
                    "fps": response.fps,
                },
                expire=settings.CACHE_TTL_SECONDS,
            )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Frame inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
