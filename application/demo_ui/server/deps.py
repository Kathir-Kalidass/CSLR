from __future__ import annotations

try:
    from gtts import gTTS
except Exception:  # pragma: no cover - optional dependency in demo
    gTTS = None

try:
    import pyttsx3
except Exception:  # pragma: no cover - optional dependency in demo
    pyttsx3 = None

try:
    import cv2
    import mediapipe as mp
    import numpy as np
    import torch
    import torchvision.transforms as transforms
except Exception:  # pragma: no cover - optional dependency in demo
    cv2 = None
    mp = None
    np = None
    torch = None
    transforms = None

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency in demo
    YOLO = None
