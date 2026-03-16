"""
Configuration Management
Loads environment variables and provides application settings
"""

import os
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    """Application settings loaded from environment variables"""
    
    # Server Configuration
    APP_NAME: str = os.getenv("APP_NAME", "CSLR_Backend")
    APP_ENV: str = os.getenv("APP_ENV", "development")
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # WebSocket Configuration
    WS_ROUTE: str = os.getenv("WS_ROUTE", "/ws/inference")
    MAX_CLIENTS: int = int(os.getenv("MAX_CLIENTS", "5"))
    
    # GPU/Device Configuration
    _requested_device: str = os.getenv("DEVICE", "cuda")
    DEVICE: str = _requested_device if (_requested_device != "cuda" or torch.cuda.is_available()) else "cpu"
    USE_AMP: bool = os.getenv("USE_AMP", "true").lower() == "true"
    TORCH_THREADS: int = int(os.getenv("TORCH_THREADS", "4"))

    # iSign dataset paths
    ISIGN_DATA_DIR: str = os.getenv("ISIGN_DATA_DIR", "dataset/isign")
    ISIGN_PROCESSED_DIR: str = os.getenv("ISIGN_PROCESSED_DIR", "dataset/isign_processed")
    ISIGN_VOCAB_FILE: str = os.getenv("ISIGN_VOCAB_FILE", f"{ISIGN_PROCESSED_DIR}/vocab.json")
    ISIGN_CHECKPOINT_PATH: str = os.getenv("ISIGN_CHECKPOINT_PATH", "checkpoints/isign/best_model.pt")
    INFERENCE_DEFAULT_VOCAB_SIZE: int = int(os.getenv("INFERENCE_DEFAULT_VOCAB_SIZE", "2000"))
    
    # Model Paths
    RGB_MODEL_PATH: str = os.getenv("RGB_MODEL_PATH", "app/models/checkpoints/rgb_model.pt")
    POSE_MODEL_PATH: str = os.getenv("POSE_MODEL_PATH", "app/models/checkpoints/pose_model.pt")
    FUSION_MODEL_PATH: str = os.getenv("FUSION_MODEL_PATH", "app/models/checkpoints/fusion_model.pt")
    SEQUENCE_MODEL_PATH: str = os.getenv("SEQUENCE_MODEL_PATH", "app/models/checkpoints/sequence_model.pt")
    TRANSLATION_MODEL_PATH: str = os.getenv("TRANSLATION_MODEL_PATH", "app/models/checkpoints/t5_model")
    
    # Inference Settings
    CLIP_LENGTH: int = int(os.getenv("CLIP_LENGTH", "32"))
    FPS_TARGET: int = int(os.getenv("FPS_TARGET", "30"))
    BEAM_WIDTH: int = int(os.getenv("BEAM_WIDTH", "5"))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
    
    # Redis Configuration (Optional)
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    ENABLE_CACHE: bool = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    USE_TRANSLATION_SERVICE: bool = os.getenv("USE_TRANSLATION_SERVICE", "false").lower() == "true"
    
    # Monitoring
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    ENABLE_PROFILER: bool = os.getenv("ENABLE_PROFILER", "true").lower() == "true"
    
    # TTS Configuration
    TTS_ENGINE: str = os.getenv("TTS_ENGINE", "pyttsx3")
    TTS_RATE: int = int(os.getenv("TTS_RATE", "160"))
    TTS_VOLUME: float = float(os.getenv("TTS_VOLUME", "1.0"))
    AUTO_TTS_INFERENCE: bool = os.getenv("AUTO_TTS_INFERENCE", "false").lower() == "true"
    AUTO_TTS_STREAM: bool = os.getenv("AUTO_TTS_STREAM", "true").lower() == "true"


# Global settings instance
settings = Settings()
