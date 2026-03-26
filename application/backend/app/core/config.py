"""
Configuration Management
Loads environment variables and provides application settings
"""

import os
from pathlib import Path
import torch

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency in training-only envs
    def load_dotenv(*args, **kwargs):
        return False

BACKEND_ROOT = Path(__file__).resolve().parents[2]

# Load environment variables from backend root first, then fallback to process cwd.
load_dotenv(BACKEND_ROOT / ".env")
load_dotenv()


class Settings:
    """Application settings loaded from environment variables"""

    BACKEND_ROOT: str = str(BACKEND_ROOT)
    
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
    ISIGN_CHECKPOINT_PATH: str = os.getenv(
        "ISIGN_CHECKPOINT_PATH",
        "checkpoints/isign_pose_only_npy/best.pt",
    )
    ISIGN_CHECKPOINT_CONFIG: str = os.getenv(
        "ISIGN_CHECKPOINT_CONFIG",
        "checkpoints/isign_pose_only_npy/train_config.json",
    )
    ISIGN_TRAINING_OUTPUT_DIR: str = os.getenv("ISIGN_TRAINING_OUTPUT_DIR", "checkpoints/isign_fast_v2")
    ISIGN_PENDING_EXT: str = os.getenv("ISIGN_PENDING_EXT", ".crdownload")
    ISIGN_STRICT_DATA_CHECK: bool = os.getenv("ISIGN_STRICT_DATA_CHECK", "true").lower() == "true"
    ISIGN_MIN_READY_POSE_FILES: int = int(os.getenv("ISIGN_MIN_READY_POSE_FILES", "100"))
    INFERENCE_DEFAULT_VOCAB_SIZE: int = int(os.getenv("INFERENCE_DEFAULT_VOCAB_SIZE", "2000"))
    
    # Model Paths
    RGB_MODEL_PATH: str = os.getenv("RGB_MODEL_PATH", "app/models/checkpoints/rgb_model.pt")
    POSE_MODEL_PATH: str = os.getenv("POSE_MODEL_PATH", "app/models/checkpoints/pose_model.pt")
    FUSION_MODEL_PATH: str = os.getenv("FUSION_MODEL_PATH", "app/models/checkpoints/fusion_model.pt")
    SEQUENCE_MODEL_PATH: str = os.getenv("SEQUENCE_MODEL_PATH", "app/models/checkpoints/sequence_model.pt")
    TRANSLATION_MODEL_PATH: str = os.getenv("TRANSLATION_MODEL_PATH", "app/models/checkpoints/t5_model")
    
    # Inference Settings
    CLIP_LENGTH: int = int(os.getenv("CLIP_LENGTH", "48"))
    FPS_TARGET: int = int(os.getenv("FPS_TARGET", "30"))
    BEAM_WIDTH: int = int(os.getenv("BEAM_WIDTH", "5"))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
    MIN_VALID_FRAMES: int = int(os.getenv("MIN_VALID_FRAMES", "8"))
    ENABLE_GLOSS_FILTER: bool = os.getenv("ENABLE_GLOSS_FILTER", "true").lower() == "true"
    GLOSS_MAX_TOKENS: int = int(os.getenv("GLOSS_MAX_TOKENS", "20"))
    GLOSS_BLOCKLIST: str = os.getenv("GLOSS_BLOCKLIST", "<blank>,<unk>,<pad>,sil,noise")
    ENABLE_TEMPORAL_GLOSS_VOTING: bool = os.getenv("ENABLE_TEMPORAL_GLOSS_VOTING", "true").lower() == "true"
    GLOSS_HISTORY_SIZE: int = int(os.getenv("GLOSS_HISTORY_SIZE", "20"))
    GLOSS_VOTE_WINDOW: int = int(os.getenv("GLOSS_VOTE_WINDOW", "5"))
    GLOSS_MIN_VOTES: int = int(os.getenv("GLOSS_MIN_VOTES", "2"))
    CTC_MIN_TOKEN_RUN: int = int(os.getenv("CTC_MIN_TOKEN_RUN", "2"))
    CTC_MIN_TOKEN_MARGIN: float = float(os.getenv("CTC_MIN_TOKEN_MARGIN", "0.04"))
    CTC_LENGTH_NORM_ALPHA: float = float(os.getenv("CTC_LENGTH_NORM_ALPHA", "0.35"))
    CTC_REPETITION_PENALTY: float = float(os.getenv("CTC_REPETITION_PENALTY", "0.15"))
    CTC_LM_PATH: str = os.getenv("CTC_LM_PATH", "")
    CTC_LM_WEIGHT: float = float(os.getenv("CTC_LM_WEIGHT", "0.0"))
    CTC_LM_TOKEN_BONUS: float = float(os.getenv("CTC_LM_TOKEN_BONUS", "0.0"))
    CTC_LM_CANDIDATES: int = int(os.getenv("CTC_LM_CANDIDATES", "20"))
    ENABLE_ADAPTIVE_FILTERING: bool = os.getenv("ENABLE_ADAPTIVE_FILTERING", "true").lower() == "true"
    ADAPTIVE_STRICTNESS_STEP_UP: float = float(os.getenv("ADAPTIVE_STRICTNESS_STEP_UP", "0.12"))
    ADAPTIVE_STRICTNESS_STEP_DOWN: float = float(os.getenv("ADAPTIVE_STRICTNESS_STEP_DOWN", "0.08"))
    ADAPTIVE_NOISE_VAR_THRESHOLD: float = float(os.getenv("ADAPTIVE_NOISE_VAR_THRESHOLD", "0.03"))
    ADAPTIVE_CONF_HIGH: float = float(os.getenv("ADAPTIVE_CONF_HIGH", "0.85"))
    ADAPTIVE_CONF_LOW: float = float(os.getenv("ADAPTIVE_CONF_LOW", "0.55"))
    ADAPTIVE_THRESHOLD_BOOST_MAX: float = float(os.getenv("ADAPTIVE_THRESHOLD_BOOST_MAX", "0.18"))
    ADAPTIVE_MAXTOK_REDUCTION: float = float(os.getenv("ADAPTIVE_MAXTOK_REDUCTION", "0.40"))
    ADAPTIVE_VOTES_BONUS_MAX: int = int(os.getenv("ADAPTIVE_VOTES_BONUS_MAX", "2"))
    ENABLE_ENSEMBLE_DECODE: bool = os.getenv("ENABLE_ENSEMBLE_DECODE", "true").lower() == "true"
    ENSEMBLE_MIN_AGREEMENT: float = float(os.getenv("ENSEMBLE_MIN_AGREEMENT", "0.55"))
    ENABLE_CONFIDENCE_CALIBRATION: bool = os.getenv("ENABLE_CONFIDENCE_CALIBRATION", "true").lower() == "true"
    CONFIDENCE_TEMPERATURE: float = float(os.getenv("CONFIDENCE_TEMPERATURE", "1.0"))
    CONFIDENCE_CALIBRATION_FILE: str = os.getenv("CONFIDENCE_CALIBRATION_FILE", "checkpoints/isign_fast_v2/confidence_calibration.json")
    
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
