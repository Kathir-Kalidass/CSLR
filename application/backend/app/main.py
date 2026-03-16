"""
FastAPI Main Entry Point
Registers all routers, initializes models, and configures the application.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import torch

from app.core.config import settings
from app.core.logging import setup_logging, logger
from app.core.exceptions import register_exception_handlers
from app.api.routes import api_router
from app.monitoring.metrics import global_metrics
from app.monitoring.gpu_monitor import gpu_monitor
from app.services.cache_service import CacheService
from app.services.streaming_service import StreamingService
from app.services.translation_service import TranslationService
from app.services.audio_service import AudioService


# Model initialization on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for FastAPI app
    - Loads models on startup
    - Cleans up on shutdown
    """
    logger.info("🚀 Starting CSLR Backend")
    logger.info(f"Device: {settings.DEVICE}")
    logger.info(f"AMP Enabled: {settings.USE_AMP}")
    
    # Initialize GPU if available
    if settings.DEVICE == "cuda" and torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.backends.cudnn.benchmark = True
    
    # Initialize InferenceService (connects all 4 modules)
    from app.services.inference_service import InferenceService
    logger.info("🔧 Initializing InferenceService...")
    try:
        app.state.inference_service = InferenceService(
            vocab_file=settings.ISIGN_VOCAB_FILE
        )
        logger.info("✅ InferenceService initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize InferenceService: {e}")
        logger.warning("⚠️ Starting server without InferenceService")
        app.state.inference_service = None

    # Optional runtime services
    app.state.cache_service = CacheService()
    app.state.streaming_service = StreamingService()
    app.state.translation_service = TranslationService()
    app.state.audio_service = AudioService()
    app.state.metrics = global_metrics
    app.state.gpu_monitor = gpu_monitor
    
    yield
    
    # Cleanup
    logger.info("🛑 Shutting down CSLR Backend")
    audio_service = getattr(app.state, "audio_service", None)
    if audio_service is not None:
        audio_service.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="Advanced Continuous Sign Language Recognition System",
    lifespan=lifespan,
    debug=settings.DEBUG
)

# Setup logging
setup_logging()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register exception handlers
register_exception_handlers(app)

# Include API routes
app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "CSLR Backend API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": settings.DEVICE,
        "cuda_available": torch.cuda.is_available()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
