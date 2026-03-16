"""
API Router
Combines all API endpoints
"""

from fastapi import APIRouter
from app.api import health, inference, websocket
from app.api.endpoints import training

api_router = APIRouter()

# Include sub-routers
api_router.include_router(health.router, prefix="/health", tags=["Health"])
api_router.include_router(inference.router, prefix="/inference", tags=["Inference"])
api_router.include_router(websocket.router, prefix="/ws", tags=["WebSocket"])
api_router.include_router(training.router, tags=["Training"])
