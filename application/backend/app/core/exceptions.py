"""
Custom Exception Handlers
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from app.core.logging import logger


class CSLRException(Exception):
    """Base exception for CSLR application"""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class ModelLoadError(CSLRException):
    """Raised when model loading fails"""
    def __init__(self, message: str):
        super().__init__(message, status_code=500)


class InferenceError(CSLRException):
    """Raised when inference fails"""
    def __init__(self, message: str):
        super().__init__(message, status_code=500)


class ValidationError(CSLRException):
    """Raised when input validation fails"""
    def __init__(self, message: str):
        super().__init__(message, status_code=400)


def register_exception_handlers(app: FastAPI):
    """Register custom exception handlers"""
    
    @app.exception_handler(CSLRException)
    async def cslr_exception_handler(request: Request, exc: CSLRException):
        logger.error(f"CSLR Exception: {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.message, "type": type(exc).__name__}
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {str(exc)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Internal server error", "detail": str(exc)}
        )
