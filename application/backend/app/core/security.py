"""
Security utilities
Rate limiting, authentication, JWT validation
"""

from typing import Optional
from fastapi import HTTPException, status, Header
from datetime import datetime, timedelta
import secrets
import hashlib
from app.core.config import settings
from app.core.logging import logger


class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}
    
    def check_rate_limit(self, client_id: str) -> bool:
        """Check if client has exceeded rate limit"""
        now = datetime.now()
        
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Remove old requests outside window
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if now - req_time < timedelta(seconds=self.window_seconds)
        ]
        
        # Check limit
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[client_id].append(now)
        return True


def validate_websocket_token(token: Optional[str]) -> bool:
    """
    Validate WebSocket connection token
    
    Args:
        token: JWT token or API key
    
    Returns:
        True if valid, False otherwise
    """
    # If no token provided and auth not required, allow
    auth_required = getattr(settings, 'REQUIRE_AUTH', False)
    if not auth_required:
        return True
    
    if not token:
        logger.warning("No token provided for WebSocket connection")
        return False
    
    # Check if token is valid API key
    valid_api_keys = getattr(settings, 'API_KEYS', [])
    if token in valid_api_keys:
        return True
    
    # Could add JWT validation here
    # try:
    #     from jose import jwt
    #     payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
    #     return True
    # except:
    #     return False
    
    logger.warning(f"Invalid token: {token[:10]}...")
    return False


def validate_frame_size(frame_bytes: int, max_size: int = 5 * 1024 * 1024) -> bool:
    """Validate incoming frame size"""
    if frame_bytes > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Frame size {frame_bytes} exceeds maximum {max_size}"
        )
    return True


def generate_api_key() -> str:
    """
    Generate a secure API key
    
    Returns:
        32-character hex API key
    """
    return secrets.token_hex(32)


def hash_api_key(api_key: str) -> str:
    """
    Hash API key for storage
    
    Args:
        api_key: Plain text API key
    
    Returns:
        SHA256 hash of key
    """
    return hashlib.sha256(api_key.encode()).hexdigest()


def verify_api_key(api_key: str, stored_hash: str) -> bool:
    """
    Verify API key against stored hash
    
    Args:
        api_key: Plain text API key from request
        stored_hash: Stored hash to compare against
    
    Returns:
        True if valid
    """
    return hash_api_key(api_key) == stored_hash


class APIKeyValidator:
    """API Key validation middleware"""
    
    def __init__(self, api_keys: list[str] = None):
        self.api_keys = api_keys or []
        self.hashed_keys = {hash_api_key(key): key for key in self.api_keys}
    
    async def __call__(self, x_api_key: Optional[str] = Header(None)):
        """
        Validate API key from header
        
        Args:
            x_api_key: API key from X-API-Key header
        
        Raises:
            HTTPException: If key is invalid
        """
        # If no keys configured, skip validation
        if not self.api_keys:
            return True
        
        if not x_api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required. Provide X-API-Key header."
            )
        
        # Check if key is valid
        if x_api_key not in self.api_keys:
            logger.warning(f"Invalid API key attempt: {x_api_key[:10]}...")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid API key"
            )
        
        return True
