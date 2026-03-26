"""
Dependencies for API endpoints
Dependency injection for FastAPI routes
"""

from typing import Generator, Optional
from fastapi import Depends, HTTPException, status, Header
from app.core.config import settings
from app.core.security import RateLimiter, APIKeyValidator
from app.core.logging import logger

# Initialize rate limiter
rate_limiter = RateLimiter(max_requests=100, window_seconds=60)

# Initialize API key validator
api_keys = getattr(settings, 'API_KEYS', [])
api_key_validator = APIKeyValidator(api_keys=api_keys)


def get_rate_limiter() -> RateLimiter:
    """Dependency to get rate limiter"""
    return rate_limiter


def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """
    Verify API key from header
    
    Args:
        x_api_key: API key from X-API-Key header
    
    Returns:
        True if valid or auth not required
    
    Raises:
        HTTPException: If auth required and key invalid
    """
    # Check if authentication is required
    require_auth = getattr(settings, 'REQUIRE_AUTH', False)
    
    if not require_auth:
        return True
    
    # Use validator
    return api_key_validator(x_api_key)
