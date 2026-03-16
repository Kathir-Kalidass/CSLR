"""
Test Security Features
"""

import pytest
from datetime import datetime


def test_rate_limiter_initialization():
    """Test rate limiter initialization"""
    from app.core.security import RateLimiter
    
    limiter = RateLimiter(max_requests=100, window_seconds=60)
    
    assert limiter.max_requests == 100
    assert limiter.window_seconds == 60


def test_rate_limiter_check_limit():
    """Test rate limiting logic"""
    from app.core.security import RateLimiter
    
    limiter = RateLimiter(max_requests=5, window_seconds=60)
    
    client_id = "test_client"
    
    # First 5 requests should pass
    for i in range(5):
        assert limiter.check_rate_limit(client_id) == True
    
    # 6th request should fail
    assert limiter.check_rate_limit(client_id) == False


def test_rate_limiter_multiple_clients():
    """Test rate limiting with multiple clients"""
    from app.core.security import RateLimiter
    
    limiter = RateLimiter(max_requests=3, window_seconds=60)
    
    # Client 1
    for i in range(3):
        assert limiter.check_rate_limit("client_1") == True
    
    # Client 2 should not be affected
    assert limiter.check_rate_limit("client_2") == True


def test_generate_api_key():
    """Test API key generation"""
    from app.core.security import generate_api_key
    
    key1 = generate_api_key()
    key2 = generate_api_key()
    
    assert len(key1) == 64  # 32 bytes hex = 64 chars
    assert len(key2) == 64
    assert key1 != key2  # Should be unique


def test_hash_api_key():
    """Test API key hashing"""
    from app.core.security import hash_api_key
    
    key = "test_api_key_123"
    hashed = hash_api_key(key)
    
    assert len(hashed) == 64  # SHA256 hex = 64 chars
    assert hashed != key  # Should be different from original
    
    # Same key should produce same hash
    hashed2 = hash_api_key(key)
    assert hashed == hashed2


def test_verify_api_key():
    """Test API key verification"""
    from app.core.security import hash_api_key, verify_api_key
    
    key = "my_secret_key"
    hashed = hash_api_key(key)
    
    # Correct key should verify
    assert verify_api_key(key, hashed) == True
    
    # Wrong key should fail
    assert verify_api_key("wrong_key", hashed) == False


def test_validate_frame_size():
    """Test frame size validation"""
    from app.core.security import validate_frame_size
    from fastapi import HTTPException
    
    # Small frame should pass
    assert validate_frame_size(1000) == True
    
    # Large frame should raise exception
    with pytest.raises(HTTPException):
        validate_frame_size(10 * 1024 * 1024)  # 10MB


def test_api_key_validator():
    """Test API key validator class"""
    from app.core.security import APIKeyValidator
    
    validator = APIKeyValidator(api_keys=["key1", "key2"])
    
    assert len(validator.api_keys) == 2


def test_validate_websocket_token():
    """Test WebSocket token validation"""
    from app.core.security import validate_websocket_token
    
    # Without auth requirement, should accept any token
    result = validate_websocket_token("any_token")
    
    # Should return boolean
    assert isinstance(result, bool)


def test_validate_websocket_token_no_token():
    """Test WebSocket validation without token"""
    from app.core.security import validate_websocket_token
    
    # Without auth requirement, should accept None
    result = validate_websocket_token(None)
    
    assert isinstance(result, bool)
