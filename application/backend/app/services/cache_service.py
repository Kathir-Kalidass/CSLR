"""
Cache Service
Redis-based caching for inference results
"""

from typing import Optional, Any
import json
from app.core.config import settings
from app.core.logging import logger

try:
    import redis.asyncio as aioredis  # type: ignore
    REDIS_AVAILABLE = True
except ImportError:
    aioredis = None  # type: ignore
    REDIS_AVAILABLE = False
    logger.warning("Redis not available - caching disabled")


class CacheService:
    """
    Caching service for inference results
    Reduces redundant computation
    """
    
    def __init__(self):
        self.enabled = REDIS_AVAILABLE
        self.redis = None
        
        if self.enabled:
            self._init_redis()
    
    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            if aioredis:
                self.redis = aioredis.from_url(
                    f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}",
                    encoding="utf-8",
                    decode_responses=True
                )
                logger.info("Redis cache initialized")
        except Exception as e:
            logger.error(f"Redis initialization failed: {e}")
            self.enabled = False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.enabled or not self.redis:
            return None
        
        try:
            value = await self.redis.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        
        return None
    
    async def set(self, key: str, value: Any, expire: int = 3600):
        """Set value in cache with expiration"""
        if not self.enabled or not self.redis:
            return False
        
        try:
            await self.redis.setex(
                key,
                expire,
                json.dumps(value)
            )
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str):
        """Delete key from cache"""
        if not self.enabled or not self.redis:
            return False
        
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    async def clear(self):
        """Clear all cache"""
        if not self.enabled or not self.redis:
            return False
        
        try:
            await self.redis.flushdb()
            return True
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
