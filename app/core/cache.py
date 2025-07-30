import redis
from app.core.config import settings
import json
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self):
        try:
            self.redis_client = redis.from_url(settings.REDIS_URL)
            # Test connection
            self.redis_client.ping()
            logger.info("✅ Redis connection established successfully!")
        except Exception as e:
            logger.warning(f"❌ Redis not available: {e}. Using in-memory cache.")
            self.redis_client = None
            self.memory_cache = {}
    
    def get(self, key: str):
        try:
            if self.redis_client:
                value = self.redis_client.get(key)
                return json.loads(value) if value else None
            else:
                return self.memory_cache.get(key)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value, ttl: int = None):
        try:
            if self.redis_client:
                ttl = ttl or settings.CACHE_TTL
                self.redis_client.setex(key, ttl, json.dumps(value))
            else:
                self.memory_cache[key] = value
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def delete(self, key: str):
        try:
            if self.redis_client:
                self.redis_client.delete(key)
            else:
                self.memory_cache.pop(key, None)
        except Exception as e:
            logger.error(f"Cache delete error: {e}")

cache = CacheManager()
