#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Caching System with Redis and In-Memory Fallback
نظام التخزين المؤقت مع Redis وبديل في الذاكرة
"""

import json
import hashlib
import logging
from typing import Any, Optional, Callable
from datetime import timedelta
from functools import wraps
import os

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Unified cache manager supporting Redis and in-memory caching
    """

    def __init__(self):
        self.redis_client = None
        self.memory_cache = {}
        self.cache_stats = {"hits": 0, "misses": 0, "errors": 0}

        # Try to connect to Redis
        try:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
            import redis
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info(f"✅ Connected to Redis at {redis_url}")
        except Exception as e:
            logger.warning(f"⚠️  Redis not available: {e}. Using in-memory cache.")

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        try:
            # Try Redis first
            if self.redis_client:
                value = self.redis_client.get(key)
                if value:
                    self.cache_stats["hits"] += 1
                    return json.loads(value)

            # Fallback to memory cache
            if key in self.memory_cache:
                self.cache_stats["hits"] += 1
                return self.memory_cache[key]

            self.cache_stats["misses"] += 1
            return None

        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            self.cache_stats["errors"] += 1
            return None

    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """
        Set value in cache

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (default 1 hour)

        Returns:
            True if successful
        """
        try:
            serialized = json.dumps(value)

            # Try Redis first
            if self.redis_client:
                self.redis_client.setex(key, ttl, serialized)
                return True

            # Fallback to memory cache
            self.memory_cache[key] = value

            # Simple TTL implementation for memory cache
            # (In production, use a proper TTL implementation)
            if len(self.memory_cache) > 1000:  # Limit size
                # Remove oldest 100 items
                keys_to_remove = list(self.memory_cache.keys())[:100]
                for k in keys_to_remove:
                    del self.memory_cache[k]

            return True

        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            self.cache_stats["errors"] += 1
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            if self.redis_client:
                self.redis_client.delete(key)

            if key in self.memory_cache:
                del self.memory_cache[key]

            return True
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False

    def clear(self) -> bool:
        """Clear all cache"""
        try:
            if self.redis_client:
                self.redis_client.flushdb()

            self.memory_cache.clear()
            logger.info("✅ Cache cleared")
            return True
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False

    def get_stats(self) -> dict:
        """Get cache statistics"""
        total = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total * 100) if total > 0 else 0

        return {
            **self.cache_stats,
            "hit_rate": round(hit_rate, 2),
            "backend": "redis" if self.redis_client else "memory",
            "size": len(self.memory_cache) if not self.redis_client else "unknown"
        }


# Global cache instance
_cache_manager = None


def get_cache() -> CacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def cache_result(ttl: int = 3600, key_prefix: str = ""):
    """
    Decorator to cache function results

    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache key

    Example:
        @cache_result(ttl=600, key_prefix="search")
        def search_knowledge(query: str):
            # Expensive operation
            return results
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()

            # Create cache key from function name and arguments
            key_parts = [key_prefix, func.__name__]

            # Add args
            for arg in args:
                if isinstance(arg, (str, int, float, bool)):
                    key_parts.append(str(arg))

            # Add kwargs
            for k, v in sorted(kwargs.items()):
                if isinstance(v, (str, int, float, bool)):
                    key_parts.append(f"{k}={v}")

            # Create hash of key
            key_str = ":".join(key_parts)
            cache_key = hashlib.md5(key_str.encode()).hexdigest()

            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_value

            # Execute function
            result = func(*args, **kwargs)

            # Store in cache
            cache.set(cache_key, result, ttl)

            return result

        return wrapper
    return decorator
