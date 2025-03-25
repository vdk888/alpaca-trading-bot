
import redis
import json
from datetime import datetime, timedelta
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class CacheService:
    def __init__(self):
        self.redis_client = None
        self.memory_cache: Dict[str, Any] = {}
        self.memory_ttl: Dict[str, datetime] = {}
        self.connect()

    def connect(self):
        try:
            self.redis_client = redis.Redis(
                host='0.0.0.0',
                port=6379,
                db=0,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            # Enable AOF persistence
            self.redis_client.config_set('appendonly', 'yes')
            self.redis_client.config_set('appendfsync', 'everysec')
        except redis.ConnectionError:
            logging.warning("Redis connection failed - falling back to in-memory cache")
            self.redis_client = None
        
    def set_with_ttl(self, key: str, data: dict, ttl_hours: int = 1):
        """Store data with TTL"""
        try:
            if self.redis_client:
                self.redis_client.setex(
                    key,
                    timedelta(hours=ttl_hours),
                    json.dumps(data)
                )
            else:
                # Fallback to memory cache
                self.memory_cache[key] = json.dumps(data)
                self.memory_ttl[key] = datetime.now() + timedelta(hours=ttl_hours)
        except Exception as e:
            logger.error(f"Error setting cache for {key}: {str(e)}")
            
    def get(self, key: str) -> dict:
        """Get data from cache"""
        try:
            if self.redis_client:
                data = self.redis_client.get(key)
                return json.loads(data) if data else None
            else:
                # Fallback to memory cache
                if key in self.memory_cache and datetime.now() < self.memory_ttl[key]:
                    return json.loads(self.memory_cache[key])
                return None
        except Exception as e:
            logger.error(f"Error getting cache for {key}: {str(e)}")
            return None
            
    def is_fresh(self, key: str, max_age_hours: int = 1) -> bool:
        """Check if cached data exists and is fresh"""
        try:
            if self.redis_client:
                ttl = self.redis_client.ttl(key)
                return ttl > 0 and ttl > (max_age_hours * 3600 - 300)  # 5 min buffer
            else:
                # Fallback to memory cache
                if key in self.memory_ttl:
                    remaining = (self.memory_ttl[key] - datetime.now()).total_seconds()
                    return remaining > 0 and remaining > (max_age_hours * 3600 - 300)
                return False
        except Exception as e:
            logger.error(f"Error checking cache freshness for {key}: {str(e)}")
            return False
