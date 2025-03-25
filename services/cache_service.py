
import redis
import json
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class CacheService:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        
    def set_with_ttl(self, key: str, data: dict, ttl_hours: int = 1):
        """Store data with TTL"""
        try:
            self.redis_client.setex(
                key,
                timedelta(hours=ttl_hours),
                json.dumps(data)
            )
        except Exception as e:
            logger.error(f"Error setting cache for {key}: {str(e)}")
            
    def get(self, key: str) -> dict:
        """Get data from cache"""
        try:
            data = self.redis_client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Error getting cache for {key}: {str(e)}")
            return None
            
    def is_fresh(self, key: str, max_age_hours: int = 1) -> bool:
        """Check if cached data exists and is fresh"""
        try:
            ttl = self.redis_client.ttl(key)
            return ttl > 0 and ttl > (max_age_hours * 3600 - 300)  # 5 min buffer
        except Exception as e:
            logger.error(f"Error checking cache freshness for {key}: {str(e)}")
            return False
