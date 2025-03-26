
import json
from datetime import datetime, timedelta
import logging
from typing import Dict, Any
from replit.object_storage import Client

logger = logging.getLogger(__name__)

class CacheService:
    def __init__(self):
        self.storage_client = None
        self.connect()

    def connect(self):
        try:
            self.storage_client = Client()
            logger.info("Successfully connected to Replit Object Storage")
        except Exception as e:
            logger.error(f"Error connecting to Object Storage: {str(e)}")
            self.storage_client = None

    def set_with_ttl(self, key: str, data: dict, ttl_hours: int = 1):
        """Store data with TTL"""
        try:
            if not self.storage_client:
                self.connect()
                if not self.storage_client:
                    logger.error("No Object Storage connection available")
                    return

            # Add TTL info to the data
            cache_data = {
                'data': data,
                'expires_at': (datetime.now() + timedelta(hours=ttl_hours)).isoformat(),
                'created_at': datetime.now().isoformat()
            }
            
            # Store in Object Storage
            self.storage_client.upload_from_text(key, json.dumps(cache_data))
            
        except Exception as e:
            logger.error(f"Error setting cache for {key}: {str(e)}")

    def get(self, key: str) -> dict:
        """Get data from cache"""
        try:
            if not self.storage_client:
                self.connect()
                if not self.storage_client:
                    logger.error("No Object Storage connection available")
                    return None

            try:
                # Get data from Object Storage
                cache_data = json.loads(self.storage_client.download_from_text(key))
                
                # Check expiration
                expires_at = datetime.fromisoformat(cache_data['expires_at'])
                if datetime.now() > expires_at:
                    # Delete expired data
                    try:
                        self.storage_client.delete(key)
                    except Exception as e:
                        logger.error(f"Error deleting expired cache for {key}: {str(e)}")
                    return None
                    
                return cache_data['data']
            except Exception as e:
                logger.debug(f"Cache miss for {key}: {str(e)}")
                return None

        except Exception as e:
            logger.error(f"Error getting cache for {key}: {str(e)}")
            return None

    def is_fresh(self, key: str, max_age_hours: int = 1) -> bool:
        """Check if cached data exists and is fresh"""
        try:
            if not self.storage_client:
                logger.error("No Object Storage connection available")
                return False

            try:
                # Get data from Object Storage
                cache_data = json.loads(self.storage_client.download_from_text(key))
                
                # Check expiration
                expires_at = datetime.fromisoformat(cache_data['expires_at'])
                remaining = (expires_at - datetime.now()).total_seconds()
                
                # Consider fresh if more than 5 min remaining
                return remaining > 300
            except:
                return False

        except Exception as e:
            logger.error(f"Error checking cache freshness for {key}: {str(e)}")
            return False
