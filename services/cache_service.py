import json
import os
import re
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, Optional
import pandas as pd

# Attempt to import Replit client, but don't fail if it's not there
try:
    from replit.object_storage import Client
    REPLIT_AVAILABLE = True
except ImportError:
    Client = None  # Define Client as None if import fails
    REPLIT_AVAILABLE = False
    logging.warning("Replit Object Storage client not found. Using local file cache fallback.")


logger = logging.getLogger(__name__)

class CacheService:
    def __init__(self, local_cache_dir: str = "data/cache/local_cache"):
        self.storage_client: Optional[Client] = None
        self.is_replit = 'REPL_ID' in os.environ and REPLIT_AVAILABLE
        self.local_cache_path = Path(local_cache_dir)
        self.connect()

    def _ensure_local_cache_dir(self):
        """Ensures the local cache directory exists."""
        try:
            self.local_cache_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating local cache directory {self.local_cache_path}: {e}")

    def _sanitize_key(self, key: str) -> str:
        """Sanitizes a key to be used as a filename."""
        # Remove potentially problematic characters for filenames
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', key)
        # Limit length if necessary (though less common for cache keys)
        return sanitized + ".json"

    def _get_local_cache_filepath(self, key: str) -> Path:
        """Gets the full path for a local cache file."""
        return self.local_cache_path / self._sanitize_key(key)

    def connect(self):
        """Connects to Replit storage if available, otherwise prepares local cache."""
        if self.is_replit and Client:
            try:
                self.storage_client = Client()
                logger.info("Successfully connected to Replit Object Storage")
            except Exception as e:
                logger.error(f"Error connecting to Replit Object Storage: {str(e)}. Falling back to local cache.")
                self.storage_client = None
                self.is_replit = False # Force fallback if connection fails
                self._ensure_local_cache_dir()
        else:
            logger.info("Running locally or Replit storage unavailable. Using local file cache.")
            self.storage_client = None
            self.is_replit = False
            self._ensure_local_cache_dir()

    def set_with_ttl(self, key: str, data: dict, ttl_hours: int = 1):
        """Store data with TTL in Replit storage or local file cache."""
        expires_at = datetime.now() + timedelta(hours=ttl_hours)
        cache_data = {
            'data': self._convert_timestamps(data),
            'expires_at': expires_at.isoformat(),
            'created_at': datetime.now().isoformat()
        }
        cache_json = json.dumps(cache_data)

        if self.is_replit and self.storage_client:
            try:
                self.storage_client.upload_from_text(key, cache_json)
                logger.debug(f"Stored {key} in Replit Object Storage.")
            except Exception as e:
                logger.error(f"Error setting Replit cache for {key}: {str(e)}. Attempting local fallback.")
                self._set_local_cache(key, cache_json) # Attempt local fallback on Replit error
        else:
            self._set_local_cache(key, cache_json)

    def _set_local_cache(self, key: str, cache_json: str):
        """Helper to store data in the local file cache."""
        filepath = self._get_local_cache_filepath(key)
        try:
            with open(filepath, 'w') as f:
                f.write(cache_json)
            logger.debug(f"Stored {key} in local cache: {filepath}")
        except Exception as e:
            logger.error(f"Error setting local cache for {key} at {filepath}: {str(e)}")


    def _convert_timestamps(self, obj):
        """Converts Timestamps to strings recursively."""
        if isinstance(obj, dict):
            return {str(k): self._convert_timestamps(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_timestamps(item) for item in obj]
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, pd.DatetimeIndex):
            return [ts.isoformat() for ts in obj]
        elif isinstance(obj, pd.Index):
            # Handle potential non-string index elements safely
            return [str(item) for item in obj]
        return obj


    def get(self, key: str) -> Optional[dict]:
        """Get data from Replit cache or local file cache."""
        if self.is_replit and self.storage_client:
            try:
                cache_json = self.storage_client.download_as_text(key)
                cache_data = json.loads(cache_json)
                logger.debug(f"Retrieved {key} from Replit Object Storage.")
                return self._validate_and_extract_data(key, cache_data, is_replit=True)
            except Exception as e:
                # Log Replit error but still try local cache if Replit fails
                logger.debug(f"Replit cache miss/error for {key}: {str(e)}. Checking local cache.")
                return self._get_local_cache(key)
        else:
            return self._get_local_cache(key)

    def _get_local_cache(self, key: str) -> Optional[dict]:
        """Helper to retrieve data from the local file cache."""
        filepath = self._get_local_cache_filepath(key)
        if not filepath.exists():
            logger.warning(f"Local cache miss for {key}: File not found at {filepath}") # Changed to WARNING
            return None

        try:
            with open(filepath, 'r') as f:
                cache_json = f.read()
            cache_data = json.loads(cache_json)
            logger.debug(f"Retrieved {key} from local cache: {filepath}")
            validation_result = self._validate_and_extract_data(key, cache_data, is_replit=False) # Store result
            if validation_result is None: # ADDED CHECK
                logger.warning(f"Validation failed for local cache file {filepath}") # ADDED LOGGING
            return validation_result # Return validation result
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from local cache file {filepath}: {e}")
            self._delete_local_cache(key) # Delete corrupted file
            return None
        except Exception as e:
            logger.error(f"Error getting local cache for {key} from {filepath}: {str(e)}")
            return None

    def _validate_and_extract_data(self, key: str, cache_data: dict, is_replit: bool) -> Optional[dict]:
        """Validates TTL and extracts data, deleting if expired."""
        try:
            expires_at = datetime.fromisoformat(cache_data['expires_at'])
            now = datetime.now(expires_at.tzinfo) # Ensure timezone comparison if applicable
            if now > expires_at:
                logger.warning(f"Cache expired for {key}. Expires: {expires_at}, Now: {now}. Deleting.") # Changed to WARNING
                if is_replit and self.storage_client:
                    self._delete_replit_cache(key)
                else:
                    self._delete_local_cache(key)
                return None
            return cache_data['data']
        except KeyError as e:
             logger.error(f"Invalid cache format for {key} (missing key: {e}). Deleting.") # Keep as ERROR
             if is_replit and self.storage_client:
                 self._delete_replit_cache(key)
             else: # Correctly indented else
                 self._delete_local_cache(key)
             return None
        except Exception as e:
            logger.error(f"Error validating cache data for {key}: {str(e)}")
            return None # Treat validation errors as misses

    def _delete_replit_cache(self, key: str):
        """Helper to delete from Replit storage."""
        if not self.storage_client: return
        try:
            self.storage_client.delete(key)
            logger.debug(f"Deleted expired/invalid Replit cache for {key}.")
        except Exception as e:
            logger.error(f"Error deleting expired/invalid Replit cache for {key}: {str(e)}")

    def _delete_local_cache(self, key: str):
        """Helper to delete from local file cache."""
        filepath = self._get_local_cache_filepath(key)
        try:
            if filepath.exists():
                filepath.unlink()
                logger.debug(f"Deleted expired/invalid local cache file for {key}: {filepath}")
        except Exception as e:
            logger.error(f"Error deleting expired/invalid local cache file for {key} at {filepath}: {str(e)}")


    def is_fresh(self, key: str, max_age_hours: int = 1) -> bool:
        """Check if cached data exists and is fresh (more than 5 min remaining)."""
        cache_data = None
        source = "unknown"

        if self.is_replit and self.storage_client:
            try:
                cache_json = self.storage_client.download_as_text(key)
                cache_data = json.loads(cache_json)
                source = "Replit"
            except Exception as e:
                 logger.debug(f"Replit cache miss/error for freshness check on {key}: {e}. Checking local.")
                 # Fall through to check local cache
        
        if cache_data is None and not self.is_replit: # Check local only if not replit or replit failed
             filepath = self._get_local_cache_filepath(key)
             if filepath.exists():
                 try:
                     with open(filepath, 'r') as f:
                         cache_json = f.read()
                     cache_data = json.loads(cache_json)
                     source = "Local"
                 except Exception as e:
                     logger.error(f"Error reading local cache for freshness check on {key} at {filepath}: {e}")
                     return False # Error reading local cache means not fresh

        if cache_data:
            try:
                expires_at = datetime.fromisoformat(cache_data['expires_at'])
                remaining_seconds = (expires_at - datetime.now()).total_seconds()
                is_fresh = remaining_seconds > 300 # Consider fresh if more than 5 min remaining
                logger.debug(f"Freshness check for {key} ({source}): Remaining={remaining_seconds:.0f}s, Fresh={is_fresh}")
                return is_fresh
            except Exception as e:
                logger.error(f"Error checking cache freshness for {key} ({source}): {str(e)}")
                return False # Error during validation means not fresh
        else:
            logger.debug(f"Freshness check for {key}: Not found in any cache.")
            return False # Not found in any cache
