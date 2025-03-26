
from services.cache_service import CacheService

def test_object_storage():
    cache_service = CacheService()

    # Test data to store
    test_key = "test_data"
    test_data = {"name": "Replit", "type": "Object Storage Test"}

    # Set data with a TTL of 1 hour
    cache_service.set_with_ttl(test_key, test_data, ttl_hours=1)
    print(f"Stored data under key '{test_key}': {test_data}")

    # Retrieve the data
    retrieved_data = cache_service.get(test_key)
    if retrieved_data is not None:
        print(f"Retrieved data: {retrieved_data}")
    else:
        print("No data retrieved or data has expired.")

    # Check if data is still fresh
    is_fresh = cache_service.is_fresh(test_key)
    if is_fresh:
        print(f"The data under key '{test_key}' is fresh.")
    else:
        print(f"The data under key '{test_key}' is not fresh or does not exist.")
    
# Run the test function
if __name__ == "__main__":
    test_object_storage()
