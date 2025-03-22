import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_json_load():
    try:
        # Try to load the JSON file using absolute path
        params_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_params.json")
        logger.info(f"Attempting to load from: {params_file}")
        logger.info(f"File exists: {os.path.exists(params_file)}")
        
        if os.path.exists(params_file):
            try:
                with open(params_file, "r") as f:
                    file_content = f.read()
                    logger.info(f"File content length: {len(file_content)}")
                    best_params_data = json.loads(file_content)
                    logger.info(f"Successfully parsed JSON with {len(best_params_data)} symbols")
                    
                    # Print first few keys
                    logger.info(f"First few keys: {list(best_params_data.keys())[:5]}")
                    
                    # Try to access a specific symbol
                    if "BTC/USD" in best_params_data:
                        logger.info("BTC/USD found in best_params_data")
                    else:
                        logger.info("BTC/USD not found in best_params_data")
                        
                    return True
            except json.JSONDecodeError as json_err:
                logger.error(f"JSON parsing error: {json_err}")
                return False
            except Exception as read_err:
                logger.error(f"File reading error: {read_err}")
                return False
        else:
            logger.error(f"File not found at {params_file}")
            return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    result = test_json_load()
    logger.info(f"Test result: {result}")
