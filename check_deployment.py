import os
import logging
import sys

logger = logging.getLogger(__name__)

def check_deployment_environment():
    """
    Check if all required environment variables are set
    Returns True if all requirements are met, False otherwise
    """
    try:
        # Check required environment variables
        required_vars = [
            'ALPACA_API_KEY', 
            'ALPACA_SECRET_KEY', 
            'TELEGRAM_BOT_TOKEN', 
            'CHAT_ID', 
            'BOT_PASSWORD', 
            'TRADE_HISTORY_FILE'
        ]

        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            logger.critical(f"Missing required environment variables: {', '.join(missing_vars)}")
            return False

        logger.info("All required environment variables are set")
        return True

    except Exception as e:
        logger.critical(f"Error checking deployment environment: {e}")
        return False

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('deployment_check.log'),
            logging.StreamHandler()
        ]
    )

    if check_deployment_environment():
        print("Deployment environment check passed")
        sys.exit(0)
    else:
        print("Deployment environment check failed")
        sys.exit(1)