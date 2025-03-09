
import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deployment_check.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_deployment_environment():
    """Verify all required environment variables for deployment are set"""
    required_vars = [
        'ALPACA_API_KEY',
        'ALPACA_SECRET_KEY',
        'TELEGRAM_BOT_TOKEN',
        'CHAT_ID',
        'BOT_PASSWORD',
        'TRADE_HISTORY_FILE'
    ]
    
    print("\n=== DEPLOYMENT ENVIRONMENT CHECK ===")
    
    # Check each variable
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Show first and last 3 characters of the value for security
            masked_value = value[:3] + '*' * (len(value)-6) + value[-3:] if len(value) > 8 else '****'
            print(f"✅ {var} is set: {masked_value}")
        else:
            print(f"❌ {var} is NOT set")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n❌ DEPLOYMENT CHECK FAILED: Missing environment variables: {', '.join(missing_vars)}")
        print("\nPlease add these missing secrets in the Deployments configuration panel before deploying.")
        print("1. Go to the Deployments panel")
        print("2. Click on 'Configuration'")
        print("3. Add each missing secret under the 'Secrets' section")
        return False
    
    print("\n✅ DEPLOYMENT CHECK PASSED: All required environment variables are set.")
    return True

if __name__ == "__main__":
    if not check_deployment_environment():
        sys.exit(1)
    print("\nDeployment environment is ready. Continuing with startup...\n")
import os
import logging
import json

logger = logging.getLogger(__name__)

def check_deployment_environment():
    """
    Verify that all required environment variables are set in the deployment
    environment and log their status
    
    Returns:
        bool: True if all required variables are set, False otherwise
    """
    required_vars = [
        'ALPACA_API_KEY', 
        'ALPACA_SECRET_KEY', 
        'TELEGRAM_BOT_TOKEN', 
        'CHAT_ID', 
        'BOT_PASSWORD',
        'TRADE_HISTORY_FILE'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    all_vars_present = len(missing_vars) == 0
    
    if all_vars_present:
        logger.info("✅ All required environment variables are set")
    else:
        logger.critical(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
    
    # Check if we're running in a deployment environment
    if os.getenv('REPLIT_DEPLOYMENT') == '1':
        logger.info("✅ Running in Replit Deployment environment")
    else:
        logger.info("ℹ️ Running in local development environment")
    
    return all_vars_present

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    result = check_deployment_environment()
    print(f"Environment check {'passed' if result else 'failed'}")
