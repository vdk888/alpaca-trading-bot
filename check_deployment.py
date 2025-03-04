
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
