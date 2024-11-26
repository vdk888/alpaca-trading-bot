import os
from dotenv import load_dotenv

def check_env():
    # Try to load .env file
    load_dotenv()
    
    # Variables to check
    required_vars = [
        'ALPACA_API_KEY',
        'ALPACA_SECRET_KEY',
        'TELEGRAM_BOT_TOKEN',
        'CHAT_ID'
    ]
    
    print("\n=== Environment Variables Check ===")
    
    # Check .env file existence
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        print("✅ .env file found at:", env_path)
    else:
        print("❌ .env file NOT found at:", env_path)
    
    # Check each variable
    all_good = True
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Show first and last 4 characters of the value for security
            masked_value = value[:4] + '*' * (len(value)-8) + value[-4:] if len(value) > 8 else '****'
            print(f"✅ {var} is set: {masked_value}")
        else:
            print(f"❌ {var} is NOT set")
            all_good = False
    
    print("\nStatus:", "✅ All good!" if all_good else "❌ Some variables are missing")

if __name__ == "__main__":
    check_env()
