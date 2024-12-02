import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_SECRET_KEY')

def check_btc_trading():
    """Check BTC/USD trading capability"""
    try:
        # Initialize trading client
        trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
        
        # Get account information
        account = trading_client.get_account()
        print("\nAccount Status:")
        print("-" * 50)
        print(f"Account ID: {account.id}")
        print(f"Cash: ${float(account.cash)}")
        print(f"Trading blocked: {account.trading_blocked}")
        print(f"Shorting enabled: {account.shorting_enabled}")
        print(f"Crypto status: {getattr(account, 'crypto_status', 'Not Available')}")
        
        # Try to get BTC/USD asset info
        try:
            btc = trading_client.get_asset("BTC/USD")
            print("\nBTC/USD Asset Info:")
            print("-" * 50)
            for field in dir(btc):
                if not field.startswith('_'):  # Skip private attributes
                    value = getattr(btc, field)
                    if not callable(value):  # Skip methods
                        print(f"{field}: {value}")
        except Exception as e:
            print(f"\nError getting BTC/USD details: {str(e)}")
            
        # Try to get positions
        try:
            positions = trading_client.get_all_positions()
            print("\nCurrent Positions:")
            print("-" * 50)
            for position in positions:
                if position.symbol == "BTC/USD":
                    print(f"BTC Position: {position.qty} @ ${position.avg_entry_price}")
        except Exception as e:
            print(f"\nError getting positions: {str(e)}")
            
    except Exception as e:
        print(f"Error connecting to Alpaca: {str(e)}")

if __name__ == "__main__":
    check_btc_trading()
