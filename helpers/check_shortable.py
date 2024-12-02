import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from config import TRADING_SYMBOLS
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_SECRET_KEY')

def check_shortable_assets():
    """Check which assets can be shorted"""
    try:
        # Initialize trading client
        trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
        
        print("\nShortable Status for Configured Symbols:")
        print("-" * 50)
        
        for symbol in TRADING_SYMBOLS.keys():
            try:
                asset = trading_client.get_asset(symbol)
                print(f"\n{symbol}:")
                print(f"  Shortable: {asset.shortable}")
                print(f"  Easy to Borrow: {asset.easy_to_borrow}")
                print(f"  Marginable: {asset.marginable}")
                print(f"  Asset Class: {asset.asset_class}")
                print(f"  Exchange: {asset.exchange}")
            except Exception as e:
                print(f"\n{symbol}: Error getting asset info - {str(e)}")
            
    except Exception as e:
        print(f"Error connecting to Alpaca: {str(e)}")

if __name__ == "__main__":
    check_shortable_assets()
