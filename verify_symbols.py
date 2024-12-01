import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from config import TRADING_SYMBOLS

# Load environment variables
load_dotenv()
API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_SECRET_KEY')

def verify_symbols():
    """Verify all configured symbols are available on Alpaca"""
    try:
        # Initialize trading client
        trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
        
        print("\nVerifying Configured Symbols:")
        print("-" * 80)
        print(f"{'Symbol':<12} {'Market':<8} {'Status':<10} {'Tradable':<10} {'Shortable':<10} {'Fractionable'}")
        print("-" * 80)
        
        all_valid = True
        
        for symbol, config in TRADING_SYMBOLS.items():
            try:
                asset = trading_client.get_asset(symbol)
                print(f"{symbol:<12} {config['market']:<8} {asset.status:<10} {str(asset.tradable):<10} {str(asset.shortable):<10} {asset.fractionable}")
                
                if not asset.tradable or asset.status != 'active':
                    all_valid = False
                    print(f"WARNING: {symbol} is not tradable or not active!")
                    
            except Exception as e:
                all_valid = False
                print(f"{symbol:<12} ERROR: {str(e)}")
        
        if all_valid:
            print("\n✅ All symbols are valid and tradable!")
        else:
            print("\n❌ Some symbols have issues. Please review the warnings above.")
            
    except Exception as e:
        print(f"Error connecting to Alpaca: {str(e)}")

if __name__ == "__main__":
    verify_symbols()
