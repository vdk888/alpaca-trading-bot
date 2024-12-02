from alpaca.trading.client import TradingClient
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER

try:
    trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)
    
    # Check account details
    account = trading_client.get_account()
    print(f"\nAccount Status:")
    print(f"Trading allowed: {not account.trading_blocked}")
    print(f"Crypto trading allowed: {getattr(account, 'crypto_status', 'Not Available')}")
    print(f"Account cash: ${float(account.cash)}")
    
    # Try to get BTC/USD details
    try:
        btc = trading_client.get_asset("BTC/USD")
        print(f"\nBTC/USD Status:")
        print(f"Tradable: {btc.tradable}")
        print(f"Status: {btc.status}")
        print(f"Asset Class: {btc.asset_class}")
    except Exception as e:
        print(f"\nError getting BTC/USD details: {str(e)}")
        
except Exception as e:
    print(f"Error connecting to Alpaca: {str(e)}")
