from alpaca.trading.client import TradingClient
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER

try:
    trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)
    account = trading_client.get_account()
    print(f"Successfully authenticated. Account status: {account.status}")
    print(f"Account ID: {account.id}")
    print(f"Cash: ${account.cash}")
except Exception as e:
    print(f"Authentication failed: {e}")
