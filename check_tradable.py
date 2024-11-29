from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize clients
trading_client = TradingClient(
    api_key=os.getenv('ALPACA_API_KEY'),
    secret_key=os.getenv('ALPACA_SECRET_KEY'),
    paper=True
)

# Symbols to check
symbols = ['SPY', 'DAX', 'NIKKEI', 'EUR/USD']

def check_tradable(symbol: str):
    try:
        # Try to get asset information
        asset = trading_client.get_asset(symbol)
        logger.info(f"\nAsset {symbol}:")
        logger.info(f"Tradable: {asset.tradable}")
        logger.info(f"Status: {asset.status}")
        logger.info(f"Class: {asset.class_}")
        logger.info(f"Exchange: {asset.exchange}")
        return asset.tradable
    except Exception as e:
        logger.error(f"Error checking {symbol}: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Checking tradable status on Alpaca...")
    
    for symbol in symbols:
        check_tradable(symbol)
