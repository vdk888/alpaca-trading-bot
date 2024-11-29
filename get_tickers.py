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
ALPACA_PAPER = os.getenv('ALPACA_PAPER', default='True')

def get_tradable_tickers():
    """Get all tradable tickers from Alpaca"""
    try:
        # Initialize trading client
        logger.info("Initializing Alpaca client...")
        trading_client = TradingClient(
            api_key=API_KEY,
            secret_key=API_SECRET,
            paper=ALPACA_PAPER.lower() == 'true'
        )
        
        # Get all assets
        logger.info("Fetching assets...")
        assets = trading_client.get_all_assets()
        
        # Filter for tradable US equities
        tradable_tickers = []
        for asset in assets:
            if (hasattr(asset, 'tradable') and asset.tradable and 
                hasattr(asset, 'status') and asset.status == 'active' and
                hasattr(asset, 'asset_class') and asset.asset_class == 'us_equity'):
                tradable_tickers.append(asset.symbol)
        
        return sorted(tradable_tickers)  # Sort alphabetically
        
    except Exception as e:
        logger.error(f"Error fetching tickers: {str(e)}")
        if "forbidden" in str(e).lower():
            logger.error("Authentication failed. Please check your API credentials.")
            logger.error(f"Using API key starting with: {API_KEY[:5] if API_KEY else 'None'}")
        raise

if __name__ == "__main__":
    try:
        logger.info("Starting ticker retrieval...")
        tickers = get_tradable_tickers()
        print(f"\nFound {len(tickers)} tradable tickers")
        print("\nFirst 10 tickers as sample:")
        print(tickers[:10])
        
        # Save to file
        output_file = "tradable_tickers.txt"
        with open(output_file, "w") as f:
            f.write("\n".join(tickers))
        print(f"\nComplete list saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
