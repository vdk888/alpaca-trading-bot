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
        
        # Filter for tradable assets (both US equities and crypto)
        tradable_tickers = []
        crypto_tickers = []
        for asset in assets:
            if (hasattr(asset, 'tradable') and asset.tradable and 
                hasattr(asset, 'status') and asset.status == 'active'):
                if hasattr(asset, 'asset_class'):
                    if asset.asset_class == 'us_equity':
                        tradable_tickers.append(asset.symbol)
                    elif asset.asset_class == 'crypto':
                        crypto_tickers.append(asset.symbol)
        
        return {
            'equities': sorted(tradable_tickers),
            'crypto': sorted(crypto_tickers)
        }
        
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
        print(f"\nFound {len(tickers['equities'])} tradable equities and {len(tickers['crypto'])} tradable crypto assets")
        print("\nFirst 10 equities as sample:")
        print(tickers['equities'][:10])
        print("\nFirst 10 crypto assets as sample:")
        print(tickers['crypto'][:10])
        
        # Save to file
        output_file_equities = "tradable_equities.txt"
        output_file_crypto = "tradable_crypto.txt"
        with open(output_file_equities, "w") as f:
            f.write("\n".join(tickers['equities']))
        with open(output_file_crypto, "w") as f:
            f.write("\n".join(tickers['crypto']))
        print(f"\nComplete list of equities saved to {output_file_equities}")
        print(f"\nComplete list of crypto assets saved to {output_file_crypto}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
