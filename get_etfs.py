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

def get_tradable_etfs():
    """Get all tradable ETFs from Alpaca"""
    try:
        # Initialize trading client
        logger.info("Initializing Alpaca client...")
        trading_client = TradingClient(
            api_key=API_KEY,
            secret_key=API_SECRET,
            paper=True
        )
        
        # Get all assets
        logger.info("Fetching assets...")
        assets = trading_client.get_all_assets()
        
        # Debug: Check first asset structure
        if assets:
            first_asset = assets[0]
            logger.info("Asset attributes:")
            for attr in dir(first_asset):
                if not attr.startswith('_'):  # Skip private attributes
                    value = getattr(first_asset, attr)
                    if not callable(value):  # Skip methods
                        logger.info(f"{attr}: {value}")
        
        # Filter for tradable ETFs
        etf_tickers = []
        for asset in assets:
            if (hasattr(asset, 'tradable') and asset.tradable and 
                hasattr(asset, 'status') and asset.status == 'active' and
                hasattr(asset, 'exchange') and asset.exchange in ['NYSE', 'NASDAQ', 'ARCA'] and
                hasattr(asset, 'asset_class') and asset.asset_class == 'us_equity' and
                hasattr(asset, 'name') and 'ETF' in asset.name.upper()):
                etf_tickers.append({
                    'symbol': asset.symbol,
                    'name': asset.name,
                    'exchange': asset.exchange
                })
        
        return sorted(etf_tickers, key=lambda x: x['symbol'])  # Sort alphabetically by symbol
        
    except Exception as e:
        logger.error(f"Error fetching ETFs: {str(e)}")
        if "forbidden" in str(e).lower():
            logger.error("Authentication failed. Please check your API credentials.")
            logger.error(f"Using API key starting with: {API_KEY[:5] if API_KEY else 'None'}")
        raise

if __name__ == "__main__":
    try:
        logger.info("Starting ETF retrieval...")
        etfs = get_tradable_etfs()
        print(f"\nFound {len(etfs)} tradable ETFs")
        print("\nFirst 10 ETFs as sample:")
        for etf in etfs[:10]:
            print(f"{etf['symbol']:<10} - {etf['name']:<50} ({etf['exchange']})")
        
        # Save to file with more details
        output_file = "tradable_etfs.txt"
        with open(output_file, "w", encoding='utf-8') as f:
            f.write("Symbol     Exchange  Name\n")
            f.write("-" * 80 + "\n")
            for etf in etfs:
                f.write(f"{etf['symbol']:<10} {etf['exchange']:<9} {etf['name']}\n")
        print(f"\nComplete list saved to {output_file}")
        
        # Save just symbols to a separate file
        symbols_file = "etf_symbols.txt"
        with open(symbols_file, "w") as f:
            f.write("\n".join(etf['symbol'] for etf in etfs))
        print(f"Symbols only list saved to {symbols_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
