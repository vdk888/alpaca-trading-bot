import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
import yfinance as yf
import pandas as pd
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_SECRET_KEY')

def get_yahoo_data(symbol):
    """Get basic volume and price data for a symbol using Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Get the required fields with default values if not found
        avg_volume = info.get('averageVolume', 0)
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        market_cap = info.get('marketCap', 0)
        name = info.get('longName', info.get('shortName', symbol))
        
        return {
            'symbol': symbol,
            'avg_volume': avg_volume,
            'current_price': current_price,
            'market_cap': market_cap,
            'name': name
        }
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def main():
    try:
        # Initialize trading client
        logger.info("Initializing Alpaca client...")
        trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
        
        # Get all assets
        logger.info("Fetching assets...")
        assets = trading_client.get_all_assets()
        
        # Filter for tradable ETFs
        etf_symbols = []
        for asset in assets:
            if (asset.tradable and 
                asset.status == 'active' and
                asset.exchange in ['NYSE', 'NASDAQ', 'ARCA'] and
                'ETF' in asset.name.upper()):
                etf_symbols.append(asset.symbol)
        
        logger.info(f"Found {len(etf_symbols)} ETFs. Fetching volume data...")
        
        # Get volume data for each ETF
        etf_data = []
        for symbol in etf_symbols:
            data = get_yahoo_data(symbol)
            if data:
                etf_data.append(data)
            time.sleep(0.1)  # Rate limiting
        
        # Sort by average volume
        etf_data = sorted(etf_data, key=lambda x: x['avg_volume'], reverse=True)
        
        # Print top 20 ETFs
        print(f"\nFound volume data for {len(etf_data)} ETFs")
        print("\nTop 20 ETFs by average daily volume:")
        print("\nSymbol     Avg Daily Volume    Price      Market Cap     Name")
        print("-" * 100)
        
        for etf in etf_data[:20]:
            print(f"{etf['symbol']:<10} {etf['avg_volume']:>15,d} "
                  f"${etf['current_price']:>8,.2f} "
                  f"${etf['market_cap']/1e9:>8,.1f}B    "
                  f"{etf['name']}")
        
        # Save to CSV
        df = pd.DataFrame(etf_data)
        df.to_csv("etf_volumes.csv", index=False)
        print(f"\nComplete data saved to etf_volumes.csv")
        
        # Save high volume symbols
        min_volume = 100000
        high_volume_symbols = [etf['symbol'] for etf in etf_data 
                             if etf['avg_volume'] >= min_volume]
        
        with open("high_volume_etfs.txt", "w") as f:
            f.write("\n".join(high_volume_symbols))
        print(f"High volume ETF symbols (>{min_volume:,} daily volume) "
              f"saved to high_volume_etfs.txt")
              
    except Exception as e:
        logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
