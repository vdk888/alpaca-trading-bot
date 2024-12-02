import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

# Load environment variables
load_dotenv()
API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_SECRET_KEY')

def get_average_volume(symbol, days=30):
    """Get average daily volume for a symbol"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start_date, end=end_date)
        return hist['Volume'].mean() if not hist.empty else 0
    except Exception as e:
        print(f"Error getting volume for {symbol}: {str(e)}")
        return 0

def get_liquid_assets():
    """Get most liquid assets from Alpaca"""
    try:
        # Initialize trading client
        trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
        
        # Get all assets
        assets = trading_client.get_all_assets()
        
        # Separate ETFs and Crypto
        etfs = []
        crypto = []
        
        for asset in assets:
            if not asset.tradable or asset.status != 'active':
                continue
                
            # Get volume data
            volume = get_average_volume(
                asset.symbol if asset.asset_class != 'crypto' else asset.symbol.replace('/', '-')
            )
            
            asset_info = {
                'symbol': asset.symbol,
                'name': asset.name,
                'exchange': asset.exchange,
                'volume': volume,
                'shortable': asset.shortable,
                'fractionable': asset.fractionable
            }
            
            if asset.asset_class == 'crypto':
                crypto.append(asset_info)
            elif 'ETF' in str(asset.name).upper():
                etfs.append(asset_info)
        
        # Sort by volume
        etfs.sort(key=lambda x: x['volume'], reverse=True)
        crypto.sort(key=lambda x: x['volume'], reverse=True)
        
        print("\nTop 15 Liquid ETFs:")
        print("-" * 80)
        print(f"{'Symbol':<10} {'Volume':<15} {'Shortable':<10} {'Fractionable':<12} {'Name'}")
        print("-" * 80)
        for etf in etfs[:15]:
            print(f"{etf['symbol']:<10} {int(etf['volume']):,<15} {str(etf['shortable']):<10} {str(etf['fractionable']):<12} {etf['name']}")
        
        print("\nTop 5 Liquid Crypto:")
        print("-" * 80)
        print(f"{'Symbol':<10} {'Volume':<15} {'Shortable':<10} {'Fractionable':<12} {'Name'}")
        print("-" * 80)
        for c in crypto[:5]:
            print(f"{c['symbol']:<10} {int(c['volume']):,<15} {str(c['shortable']):<10} {str(c['fractionable']):<12} {c['name']}")
            
        return etfs[:15], crypto[:5]
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return [], []

if __name__ == "__main__":
    get_liquid_assets()
