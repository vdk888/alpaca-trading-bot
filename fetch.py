import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional

def fetch_historical_data(symbol: str, interval: str = '5m', days: int = 7) -> pd.DataFrame:
    """
    Fetch historical data from Yahoo Finance
    
    Args:
        symbol: Stock symbol
        interval: Data interval ('1m', '5m', '15m', '30m', '60m', '1d')
        days: Number of days of historical data to fetch
    
    Returns:
        DataFrame with OHLCV data
    """
    ticker = yf.Ticker(symbol)
    
    # Calculate start and end dates
    end = datetime.now()
    start = end - timedelta(days=days)
    
    # Fetch data
    df = ticker.history(start=start, end=end, interval=interval)
    
    # Clean and format the data
    df.columns = [col.lower() for col in df.columns]
    return df[['open', 'high', 'low', 'close', 'volume']]

def get_latest_data(symbol: str, interval: str = '5m', limit: int = 100) -> pd.DataFrame:
    """
    Get the most recent data points
    
    Args:
        symbol: Stock symbol
        interval: Data interval
        limit: Number of data points to return
    
    Returns:
        DataFrame with the most recent data points
    """
    df = fetch_historical_data(symbol, interval)
    return df.tail(limit)

def is_market_open() -> bool:
    """Check if US market is currently open"""
    now = datetime.now()
    # Check if it's a weekday
    if now.weekday() in [5, 6]:  # Saturday = 5, Sunday = 6
        return False
    
    # Convert current time to US Eastern time
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= now <= market_close
