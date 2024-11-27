import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional

def fetch_historical_data(symbol: str, interval: str = '5m', days: int = 3) -> pd.DataFrame:
    """
    Fetch historical data from Yahoo Finance
    
    Args:
        symbol: Stock symbol
        interval: Data interval ('1m', '5m', '15m', '30m', '60m', '1d')
        days: Number of days of historical data to fetch (default: 3)
    
    Returns:
        DataFrame with OHLCV data
    """
    ticker = yf.Ticker(symbol)
    
    # Calculate start and end dates
    end = datetime.now()
    start = end - timedelta(days=days)
    
    # Fetch data with retry mechanism
    max_retries = 3
    for attempt in range(max_retries):
        try:
            df = ticker.history(start=start, end=end, interval=interval)
            if not df.empty:
                break
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            continue
    
    # Ensure we have enough data
    min_required_bars = 700  # Minimum bars needed for weekly signals
    if len(df) < min_required_bars:
        # Try fetching more data
        start = end - timedelta(days=days + 2)
        df = ticker.history(start=start, end=end, interval=interval)
    
    # Clean and format the data
    df.columns = [col.lower() for col in df.columns]
    df = df[['open', 'high', 'low', 'close', 'volume']]
    
    # Add logging for data quality
    print(f"Fetched {len(df)} bars of {interval} data for {symbol}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    return df

def get_latest_data(symbol: str, interval: str = '5m', limit: Optional[int] = None) -> pd.DataFrame:
    """
    Get the most recent data points
    
    Args:
        symbol: Stock symbol
        interval: Data interval
        limit: Number of data points to return (default: None = all available data)
    
    Returns:
        DataFrame with the most recent data points
    """
    # Fetch at least 3 days of data for proper weekly signal calculation
    df = fetch_historical_data(symbol, interval, days=3)
    
    # Apply limit if specified
    if limit is not None:
        return df.tail(limit)
    return df

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
