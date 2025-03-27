import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional
from config import TRADING_SYMBOLS, DEFAULT_INTERVAL, DEFAULT_INTERVAL_WEEKLY, default_interval_yahoo, default_backtest_interval
import logging
import pytz
import json
from services.cache_service import CacheService

logger = logging.getLogger(__name__)
cache_service = CacheService()

def get_cache_key(symbol: str, interval: str, days: int) -> str:
    """Generate standardized cache key for price data"""
    return f"price_data:{symbol}:{interval}:{days}"

def fetch_historical_data(symbol: str, interval: str = default_interval_yahoo, days: int = default_backtest_interval, use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch historical data from Yahoo Finance

    Args:
        symbol: Stock symbol
        interval: Data interval ('1m', '5m', '15m', '30m', '60m', '1d')
        days: Number of days of historical data to fetch (default: 3)

    Returns:
        DataFrame with OHLCV data
    """
    # Check cache first if enabled
    if use_cache:
        cache_key = get_cache_key(symbol, interval, days)
        cached_data = cache_service.get(cache_key)
        if cached_data and cache_service.is_fresh(cache_key):
            logger.debug(f"Using cached data for {symbol}")
            if isinstance(cached_data, dict) and 'data' in cached_data and 'index' in cached_data:
                df = pd.DataFrame.from_dict(cached_data['data'])
                # Convert string timestamps back to datetime with timezone
                df.index = pd.to_datetime(cached_data['index']).tz_localize('UTC')
                return df
            # If cached data is already a DataFrame
            df = pd.DataFrame.from_dict(cached_data)
            if df.index.tz is None:
                df.index = pd.to_datetime(df.index).tz_localize('UTC')
            return df

    # Get the correct Yahoo Finance symbol
    yf_symbol = TRADING_SYMBOLS[symbol]['yfinance']
    ticker = yf.Ticker(yf_symbol)

    # Calculate start and end dates
    end = datetime.now(pytz.UTC)
    start = end - timedelta(days=days)

    # Debug logging
    logger.debug(f"Attempting to fetch {interval} data for {symbol} ({yf_symbol})")
    logger.debug(f"Date range: {start} to {end}")
    logger.debug(f"Requested days: {days}")

    # Fetch data with retry mechanism
    max_retries = 3
    for attempt in range(max_retries):
        try:
            df = ticker.history(start=start, end=end, interval=interval)
            logger.debug(f"Successfully fetched {len(df)} bars of {interval} data")
            if not df.empty:
                break
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                logger.error(f"Failed to fetch data for {symbol} ({yf_symbol}): {str(e)}")
                raise e
            continue

    if df.empty:
        logger.error(f"No data available for {symbol} ({yf_symbol})")
        raise ValueError(f"No data available for {symbol} ({yf_symbol})")

    # Ensure we have enough data
    min_required_bars = 700  # Minimum bars needed for weekly signals
    if len(df) < min_required_bars:
        logger.debug(f"Only {len(df)} bars found, fetching more data")
        start = end - timedelta(days=days + 2)
        df = ticker.history(start=start, end=end, interval=interval)

    # Clean and format the data
    df.columns = [col.lower() for col in df.columns]
    df = df[['open', 'high', 'low', 'close', 'volume']]

    # Add logging for data quality
    logger.info(f"Fetched {len(df)} bars of {interval} data for {symbol} ({yf_symbol})")
    logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")

    # Store in cache if enabled
    if use_cache:
        cache_key = get_cache_key(symbol, interval, days)
        # Ensure timezone awareness before caching
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        # Convert DataFrame to dict with string timestamps
        df_dict = df.copy()
        df_dict.index = df_dict.index.strftime('%Y-%m-%d %H:%M:%S%z')
        cache_dict = {
            'data': df_dict.to_dict(),
            'index': df_dict.index.tolist()
        }
        cache_service.set_with_ttl(cache_key, cache_dict, ttl_hours=1)
        logger.debug(f"Stored data in cache for {symbol}")

    return df

def get_latest_data(symbol: str, interval: str = default_interval_yahoo, limit: Optional[int] = None, days: int = default_backtest_interval, use_cache: bool = True) -> pd.DataFrame:
    """
    Get the most recent data points

    Args:
        symbol: Stock symbol
        interval: Data interval
        limit: Number of data points to return (default: None = all available data)
        days: Number of days of historical data to fetch (default: 3)

    Returns:
        DataFrame with the most recent data points
    """
    # Get the correct interval from config
    config_interval = TRADING_SYMBOLS[symbol].get('interval', interval)

    try:
        # Fetch data for the specified number of days
        logger.info(f"Fetching {days} days of data for {symbol}")
        df = fetch_historical_data(symbol, config_interval, days=days, use_cache=use_cache)

        # Filter for market hours
        market_hours = TRADING_SYMBOLS[symbol]['market_hours']
        if market_hours['start'] != '00:00' or market_hours['end'] != '23:59':
            # Convert index to market timezone
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            market_tz = market_hours['timezone']
            df.index = df.index.tz_convert(market_tz)

            # Create time masks for market hours
            start_time = pd.Timestamp.strptime(market_hours['start'], '%H:%M').time()
            end_time = pd.Timestamp.strptime(market_hours['end'], '%H:%M').time()

            # Filter for market hours
            df = df[
                (df.index.time >= start_time) & 
                (df.index.time <= end_time) &
                (df.index.weekday < 5)  # Monday = 0, Friday = 4
            ]

        # Apply limit if specified
        if limit is not None:
            return df.tail(limit)
        return df

    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        raise

def is_market_open(symbol: str = 'SPY') -> bool:
    """Check if market is currently open for the given symbol"""
    try:
        market_hours = TRADING_SYMBOLS[symbol]['market_hours']
        now = datetime.now(pytz.UTC)

        # For 24/7 markets
        if market_hours['start'] == '00:00' and market_hours['end'] == '23:59':
            return True

        # Convert current time to market timezone
        market_tz = market_hours['timezone']
        market_time = now.astimezone(pytz.timezone(market_tz))

        # Check if it's a weekday
        if market_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Parse market hours
        start_time = datetime.strptime(market_hours['start'], '%H:%M').time()
        end_time = datetime.strptime(market_hours['end'], '%H:%M').time()
        current_time = market_time.time()

        return start_time <= current_time <= end_time

    except Exception as e:
        logger.error(f"Error checking market hours for {symbol}: {str(e)}")
        return False