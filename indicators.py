import pandas as pd
import numpy as np
from typing import Dict, Union, List, Tuple

def calculate_macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """Calculate MACD (Moving Average Convergence Divergence)"""
    ema_fast = data['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line  # MACD histogram

def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate RSI (Relative Strength Index)"""
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_stochastic(data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.Series:
    """Calculate Stochastic Oscillator"""
    low_min = data['low'].rolling(window=k_period).min()
    high_max = data['high'].rolling(window=k_period).max()
    k = 100 * (data['close'] - low_min) / (high_max - low_min)
    d = k.rolling(window=d_period).mean()
    return k - d  # Similar to MACD, we return K-D

def calculate_composite_indicator(data: pd.DataFrame, params: Dict[str, Union[float, int]], reactivity: float = 1.0, is_weekly: bool = False) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    # Print input data stats for debugging
    print(f"Debug: calculate_composite_indicator input stats:")
    print(f"Data points: {len(data)}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Calculate indicators
    macd = calculate_macd(data, fast=params['macd_fast'], slow=params['macd_slow'], signal=params['macd_signal'])
    rsi = calculate_rsi(data, period=params['rsi_period'])
    stoch = calculate_stochastic(data, k_period=params['stochastic_k_period'], d_period=params['stochastic_d_period'])
    
    # Print indicator stats
    print(f"MACD range: {macd.min():.4f} to {macd.max():.4f}")
    print(f"RSI range: {rsi.min():.4f} to {rsi.max():.4f}")
    print(f"Stochastic range: {stoch.min():.4f} to {stoch.max():.4f}")
    
    # Handle NaN values safely
    macd = macd.fillna(0)
    rsi = rsi.fillna(50)  # Neutral RSI value
    stoch = stoch.fillna(0)
    
    # Normalize each indicator with safe division
    norm_macd = macd / (macd.std() if macd.std() != 0 else 1)
    norm_rsi = (rsi - 50) / 25  # Center around 0 and scale
    norm_stoch = stoch / (100 if stoch.max() != 0 else 1)  # Scale to [-1, 1]
    
    # Print normalized stats
    print(f"Normalized MACD range: {norm_macd.min():.4f} to {norm_macd.max():.4f}")
    print(f"Normalized RSI range: {norm_rsi.min():.4f} to {norm_rsi.max():.4f}")
    print(f"Normalized Stoch range: {norm_stoch.min():.4f} to {norm_stoch.max():.4f}")
    
    # Combine indicators with weighted approach
    if is_weekly:
        # Weekly composite puts more weight on longer-term indicators
        composite = (0.4 * norm_macd + 0.4 * norm_rsi + 0.2 * norm_stoch)
    else:
        # Daily composite weights all indicators equally
        composite = (norm_macd + norm_rsi + norm_stoch) / 3
    
    composite = composite.fillna(0)
    
    # Calculate the rolling standard deviation for thresholds
    window = int(params['sell_rolling_std'] if is_weekly else params['buy_rolling_std'])
    rolling_std = composite.rolling(window=window, min_periods=5).std()
    rolling_std = rolling_std.fillna(composite.std())  # Use overall std for initial values
    
    # Ensure rolling_std is never zero to avoid division issues
    rolling_std = rolling_std.clip(lower=0.0001)
    
    # Calculate dynamic threshold lines
    if is_weekly:
        # Weekly thresholds are more conservative
        down_lim_line = -rolling_std * reactivity
        up_lim_line = rolling_std * reactivity
    else:
        # Daily thresholds use parameter-based multipliers
        down_lim_line = params['sell_down_lim'] * rolling_std * reactivity
        up_lim_line = params['buy_up_lim'] * rolling_std * reactivity
    
    # Print final composite stats
    print(f"{'Weekly' if is_weekly else 'Daily'} Composite stats:")
    print(f"Mean: {composite.mean():.4f}")
    print(f"Std: {composite.std():.4f}")
    print(f"Range: {composite.min():.4f} to {composite.max():.4f}")
    print(f"Threshold range: {down_lim_line.min():.4f} to {up_lim_line.max():.4f}")

    return pd.DataFrame({
        'Composite': composite,
        'Down_Lim': down_lim_line,
        'Up_Lim': up_lim_line
    }), composite, rolling_std

def generate_signals(data: pd.DataFrame, params: Dict[str, Union[float, int]], reactivity: float = 1.0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"Debug: Starting signal generation with {len(data)} data points")
    
    # Calculate daily composite and thresholds (5-minute timeframe)
    daily_data, daily_composite, daily_std = calculate_composite_indicator(data, params, reactivity)
    
    # Calculate weekly composite (35-minute timeframe = 7 * 5min)
    weekly_data = data.resample('35min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()  # Remove any NaN rows
    
    # Debug information
    print(f"Debug: Generated {len(weekly_data)} weekly bars")
    print(f"Debug: Weekly data range: {weekly_data.index[0]} to {weekly_data.index[-1]}")
    print(f"Debug: Weekly data sample:")
    print(weekly_data.head())
    
    # Verify we have enough data for weekly calculations
    min_weekly_bars = 20  # Minimum bars needed for reliable signals
    if len(weekly_data) < min_weekly_bars:
        print(f"Warning: Insufficient weekly data. Have {len(weekly_data)} bars, need {min_weekly_bars}")
        # Still calculate but with a warning
    
    # Calculate weekly indicators
    weekly_indicators, weekly_composite, weekly_std = calculate_composite_indicator(weekly_data, params, reactivity, is_weekly=True)
    
    # Forward fill the weekly data to match the 5-minute timeframe
    weekly_resampled = pd.DataFrame(index=data.index)
    for col in weekly_indicators.columns:
        weekly_resampled[col] = weekly_indicators[col].reindex(data.index, method='ffill')
    
    # Print weekly composite stats for debugging
    non_zero = weekly_composite[weekly_composite != 0]
    print(f"Debug: Weekly Composite detailed stats:")
    print(f"Mean: {weekly_composite.mean():.4f}")
    print(f"Std: {weekly_composite.std():.4f}")
    print(f"Non-zero values: {len(non_zero)}")
    if len(non_zero) > 0:
        print(f"Non-zero range: {non_zero.min():.4f} to {non_zero.max():.4f}")
    
    # Initialize signals DataFrame with zeros
    signals = pd.DataFrame(0, index=data.index, columns=['signal', 'daily_composite', 'daily_down_lim', 'daily_up_lim', 'weekly_composite', 'weekly_down_lim', 'weekly_up_lim'])
    
    # Assign values without chaining
    signals = signals.assign(
        daily_composite=daily_data['Composite'],
        daily_down_lim=daily_data['Down_Lim'],
        daily_up_lim=daily_data['Up_Lim'],
        weekly_composite=weekly_resampled['Composite'],
        weekly_down_lim=weekly_resampled['Down_Lim'],
        weekly_up_lim=weekly_resampled['Up_Lim']
    )
    
    # Generate buy signals (daily crossing above upper limit)
    buy_mask = (daily_data['Composite'] > daily_data['Up_Lim']) & (daily_data['Composite'].shift(1) <= daily_data['Up_Lim'].shift(1))
    signals.loc[buy_mask, 'signal'] = 1
    
    # Generate sell signals (weekly crossing below lower limit)
    sell_mask = (weekly_resampled['Composite'] < weekly_resampled['Down_Lim']) & (weekly_resampled['Composite'].shift(1) >= weekly_resampled['Down_Lim'].shift(1))
    signals.loc[sell_mask, 'signal'] = -1
    
    # Print final signal counts for debugging
    buy_count = (signals['signal'] == 1).sum()
    sell_count = (signals['signal'] == -1).sum()
    print(f"Debug: Generated {buy_count} buy signals and {sell_count} sell signals")
    
    return signals, daily_data, weekly_resampled

def get_default_params():
    return {
        'percent_increase_buy': 0.02,
        'percent_decrease_sell': 0.02,
        'sell_down_lim': 2.0,
        'sell_rolling_std': 20,
        'buy_up_lim': -2.0,
        'buy_rolling_std': 20,
        'rsi_period': 14,
        'stochastic_k_period': 14,
        'stochastic_d_period': 3,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'reactivity': 1.0  # Add this line
    }

# Example usage
if __name__ == "__main__":
    # This is just a placeholder. In a real scenario, you would import data from fetch_data.py
    data = pd.DataFrame({
        "open": np.random.randn(100).cumsum() + 100,
        "high": np.random.randn(100).cumsum() + 102,
        "low": np.random.randn(100).cumsum() + 98,
        "close": np.random.randn(100).cumsum() + 101,
        "Volume": np.random.randint(1000, 5000, 100)
    }, index=pd.date_range(start="2023-01-01", periods=100))
    
    params = get_default_params()
    
    result = generate_signals(data, params)
    print(result[0].tail(10))
    
    # Count buy and sell signals
    buy_signals = (result[0]['signal'] == 1).sum()
    sell_signals = (result[0]['signal'] == -1).sum()
    print(f"\nTotal buy signals: {buy_signals}")
    print(f"Total sell signals: {sell_signals}")