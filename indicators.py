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
    macd = calculate_macd(data, fast=params['macd_fast'], slow=params['macd_slow'], signal=params['macd_signal'])
    rsi = calculate_rsi(data, period=params['rsi_period'])
    stoch = calculate_stochastic(data, k_period=params['stochastic_k_period'], d_period=params['stochastic_d_period'])
    
    # Normalize each indicator
    norm_macd = (macd - macd.mean()) / macd.std()
    norm_rsi = (rsi - 50) / 25  # Center around 0 and scale
    norm_stoch = stoch / 100  # Already centered around 0, just scale
    
    # Combine indicators (equal weighting for simplicity)
    composite = ((norm_macd + norm_rsi + norm_stoch) / 3) 
    composite = composite.ffill().fillna(0)  # Fill NaNs
    
    # Calculate the standard deviation for thresholds
    rolling_std = composite.rolling(window=int(params['sell_rolling_std'] if is_weekly else params['buy_rolling_std'])).std().ffill().fillna(0)
    
    # Calculate the threshold lines
    down_lim_line = composite.mean() + (params['sell_down_lim'] * rolling_std * reactivity)
    up_lim_line = composite.mean() + (params['buy_up_lim'] * rolling_std * reactivity)

    return pd.DataFrame({
        'Composite': composite,
        'Down_Lim': down_lim_line,
        'Up_Lim': up_lim_line
    }), composite, rolling_std

def generate_signals(data: pd.DataFrame, params: Dict[str, Union[float, int]], reactivity: float = 1.0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Calculate daily composite and thresholds (now 5-minute timeframe)
    daily_data, daily_composite, daily_std = calculate_composite_indicator(data, params, reactivity)
    
    # Calculate "weekly" composite (35-minute timeframe = 7 * 5min)
    weekly_data = data.resample('35min').last()  # Use 'min' instead of 'T'
    weekly_data, weekly_composite, weekly_std = calculate_composite_indicator(weekly_data, params, reactivity, is_weekly=True)
    weekly_data = weekly_data.reindex(data.index, method='ffill')
    print(f"Debug: generate_signals called with reactivity={reactivity}")

    # Initialize signals DataFrame with zeros
    signals = pd.DataFrame(0, index=data.index, columns=['Signal', 'Daily_Composite', 'Daily_Down_Lim', 'Daily_Up_Lim', 'Weekly_Composite', 'Weekly_Down_Lim', 'Weekly_Up_Lim'])
    
    # Assign values without chaining
    signals = signals.assign(
        Daily_Composite=daily_data['Composite'],
        Daily_Down_Lim=daily_data['Down_Lim'],
        Daily_Up_Lim=daily_data['Up_Lim'],
        Weekly_Composite=weekly_data['Composite'],
        Weekly_Down_Lim=weekly_data['Down_Lim'],
        Weekly_Up_Lim=weekly_data['Up_Lim']
    )
    
    # Generate buy signals (daily crossing above upper limit)
    buy_mask = (daily_data['Composite'] > daily_data['Up_Lim']) & (daily_data['Composite'].shift(1) <= daily_data['Up_Lim'].shift(1))
    signals.loc[buy_mask, 'Signal'] = 1
    
    # Generate sell signals (weekly crossing below lower limit)
    sell_mask = (weekly_data['Composite'] < weekly_data['Down_Lim']) & (weekly_data['Composite'].shift(1) >= weekly_data['Down_Lim'].shift(1))
    signals.loc[sell_mask, 'Signal'] = -1
    
    return signals, daily_data, weekly_data

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
    buy_signals = (result[0]['Signal'] == 1).sum()
    sell_signals = (result[0]['Signal'] == -1).sum()
    print(f"\nTotal buy signals: {buy_signals}")
    print(f"Total sell signals: {sell_signals}")