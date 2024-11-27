import matplotlib.pyplot as plt
from matplotlib.dates import HourLocator, DateFormatter
from datetime import datetime, timedelta
import yfinance as yf
from indicators import generate_signals, get_default_params
import io
import pandas as pd
import pytz
import numpy as np

def is_market_hours(timestamp):
    """Check if timestamp is during market hours (9:30 AM - 4:00 PM ET, weekdays)"""
    et_tz = pytz.timezone('US/Eastern')
    ts_et = timestamp.astimezone(et_tz)
    
    # Check if it's a weekday
    if ts_et.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False
    
    # Check if it's between 9:30 AM and 4:00 PM ET
    market_start = ts_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_end = ts_et.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_start <= ts_et <= market_end

def split_into_sessions(data):
    """Split data into continuous market sessions"""
    sessions = []
    current_session = []
    last_timestamp = None
    
    for timestamp, row in data.iterrows():
        if last_timestamp is not None:
            # Check if there's a gap larger than 5 minutes (typical interval)
            time_diff = (timestamp - last_timestamp).total_seconds() / 60
            if time_diff > 6:  # Allow for small delays
                if current_session:
                    sessions.append(pd.DataFrame(current_session))
                current_session = []
        
        current_session.append(row)
        last_timestamp = timestamp
    
    if current_session:
        sessions.append(pd.DataFrame(current_session))
    
    return sessions

def create_strategy_plot(symbol='SPY', days=5):
    """Create a strategy visualization plot and return it as bytes"""
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Create Ticker object
    ticker = yf.Ticker(symbol)
    
    # Fetch data with explicit columns
    data = ticker.history(
        start=start_date,
        end=end_date,
        interval='5m',
        actions=False
    )
    
    if len(data) == 0:
        raise ValueError(f"No data available for {symbol} in the specified date range")
    
    # Filter for market hours only
    data = data[data.index.map(is_market_hours)]
    
    # Convert column names to lowercase
    data.columns = data.columns.str.lower()
    
    # Ensure required columns exist
    required_columns = ['close', 'open', 'high', 'low', 'volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}. Available columns: {data.columns.tolist()}")
    
    # Generate signals
    params = get_default_params()
    signals, daily_data, weekly_data = generate_signals(data, params)
    
    # Split data into sessions
    sessions = split_into_sessions(data)
    
    # Create the plot
    fig = plt.figure(figsize=(15, 12))
    
    # Plot 1: Price and Signals
    ax1 = plt.subplot(3, 1, 1)
    
    # Plot each session separately and collect x-limits
    all_timestamps = []
    session_boundaries = []
    last_timestamp = None
    shifted_data = pd.DataFrame()
    
    # Store the original session start times for labeling
    session_start_times = []
    
    # First, collect all original timestamps and determine trading sessions
    trading_sessions = []
    current_session = []
    
    for idx, row in data.iterrows():
        if not current_session or (idx - current_session[-1].name).total_seconds() <= 300:  # 5 minutes
            current_session.append(row)
        else:
            trading_sessions.append(pd.DataFrame(current_session))
            current_session = [row]
    if current_session:
        trading_sessions.append(pd.DataFrame(current_session))
    
    # Now plot each session with proper time labels
    for i, session in enumerate(trading_sessions):
        session_df = session.copy()
        
        if last_timestamp is not None:
            # Add a small gap between sessions
            gap = pd.Timedelta(minutes=5)
            time_shift = (last_timestamp + gap) - session_df.index[0]
            session_df.index = session_df.index + time_shift
        
        # Store original start time of session
        session_start_times.append((session_df.index[0], session.index[0]))
        
        ax1.plot(session_df.index, session_df['close'],
                label='Price' if i == 0 else "",
                color='blue', alpha=0.6)
        
        all_timestamps.extend(session_df.index)
        session_boundaries.append(session_df.index[0])
        last_timestamp = session_df.index[-1]
        
        # Store the shifted data for signals
        shifted_data = pd.concat([shifted_data, session_df])
    
    # Create timestamp mapping for signals
    original_to_shifted = {}
    for orig_session, shifted_session in zip(trading_sessions, session_boundaries):
        time_diff = shifted_session - orig_session.index[0]
        for orig_time in orig_session.index:
            original_to_shifted[orig_time] = orig_time + time_diff
    
    # Plot signals with correct timestamps
    buy_signals = signals[signals['signal'] == 1].copy()
    if len(buy_signals) > 0:
        buy_signals['close'] = data.loc[buy_signals.index, 'close']  # Get close prices
        shifted_indices = [original_to_shifted[idx] for idx in buy_signals.index]
        ax1.scatter(shifted_indices, buy_signals['close'],
                   marker='^', color='green', s=100, label='Buy Signal')
        for idx, shifted_idx in zip(buy_signals.index, shifted_indices):
            ax1.annotate(f'${buy_signals.loc[idx, "close"]:.2f}',
                        (shifted_idx, buy_signals.loc[idx, 'close']),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom')
    
    # Plot sell signals
    sell_signals = signals[signals['signal'] == -1].copy()
    if len(sell_signals) > 0:
        sell_signals['close'] = data.loc[sell_signals.index, 'close']  # Get close prices
        shifted_indices = [original_to_shifted[idx] for idx in sell_signals.index]
        ax1.scatter(shifted_indices, sell_signals['close'],
                   marker='v', color='red', s=100, label='Sell Signal')
        for idx, shifted_idx in zip(sell_signals.index, shifted_indices):
            ax1.annotate(f'${sell_signals.loc[idx, "close"]:.2f}',
                        (shifted_idx, sell_signals.loc[idx, 'close']),
                        xytext=(0, -10), textcoords='offset points',
                        ha='center', va='top')
    
    # Format x-axis to show dates without gaps
    def format_date(x, p):
        try:
            # Find the closest session start time
            for shifted_time, original_time in session_start_times:
                if abs((pd.Timestamp(x) - shifted_time).total_seconds()) < 300:  # Within 5 minutes
                    # Show full date at session boundaries
                    return original_time.strftime('%Y-%m-%d\n%H:%M')
            
            # For other times, find the corresponding original time
            for shifted_time, original_time in session_start_times:
                if pd.Timestamp(x) >= shifted_time:
                    last_session_start = shifted_time
                    last_original_start = original_time
                    break
            
            time_since_session_start = pd.Timestamp(x) - last_session_start
            original_time = last_original_start + time_since_session_start
            return original_time.strftime('%H:%M')
            
        except Exception as e:
            print(f"Error formatting date: {e}")
            return ''
    
    # Set up axis formatting
    ax1.xaxis.set_major_locator(HourLocator(interval=1))
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(format_date))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    
    ax1.set_title(f'{symbol} Price and Signals - Last {days} Trading Days')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Daily Composite
    ax2 = plt.subplot(3, 1, 2)
    sessions_signals = split_into_sessions(signals)
    last_timestamp = None
    
    for session_data in sessions_signals:
        if last_timestamp is not None:
            gap = pd.Timedelta(minutes=5)
            session_data.index = session_data.index.shift(-1, freq=(session_data.index[0] - (last_timestamp + gap)))
            
        ax2.plot(session_data.index, session_data['daily_composite'], 
                label='Daily Composite' if session_data is sessions_signals[0] else "", 
                color='blue')
        ax2.plot(session_data.index, session_data['daily_up_lim'], '--', 
                label='Upper Limit' if session_data is sessions_signals[0] else "", 
                color='green')
        ax2.plot(session_data.index, session_data['daily_down_lim'], '--', 
                label='Lower Limit' if session_data is sessions_signals[0] else "", 
                color='red')
        
        last_timestamp = session_data.index[-1]
    
    # Apply the same time axis formatting to other plots
    ax2.xaxis.set_major_locator(HourLocator(interval=1))
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(format_date))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    ax2.set_xlim(min(all_timestamps), max(all_timestamps))
    
    # Add vertical lines between sessions
    for boundary in session_boundaries[1:]:
        ax2.axvline(x=boundary, color='gray', linestyle='--', alpha=0.3)
    
    ax2.set_title('Daily Composite Indicator')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Weekly Composite
    ax3 = plt.subplot(3, 1, 3)
    last_timestamp = None
    
    for session_data in sessions_signals:
        if last_timestamp is not None:
            gap = pd.Timedelta(minutes=5)
            session_data.index = session_data.index.shift(-1, freq=(session_data.index[0] - (last_timestamp + gap)))
            
        ax3.plot(session_data.index, session_data['weekly_composite'], 
                label='Weekly Composite' if session_data is sessions_signals[0] else "", 
                color='purple')
        ax3.plot(session_data.index, session_data['weekly_up_lim'], '--', 
                label='Upper Limit' if session_data is sessions_signals[0] else "", 
                color='green')
        ax3.plot(session_data.index, session_data['weekly_down_lim'], '--', 
                label='Lower Limit' if session_data is sessions_signals[0] else "", 
                color='red')
        
        last_timestamp = session_data.index[-1]
    
    # Apply the same time axis formatting to other plots
    ax3.xaxis.set_major_locator(HourLocator(interval=1))
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(format_date))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    ax3.set_xlim(min(all_timestamps), max(all_timestamps))
    
    # Add vertical lines between sessions
    for boundary in session_boundaries[1:]:
        ax3.axvline(x=boundary, color='gray', linestyle='--', alpha=0.3)
    
    ax3.set_title('Weekly Composite Indicator (35-min bars)')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    
    # Convert plot to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    # Calculate trading days (excluding non-market hours)
    unique_dates = pd.Series([idx.date() for idx in data.index]).nunique()
    
    # Prepare statistics
    stats = {
        'buy_signals': len(buy_signals),
        'sell_signals': len(sell_signals),
        'daily_composite_mean': signals['daily_composite'].mean(),
        'daily_composite_std': signals['daily_composite'].std(),
        'weekly_composite_mean': signals['weekly_composite'].mean(),
        'weekly_composite_std': signals['weekly_composite'].std(),
        'current_price': data['close'].iloc[-1],
        'price_change': (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100,
        'trading_days': unique_dates
    }
    
    return buf, stats
