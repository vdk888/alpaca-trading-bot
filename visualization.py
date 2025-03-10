import matplotlib.pyplot as plt
from matplotlib.dates import HourLocator, DateFormatter, num2date
from datetime import datetime, timedelta
import yfinance as yf
from indicators import generate_signals, get_default_params
import io
import pandas as pd
import pytz
import numpy as np
import logging
import matplotlib
import json
import config

logger = logging.getLogger(__name__)

def is_market_hours(timestamp, market_config):
    """Check if timestamp is during market hours for the specific market"""
    market_tz = pytz.timezone(market_config['timezone'])
    ts_market = timestamp.astimezone(market_tz)
    
    # Parse market hours
    start_time = datetime.strptime(market_config['start'], '%H:%M').time()
    end_time = datetime.strptime(market_config['end'], '%H:%M').time()
    
    # For 24/7 markets like Forex
    if start_time == datetime.strptime('00:00', '%H:%M').time() and \
       end_time == datetime.strptime('23:59', '%H:%M').time():
        return True
    
    # Check if it's a weekday
    if ts_market.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False
    
    # Create datetime objects for comparison
    market_start = ts_market.replace(
        hour=start_time.hour,
        minute=start_time.minute,
        second=0,
        microsecond=0
    ).replace(tzinfo=ts_market.tzinfo)  # Preserve timezone
    
    market_end = ts_market.replace(
        hour=end_time.hour,
        minute=end_time.minute,
        second=0,
        microsecond=0
    ).replace(tzinfo=ts_market.tzinfo)  # Preserve timezone
    
    return market_start <= ts_market <= market_end

def split_into_sessions(data):
    """Split data into continuous market sessions"""
    if len(data) == 0:
        return []
    
    # Ensure data index is timezone-aware
    if data.index.tz is None:
        data.index = data.index.tz_localize('UTC')
    
    # Calculate typical time interval between data points
    if len(data) > 1:
        typical_interval = (data.index[1] - data.index[0]).total_seconds() / 60
    else:
        typical_interval = 1  # Default to 1 minute if there's only one data point
    
    sessions = []
    current_session = []
    last_timestamp = None
    
    for timestamp, row in data.iterrows():
        if last_timestamp is not None:
            time_diff = (timestamp - last_timestamp).total_seconds() / 60
            if time_diff > typical_interval * 10:  # More lenient gap threshold
                if current_session:
                    sessions.append(pd.DataFrame(current_session))
                current_session = []
        
        current_session.append(row)
        last_timestamp = timestamp
    
    if current_session:
        sessions.append(pd.DataFrame(current_session))
    
    return sessions

def plot_symbol_data(df, symbol, days=None):
    """Create a plot of symbol price data with indicators."""
    plt.figure(figsize=(12, 8))
    
    # Plot price data
    plt.plot(df.index, df['close'], label='Price')
    
    # Plot indicators if they exist
    if 'SMA_20' in df.columns:
        plt.plot(df.index, df['SMA_20'], label='SMA 20', alpha=0.7)
    if 'SMA_50' in df.columns:
        plt.plot(df.index, df['SMA_50'], label='SMA 50', alpha=0.7)
    
    period_text = f"{days} days" if days else "Period"
    plt.title(f'{symbol} Price Chart - {period_text}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.legend()
    
    # Format dates on x-axis
    plt.gcf().autofmt_xdate()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf

def create_strategy_plot(symbol='SPY', days=5, return_data=False):
    """Create a strategy visualization plot for a single symbol and return it as bytes"""
    # Get the correct Yahoo Finance symbol and market configuration
    from config import TRADING_SYMBOLS
    symbol_config = TRADING_SYMBOLS[symbol]
    yf_symbol = symbol_config['yfinance']
    
    # Calculate date range with extra days to account for market closures
    end_date = datetime.now(pytz.UTC)
    start_date = end_date - timedelta(days=days + 2)  # Add buffer days
    
    # Create Ticker object
    ticker = yf.Ticker(yf_symbol)
    
    try:
        # Fetch data with explicit columns
        data = ticker.history(
            start=start_date,
            end=end_date,
            interval=config.default_interval_yahoo,
            actions=True
        )
        
        logger.info(f"Fetched data for {symbol} ({yf_symbol}): {len(data)} rows")
        logger.info(f"Columns: {data.columns.tolist()}")
        
        if len(data) == 0:
            raise ValueError(f"No data available for {symbol} ({yf_symbol}) in the specified date range")
        
        # Localize timezone if not already localized
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC')
        
        # Filter for market hours only if not a 24/7 market
        if symbol_config['market_hours']['start'] != '00:00' and symbol_config['market_hours']['end'] != '23:59':
            data = data[data.index.map(lambda x: is_market_hours(x, symbol_config['market_hours']))]
        
        logger.info(f"Data after market hours filtering: {len(data)} rows")
        
        # Ensure we have enough data after filtering
        if len(data) == 0:
            raise ValueError(f"No market hours data available for {symbol} in the specified date range")
        
        # Convert column names to lowercase
        data.columns = data.columns.str.lower()
        
        # Ensure required columns exist
        required_columns = ['close', 'open', 'high', 'low', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}. Available columns: {data.columns.tolist()}")
        
        # Generate signals
        try:
            with open("best_params.json", "r") as f:
                best_params_data = json.load(f)
                if symbol in best_params_data:
                    params = best_params_data[symbol]['best_params']
                    logger.info(f"Using best parameters for {symbol}: {params}")
                else:
                    logger.info(f"No best parameters found for {symbol}. Using default parameters.")
                    params = get_default_params()
        except FileNotFoundError:
            logger.warning("Best parameters file not found. Using default parameters.")
            params = get_default_params()
        
        signals, daily_data, weekly_data = generate_signals(data, params)
        
        logger.info(f"Generated signals: {len(signals)} rows")
        logger.info(f"Signal distribution: {signals['signal'].value_counts().to_dict()}")
        
        if return_data:
            return data, signals
        
        # Calculate trading days using pandas
        trading_days = len(pd.Series([idx.date() for idx in data.index]).unique())
        
        if trading_days == 0:
            raise ValueError(f"No trading days found for {symbol} in the specified date range")
            
        # Calculate statistics
        stats = {
            'trading_days': trading_days,
            'price_change': ((data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0] * 100),
            'buy_signals': len(signals[signals['signal'] == 1]),
            'sell_signals': len(signals[signals['signal'] == -1])
        }
        
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
                # Ensure both timestamps are timezone-aware
                session_start = session_df.index[0]
                if session_start.tz is None:
                    session_start = session_start.tz_localize('UTC')
                if last_timestamp.tz is None:
                    last_timestamp = last_timestamp.tz_localize('UTC')
                time_shift = (last_timestamp + gap) - session_start
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
            # Ensure timestamps are timezone-aware
            orig_start = orig_session.index[0]
            shifted_start = shifted_session
            if orig_start.tz is None:
                orig_start = orig_start.tz_localize('UTC')
            if shifted_start.tz is None:
                shifted_start = shifted_start.tz_localize('UTC')
            time_diff = shifted_start - orig_start
            for orig_time in orig_session.index:
                if orig_time.tz is None:
                    orig_time = orig_time.tz_localize('UTC')
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
                # Convert matplotlib's date format to pandas timestamp
                x_ts = pd.Timestamp(num2date(x, tz=pytz.UTC))
                
                # Find the closest session start time
                for shifted_time, original_time in session_start_times:
                    if abs((x_ts - shifted_time).total_seconds()) < 300:  # Within 5 minutes
                        # Show full date at session boundaries
                        return original_time.strftime('%Y-%m-%d\n%H:%M')
                
                # For other times, find the corresponding original time
                for shifted_time, original_time in session_start_times:
                    if x_ts >= shifted_time:
                        last_session_start = shifted_time
                        last_original_start = original_time
                        break
                else:
                    return ''  # No matching session found
                
                time_since_session_start = x_ts - last_session_start
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
            ax2.plot(session_data.index, session_data['daily_up_lim_2std'], ':', 
                    label='Upper 2 STD' if session_data is sessions_signals[0] else "", 
                    color='green', alpha=0.7)
            ax2.plot(session_data.index, session_data['daily_down_lim_2std'], ':', 
                    label='Lower 2 STD' if session_data is sessions_signals[0] else "", 
                    color='red', alpha=0.7)
            
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
            ax3.plot(session_data.index, session_data['weekly_up_lim_2std'], ':', 
                    label='Upper 2 STD' if session_data is sessions_signals[0] else "", 
                    color='green', alpha=0.7)
            ax3.plot(session_data.index, session_data['weekly_down_lim_2std'], ':', 
                    label='Lower 2 STD' if session_data is sessions_signals[0] else "", 
                    color='red', alpha=0.7)
            
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
        
        # Save plot to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        return buf, stats
        
    except Exception as e:
        logger.error(f"Error processing {symbol} ({yf_symbol}): {str(e)}")
        raise ValueError(f"Error processing {symbol} ({yf_symbol}): {str(e)}")

def create_multi_symbol_plot(symbols=None, days=5):
    """Create strategy visualization plots for multiple symbols and return them as bytes"""
    if symbols is None:
        from config import TRADING_SYMBOLS
        symbols = list(TRADING_SYMBOLS.keys())
    
    # Create subplots based on number of symbols
    n_symbols = len(symbols)
    n_cols = min(2, n_symbols)  # Maximum 2 columns
    n_rows = (n_symbols + 1) // 2  # Ceiling division for number of rows
    
    # Create figure with enough height for all symbols
    fig = plt.figure(figsize=(15 * n_cols, 12 * n_rows))
    
    for idx, symbol in enumerate(symbols):
        try:
            # Calculate subplot position
            ax_idx = idx + 1
            
            # Create Ticker object and fetch data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=config.default_interval_yahoo,
                actions=True
            )
            
            if len(data) == 0:
                continue
                
            # Filter for market hours only
            data = data[data.index.map(lambda x: is_market_hours(x, {'start': '09:30', 'end': '16:00', 'timezone': 'US/Eastern'}))]
            data.columns = data.columns.str.lower()
            
            # Generate signals
            try:
                with open("best_params.json", "r") as f:
                    best_params_data = json.load(f)
                    if self.symbol in best_params_data:
                        params = best_params_data[self.symbol]['best_params']
                        print(f"Using best parameters for {self.symbol}: {params}")
                    else:
                        print(f"No best parameters found for {self.symbol}. Using default parameters.")
                        params = get_default_params()
            except FileNotFoundError:
                print("Best parameters file not found. Using default parameters.")
                params = get_default_params()        

            signals, daily_data, weekly_data = generate_signals(data, params)
            
            # Split data into sessions
            sessions = split_into_sessions(data)
            
            # Create subplots for this symbol
            ax1 = plt.subplot(n_rows, n_cols, ax_idx)
            ax2 = ax1.twinx()
            
            # Plot price and volume
            all_timestamps = []
            for session in sessions:
                timestamps = session.index
                all_timestamps.extend(timestamps)
                ax1.plot(timestamps, session['close'], color='blue', linewidth=1)
                ax2.bar(timestamps, session['volume'], color='gray', alpha=0.3)
            
            # Plot signals
            buy_signals = signals[signals['signal'] == 1].index
            sell_signals = signals[signals['signal'] == -1].index
            ax1.scatter(buy_signals, data.loc[buy_signals, 'close'], color='green', marker='^', s=100, label='Buy')
            ax1.scatter(sell_signals, data.loc[sell_signals, 'close'], color='red', marker='v', s=100, label='Sell')
            
            # Plot daily and weekly composites
            ax3 = plt.subplot(n_rows, n_cols, ax_idx)
            ax3.plot(signals.index, signals['daily_composite'], color='orange', label='Daily Composite')
            ax3.plot(signals.index, signals['weekly_composite'], color='purple', label='Weekly Composite')
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            
            # Customize appearance
            ax1.set_title(f'{symbol} - Price and Signals')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            ax2.set_ylabel('Volume', color='gray')
            ax3.set_title(f'{symbol} - Indicators')
            ax3.legend(loc='upper left')
            ax3.grid(True, alpha=0.3)
            
            # Format x-axis
            ax1.xaxis.set_major_locator(HourLocator(interval=4))
            ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
        except Exception as e:
            plt.text(0.5, 0.5, f'Error plotting {symbol}: {str(e)}', 
                    ha='center', va='center', transform=fig.transFigure)
    
    plt.tight_layout()
    
    # Save plot to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf
