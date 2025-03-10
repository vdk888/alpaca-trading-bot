import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import pytz
from indicators import generate_signals, get_default_params
from config import TRADING_SYMBOLS, DEFAULT_INTERVAL, DEFAULT_INTERVAL_WEEKLY, default_interval_yahoo, default_backtest_interval, per_symbol_capital, PER_SYMBOL_CAPITAL_MULTIPLIER, TRADING_COSTS
import matplotlib
matplotlib.use('Agg')  # Use Agg backend - must be before importing pyplot
import matplotlib.pyplot as plt
import io
import matplotlib.dates as mdates
from matplotlib.dates import HourLocator, num2date
import json
from backtest_individual import run_backtest as run_individual_backtest

def is_market_hours(timestamp, market_hours):
    """Check if given timestamp is within market hours"""
    if timestamp.tz is None:
        timestamp = timestamp.tz_localize('UTC')
    
    # Convert to market timezone
    market_tz = pytz.timezone(market_hours['timezone'])
    local_time = timestamp.astimezone(market_tz)
    
    # Parse market hours
    market_start = pd.Timestamp(f"{local_time.date()} {market_hours['start']}").tz_localize(market_tz)
    market_end = pd.Timestamp(f"{local_time.date()} {market_hours['end']}").tz_localize(market_tz)
    
    return market_start <= local_time <= market_end

def run_backtest(symbol: str, days: int = default_backtest_interval, initial_capital: float = 100000) -> dict:
    """Run backtest simulation for a symbol over specified number of days"""
    # Load the best parameters from Object Storage based on the symbol
    try:
        from replit.object_storage import Client
        
        # Initialize Object Storage client
        client = Client()
        
        # Try to get parameters from Object Storage
        try:
            json_content = client.download_as_text("best_params.json")
            best_params_data = json.loads(json_content)
            if symbol in best_params_data:
                params = best_params_data[symbol]['best_params']
                print(f"Using best parameters for {symbol}: {params}")
            else:
                print(f"No best parameters found for {symbol}. Using default parameters.")
                params = get_default_params()
        except Exception as e:
            print(f"Could not read from Object Storage: {e}")
            # Try local file as fallback
            try:
                with open("best_params.json", "r") as f:
                    best_params_data = json.load(f)
                    if symbol in best_params_data:
                        params = best_params_data[symbol]['best_params']
                        print(f"Using best parameters for {symbol}: {params}")
                    else:
                        print(f"No best parameters found for {symbol}. Using default parameters.")
                        params = get_default_params()
            except FileNotFoundError:
                print("Best parameters file not found. Using default parameters.")
                params = get_default_params()
    except Exception as e:
        print(f"Error loading parameters: {e}")
        params = get_default_params()

    # Call the individual backtest with our parameters
    result = run_individual_backtest(symbol=symbol, days=days, params=params, is_simulating=False)
    
    # Add all required data to the DataFrame to maintain compatibility
    data = result['data'].copy()
    
    # Add signals to data
    data['signal'] = result['signals']['signal']
    
    # Scale factor to adjust from 100k to actual initial capital
    scale_factor = initial_capital / 100000.0
    
    # Add shares data (scale down shares proportionally)
    shares_list = result['shares']
    if len(shares_list) == len(data) + 1:  # Individual backtest includes initial position
        shares_list = shares_list[1:]  # Remove initial position
    data['shares'] = [s * scale_factor for s in shares_list]
    
    # Scale down portfolio values
    portfolio_values = result['portfolio_value']
    if len(portfolio_values) == len(data) + 1:  # Remove initial value if present
        portfolio_values = portfolio_values[1:]
    portfolio_values = [pv * scale_factor for pv in portfolio_values]
    
    # Add all required columns exactly as in original implementation
    data['portfolio_value'] = portfolio_values
    data['position_value'] = data['shares'] * data['close']  # Calculate position value
    data['cash'] = data['portfolio_value'] - data['position_value']  # Calculate cash as difference
    
    # Scale down trade values in the trades list
    scaled_trades = []
    for trade in result['trades']:
        scaled_trade = trade.copy()
        scaled_trade['shares'] = trade['shares'] * scale_factor
        scaled_trade['value'] = trade['value'] * scale_factor
        scaled_trade['total_position'] = trade['total_position'] * scale_factor
        scaled_trades.append(scaled_trade)
    
    # Calculate portfolio turnover
    turnover = 0
    if len(portfolio_values) > 1:
        buys = sum(t['value'] for t in scaled_trades if data.loc[t['time'], 'signal'] == 1)
        sells = sum(t['value'] for t in scaled_trades if data.loc[t['time'], 'signal'] == -1)
        avg_portfolio_value = np.mean(portfolio_values)
        turnover = min(buys, sells) / avg_portfolio_value
    
    # Calculate total trading costs
    total_trading_costs = sum(t.get('trading_costs', 0) for t in scaled_trades)
    
    # Add turnover and trading costs to stats
    result['stats']['turnover'] = turnover
    result['stats']['trading_costs'] = total_trading_costs
    
    # Format the result to match the original function's output format exactly
    return {
        'symbol': result['symbol'],
        'data': data,
        'trades': scaled_trades,
        'stats': {
            'initial_capital': initial_capital,  # Use actual initial capital
            'final_value': data['portfolio_value'].iloc[-1],  # Use scaled final value
            'total_return': result['stats']['total_return'],  # Return % stays the same
            'total_trades': result['stats']['total_trades'],
            'win_rate': result['stats']['win_rate'],
            'max_drawdown': result['stats']['max_drawdown'],
            'sharpe_ratio': result['stats']['sharpe_ratio'],
            'turnover': turnover,
            'trading_costs': total_trading_costs
        }
    }

def run_portfolio_backtest(symbols: list, days: int = default_backtest_interval, progress_callback=None) -> dict:
    """Run backtest simulation for multiple symbols as a portfolio"""
    # Set much higher per-symbol capital to allow for greater exposure
    initial_capital = 100000  # Total portfolio capital
    per_symbol_capital = initial_capital / len(symbols) * PER_SYMBOL_CAPITAL_MULTIPLIER
    
    # Run individual backtests
    individual_results = {}
    all_dates = set()
    for symbol in symbols:
        # Call progress callback if provided
        if progress_callback:
            progress_callback(symbol)
            
        result = run_backtest(symbol, days, initial_capital=per_symbol_capital)
        individual_results[symbol] = result
        all_dates.update(result['data'].index)
    
    # Create unified timeline
    timeline = sorted(all_dates)
    portfolio_data = pd.DataFrame(index=timeline)
    
    # Initialize portfolio tracking
    portfolio_data['total_value'] = 0
    portfolio_data['total_cash'] = 0
    
    # Aggregate data from all symbols
    for symbol in symbols:
        result = individual_results[symbol]
        symbol_data = result['data']
        
        # Forward fill symbol data to match portfolio timeline
        symbol_data = symbol_data.reindex(timeline).ffill()
        
        # Add symbol-specific columns
        portfolio_data[f'{symbol}_price'] = symbol_data['close']
        portfolio_data[f'{symbol}_shares'] = symbol_data['shares']
        portfolio_data[f'{symbol}_value'] = symbol_data['position_value']
        portfolio_data[f'{symbol}_cash'] = symbol_data['cash']
        portfolio_data[f'{symbol}_signal'] = symbol_data['signal']
        
        # Add to portfolio totals
        portfolio_data['total_value'] += symbol_data['position_value']
        portfolio_data['total_cash'] += symbol_data['cash'] - (per_symbol_capital - initial_capital / len(symbols))
    
    # Calculate portfolio metrics
    portfolio_data['portfolio_total'] = portfolio_data['total_value'] + portfolio_data['total_cash']
    
    # Calculate returns and drawdown
    portfolio_data['portfolio_return'] = (portfolio_data['portfolio_total'] / initial_capital - 1) * 100
    portfolio_data['high_watermark'] = portfolio_data['portfolio_total'].cummax()
    portfolio_data['drawdown'] = (portfolio_data['portfolio_total'] - portfolio_data['high_watermark']) / portfolio_data['high_watermark'] * 100
    
    # Save complete dataset
    portfolio_data.to_csv('portfolio backtest.csv')
    
    # Prepare result dictionary
    result = {
        'data': portfolio_data,
        'individual_results': individual_results,
        'metrics': {
            'initial_capital': initial_capital,
            'final_value': portfolio_data['portfolio_total'].iloc[-1],
            'total_return': portfolio_data['portfolio_return'].iloc[-1],
            'max_drawdown': portfolio_data['drawdown'].min(),
            'symbol_returns': {
                symbol: (individual_results[symbol]['data']['portfolio_value'].iloc[-1] -
                        per_symbol_capital) / per_symbol_capital * 100
                for symbol in symbols
            }
        }
    }
    
    # Calculate portfolio trading costs
    total_trading_costs = 0
    
    # Calculate actual allocation percentages for each symbol over time
    symbol_allocations = {}
    for symbol in symbols:
        symbol_values = portfolio_data[f'{symbol}_value']
        total_portfolio = portfolio_data['portfolio_total']
        symbol_allocations[symbol] = symbol_values / total_portfolio
    
    # Calculate scaled trading costs based on actual portfolio allocation
    for symbol in symbols:
        symbol_trades = individual_results[symbol]['trades']
        market_type = TRADING_SYMBOLS[symbol]['market']
        costs = TRADING_COSTS.get(market_type, TRADING_COSTS['DEFAULT'])
        
        for trade in symbol_trades:
            # Get the allocation percentage at the time of the trade
            trade_time = trade['time']
            if trade_time in symbol_allocations[symbol].index:
                allocation_pct = symbol_allocations[symbol].loc[trade_time]
                # If allocation is zero, use the next available value
                if pd.isna(allocation_pct) or allocation_pct == 0:
                    allocation_pct = symbol_allocations[symbol].loc[trade_time:].dropna().iloc[0] if not symbol_allocations[symbol].loc[trade_time:].dropna().empty else 0
            else:
                # If time not found, use the closest previous time
                prev_times = symbol_allocations[symbol].index[symbol_allocations[symbol].index <= trade_time]
                allocation_pct = symbol_allocations[symbol].loc[prev_times[-1]] if len(prev_times) > 0 else 0
            
            # Scale the trading costs by the actual allocation percentage
            if 'trading_costs' in trade:
                scaled_cost = trade['trading_costs'] * (allocation_pct if allocation_pct > 0 else 1/len(symbols))
            else:
                if trade['type'] == 'buy':
                    # For buys: full spread + fee
                    cost = trade['value'] * (costs['trading_fee'] + costs['spread'])
                else:
                    # For sells: half spread + fee
                    cost = trade['value'] * (costs['trading_fee'] + costs['spread'] / 2)
                scaled_cost = cost * (allocation_pct if allocation_pct > 0 else 1/len(symbols))
            
            total_trading_costs += scaled_cost

    # Add trading costs to metrics
    result['metrics']['trading_costs'] = total_trading_costs
    
    # Calculate portfolio turnover based on actual portfolio changes
    if len(portfolio_data) > 1:
        # Calculate the sum of actual position changes (not individual trades)
        position_values = {}
        for symbol in symbols:
            position_values[symbol] = portfolio_data[f'{symbol}_value']
        
        # Calculate daily position changes
        daily_changes = pd.DataFrame(index=portfolio_data.index)
        for symbol in symbols:
            daily_changes[symbol] = position_values[symbol].diff().abs()
        
        # Sum up all absolute changes
        total_position_changes = daily_changes.sum().sum() / 2  # Divide by 2 to count only buys or sells
        
        # Calculate average portfolio value
        avg_portfolio_value = portfolio_data['portfolio_total'].mean()
        
        # Calculate turnover as total changes divided by average portfolio value
        turnover = total_position_changes / avg_portfolio_value if avg_portfolio_value > 0 else 0
        
        turnover_metrics = {
            'turnover': turnover,
            'total_position_changes': total_position_changes,
            'avg_portfolio_value': avg_portfolio_value
        }
        
        result['metrics']['turnover'] = turnover_metrics
    
    return result

def split_into_sessions(data):
    """Split data into continuous market sessions"""
    sessions = []
    current_session = []
    
    for idx, row in data.iterrows():
        if not current_session or (idx - current_session[-1].name).total_seconds() <= 300:  # 5 minutes
            current_session.append(row)
        else:
            sessions.append(pd.DataFrame(current_session))
            current_session = [row]
    
    if current_session:
        sessions.append(pd.DataFrame(current_session))
    
    return sessions

def create_backtest_plot(backtest_result: dict) -> tuple:
    """Create visualization of backtest results"""
    data = backtest_result['data']
    signals = backtest_result['data']['signal']
    daily_data = None
    weekly_data = None
    portfolio_value = backtest_result['data']['portfolio_value']
    shares_owned = backtest_result['data']['shares']
    stats = backtest_result['stats']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 20))
    gs = plt.GridSpec(4, 1, height_ratios=[3, 1.5, 1.5, 3], hspace=0.3)
    
    # Plot 1: Price and Signals
    ax1 = plt.subplot(gs[0])
    ax1_volume = ax1.twinx()
    
    # Split data into sessions
    sessions = split_into_sessions(data)
    
    # Plot each session separately
    all_timestamps = []
    session_boundaries = []
    last_timestamp = None
    shifted_data = pd.DataFrame()
    session_start_times = []
    
    # Plot each session
    for i, session in enumerate(sessions):
        session_df = session.copy()
        
        if last_timestamp is not None:
            # Add a small gap between sessions
            gap = pd.Timedelta(minutes=5)
            time_shift = (last_timestamp + gap) - session_df.index[0]
            session_df.index = session_df.index + time_shift
        
        # Store original and shifted start times
        session_start_times.append((session_df.index[0], session.index[0]))
        
        # Plot price
        ax1.plot(session_df.index, session_df['close'], color='blue', alpha=0.7)
        
        # Plot volume
        volume_data = session_df['volume'].rolling(window=5).mean()
        ax1_volume.fill_between(session_df.index, volume_data, color='gray', alpha=0.3)
        
        all_timestamps.extend(session_df.index)
        session_boundaries.append(session_df.index[0])
        last_timestamp = session_df.index[-1]
        shifted_data = pd.concat([shifted_data, session_df])
    
    # Create timestamp mapping for signals
    original_to_shifted = {}
    for orig_session, shifted_session in zip(sessions, session_boundaries):
        time_diff = shifted_session - orig_session.index[0]
        for orig_time in orig_session.index:
            original_to_shifted[orig_time] = orig_time + time_diff
    
    # Plot signals with correct timestamps
    buy_signals = data[data['signal'] == 1]
    sell_signals = data[data['signal'] == -1]
    
    for signals_df, color, marker, va, offset in [
        (buy_signals, 'green', '^', 'bottom', 10),
        (sell_signals, 'red', 'v', 'top', -10)
    ]:
        if len(signals_df) > 0:
            signals_df = signals_df.copy()
            signals_df['close'] = data.loc[signals_df.index, 'close']
            shifted_indices = [original_to_shifted[idx] for idx in signals_df.index]
            ax1.scatter(shifted_indices, signals_df['close'], 
                       color=color, marker=marker, s=100)
            
            for idx, shifted_idx in zip(signals_df.index, shifted_indices):
                ax1.annotate(f'${signals_df.loc[idx, "close"]:.2f}',
                            (shifted_idx, signals_df.loc[idx, "close"]),
                            xytext=(0, offset), textcoords='offset points',
                            ha='center', va=va, color=color)
    
    # Format x-axis
    def format_date(x, p):
        try:
            x_ts = pd.Timestamp(num2date(x, tz=pytz.UTC))
            
            # Find the closest session start time
            for shifted_time, original_time in session_start_times:
                if abs((x_ts - shifted_time).total_seconds()) < 300:
                    return original_time.strftime('%Y-%m-%d\n%H:%M')
            
            # For other times, find the corresponding original time
            for shifted_time, original_time in session_start_times:
                if x_ts >= shifted_time:
                    last_session_start = shifted_time
                    last_original_start = original_time
                    break
            else:
                return ''
            
            time_since_session_start = x_ts - last_session_start
            original_time = last_original_start + time_since_session_start
            return original_time.strftime('%H:%M')
            
        except Exception:
            return ''
    
    ax1.xaxis.set_major_locator(HourLocator(interval=1))
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(format_date))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    ax1.set_title('Price Action with Trading Signals')
    ax1.set_ylabel('Price')
    ax1_volume.set_ylabel('Volume')
    ax1.legend(['Price', 'Buy Signal', 'Sell Signal'])
    
    # Plot 2: Daily Composite (reduced height)
    ax2 = plt.subplot(gs[1])
    sessions_daily = split_into_sessions(daily_data)
    last_timestamp = None
    
    for session_data in sessions_daily:
        if last_timestamp is not None:
            gap = pd.Timedelta(minutes=5)
            session_data.index = session_data.index.shift(-1, freq=(session_data.index[0] - (last_timestamp + gap)))
        
        ax2.plot(session_data.index, session_data['Composite'], color='blue')
        ax2.plot(session_data.index, session_data['Up_Lim'], '--', color='green', alpha=0.6)
        ax2.plot(session_data.index, session_data['Down_Lim'], '--', color='red', alpha=0.6)
        ax2.fill_between(session_data.index, session_data['Up_Lim'], session_data['Down_Lim'], 
                        color='gray', alpha=0.1)
        last_timestamp = session_data.index[-1]
    
    ax2.set_title('Daily Composite Indicator')
    ax2.legend(['Daily Composite', 'Upper Limit', 'Lower Limit'])
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(format_date))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 3: Weekly Composite (reduced height)
    ax3 = plt.subplot(gs[2])
    sessions_weekly = split_into_sessions(weekly_data)
    last_timestamp = None
    
    for session_data in sessions_weekly:
        if last_timestamp is not None:
            gap = pd.Timedelta(minutes=5)
            session_data.index = session_data.index.shift(-1, freq=(session_data.index[0] - (last_timestamp + gap)))
        
        ax3.plot(session_data.index, session_data['Composite'], color='purple')
        ax3.plot(session_data.index, session_data['Up_Lim'], '--', color='green', alpha=0.6)
        ax3.plot(session_data.index, session_data['Down_Lim'], '--', color='red', alpha=0.6)
        ax3.fill_between(session_data.index, session_data['Up_Lim'], session_data['Down_Lim'], 
                        color='gray', alpha=0.1)
        last_timestamp = session_data.index[-1]
    
    ax3.set_title('Weekly Composite Indicator')
    ax3.legend(['Weekly Composite', 'Upper Limit', 'Lower Limit'])
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(format_date))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 4: Portfolio Performance and Position Size
    ax4 = plt.subplot(gs[3])
    ax4_shares = ax4.twinx()
    
    # Create a DataFrame with portfolio data
    portfolio_df = pd.DataFrame({
        'value': portfolio_value[1:],  # Skip initial value
        'shares': shares_owned[1:]  # Skip initial shares
    }, index=data.index)
    
    # Split portfolio data into sessions
    sessions_portfolio = split_into_sessions(portfolio_df)
    last_timestamp = None
    
    for session_data in sessions_portfolio:
        if last_timestamp is not None:
            gap = pd.Timedelta(minutes=5)
            session_data.index = session_data.index.shift(-1, freq=(session_data.index[0] - (last_timestamp + gap)))
        
        ax4.plot(session_data.index, session_data['value'], color='green')
        ax4_shares.plot(session_data.index, session_data['shares'], color='blue', alpha=0.5)
        last_timestamp = session_data.index[-1]
    
    ax4.set_ylabel('Portfolio Value ($)')
    ax4_shares.set_ylabel('Shares Owned')
    ax4.set_title('Portfolio Performance and Position Size')
    
    # Add both legends
    ax4_shares.legend(['Portfolio Value', 'Shares Owned'], loc='upper left')
    
    ax4.xaxis.set_major_formatter(plt.FuncFormatter(format_date))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save plot to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    return buf, backtest_result['stats']

def create_portfolio_backtest_plot(backtest_result: dict) -> io.BytesIO:
    """Create visualization of portfolio backtest results"""
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])
    
    data = backtest_result['data']
    
    # Calculate benchmark (equal-weight portfolio)
    price_columns = [col for col in data.columns if col.endswith('_price')]
    initial_prices = data[price_columns].iloc[0]
    
    # Calculate returns for each asset
    asset_returns = data[price_columns].div(initial_prices) - 1
    
    # Equal-weight benchmark return
    benchmark_return = asset_returns.mean(axis=1)
    initial_capital = backtest_result['metrics']['initial_capital']
    benchmark_value = (1 + benchmark_return) * initial_capital
    
    # Portfolio Performance Plot (top)
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(data.index, data['portfolio_total'], 
             label='Portfolio Value', linewidth=2, color='blue')
    ax1.plot(data.index, benchmark_value,
             label='Benchmark (Equal-Weight)', linewidth=2, color='red', linestyle='--')
    
    # Format y-axis to show dollar values
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${int(x):,}'))
    
    # Add some padding to y-axis
    ymin = min(data['portfolio_total'].min(), benchmark_value.min()) * 0.99
    ymax = max(data['portfolio_total'].max(), benchmark_value.max()) * 1.01
    ax1.set_ylim(ymin, ymax)
    
    ax1.set_title('Portfolio Performance')
    ax1.set_ylabel('Total Value ($)')
    ax1.grid(True)
    ax1.legend()
    
    # Asset Allocation Plot (bottom)
    ax2 = fig.add_subplot(gs[1])
    
    # Calculate percentage allocation for each symbol and cash
    # Filter out columns that end with '_value' but exclude 'total_value'
    symbol_values = [col for col in data.columns if col.endswith('_value') 
                    and not col.startswith('total')]
    symbols = [col.replace('_value', '') for col in symbol_values]
    
    # Include both position values and cash in total
    total_portfolio = data[symbol_values].sum(axis=1) + data['total_cash']
    allocations = []
    
    # Add cash allocation first
    cash_allocation = (data['total_cash'] / total_portfolio * 100).fillna(0)
    allocations.append(cash_allocation)
    
    # Add symbol allocations
    for symbol in symbols:
        allocation = (data[f'{symbol}_value'] / total_portfolio * 100).fillna(0)
        allocations.append(allocation)
    
    # Plot stacked area chart for allocations with cash
    ax2.stackplot(data.index, allocations, labels=['Cash'] + symbols, alpha=0.8)
    
    ax2.set_title('Asset Allocation')
    ax2.set_ylabel('Allocation (%)')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.grid(True)
    
    # Format x-axis for both plots
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Adjust layout to prevent overlapping
    plt.tight_layout()
    
    # Save to buffer with high DPI for better quality
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf

def create_portfolio_with_prices_plot(backtest_result: dict) -> io.BytesIO:
    """Create visualization of portfolio value with individual asset prices, all normalized to base 100"""
    # Create figure
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111)
    
    data = backtest_result['data']
    
    # Get symbol columns (those ending with '_price')
    symbol_prices = [col for col in data.columns if col.endswith('_price')]
    
    # Plot individual assets first (behind portfolio line)
    for price_col in symbol_prices:
        symbol = price_col.replace('_price', '')
        # Get first non-NaN value for normalization
        initial_price = data[price_col].dropna().iloc[0]
        normalized_prices = data[price_col] / initial_price * 100
        
        # Check if symbol is crypto or ETF
        is_crypto = TRADING_SYMBOLS[symbol]['market'] == 'CRYPTO'
        
        ax.plot(data.index, normalized_prices, 
                label=f'{symbol} Price', 
                alpha=0.3,
                linestyle='' if is_crypto else '--',  # Solid for crypto, dashed for ETF
                marker='.' if is_crypto else None,    # Dots for crypto
                markersize=1 if is_crypto else None,  # Small dots
                zorder=1)  # Put asset lines behind portfolio line
    
    # Normalize portfolio value to base 100 using first non-NaN value
    initial_portfolio = data['portfolio_total'].dropna().iloc[0]
    normalized_portfolio = data['portfolio_total'] / initial_portfolio * 100
    
    # Plot portfolio value last (on top) with increased visibility
    ax.plot(data.index, normalized_portfolio, 
            label='Portfolio Value', 
            linewidth=4.0,
            color='navy',
            alpha=1.0,
            zorder=10)
    
    # Format y-axis to show percentage values
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}'))
    
    ax.set_title('Portfolio and Asset Performance (Base 100)')
    ax.set_ylabel('Value (Base 100)')
    ax.grid(True, alpha=0.2)
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Add legend with portfolio first
    handles, labels = ax.get_legend_handles_labels()
    if 'Portfolio Value' in labels:
        portfolio_idx = labels.index('Portfolio Value')
        handles = [handles[portfolio_idx]] + handles[:portfolio_idx] + handles[portfolio_idx+1:]
        labels = [labels[portfolio_idx]] + labels[:portfolio_idx] + labels[portfolio_idx+1:]
    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    
    # Adjust layout to prevent overlapping
    plt.tight_layout()
    
    # Save to buffer with high DPI for better quality
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf
