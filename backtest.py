import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import pytz
from indicators import generate_signals, get_default_params
from config import TRADING_SYMBOLS
import matplotlib.pyplot as plt
import io
import matplotlib.dates as mdates
from matplotlib.dates import HourLocator, num2date

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

def run_backtest(symbol: str, days: int = 5, initial_capital: float = 100000) -> dict:
    """Run backtest simulation for a symbol over specified number of days"""
    # Get symbol configuration
    symbol_config = TRADING_SYMBOLS[symbol]
    yf_symbol = symbol_config['yfinance']
    
    # Handle crypto symbols with forward slashes
    if '/' in yf_symbol:
        yf_symbol = yf_symbol.replace('/', '-')

    # Calculate date range
    end_date = datetime.now(pytz.UTC)
    start_date = end_date - timedelta(days=days + 2)  # Add buffer days
    
    # Fetch historical data
    ticker = yf.Ticker(yf_symbol)
    data = ticker.history(
        start=start_date,
        end=end_date,
        interval=symbol_config.get('interval', '5m'),
        actions=False
    )
    
    if len(data) == 0:
        raise ValueError(f"No data available for {symbol} in the specified date range")
    
    # Localize timezone if needed
    if data.index.tz is None:
        data.index = data.index.tz_localize('UTC')
    
    # Filter for market hours
    data = data[data.index.map(lambda x: is_market_hours(x, symbol_config['market_hours']))]
    data.columns = data.columns.str.lower()
    
    # Generate signals
    params = get_default_params()
    signals, daily_data, weekly_data = generate_signals(data, params)
    
    # Add signals to data
    data['signal'] = signals['signal']
    
    # Initialize portfolio tracking
    data['shares'] = 0  # Current position in shares
    data['cash'] = initial_capital  # Available cash
    data['position_value'] = 0  # Value of current position
    data['portfolio_value'] = initial_capital  # Total portfolio value
    
    position = 0  # Current position in shares
    cash = initial_capital
    trades = []  # Track individual trades
    
    # Simulate trading
    for i in range(len(data)):
        current_price = data['close'].iloc[i]
        current_time = data.index[i]
        
        # Update position value
        position_value = position * current_price
        
        if i > 0:  # Skip first bar for signal processing
            signal = signals['signal'].iloc[i]
            
            # Process signals
            if signal == 1:  # Buy signal
                # Calculate maximum position value (100% of initial capital)
                max_position_value = initial_capital
                
                # If total position is less than max, allow adding 20% more
                if position_value < max_position_value:
                    # Calculate position size as 20% of initial capital
                    capital_to_use = initial_capital * 0.20
                    shares_to_buy = capital_to_use / current_price
                    
                    # Round based on market type
                    if symbol_config['market'] == 'CRYPTO':
                        shares_to_buy = round(shares_to_buy, 8)  # Round to 8 decimal places for crypto
                    else:
                        shares_to_buy = int(shares_to_buy)  # Round down to whole shares for stocks
                    
                    # Ensure minimum position size
                    min_qty = 1 if symbol_config['market'] != 'CRYPTO' else 0.0001
                    if shares_to_buy < min_qty:
                        shares_to_buy = min_qty
                    
                    # Check if adding this position would exceed max position value
                    new_total_value = position_value + (shares_to_buy * current_price)
                    if new_total_value > max_position_value:
                        # Adjust shares to not exceed max position
                        shares_to_buy = (max_position_value - position_value) / current_price
                        if symbol_config['market'] == 'CRYPTO':
                            shares_to_buy = round(shares_to_buy, 8)
                        else:
                            shares_to_buy = int(shares_to_buy)
                    
                    cost = shares_to_buy * current_price
                    if cost <= cash and shares_to_buy > 0:  # Check if we have enough cash and shares to buy
                        position += shares_to_buy  # Add to existing position
                        cash -= cost
                        trades.append({
                            'time': current_time,
                            'type': 'buy',
                            'price': current_price,
                            'shares': shares_to_buy,
                            'value': cost,
                            'total_position': position
                        })
            
            elif signal == -1 and position > 0:  # Sell signal
                # Sell entire position
                sale_value = position * current_price
                cash += sale_value
                trades.append({
                    'time': current_time,
                    'type': 'sell',
                    'price': current_price,
                    'shares': position,
                    'value': sale_value,
                    'total_position': 0
                })
                position = 0
        
        # Update data for this timestamp
        data.loc[current_time, 'shares'] = position
        data.loc[current_time, 'cash'] = cash
        data.loc[current_time, 'position_value'] = position * current_price
        data.loc[current_time, 'portfolio_value'] = cash + (position * current_price)
    
    # Calculate performance metrics
    final_value = cash + (position * data['close'].iloc[-1])
    total_return = ((final_value - initial_capital) / initial_capital) * 100
    
    if trades:
        trades_df = pd.DataFrame(trades)
        if len(trades_df) > 0:
            # Calculate win rate
            buy_trades = trades_df[trades_df['type'] == 'buy']
            sell_trades = trades_df[trades_df['type'] == 'sell']
            
            if len(buy_trades) > 0 and len(sell_trades) > 0:
                min_trades = min(len(buy_trades), len(sell_trades))
                profits = sell_trades['value'].iloc[:min_trades].values - buy_trades['value'].iloc[:min_trades].values
                win_rate = (len(profits[profits > 0]) / len(profits)) * 100 if len(profits) > 0 else 0
            else:
                win_rate = 0
        else:
            win_rate = 0
        
        # Calculate max drawdown
        portfolio_series = data['portfolio_value']
        rolling_max = portfolio_series.expanding().max()
        drawdowns = (portfolio_series - rolling_max) / rolling_max * 100
        max_drawdown = abs(drawdowns.min())
        
        # Calculate Sharpe Ratio (assuming risk-free rate of 2%)
        returns = data['portfolio_value'].pct_change().dropna()
        excess_returns = returns - 0.02/252  # Daily risk-free rate
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if len(returns) > 0 else 0
    else:
        win_rate = 0
        max_drawdown = 0
        sharpe_ratio = 0
    
    return {
        'symbol': symbol,
        'data': data,
        'trades': trades,
        'stats': {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }
    }

def run_portfolio_backtest(symbols: list, days: int = 5) -> dict:
    """Run backtest simulation for multiple symbols as a portfolio"""
    # Calculate per-symbol capital
    initial_capital = 100000  # Total portfolio capital
    per_symbol_capital = initial_capital / len(symbols)
    
    # Run individual backtests
    individual_results = {}
    all_dates = set()
    for symbol in symbols:
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
        portfolio_data['total_cash'] += symbol_data['cash']
    
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
    
    # Portfolio Performance Plot (top)
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(data.index, data['portfolio_total'], 
             label='Portfolio Value', linewidth=2, color='blue')
    
    # Format y-axis to show dollar values
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${int(x):,}'))
    
    # Add some padding to y-axis
    ymin = data['portfolio_total'].min() * 0.99
    ymax = data['portfolio_total'].max() * 1.01
    ax1.set_ylim(ymin, ymax)
    
    ax1.set_title('Portfolio Performance')
    ax1.set_ylabel('Total Value ($)')
    ax1.grid(True)
    
    # Asset Allocation Plot (bottom)
    ax2 = fig.add_subplot(gs[1])
    
    # Calculate percentage allocation for each symbol
    # Filter out columns that end with '_value' but exclude 'total_value'
    symbol_values = [col for col in data.columns if col.endswith('_value') 
                    and not col.startswith('total')]
    symbols = [col.replace('_value', '') for col in symbol_values]
    
    # Calculate allocations based on position values only
    total_invested = data[symbol_values].sum(axis=1)
    allocations = []
    
    for symbol in symbols:
        # Avoid division by zero
        safe_total = total_invested.replace(0, np.nan)
        allocation = (data[f'{symbol}_value'] / safe_total * 100).fillna(0)
        allocations.append(allocation)
    
    # Plot stacked area chart for allocations
    ax2.stackplot(data.index, allocations, labels=symbols, alpha=0.8)
    
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
