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

def run_individual_backtest(symbol: str, days: int = 5) -> dict:
    """Run backtest simulation for a single symbol over specified number of days"""
    try:
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
        
        if data is None or len(data) == 0:
            raise ValueError(f"No data available for {symbol} in the specified date range")
        
        # Localize timezone if needed
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC')
        
        # Filter for market hours
        data = data[data.index.map(lambda x: is_market_hours(x, symbol_config['market_hours']))]
        if len(data) == 0:
            raise ValueError(f"No market hours data available for {symbol} in the specified date range")
            
        data.columns = data.columns.str.lower()
        
        # Generate signals
        params = get_default_params()
        signals, daily_data, weekly_data = generate_signals(data, params)
        
        if signals is None or len(signals) == 0:
            raise ValueError(f"Could not generate signals for {symbol}")
        
        # Initialize portfolio tracking
        initial_capital = 100000  # $100k initial capital
        position = 0  # Current position in shares
        cash = initial_capital
        portfolio_value = [initial_capital]  # Start with initial capital
        shares_owned = [0]  # Start with no shares
        trades = []  # Track individual trades
        total_position_value = 0  # Track total position value for position sizing
        
        # Simulate trading
        for i in range(len(data)):
            current_price = data['close'].iloc[i]
            current_time = data.index[i]
            
            # Update total position value
            total_position_value = position * current_price
            
            if i > 0:  # Skip first bar for signal processing
                signal = signals['signal'].iloc[i]
                
                # Process signals
                if signal == 1:  # Buy signal
                    # Calculate maximum position value (100% of initial capital)
                    max_position_value = initial_capital
                    
                    # If total position is less than max, allow adding 20% more
                    if total_position_value < max_position_value:
                        # Calculate position size as 20% of initial capital
                        capital_to_use = initial_capital * 0.20
                        shares_to_buy = capital_to_use / current_price
                        
                        # Round based on market type
                        if symbol_config['market'] == 'CRYPTO':
                            shares_to_buy = round(shares_to_buy, 8)  # Round to 8 decimal places for crypto
                        else:
                            shares_to_buy = int(shares_to_buy)  # Round down to whole shares for stocks
                        
                        if shares_to_buy > 0:
                            cost = shares_to_buy * current_price
                            if cost <= cash:  # Check if we have enough cash
                                position += shares_to_buy
                                cash -= cost
                                trades.append({
                                    'time': current_time,
                                    'type': 'buy',
                                    'shares': shares_to_buy,
                                    'price': current_price,
                                    'value': cost
                                })
                
                elif signal == -1:  # Sell signal
                    if position > 0:
                        value = position * current_price
                        trades.append({
                            'time': current_time,
                            'type': 'sell',
                            'shares': position,
                            'price': current_price,
                            'value': value
                        })
                        cash += value
                        position = 0
            
            # Track portfolio value and position
            current_value = cash + (position * current_price)
            portfolio_value.append(current_value)
            shares_owned.append(position)
        
        # Calculate returns and other metrics
        portfolio_value = pd.Series(portfolio_value, index=[data.index[0]] + list(data.index))
        shares_owned = pd.Series(shares_owned, index=[data.index[0]] + list(data.index))
        
        returns = (portfolio_value.iloc[-1] - initial_capital) / initial_capital
        
        # Prepare and return results
        return {
            'symbol': symbol,
            'initial_capital': initial_capital,
            'final_value': portfolio_value.iloc[-1],
            'returns': returns,
            'trades': trades,
            'portfolio_value': portfolio_value,
            'shares_owned': shares_owned,
            'price_data': data,
            'signals': signals
        }
    except Exception as e:
        raise ValueError(f"Error running backtest for {symbol}: {str(e)}")

def create_individual_backtest_plot(backtest_result: dict):
    """Create visualization of backtest results for individual symbol"""
    # Create figure with subplots
    fig = plt.figure(figsize=(12, 8))
    
    # Add GridSpec
    gs = plt.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.3)
    
    # Portfolio value subplot
    ax1 = plt.subplot(gs[0])
    ax1.plot(backtest_result['portfolio_value'].index, backtest_result['portfolio_value'].values, label='Portfolio Value', color='blue')
    
    # Format x-axis
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add grid and legend
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title('Portfolio Value Over Time')
    
    # Price and signals subplot
    ax2 = plt.subplot(gs[1])
    ax2.plot(backtest_result['price_data'].index, backtest_result['price_data']['close'], label='Price', color='gray', alpha=0.7)
    
    # Plot buy signals
    buy_signals = backtest_result['signals'][backtest_result['signals']['signal'] == 1]
    if len(buy_signals) > 0:
        ax2.scatter(buy_signals.index, backtest_result['price_data']['close'][buy_signals.index],
                   marker='^', color='green', label='Buy Signal')
    
    # Plot sell signals
    sell_signals = backtest_result['signals'][backtest_result['signals']['signal'] == -1]
    if len(sell_signals) > 0:
        ax2.scatter(sell_signals.index, backtest_result['price_data']['close'][sell_signals.index],
                   marker='v', color='red', label='Sell Signal')
    
    # Format x-axis
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add grid and legend
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_title('Price and Signals')
    
    # Adjust layout
    plt.tight_layout()
    
    # Calculate statistics
    total_return = ((backtest_result['final_value'] - backtest_result['initial_capital']) / backtest_result['initial_capital']) * 100
    
    # Calculate win rate
    profitable_trades = len([t for t in backtest_result['trades'] if t['type'] == 'sell' and t['value'] > t['shares'] * backtest_result['trades'][backtest_result['trades'].index(t)-1]['price']])
    total_trades = len([t for t in backtest_result['trades'] if t['type'] == 'sell'])
    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Calculate max drawdown
    portfolio_value = backtest_result['portfolio_value']
    rolling_max = portfolio_value.expanding().max()
    drawdowns = (portfolio_value - rolling_max) / rolling_max * 100
    max_drawdown = abs(drawdowns.min())
    
    # Calculate Sharpe ratio (simplified)
    returns = portfolio_value.pct_change().dropna()
    sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 else 0
    
    # Save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    # Return buffer and statistics
    return buf, {
        'total_return': total_return,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio
    }

def create_backtest_plot(backtest_result: dict):
    """Create visualization of backtest results"""
    # Create figure with subplots
    fig = plt.figure(figsize=(12, 8))
    
    # Add GridSpec
    gs = plt.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.3)
    
    # Portfolio value subplot
    ax1 = plt.subplot(gs[0])
    ax1.plot(backtest_result['portfolio_value'].index, backtest_result['portfolio_value'].values, label='Portfolio Value', color='blue')
    
    # Format x-axis
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add grid and legend
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title('Portfolio Value Over Time')
    
    # Price and signals subplot
    ax2 = plt.subplot(gs[1])
    ax2.plot(backtest_result['price_data'].index, backtest_result['price_data']['close'], label='Price', color='gray', alpha=0.7)
    
    # Plot buy signals
    buy_signals = backtest_result['signals'][backtest_result['signals']['signal'] == 1]
    if len(buy_signals) > 0:
        ax2.scatter(buy_signals.index, backtest_result['price_data']['close'][buy_signals.index],
                   marker='^', color='green', label='Buy Signal')
    
    # Plot sell signals
    sell_signals = backtest_result['signals'][backtest_result['signals']['signal'] == -1]
    if len(sell_signals) > 0:
        ax2.scatter(sell_signals.index, backtest_result['price_data']['close'][sell_signals.index],
                   marker='v', color='red', label='Sell Signal')
    
    # Format x-axis
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add grid and legend
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_title('Price and Signals')
    
    # Adjust layout
    plt.tight_layout()
    
    # Calculate statistics
    total_return = ((backtest_result['final_value'] - backtest_result['initial_capital']) / backtest_result['initial_capital']) * 100
    
    # Calculate win rate
    profitable_trades = len([t for t in backtest_result['trades'] if t['type'] == 'sell' and t['value'] > t['shares'] * backtest_result['trades'][backtest_result['trades'].index(t)-1]['price']])
    total_trades = len([t for t in backtest_result['trades'] if t['type'] == 'sell'])
    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Calculate max drawdown
    portfolio_value = backtest_result['portfolio_value']
    rolling_max = portfolio_value.expanding().max()
    drawdowns = (portfolio_value - rolling_max) / rolling_max * 100
    max_drawdown = abs(drawdowns.min())
    
    # Calculate Sharpe ratio (simplified)
    returns = portfolio_value.pct_change().dropna()
    sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 else 0
    
    # Save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    # Return buffer and statistics
    return buf, {
        'total_return': total_return,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio
    }

def split_into_sessions(data: pd.DataFrame, session_length: int = 60) -> list:
    """Split data into sessions of specified length"""
    sessions = []
    for i in range(0, len(data), session_length):
        sessions.append(data.iloc[i:i+session_length])
    return sessions
