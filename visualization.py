import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
from indicators import generate_signals, get_default_params
import io

def create_strategy_plot(symbol='SPY', days=5):
    """Create a strategy visualization plot and return it as bytes"""
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Fetch data
    data = yf.download(symbol, start=start_date, end=end_date, interval='5m')
    if len(data) == 0:
        raise ValueError(f"No data available for {symbol} in the specified date range")
    
    # Convert column names to lowercase
    data.columns = data.columns.str.lower()
    
    # Generate signals
    params = get_default_params()
    signals, daily_data, weekly_data = generate_signals(data, params)
    
    # Create the plot
    fig = plt.figure(figsize=(15, 12))
    
    # Plot 1: Price and Signals
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(data.index, data['close'], label='Price', color='blue', alpha=0.6)
    
    # Plot buy signals
    buy_signals = signals[signals['signal'] == 1]
    if len(buy_signals) > 0:
        ax1.scatter(buy_signals.index, data.loc[buy_signals.index, 'close'], 
                   marker='^', color='green', s=100, label='Buy Signal')
        # Add price labels for buy signals
        for idx in buy_signals.index:
            ax1.annotate(f'${data.loc[idx, "close"]:.2f}', 
                        (idx, data.loc[idx, 'close']),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom')
    
    # Plot sell signals
    sell_signals = signals[signals['signal'] == -1]
    if len(sell_signals) > 0:
        ax1.scatter(sell_signals.index, data.loc[sell_signals.index, 'close'], 
                   marker='v', color='red', s=100, label='Sell Signal')
        # Add price labels for sell signals
        for idx in sell_signals.index:
            ax1.annotate(f'${data.loc[idx, "close"]:.2f}', 
                        (idx, data.loc[idx, 'close']),
                        xytext=(0, -10), textcoords='offset points',
                        ha='center', va='top')
    
    ax1.set_title(f'{symbol} Price and Signals - Last {days} Days')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Daily Composite
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(signals.index, signals['daily_composite'], label='Daily Composite', color='blue')
    ax2.plot(signals.index, signals['daily_up_lim'], '--', label='Upper Limit', color='green')
    ax2.plot(signals.index, signals['daily_down_lim'], '--', label='Lower Limit', color='red')
    ax2.fill_between(signals.index, signals['daily_up_lim'], signals['daily_down_lim'], 
                     color='gray', alpha=0.1)
    ax2.set_title('Daily Composite Indicator')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Weekly Composite
    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(signals.index, signals['weekly_composite'], label='Weekly Composite', color='purple')
    ax3.plot(signals.index, signals['weekly_up_lim'], '--', label='Upper Limit', color='green')
    ax3.plot(signals.index, signals['weekly_down_lim'], '--', label='Lower Limit', color='red')
    ax3.fill_between(signals.index, signals['weekly_up_lim'], signals['weekly_down_lim'], 
                     color='gray', alpha=0.1)
    ax3.set_title('Weekly Composite Indicator (35-min bars)')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    
    # Convert plot to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    # Prepare statistics
    stats = {
        'buy_signals': len(buy_signals),
        'sell_signals': len(sell_signals),
        'daily_composite_mean': signals['daily_composite'].mean(),
        'daily_composite_std': signals['daily_composite'].std(),
        'weekly_composite_mean': signals['weekly_composite'].mean(),
        'weekly_composite_std': signals['weekly_composite'].std(),
        'current_price': data['close'].iloc[-1],
        'price_change': (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100
    }
    
    return buf, stats
