import pandas as pd
import numpy as np
from indicators import generate_signals, get_default_params
from strategy import TradingStrategy
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import logging
from config import TRADING_SYMBOLS

logger = logging.getLogger(__name__)

class BacktestResult:
    def __init__(self, symbol: str, period: int):
        self.symbol = symbol
        self.period = period
        self.trades = []
        self.positions = []
        self.performance = []
        self.signals = None
        self.data = None
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }

def run_backtest(symbol: str, period: int = 5) -> BacktestResult:
    """
    Run backtest simulation for a symbol
    
    Args:
        symbol: Trading symbol
        period: Number of days to backtest
        
    Returns:
        BacktestResult object containing performance metrics
    """
    try:
        result = BacktestResult(symbol, period)
        
        # Initialize strategy
        strategy = TradingStrategy(symbol)
        
        # Get historical data
        end_date = datetime.now(pytz.UTC)
        start_date = end_date - timedelta(days=period)
        
        # Use yfinance for historical data
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date, interval='5m')
        
        if data.empty:
            raise ValueError(f"No data available for {symbol}")
            
        # Generate signals
        signals, daily_data, weekly_data = generate_signals(data, get_default_params())
        result.signals = signals
        result.data = data
        
        # Initialize simulation variables
        position = 0
        entry_price = 0
        cash = 10000  # Starting with $10,000
        shares = 0
        
        # Track positions and performance
        for i in range(len(signals)):
            current_price = data.iloc[i]['Close']
            signal = signals.iloc[i]['signal']
            
            # Update performance tracking
            portfolio_value = cash + (shares * current_price)
            result.performance.append(portfolio_value)
            result.positions.append(shares)
            
            # Process signals
            if signal == 1 and position == 0:  # Buy signal
                # Calculate position size (2% risk)
                trade_amount = min(portfolio_value * 0.02, cash)
                shares = int(trade_amount / current_price)
                
                if shares > 0:
                    cash -= shares * current_price
                    position = 1
                    entry_price = current_price
                    
                    result.trades.append({
                        'type': 'buy',
                        'price': current_price,
                        'shares': shares,
                        'timestamp': data.index[i]
                    })
                    result.stats['total_trades'] += 1
                    
            elif signal == -1 and position == 1:  # Sell signal
                if shares > 0:
                    cash += shares * current_price
                    trade_return = ((current_price - entry_price) / entry_price) * 100
                    
                    if trade_return > 0:
                        result.stats['winning_trades'] += 1
                    else:
                        result.stats['losing_trades'] += 1
                        
                    result.trades.append({
                        'type': 'sell',
                        'price': current_price,
                        'shares': shares,
                        'timestamp': data.index[i],
                        'return': trade_return
                    })
                    
                    position = 0
                    shares = 0
        
        # Calculate final statistics
        initial_value = result.performance[0]
        final_value = result.performance[-1]
        result.stats['total_return'] = ((final_value - initial_value) / initial_value) * 100
        
        # Calculate max drawdown
        peak = result.performance[0]
        max_drawdown = 0
        
        for value in result.performance:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        result.stats['max_drawdown'] = max_drawdown
        
        # Calculate Sharpe ratio (simplified)
        returns = pd.Series(result.performance).pct_change().dropna()
        if len(returns) > 0:
            result.stats['sharpe_ratio'] = np.sqrt(252) * (returns.mean() / returns.std())
        
        return result
        
    except Exception as e:
        logger.error(f"Error in backtest for {symbol}: {str(e)}")
        raise

def create_backtest_plot(result: BacktestResult) -> tuple:
    """
    Create an interactive plot for backtest results
    
    Returns:
        Tuple of (plot buffer, statistics)
    """
    try:
        # Create figure with secondary y-axis
        fig = make_subplots(rows=3, cols=1, 
                          shared_xaxes=True,
                          vertical_spacing=0.05,
                          subplot_titles=('Price and Signals', 'Portfolio Value', 'Position Size'),
                          row_heights=[0.5, 0.25, 0.25])

        # Add candlestick
        fig.add_trace(
            go.Candlestick(
                x=result.data.index,
                open=result.data['Open'],
                high=result.data['High'],
                low=result.data['Low'],
                close=result.data['Close'],
                name='OHLC'
            ),
            row=1, col=1
        )

        # Add buy signals
        buy_points = [t for t in result.trades if t['type'] == 'buy']
        if buy_points:
            fig.add_trace(
                go.Scatter(
                    x=[t['timestamp'] for t in buy_points],
                    y=[t['price'] for t in buy_points],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(
                        symbol='triangle-up',
                        size=10,
                        color='green'
                    )
                ),
                row=1, col=1
            )

        # Add sell signals
        sell_points = [t for t in result.trades if t['type'] == 'sell']
        if sell_points:
            fig.add_trace(
                go.Scatter(
                    x=[t['timestamp'] for t in sell_points],
                    y=[t['price'] for t in sell_points],
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(
                        symbol='triangle-down',
                        size=10,
                        color='red'
                    )
                ),
                row=1, col=1
            )

        # Add portfolio value
        fig.add_trace(
            go.Scatter(
                x=result.data.index,
                y=result.performance,
                name='Portfolio Value',
                line=dict(color='blue')
            ),
            row=2, col=1
        )

        # Add position size
        fig.add_trace(
            go.Scatter(
                x=result.data.index,
                y=result.positions,
                name='Shares Held',
                line=dict(color='purple')
            ),
            row=3, col=1
        )

        # Update layout
        fig.update_layout(
            title=f'{result.symbol} Backtest Results ({result.period} days)',
            xaxis_title='Date',
            height=1000,
            showlegend=True
        )

        # Update y-axes labels
        fig.update_yaxes(title_text='Price', row=1, col=1)
        fig.update_yaxes(title_text='Portfolio Value ($)', row=2, col=1)
        fig.update_yaxes(title_text='Shares', row=3, col=1)

        # Save to buffer
        buf = io.BytesIO()
        fig.write_image(buf, format='png', width=1200, height=1000)
        buf.seek(0)

        # Prepare statistics
        stats = {
            'symbol': result.symbol,
            'period': result.period,
            'total_trades': result.stats['total_trades'],
            'winning_trades': result.stats['winning_trades'],
            'win_rate': (result.stats['winning_trades'] / result.stats['total_trades'] * 100) if result.stats['total_trades'] > 0 else 0,
            'total_return': result.stats['total_return'],
            'max_drawdown': result.stats['max_drawdown'],
            'sharpe_ratio': result.stats['sharpe_ratio']
        }

        return buf, stats

    except Exception as e:
        logger.error(f"Error creating backtest plot: {str(e)}")
        raise
