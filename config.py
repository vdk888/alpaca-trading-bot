import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
import yfinance as yf

ALPACA_PAPER = True
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')

if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    raise ValueError("Alpaca API credentials not found in environment variables")

# Default trading parameters
DEFAULT_RISK_PERCENT = 0.95
DEFAULT_INTERVAL = '1h' # available intervals: 1min, 5min, 15min, 30min, 1h, 4h, 1d
DEFAULT_INTERVAL_WEEKLY = '4h'

default_interval_yahoo = '1h' # available intervals: 1m, 5m, 15m, 30m, 1h, 4h, 1d

# Bars per day for each interval
BARS_PER_DAY = {
    '1m': 1440,
    '5m': 288,
    '15m': 96,
    '30m': 48,
    '60m': 24,
    '1h': 24,
    '1d': 1
}

# Maximum data points per request
MAX_DATA_POINTS = 2000

# Calculate maximum number of days for a given interval
def get_max_days(interval: str) -> int:
    """
    Calculate maximum number of days for a given interval
    
    Args:
        interval: Data interval
    
    Returns:
        Maximum number of days
    """
    bars_per_day = BARS_PER_DAY.get(interval, 24)  # Default to 24 bars/day
    max_days = MAX_DATA_POINTS // bars_per_day
    if interval == '1h':
        return min(730, max_days)
    return min(60, max_days)

# Interval to maximum days mapping
INTERVAL_MAX_DAYS = {interval: get_max_days(interval) for interval in BARS_PER_DAY}

# Default backtest interval based on DEFAULT_INTERVAL
default_backtest_interval = INTERVAL_MAX_DAYS.get(DEFAULT_INTERVAL.replace('min', 'm')) if INTERVAL_MAX_DAYS.get(DEFAULT_INTERVAL.replace('min', 'm')) else 365 * 0.4  # Default to 2 years if no limit
lookback_days_param = default_backtest_interval/4

#1-minute interval: Maximum of 7 days of historical data.
#5-minute interval: Maximum of 60 days of historical data.
#15-minute interval: Maximum of 60 days of historical data.
#30-minute interval: Maximum of 60 days of historical data.
#1-hour interval: Maximum of 730 days (2 years) of historical data.
#Daily interval: No strict limit, can fetch data for the entire available history.





# Trading symbols configuration
TRADING_SYMBOLS = {
    # Cryptocurrencies
    'BTC/USD': {
        'name': 'Bitcoin',
        'market': 'CRYPTO',
        'yfinance': 'BTC-USD',
        'interval': default_interval_yahoo,
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'ETH/USD': {
        'name': 'Ethereum',
        'market': 'CRYPTO',
        'yfinance': 'ETH-USD',
        'interval': default_interval_yahoo,
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'SOL/USD': {
        'name': 'Solana',
        'market': 'CRYPTO',
        'yfinance': 'SOL-USD',
        'interval': default_interval_yahoo,       
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    }, 
    'AVAX/USD': {
        'name': 'Avalanche',
        'market': 'CRYPTO',
        'yfinance': 'AVAX-USD',
        'interval': default_interval_yahoo,
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'DOT/USD': {
        'name': 'Polkadot',
        'market': 'CRYPTO',
        'yfinance': 'DOT-USD',
        'interval': default_interval_yahoo,
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'LINK/USD': {
        'name': 'Chainlink',
        'market': 'CRYPTO',
        'yfinance': 'LINK-USD',
        'interval': default_interval_yahoo,
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'DOGE/USD': {
        'name': 'Dogecoin',
        'market': 'CRYPTO',
        'yfinance': 'DOGE-USD',
        'interval': default_interval_yahoo,
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'AAVE/USD': {
        'name': 'Aave',
        'market': 'CRYPTO',
        'yfinance': 'AAVE-USD',
        'interval': default_interval_yahoo,
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'UNI/USD': {
        'name': 'Uniswap',
        'market': 'CRYPTO',
        'yfinance': 'UNI7083-USD',
        'interval': default_interval_yahoo,
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'LTC/USD': {
        'name': 'Litecoin',
        'market': 'CRYPTO',
        'yfinance': 'LTC-USD',
        'interval': default_interval_yahoo,
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'SHIB/USD': {
        'name': 'Shiba Inu',
        'market': 'CRYPTO',
        'yfinance': 'SHIB-USD',
        'interval': default_interval_yahoo,
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'BAT/USD': {
        'name': 'Basic Attention Token',
        'market': 'CRYPTO',
        'yfinance': 'BAT-USD',
        'interval': default_interval_yahoo,
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'BCH/USD': {
        'name': 'Bitcoin Cash',
        'market': 'CRYPTO',
        'yfinance': 'BCH-USD',
        'interval': default_interval_yahoo,
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'CRV/USD': {
        'name': 'Curve DAO Token',
        'market': 'CRYPTO',
        'yfinance': 'CRV-USD',
        'interval': default_interval_yahoo,
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'GRT/USD': {
        'name': 'The Graph',
        'market': 'CRYPTO',
        'yfinance': 'GRT6719-USD',
        'interval': default_interval_yahoo,
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'MKR/USD': {
        'name': 'Maker',
        'market': 'CRYPTO',
        'yfinance': 'MKR-USD',
        'interval': default_interval_yahoo,
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'SUSHI/USD': {
        'name': 'SushiSwap',
        'market': 'CRYPTO',
        'yfinance': 'SUSHI-USD',
        'interval': default_interval_yahoo,
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'XTZ/USD': {
        'name': 'Tezos',
        'market': 'CRYPTO',
        'yfinance': 'XTZ-USD',
        'interval': default_interval_yahoo,
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'YFI/USD': {
        'name': 'yearn.finance',
        'market': 'CRYPTO',
        'yfinance': 'YFI-USD',
        'interval': default_interval_yahoo,
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'XRP/USD': {
        'name': 'Ripple',
        'market': 'CRYPTO',
        'yfinance': 'XRP-USD',
        'interval': default_interval_yahoo,
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    }
}





# Trading costs configuration
TRADING_COSTS = {
    'DEFAULT': {
        'trading_fee': 0.001,  # 0.1% trading fee
        'spread': 0.006,  # 0.6% spread (bid-ask)
    },
    'CRYPTO': {
        'trading_fee': 0.003,  # 0.3% taker fee
        'spread': 0.002,  # 0.2% maker fee
    },
    'STOCK': {
        'trading_fee': 0.0005,  # 0.05% trading fee
        'spread': 0.001,  # 0.1% spread (bid-ask)
    }
}

param_grid = {
    'percent_increase_buy': [0.02],
    'percent_decrease_sell': [0.02],
    'sell_down_lim': [2.0],
    'sell_rolling_std': [20],
    'buy_up_lim': [-2.0],
    'buy_rolling_std': [20],
    'macd_fast': [12],
    'macd_slow': [26],
    'macd_signal': [9],
    'rsi_period': [14],
    'stochastic_k_period': [14],
    'stochastic_d_period': [3],
    'fractal_window': [50, 100, 150],
    'fractal_lags': [[5, 10, 20], [10, 20, 40], [15, 30, 60]],
    'reactivity': [0.8, 0.9, 1.0, 1.1, 1.2],
    'weights': [
        {'weekly_macd_weight': 0.1, 'weekly_rsi_weight': 0.1, 'weekly_stoch_weight': 0.1, 'weekly_complexity_weight': 0.7,'macd_weight': 0.1, 'rsi_weight': 0.1, 'stoch_weight': 0.1, 'complexity_weight': 0.7},
        {'weekly_macd_weight': 0.15, 'weekly_rsi_weight': 0.15, 'weekly_stoch_weight': 0.15, 'weekly_complexity_weight': 0.55,'macd_weight': 0.15, 'rsi_weight': 0.15, 'stoch_weight': 0.15, 'complexity_weight': 0.55},
        {'weekly_macd_weight': 0.2, 'weekly_rsi_weight': 0.2, 'weekly_stoch_weight': 0.2, 'weekly_complexity_weight': 0.4,'macd_weight': 0.2, 'rsi_weight': 0.2, 'stoch_weight': 0.2, 'complexity_weight': 0.4},
        {'weekly_macd_weight': 0.25, 'weekly_rsi_weight': 0.25, 'weekly_stoch_weight': 0.25, 'weekly_complexity_weight': 0.25,'macd_weight': 0.25, 'rsi_weight': 0.25, 'stoch_weight': 0.25, 'complexity_weight': 0.25},
        {'weekly_macd_weight': 0.3, 'weekly_rsi_weight': 0.3, 'weekly_stoch_weight': 0.3, 'weekly_complexity_weight': 0.1,'macd_weight': 0.3, 'rsi_weight': 0.3, 'stoch_weight': 0.3, 'complexity_weight': 0.1},
        {'weekly_macd_weight': 0.3, 'weekly_rsi_weight': 0.3, 'weekly_stoch_weight': 0.3, 'weekly_complexity_weight': 0.1,'macd_weight': 0.1, 'rsi_weight': 0.1, 'stoch_weight': 0.1, 'complexity_weight': 0.7},

    ]
}



# Function to calculate dynamic capital multiplier based on asset performance
def calculate_capital_multiplier(lookback_days=default_backtest_interval/2):
    """
    Calculate a dynamic capital multiplier based on asset performance.
    
    Args:
        lookback_days: Number of days to look back for performance calculation
        
    Returns:
        float: Capital multiplier between 1.0 and 3.0
    """
    try:
        # Default return if calculation fails
        default_multiplier = 2.0
        
        # Get end and start dates
        end_date = datetime.now(pytz.UTC)
        start_date = end_date - timedelta(days=lookback_days * 2)  # Double lookback for MA calculation
        
        # Collect performance data for all assets
        performances = []
        
        for symbol, config in TRADING_SYMBOLS.items():
            try:
                # Get the yfinance symbol
                yf_symbol = config['yfinance']
                if '/' in yf_symbol:
                    yf_symbol = yf_symbol.replace('/', '-')
                
                # Fetch historical data
                ticker = yf.Ticker(yf_symbol)
                data = ticker.history(start=start_date, end=end_date, interval=default_interval_yahoo)
                
                if len(data) >= 2:
                    # Calculate performance as percent change
                    start_price = data['Close'].iloc[0]
                    end_price = data['Close'].iloc[-1]
                    perf = ((end_price - start_price) / start_price) * 100
                    performances.append(perf)
            except Exception:
                continue
        
        if not performances:
            return default_multiplier
        
        # Convert to numpy array for calculations
        performances = np.array(performances)
        
        # Calculate average performance
        avg_perf = np.mean(performances)
        
        # Calculate moving average (simple approach for minimal code change)
        # Use half the data points for the "previous" period
        half_idx = len(performances) // 2
        if half_idx > 0:
            prev_avg = np.mean(performances[:half_idx])
            # Calculate the standard deviation of the difference
            std_diff = np.std([avg_perf, prev_avg])
        else:
            prev_avg = avg_perf
            std_diff = 1.0  # Default if not enough data
        
        # Calculate difference between current average and moving average
        diff = avg_perf - prev_avg
        
        # Normalize the difference by std_diff with bounds at -2std and 2std
        if std_diff > 0:
            normalized_diff = max(min(diff / (2 * std_diff), 2), -2)
        else:
            normalized_diff = 0
        
        # Apply sigmoid function to get a value between 0 and 1
        sigmoid = 1 / (1 + np.exp(-normalized_diff))
        
        # Scale to range [1, 3]
        multiplier = 1.0 + 2.0 * sigmoid
        
        return multiplier
        
    except Exception as e:
        print(f"Error calculating capital multiplier: {str(e)}")
        return default_multiplier


# Set capital multiplier (computed once at module import)
PER_SYMBOL_CAPITAL_MULTIPLIER = calculate_capital_multiplier(lookback_days_param/2)


initial_capital = 100000
symbols = list(TRADING_SYMBOLS.keys())
per_symbol_capital = initial_capital / len(symbols) * PER_SYMBOL_CAPITAL_MULTIPLIER  # Allow each symbol to potentially use full capital
