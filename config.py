# config.py

import os

ALPACA_PAPER = True
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')

if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    raise ValueError("Alpaca API credentials not found in environment variables")

# Trading symbols configuration
TRADING_SYMBOLS = {
    # Major Index ETFs
    'SPY': {  # S&P 500
        'name': 'S&P 500',
        'market': 'US',
        'yfinance': 'SPY',
        'interval': '5m',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'US/Eastern'
        }
    },
    'QQQ': {  # Nasdaq 100
        'name': 'Nasdaq 100',
        'market': 'US',
        'yfinance': 'QQQ',
        'interval': '5m',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'US/Eastern'
        }
    },
    'IWM': {  # Russell 2000
        'name': 'Russell 2000',
        'market': 'US',
        'yfinance': 'IWM',
        'interval': '5m',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'US/Eastern'
        }
    },
    'DIA': {  # Dow Jones
        'name': 'Dow Jones',
        'market': 'US',
        'yfinance': 'DIA',
        'interval': '5m',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'US/Eastern'
        }
    },
    
    # Sector ETFs
    'XLK': {  # Technology
        'name': 'Technology Sector',
        'market': 'US',
        'yfinance': 'XLK',
        'interval': '5m',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'US/Eastern'
        }
    },
    'XLF': {  # Financials
        'name': 'Financial Sector',
        'market': 'US',
        'yfinance': 'XLF',
        'interval': '5m',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'US/Eastern'
        }
    },
    'XLE': {  # Energy
        'name': 'Energy Sector',
        'market': 'US',
        'yfinance': 'XLE',
        'interval': '5m',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'US/Eastern'
        }
    },
    
    # Volatility ETFs
    'UVXY': {
        'name': 'ProShares Ultra VIX',
        'market': 'US',
        'yfinance': 'UVXY',
        'interval': '5m',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'US/Eastern'
        }
    },
    
    # Leveraged ETFs
    'TQQQ': {
        'name': '3x Nasdaq 100',
        'market': 'US',
        'yfinance': 'TQQQ',
        'interval': '5m',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'US/Eastern'
        }
    },

    # International ETFs
    'EEM': {
        'name': 'Emerging Markets',
        'market': 'US',
        'yfinance': 'EEM',
        'interval': '5m',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'US/Eastern'
        }
    },
    'EFA': {
        'name': 'Developed Markets',
        'market': 'US',
        'yfinance': 'EFA',
        'interval': '5m',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'US/Eastern'
        }
    },
    
    # Fixed Income ETFs
    'TLT': {
        'name': '20+ Year Treasury',
        'market': 'US',
        'yfinance': 'TLT',
        'interval': '5m',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'US/Eastern'
        }
    },
    'HYG': {
        'name': 'High Yield Bonds',
        'market': 'US',
        'yfinance': 'HYG',
        'interval': '5m',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'US/Eastern'
        }
    },
    
    # Commodities
    'GLD': {
        'name': 'Gold',
        'market': 'US',
        'yfinance': 'GLD',
        'interval': '5m',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'US/Eastern'
        }
    },
    
    # Cryptocurrencies
    'BTC/USD': {
        'name': 'Bitcoin',
        'market': 'CRYPTO',
        'yfinance': 'BTC-USD',
        'interval': '5m',
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
        'interval': '5m',
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
        'interval': '5m',
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
        'interval': '5m',
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
        'interval': '5m',
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
        'interval': '5m',
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
        'interval': '5m',
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
        'interval': '5m',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'UNI/USD': {
        'name': 'Uniswap',
        'market': 'CRYPTO',
        'yfinance': 'UNI-USD',
        'interval': '5m',
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
        'interval': '5m',
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
        'interval': '5m',
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
        'interval': '5m',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    }
}

# Default trading parameters
DEFAULT_RISK_PERCENT = 0.95
DEFAULT_INTERVAL = '5m'

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