# config.py

ALPACA_PAPER = True

# Trading symbols configuration
TRADING_SYMBOLS = {
    'SPY': {
        'market': 'US',
        'yfinance': 'SPY',  # Yahoo Finance symbol
        'interval': '5m',
        'market_hours': {
            'start': '09:30',
            'end': '16:00',
            'timezone': 'US/Eastern'
        }
    },
    'NDAQ': {
        'market': 'US',
        'yfinance': 'NDAQ',  # Yahoo Finance symbol for NASDAQ
        'interval': '5m',
        'market_hours': {
            'start': '09:00',
            'end': '16:00',
            'timezone': 'US/Eastern'
        }
    },
    'BTC/USD': {
        'market': 'CRYPTO',
        'yfinance': 'BTC-USD',  # Yahoo Finance symbol for Bitcoin
        'interval': '5m',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'ETH/USD': {
        'market': 'CRYPTO',
        'yfinance': 'ETH-USD',  # Yahoo Finance symbol for Ethereum
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