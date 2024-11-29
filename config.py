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
    }
}

# Default trading parameters
DEFAULT_RISK_PERCENT = 0.95
DEFAULT_INTERVAL = '5m'