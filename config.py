# config.py

ALPACA_API_KEY = 'PKA8KACYKMLGOT5X9QEL'
ALPACA_SECRET_KEY = 'MYwOauUZgNgvPqDle9H9oqdTWmvnFgHx0GD3yhBg'
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
    'DAX': {
        'market': 'EU',
        'yfinance': '^GDAXI',  # Yahoo Finance symbol for DAX
        'interval': '5m',
        'market_hours': {
            'start': '09:00',
            'end': '17:30',
            'timezone': 'Europe/Berlin'
        }
    },
    'NIKKEI': {
        'market': 'JP',
        'yfinance': '^N225',  # Yahoo Finance symbol for Nikkei
        'interval': '5m',
        'market_hours': {
            'start': '09:00',
            'end': '15:15',
            'timezone': 'Asia/Tokyo'
        }
    },
    'EUR/USD': {
        'market': 'FX',
        'yfinance': 'EURUSD=X',  # Yahoo Finance symbol for EUR/USD
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