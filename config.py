# config.py

ALPACA_API_KEY = 'PKA8KACYKMLGOT5X9QEL'
ALPACA_SECRET_KEY = 'MYwOauUZgNgvPqDle9H9oqdTWmvnFgHx0GD3yhBg'
ALPACA_PAPER = True

# Trading symbols configuration
TRADING_SYMBOLS = {
    'SPY': {
        'market': 'US',
        'interval': '5m',
        'market_hours': {
            'start': '09:30',
            'end': '16:00',
            'timezone': 'US/Eastern'
        }
    },
    'DAX': {
        'market': 'EU',
        'interval': '5m',
        'market_hours': {
            'start': '09:00',
            'end': '17:30',
            'timezone': 'Europe/Berlin'
        }
    },
    'NIKKEI': {  # Using ^N225 as the symbol for Nikkei
        'market': 'JP',
        'interval': '5m',
        'market_hours': {
            'start': '09:00',
            'end': '15:15',
            'timezone': 'Asia/Tokyo'
        }
    },
    'EUR/USD': {
        'market': 'FX',
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