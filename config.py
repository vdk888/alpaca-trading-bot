# config.py

ALPACA_PAPER = True

# Trading symbols configuration
TRADING_SYMBOLS = {
    # Major Index ETFs
    'SPY': {  # S&P 500
        'market': 'US',
        'yfinance': 'SPY',
        'interval': '5m',
        'market_hours': {
            'start': '09:30',
            'end': '16:00',
            'timezone': 'US/Eastern'
        }
    },
    'QQQ': {  # Nasdaq 100
        'market': 'US',
        'yfinance': 'QQQ',
        'interval': '5m',
        'market_hours': {
            'start': '09:30',
            'end': '16:00',
            'timezone': 'US/Eastern'
        }
    },
    'IWM': {  # Russell 2000
        'market': 'US',
        'yfinance': 'IWM',
        'interval': '5m',
        'market_hours': {
            'start': '09:30',
            'end': '16:00',
            'timezone': 'US/Eastern'
        }
    },
    'DIA': {  # Dow Jones
        'market': 'US',
        'yfinance': 'DIA',
        'interval': '5m',
        'market_hours': {
            'start': '09:30',
            'end': '16:00',
            'timezone': 'US/Eastern'
        }
    },
    
    # Sector ETFs
    'XLK': {  # Technology
        'market': 'US',
        'yfinance': 'XLK',
        'interval': '5m',
        'market_hours': {
            'start': '09:30',
            'end': '16:00',
            'timezone': 'US/Eastern'
        }
    },
    'XLF': {  # Financials
        'market': 'US',
        'yfinance': 'XLF',
        'interval': '5m',
        'market_hours': {
            'start': '09:30',
            'end': '16:00',
            'timezone': 'US/Eastern'
        }
    },
    'XLE': {  # Energy
        'market': 'US',
        'yfinance': 'XLE',
        'interval': '5m',
        'market_hours': {
            'start': '09:30',
            'end': '16:00',
            'timezone': 'US/Eastern'
        }
    },
    
    # Volatility ETFs
    'UVXY': {  # ProShares Ultra VIX Short-Term Futures
        'market': 'US',
        'yfinance': 'UVXY',
        'interval': '5m',
        'market_hours': {
            'start': '09:30',
            'end': '16:00',
            'timezone': 'US/Eastern'
        }
    },
    'TQQQ': {  # 3x Nasdaq
        'market': 'US',
        'yfinance': 'TQQQ',
        'interval': '5m',
        'market_hours': {
            'start': '09:30',
            'end': '16:00',
            'timezone': 'US/Eastern'
        }
    },
    'SQQQ': {  # -3x Nasdaq
        'market': 'US',
        'yfinance': 'SQQQ',
        'interval': '5m',
        'market_hours': {
            'start': '09:30',
            'end': '16:00',
            'timezone': 'US/Eastern'
        }
    },
    
    # International ETFs
    'EEM': {  # Emerging Markets
        'market': 'US',
        'yfinance': 'EEM',
        'interval': '5m',
        'market_hours': {
            'start': '09:30',
            'end': '16:00',
            'timezone': 'US/Eastern'
        }
    },
    'EFA': {  # Developed Markets
        'market': 'US',
        'yfinance': 'EFA',
        'interval': '5m',
        'market_hours': {
            'start': '09:30',
            'end': '16:00',
            'timezone': 'US/Eastern'
        }
    },
    
    # Fixed Income ETFs
    'TLT': {  # 20+ Year Treasury
        'market': 'US',
        'yfinance': 'TLT',
        'interval': '5m',
        'market_hours': {
            'start': '09:30',
            'end': '16:00',
            'timezone': 'US/Eastern'
        }
    },
    'HYG': {  # High Yield Corporate Bonds
        'market': 'US',
        'yfinance': 'HYG',
        'interval': '5m',
        'market_hours': {
            'start': '09:30',
            'end': '16:00',
            'timezone': 'US/Eastern'
        }
    },
    
    # Commodity ETFs
    'GLD': {  # Gold
        'market': 'US',
        'yfinance': 'GLD',
        'interval': '5m',
        'market_hours': {
            'start': '09:30',
            'end': '16:00',
            'timezone': 'US/Eastern'
        }
    },
    
    # Cryptocurrencies
    'BTC/USD': {
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
        'market': 'CRYPTO',
        'yfinance': 'ETH-USD',
        'interval': '5m',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'SOL/USD': {  # Solana
        'market': 'CRYPTO',
        'yfinance': 'SOL-USD',
        'interval': '5m',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'AVAX/USD': {  # Avalanche
        'market': 'CRYPTO',
        'yfinance': 'AVAX-USD',
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