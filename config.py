# config.py

ALPACA_PAPER = True

# Trading symbols configuration
TRADING_SYMBOLS = {
    # Major Index ETFs
    'SPY': {  # S&P 500
        'name': 'S&P 500',
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
        'name': 'Nasdaq 100',
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
        'name': 'Russell 2000',
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
        'name': 'Dow Jones',
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
        'name': 'Technology Sector',
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
        'name': 'Financial Sector',
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
        'name': 'Energy Sector',
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
    'UVXY': {
        'name': 'ProShares Ultra VIX',
        'market': 'US',
        'yfinance': 'UVXY',
        'interval': '5m',
        'market_hours': {
            'start': '09:30',
            'end': '16:00',
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
            'start': '09:30',
            'end': '16:00',
            'timezone': 'US/Eastern'
        }
    },
    'SQQQ': {
        'name': '-3x Nasdaq 100',
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
    'EEM': {
        'name': 'Emerging Markets',
        'market': 'US',
        'yfinance': 'EEM',
        'interval': '5m',
        'market_hours': {
            'start': '09:30',
            'end': '16:00',
            'timezone': 'US/Eastern'
        }
    },
    'EFA': {
        'name': 'Developed Markets',
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
    'TLT': {
        'name': '20+ Year Treasury',
        'market': 'US',
        'yfinance': 'TLT',
        'interval': '5m',
        'market_hours': {
            'start': '09:30',
            'end': '16:00',
            'timezone': 'US/Eastern'
        }
    },
    'HYG': {
        'name': 'High Yield Bonds',
        'market': 'US',
        'yfinance': 'HYG',
        'interval': '5m',
        'market_hours': {
            'start': '09:30',
            'end': '16:00',
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
            'start': '09:30',
            'end': '16:00',
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