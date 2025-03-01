# Alpaca Trading Bot

A Python-based trading bot that uses technical indicators to make trading decisions on the Alpaca trading platform.

## Features

- Technical analysis using MACD, RSI, and Stochastic indicators
- Composite indicator for trade signals
- Telegram bot integration for monitoring and control
- 5-minute timeframe trading with 35-minute composite signals
- Automated trade execution via Alpaca API

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your Alpaca credentials:
```
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
```

3. Run the bot:
```bash
python run_market_hours.py
```

## Components

- `trading.py`: Main trading execution logic
- `indicators.py`: Technical indicators and signal generation
- `strategy.py`: Trading strategy implementation
- `fetch.py`: Data fetching from Alpaca
- `telegram_bot.py`: Telegram bot for monitoring and control

## License

MIT
