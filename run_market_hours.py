import asyncio
import datetime
import pytz
from trading import TradingExecutor
from fetch import fetch_historical_data, get_latest_data, is_market_open
from strategy import TradingStrategy
from telegram_bot import TradingBot
from alpaca.trading.client import TradingClient
import os
from dotenv import load_dotenv
import logging
from telegram import Update
from config import TRADING_SYMBOLS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def is_market_hours():
    """Check if it's currently market hours (9:30 AM - 4:00 PM Eastern, Monday-Friday)"""
    et_tz = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(et_tz)
    
    # Check if it's a weekday
    if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False
    
    # Market hours are 9:30 AM - 4:00 PM Eastern
    market_start = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_end = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_start <= now <= market_end

async def run_bot():
    """Main function to run the trading bot"""
    # Try to load from .env file, but continue if file not found
    try:
        load_dotenv()
    except Exception as e:
        logger.warning(f"Could not load .env file: {e}")
    
    # Check for required environment variables
    required_vars = ['ALPACA_API_KEY', 'ALPACA_SECRET_KEY', 'TELEGRAM_BOT_TOKEN', 'CHAT_ID']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Initialize clients
    trading_client = TradingClient(
        os.getenv('ALPACA_API_KEY'),
        os.getenv('ALPACA_SECRET_KEY')
    )
    
    # Initialize strategies for each symbol
    symbols = list(TRADING_SYMBOLS.keys())
    strategies = {symbol: TradingStrategy(symbol) for symbol in symbols}
    trading_executors = {symbol: TradingExecutor(trading_client, symbol) for symbol in symbols}
    
    # Initialize the Telegram bot with all symbols and strategies
    trading_bot = TradingBot(trading_client, strategies, symbols)
    
    # Start the Telegram bot
    logger.info("Starting Telegram bot...")
    await trading_bot.start()
    
    logger.info(f"Bot started, monitoring symbols: {', '.join(symbols)}")
    
    async def trading_loop():
        """Background task for trading logic"""
        while True:
            try:
                for symbol in symbols:
                    try:
                        analysis = strategies[symbol].analyze()
                        if analysis['signal'] != 0:  # If there's a trading signal
                            signal_type = "LONG" if analysis['signal'] == 1 else "SHORT"
                            message = f"""
🔔 Trading Signal for {symbol}:
Signal: {signal_type}
Price: ${analysis['current_price']:.2f}
Daily Score: {analysis['daily_composite']:.4f}
Weekly Score: {analysis['weekly_composite']:.4f}
                            """
                            await trading_bot.send_message(message)
                    except Exception as e:
                        logger.error(f"Error analyzing {symbol}: {str(e)}")
                        continue
                
                await asyncio.sleep(300)  # Wait 5 minutes between iterations
            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                await asyncio.sleep(60)  # Wait a minute before retrying

    try:
        # Start the trading loop
        trading_task = asyncio.create_task(trading_loop())
        
        # Keep the main task running
        while True:
            await asyncio.sleep(1)
            
    except Exception as e:
        logger.error(f"Error in main loop: {str(e)}")
    finally:
        # Cleanup
        if 'trading_task' in locals():
            trading_task.cancel()
            try:
                await trading_task
            except asyncio.CancelledError:
                pass
        await trading_bot.stop()

if __name__ == "__main__":
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot stopped due to error: {str(e)}")
