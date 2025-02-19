import asyncio
import os
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from alpaca.trading.client import TradingClient
from strategy import TradingStrategy
from telegram_bot import TradingBot
from trading import TradingExecutor
from fetch import is_market_open

# Load environment variables
load_dotenv()
API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_SECRET_KEY')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('CHAT_ID')
SYMBOL = 'SPXL'

# Initialize logging
logging.basicConfig(
    filename="trading.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize components
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
strategy = TradingStrategy(SYMBOL, interval=config.DEFAULT_INTERVAL)
executor = TradingExecutor(trading_client, SYMBOL)
telegram_bot = TradingBot(trading_client, strategy, SYMBOL)

async def run_strategy():
    """Main strategy loop"""
    logger.info("Starting strategy loop...")
    await telegram_bot.send_message(f" Strategy started for {SYMBOL}")
    
    while True:
        try:
            if not is_market_open():
                logger.info("Market is closed. Waiting...")
                await asyncio.sleep(60)  # Check every minute
                continue
            
            # Analyze market and get signals
            analysis = strategy.analyze()
            should_trade, action = strategy.should_trade(analysis)
            
            if should_trade:
                await executor.execute_trade(action, analysis, telegram_bot.send_message)
                strategy.update_position(action)
                logger.info(f"Trade executed: {action}")
            else:
                logger.debug(f"No trade signal generated")
            
            # Wait for 5 minutes before next analysis
            await asyncio.sleep(300)
            
        except Exception as e:
            error_msg = f"Error in strategy loop: {str(e)}"
            logger.error(error_msg)
            await telegram_bot.send_message(f" {error_msg}")
            await asyncio.sleep(60)

async def main():
    """Start the bot and strategy"""
    try:
        # Start the bot (this will handle command registration internally)
        await telegram_bot.start()
        
        # Start the strategy loop
        await run_strategy()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
