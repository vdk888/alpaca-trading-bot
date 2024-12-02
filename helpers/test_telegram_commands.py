import asyncio
import os
from dotenv import load_dotenv
from telegram_bot import TradingBot
from alpaca.trading.client import TradingClient
from strategy import TradingStrategy
from config import TRADING_SYMBOLS
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockMessage:
    async def reply_text(self, text):
        logger.info(f"Response: {text}")
        
class MockUpdate:
    def __init__(self):
        self.message = MockMessage()
            
class MockContext:
    def __init__(self, args=None):
        self.args = args or []

async def test_command(bot, command: str, args: list = None):
    """Test a specific command and log the result"""
    try:
        # Create mock objects
        update = MockUpdate()
        context = MockContext(args)
            
        # Get the command handler
        handler = getattr(bot, f"{command}_command")
        if not handler:
            logger.error(f"Command {command} not found")
            return
            
        # Execute the command
        logger.info(f"\nTesting command: /{command} {' '.join(args) if args else ''}")
        await handler(update, context)
        logger.info("Command executed successfully\n")
        
    except Exception as e:
        logger.error(f"Error testing {command}: {str(e)}")

async def main():
    # Load environment variables
    try:
        load_dotenv()
    except Exception as e:
        logger.warning(f"Could not load .env file: {e}")
    
    # Check for required environment variables
    required_vars = ['ALPACA_API_KEY', 'ALPACA_SECRET_KEY', 'TELEGRAM_BOT_TOKEN', 'CHAT_ID']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Initialize trading client
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_SECRET_KEY')
    trading_client = TradingClient(api_key, api_secret, paper=True)
    
    # Initialize strategies
    strategies = {symbol: TradingStrategy(trading_client, symbol) 
                 for symbol in TRADING_SYMBOLS.keys()}
    
    # Initialize bot
    bot = TradingBot(trading_client, strategies, list(TRADING_SYMBOLS.keys()))
    
    # Test basic commands
    await test_command(bot, "start")
    await test_command(bot, "help")
    await test_command(bot, "symbols")
    
    # Test status commands
    await test_command(bot, "status")
    await test_command(bot, "status", ["BTC"])
    await test_command(bot, "position")
    await test_command(bot, "balance")
    await test_command(bot, "performance")
    
    # Test analysis commands
    await test_command(bot, "indicators", ["BTC"])
    await test_command(bot, "plot", ["BTC", "5"])
    await test_command(bot, "signals")
    await test_command(bot, "backtest", ["BTC", "5"])
    
    # Test market information
    await test_command(bot, "markets")
    
    # Test trading commands (with small amount)
    await test_command(bot, "open", ["BTC", "10"])  # Open $10 position
    await test_command(bot, "position", ["BTC"])  # Verify position
    await test_command(bot, "close", ["BTC"])  # Close position
    
    logger.info("All commands tested!")

if __name__ == "__main__":
    asyncio.run(main())
