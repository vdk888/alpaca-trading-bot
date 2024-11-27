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
from visualization import create_strategy_plot

# Load environment variables
load_dotenv()
API_KEY = os.getenv('ALPACA_KEY_ID')
API_SECRET = os.getenv('ALPACA_SECRET_KEY')
TELEGRAM_BOT_TOKEN = os.getenv('BOT_KEY')
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
strategy = TradingStrategy(SYMBOL, interval='5m')
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

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    help_text = """
Available commands:
/start - Start the bot
/help - Show this help message
/status - Check trading bot status
/position - Check current position
/plot [symbol] [days] - Generate strategy visualization (default: SPY, 5 days)
"""
    await update.message.reply_text(help_text)

async def plot_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generate and send a strategy visualization plot."""
    try:
        # Parse arguments
        args = context.args
        symbol = args[0].upper() if len(args) > 0 else 'SPY'
        days = int(args[1]) if len(args) > 1 else 5
        
        # Validate days
        if days <= 0 or days > 30:
            await update.message.reply_text("Days must be between 1 and 30")
            return
        
        # Send "generating" message
        status_message = await update.message.reply_text(
            f"📊 Generating visualization for {symbol} (last {days} days)..."
        )
        
        # Generate plot
        plot_bytes, stats = create_strategy_plot(symbol, days)
        
        # Prepare statistics message
        stats_message = f"""
📈 {symbol} Analysis (Last {days} days):
• Current Price: ${stats['current_price']:.2f}
• Price Change: {stats['price_change']:.2f}%
• Buy Signals: {stats['buy_signals']}
• Sell Signals: {stats['sell_signals']}

📊 Indicators:
• Daily Composite: {stats['daily_composite_mean']:.4f} (±{stats['daily_composite_std']:.4f})
• Weekly Composite: {stats['weekly_composite_mean']:.4f} (±{stats['weekly_composite_std']:.4f})
"""
        
        # Send plot and statistics
        await update.message.reply_photo(
            photo=plot_bytes,
            caption=stats_message
        )
        
        # Delete the "generating" message
        await status_message.delete()
        
    except Exception as e:
        await update.message.reply_text(f"Error generating plot: {str(e)}")

async def main():
    """Start the bot and strategy"""
    try:
        # Create the Application
        application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        
        # Setup telegram handlers
        telegram_bot.setup_handlers(application)
        
        # Add command handlers
        application.add_handler(CommandHandler("start", telegram_bot.start))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("status", telegram_bot.status))
        application.add_handler(CommandHandler("position", telegram_bot.position))
        application.add_handler(CommandHandler("plot", plot_command))
        
        # Log startup
        logger.info(f"Starting trading bot for {SYMBOL}")
        print(f"Starting trading bot for {SYMBOL}...")
        print("Telegram bot initialized. Use /start to begin.")
        
        # Start both the application and strategy
        async with application:
            await application.initialize()
            await application.start()
            await application.updater.start_polling()
            
            # Run the strategy concurrently
            await run_strategy()
            
    except Exception as e:
        logger.critical(f"Critical error in main: {str(e)}")
        print(f"Critical error: {str(e)}")
        raise
    finally:
        # Cleanup
        logger.info("Shutting down...")
        await application.stop()

if __name__ == "__main__":
    asyncio.run(main())