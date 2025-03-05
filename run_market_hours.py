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
from telegram import Update, Bot
from backtest_individual import find_best_params
from config import TRADING_SYMBOLS, param_grid, lookback_days_param
import json
from backtest_individual import run_backtest, create_backtest_plot
import io
import matplotlib.pyplot as plt
from indicators import get_default_params

# Add Flask server for Replit deployment
from flask import Flask
import threading

app = Flask(__name__)

@app.route('/')
def home():
    return "Trading Bot is running!"

def run_flask():
    app.run(host='0.0.0.0', port=8080)

# Start Flask server in a daemon thread
flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()


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
    market_start = now.astimezone(et_tz).replace(hour=7, minute=30, second=0, microsecond=0)
    market_end = now.astimezone(et_tz).replace(hour=22, minute=0, second=0, microsecond=0)
    
    return market_start <= now.astimezone(et_tz) <= market_end

async def run_bot():
    """Main function to run the trading bot"""
    # Try to load from .env file, but continue if file not found
    try:
        load_dotenv()
    except Exception as e:
        logger.warning(f"Could not load .env file: {e}")
    
    # Check for required environment variables
    required_vars = ['ALPACA_API_KEY', 'ALPACA_SECRET_KEY', 'TELEGRAM_BOT_TOKEN', 'CHAT_ID', 'BOT_PASSWORD', 'TRADE_HISTORY_FILE']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        error_msg = f"DEPLOYMENT ERROR: Missing required environment variables: {', '.join(missing_vars)}"
        logger.error(error_msg)
        # Send emergency notification if possible before failing
        try:
            if 'TELEGRAM_BOT_TOKEN' not in missing_vars and 'CHAT_ID' not in missing_vars:
                bot = Bot(token=os.getenv('TELEGRAM_BOT_TOKEN'))
                asyncio.create_task(bot.send_message(
                    chat_id=os.getenv('CHAT_ID'),
                    text=f"🚨 DEPLOYMENT ERROR: Missing environment variables: {', '.join(missing_vars)}"
                ))
        except Exception as e:
            logger.error(f"Failed to send emergency notification: {e}")
            
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

    # Create mock Update object for /start command
    class MockUpdate:
        def __init__(self, bot):
            self.message = MockMessage(bot)

    class MockMessage:
        def __init__(self, bot):
            self.bot = bot

        async def reply_text(self, text):
            await self.bot.send_message(chat_id=os.getenv('CHAT_ID'), text=text)

    # Send startup message with /start command
    await trading_bot.start_command(MockUpdate(trading_bot.bot), None)

    async def backtest_loop():
        """Background task for running backtests"""
        while True:
            try:
                for symbol in TRADING_SYMBOLS:
                    try:
                        # Check if we need to update parameters
                        needs_update = True
                        try:
                            with open("best_params.json", "r") as f:
                                best_params_data = json.load(f)
                                if symbol in best_params_data:
                                    last_update = datetime.datetime.strptime(best_params_data[symbol].get('date', '2000-01-01'), "%Y-%m-%d")
                                    days_since_update = (datetime.datetime.now() - last_update).days
                                    needs_update = days_since_update >= 7  # Update weekly
                        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                            logger.warning(f"Could not read best_params.json for {symbol}: {str(e)}")
                            needs_update = True
                            
                        if needs_update:
                            logger.info(f"Running backtest for {symbol}...")
                            await trading_bot.send_message(f"🔄 Running background optimization for {symbol}...")
                            try:
                                logger.info(f"Starting optimization for {symbol} with param_grid: {param_grid}")
                                # Run the CPU-intensive backtest in a thread pool
                                loop = asyncio.get_event_loop()
                                best_params = await loop.run_in_executor(
                                    None,  # Use default executor
                                    find_best_params,
                                    symbol,
                                    param_grid,
                                    30
                                )
                                
                                if best_params:
                                    logger.info(f"Successfully found best params for {symbol}: {best_params}")
                                    await trading_bot.send_message(f"✅ Optimization complete for {symbol}")
                                    logger.info(f"New optimal parameters for {symbol}: {best_params}")
                                else:
                                    error_msg = f"Failed to find best parameters for {symbol} - no valid results returned"
                                    logger.error(error_msg)
                                    await trading_bot.send_message(f"❌ {error_msg}")
                            except Exception as e:
                                error_msg = f"Failed to optimize {symbol}: {str(e)}"
                                logger.error(f"Full optimization error for {symbol}: {str(e)}", exc_info=True)
                                await trading_bot.send_message(f"❌ {error_msg}")
                                input("Press Enter to continue...")
                            # Small delay between symbols to prevent overload
                            await asyncio.sleep(1)
                    except Exception as e:
                        logger.error(f"Error in backtest for {symbol}: {str(e)}")
                        continue
                
                # Sleep for 1 hour before checking again
                await asyncio.sleep(3600)
            except Exception as e:
                logger.error(f"Error in backtest loop: {str(e)}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying
    
    async def trading_loop():
        """Background task for trading logic"""
        symbol_last_check = {symbol: None for symbol in symbols}
        
        while True:
            try:
                current_time = datetime.datetime.now(pytz.UTC)
                
                for symbol in symbols:
                    try:
                        # Check if 5 minutes have passed since last check for this symbol
                        if (symbol_last_check[symbol] is not None and 
                            (current_time - symbol_last_check[symbol]).total_seconds() < 300):
                            continue
                            
                        # Generate signals
                        try:
                            with open("best_params.json", "r") as f:
                                best_params_data = json.load(f)
                                if symbol in best_params_data:
                                    params = best_params_data[symbol]['best_params']
                                    print(f"Using best parameters for {symbol}: {params}")
                                else:
                                    print(f"No best parameters found for {symbol}. Using default parameters.")
                                    params = get_default_params()
                        except FileNotFoundError:
                            print("Best parameters file not found. Using default parameters.")
                            params = get_default_params()
                        
                        try:
                            analysis = strategies[symbol].analyze()
                            if analysis and analysis['signal'] != 0:  # If there's a trading signal
                                signal_type = "LONG" if analysis['signal'] == 1 else "SHORT"
                                message = f"""
🔔 Trading Signal for {symbol}:
Signal: {signal_type}
Price: ${analysis['current_price']:.2f}
Daily Score: {analysis['daily_composite']:.4f}
Weekly Score: {analysis['weekly_composite']:.4f}
Parameters: {params}
Bar Time: {analysis['bar_time']}
                                """
                                await trading_bot.send_message(message)
                           
                                # Execute trade with notifications through telegram bot
                                action = "BUY" if analysis['signal'] == 1 else "SELL"
                                await trading_executors[symbol].execute_trade(
                                    action=action,
                                    analysis=analysis,
                                    notify_callback=trading_bot.send_message
                                )
                                
                                # Run and send backtest results
                                await run_and_send_backtest(symbol, trading_bot)
                                
                        except Exception as e:
                            logger.error(f"Error analyzing {symbol}: {str(e)}")
                            continue
                            
                        # Update last check time for this symbol
                        symbol_last_check[symbol] = current_time
                        
                        # Small delay between symbols to prevent overload
                        await asyncio.sleep(1)
                        
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {str(e)}")
                        continue
                
                # Calculate time to sleep until next check
                elapsed_time = (datetime.datetime.now(pytz.UTC) - current_time).total_seconds()
                sleep_time = max(60, 300 - elapsed_time)  # At least 1 minute, at most 5 minutes
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                await asyncio.sleep(60)  # Wait a minute before retrying

    logger.info(f"Bot started, monitoring symbols: {', '.join(symbols)}")
    
    try:
        # Start both the trading and backtest loops
        trading_task = asyncio.create_task(trading_loop())
        backtest_task = asyncio.create_task(backtest_loop())
        
        # Keep the main task running
        while True:
            await asyncio.sleep(1)
            
    except Exception as e:
        logger.error(f"Error in main loop: {str(e)}")
    finally:
        # Cleanup
        tasks_to_cancel = []
        if 'trading_task' in locals():
            tasks_to_cancel.append(trading_task)
        if 'backtest_task' in locals():
            tasks_to_cancel.append(backtest_task)
            
        for task in tasks_to_cancel:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            
        await trading_bot.stop()

async def run_and_send_backtest(symbol: str, trading_bot, days: int = lookback_days_param):
    """Run a backtest for the symbol and send the results through telegram"""
    try:
        # Run backtest
        backtest_result = run_backtest(symbol, days=days)
        
        # Create plot and get stats
        plot_buffer, stats = create_backtest_plot(backtest_result)
        
        # Send stats message
        stats_message = f"""
📊 Backtest Results for {symbol} (Last {days} days):
Total Return: {stats['total_return']:.2f}%
Total Trades: {stats['total_trades']}
Win Rate: {stats['win_rate']:.2f}%
Sharpe Ratio: {stats['sharpe_ratio']:.2f}
Max Drawdown: {stats['max_drawdown']:.2f}%
"""
        await trading_bot.send_message(stats_message)
        
        # Send the plot
        await trading_bot.send_photo(plot_buffer)
        
    except Exception as e:
        await trading_bot.send_message(f"Error running backtest for {symbol}: {str(e)}")

async def send_stop_notification(reason: str):
    """Send a Telegram notification about program stopping"""
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('CHAT_ID')
    if bot_token and chat_id:
        bot = Bot(token=bot_token)
        try:
            await bot.send_message(
                chat_id=chat_id,
                text=f"🔴 Trading program stopped: {reason}"
            )
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")

if __name__ == "__main__":
    # Check deployment environment first
    try:
        from check_deployment import check_deployment_environment
        environment_ok = check_deployment_environment()
        if not environment_ok:
            logger.critical("Deployment environment check failed. Exiting.")
            sys.exit(1)
    except ImportError:
        logger.warning("Deployment environment checker not found. Continuing without check.")
    
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        asyncio.run(send_stop_notification("Stopped by user"))
    except Exception as e:
        error_msg = f"Bot stopped due to error: {str(e)}"
        logger.error(error_msg)
        asyncio.run(send_stop_notification(error_msg))
    else:
        asyncio.run(send_stop_notification("Normal termination"))
