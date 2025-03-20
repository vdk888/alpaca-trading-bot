import os
import asyncio
import logging
import json
import datetime
import pytz
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional, List

from config import (
    TRADING_SYMBOLS, param_grid, lookback_days_param, ALPACA_PAPER, get_update_interval, DEFAULT_INTERVAL
)
from trading import TradingExecutor
from fetch import fetch_historical_data, get_latest_data, is_market_open
from strategy import TradingStrategy
from telegram_bot import TradingBot
from alpaca.trading.client import TradingClient
from dotenv import load_dotenv
from telegram import Update, Bot
from backtest_individual import find_best_params
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
        os.getenv('ALPACA_SECRET_KEY'),
        paper=ALPACA_PAPER
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
                            from replit.object_storage import Client
                            
                            # Initialize Object Storage client
                            client = Client()
                            
                            # Try to get parameters from Object Storage
                            try:
                                json_content = client.download_as_text("best_params.json")
                                best_params_data = json.loads(json_content)
                                if symbol in best_params_data:
                                    last_update = datetime.datetime.strptime(best_params_data[symbol].get('date', '2000-01-01'), "%Y-%m-%d")
                                    days_since_update = (datetime.datetime.now() - last_update).days
                                    needs_update = days_since_update >= 7  # Update weekly
                            except Exception as e:
                                logger.warning(f"Could not read from Object Storage: {str(e)}")
                                # Try local file as fallback
                                try:
                                    with open("best_params.json", "r") as f:
                                        best_params_data = json.load(f)
                                        if symbol in best_params_data:
                                            last_update = datetime.datetime.strptime(best_params_data[symbol].get('date', '2000-01-01'), "%Y-%m-%d")
                                            days_since_update = (datetime.datetime.now() - last_update).days
                                            needs_update = days_since_update >= 7  # Update weekly
                                except FileNotFoundError:
                                    logger.warning(f"Local best_params.json not found for {symbol}")
                                    needs_update = True
                        except (json.JSONDecodeError, KeyError) as e:
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
                                    lookback_days_param
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
                        # Get update interval based on timeframe
                        update_interval = get_update_interval(DEFAULT_INTERVAL)
                        
                        # Check if enough time has passed since last check for this symbol
                        if (symbol_last_check[symbol] is not None and 
                            (current_time - symbol_last_check[symbol]).total_seconds() < update_interval):
                            continue
                            
                        # Generate signals
                        try:
                            from replit.object_storage import Client
                            
                            # Initialize Object Storage client
                            client = Client()
                            
                            # Try to get parameters from Object Storage
                            try:
                                json_content = client.download_as_text("best_params.json")
                                best_params_data = json.loads(json_content)
                                print("Successfully loaded best parameters from Object Storage")
                                if symbol in best_params_data:
                                    params = best_params_data[symbol]['best_params']
                                    print(f"Using best parameters for {symbol}: {params}")
                                else:
                                    print(f"No best parameters found for {symbol}. Using default parameters.")
                                    params = get_default_params()
                            except Exception as e:
                                print(f"Could not read from Object Storage: {e}")
                                # Try local file as fallback
                                try:
                                    with open("best_params.json", "r") as f:
                                        best_params_data = json.load(f)
                                        print("Loaded best parameters from local file")
                                        if symbol in best_params_data:
                                            params = best_params_data[symbol]['best_params']
                                            print(f"Using best parameters for {symbol}: {params}")
                                        else:
                                            print(f"No best parameters found for {symbol}. Using default parameters.")
                                            params = get_default_params()
                                except FileNotFoundError:
                                    print("Best parameters file not found. Using default parameters.")
                                    params = get_default_params()
                        except Exception as e:
                            print(f"Error loading parameters: {e}")
                            params = get_default_params()
                        
                        try:
                            analysis = strategies[symbol].analyze()
                            if analysis and analysis['signal'] != 0:  # If there's a trading signal
                                signal_type = "LONG" if analysis['signal'] == 1 else "SHORT"
                                
                                # Save signal data and create visualizations
                                csv_path, plot_path = await save_signal_data(symbol, strategies[symbol], analysis)
                                
                                message = f"""
🔔 Trading Signal for {symbol}:
Signal: {signal_type}
Price: ${analysis['current_price']:.2f}
Daily Score: {analysis['daily_composite']:.4f}
Weekly Score: {analysis['weekly_composite']:.4f}
Parameters: {params}
Bar Time: {analysis['bar_time']}

📊 Analysis files saved:
Data: {csv_path}
Chart: {plot_path}
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

async def save_signal_data(symbol: str, strategy, analysis: dict):
    """
    Save all signal-related data and create visualizations
    
    Args:
        symbol: Trading symbol
        strategy: TradingStrategy instance containing historical data
        analysis: Analysis dictionary from strategy.analyze()
    """
    # Create timestamp and directory
    timestamp = datetime.now(pytz.UTC).strftime('%Y-%m-%d_%H-%M-%S')
    signal_dir = os.path.join('signals', symbol, timestamp)
    os.makedirs(signal_dir, exist_ok=True)
    
    # Get historical data with all indicators
    df = strategy.data.copy()
    
    # Add composite scores to the dataframe
    df['daily_composite'] = analysis['daily_composite']
    df['weekly_composite'] = analysis['weekly_composite']
    df['signal'] = 0
    df.loc[df.index[-1], 'signal'] = analysis['signal']
    
    # Save complete data to CSV
    csv_path = os.path.join(signal_dir, 'data.csv')
    df.to_csv(csv_path)
    
    # Create interactive plot
    fig = make_subplots(rows=4, cols=1, 
                       shared_xaxis=True,
                       vertical_spacing=0.05,
                       row_heights=[0.5, 0.2, 0.15, 0.15])
    
    # Main price chart
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                name='Price'),
                  row=1, col=1)
    
    # Volume
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'),
                 row=2, col=1)
    
    # Composite scores
    fig.add_trace(go.Scatter(x=df.index, y=df['daily_composite'], 
                            name='Daily Score', line=dict(color='blue')),
                 row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['weekly_composite'], 
                            name='Weekly Score', line=dict(color='green')),
                 row=3, col=1)
    
    # Technical indicators
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], 
                                name='RSI', line=dict(color='purple')),
                     row=4, col=1)
    if 'MACD' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], 
                                name='MACD', line=dict(color='orange')),
                     row=4, col=1)
    
    # Mark the signal point
    signal_color = 'green' if analysis['signal'] == 1 else 'red'
    fig.add_vline(x=df.index[-1], line_dash="dash", line_color=signal_color)
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} Signal Analysis - {timestamp}',
        xaxis_rangeslider_visible=False,
        height=1200
    )
    
    # Save plot
    plot_path = os.path.join(signal_dir, 'signal_analysis.html')
    fig.write_html(plot_path)
    
    return csv_path, plot_path

async def run_and_send_backtest(symbol: str, trading_bot, days: int = lookback_days_param):
    """Run a backtest for the symbol and send the results through telegram"""
    try:
        # Run backtest with data export
        backtest_result = run_backtest_with_export(symbol=symbol,
                                                  days=days,
                                                  params=None,  # Will use default or best params
                                                  is_simulating=False,
                                                  lookback_days_param=lookback_days_param)
        
        # Create message with stats and file paths
        stats = backtest_result['stats']
        message = f"""
📊 Backtest Results for {symbol}:
Total Return: {stats['total_return']:.2%}
Sharpe Ratio: {stats['sharpe_ratio']:.2f}
Max Drawdown: {stats['max_drawdown']:.2%}
Win Rate: {stats['win_rate']:.2%}
Profit Factor: {stats['profit_factor']:.2f}
Number of Trades: {stats['num_trades']}

Files have been saved to the backtests directory.
        """
        
        await trading_bot.send_message(message)
        
    except Exception as e:
        error_msg = f"Error running backtest for {symbol}: {str(e)}"
        logger.error(error_msg)
        await trading_bot.send_message(error_msg)

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
