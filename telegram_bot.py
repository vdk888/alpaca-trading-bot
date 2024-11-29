import os
from datetime import datetime
import logging
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes
from strategy import TradingStrategy
from alpaca.trading.client import TradingClient
from visualization import create_strategy_plot, create_multi_symbol_plot
from config import TRADING_SYMBOLS
from trading import TradingExecutor
import pandas as pd

logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self, trading_client: TradingClient, strategies: dict, symbols: list):
        self.trading_client = trading_client
        self.strategies = strategies  # Dict of symbol -> TradingStrategy
        self.symbols = symbols
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('CHAT_ID')
        
        if not self.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN not found in environment variables")
        if not self.chat_id:
            raise ValueError("CHAT_ID not found in environment variables")
            
        # Initialize trading executors for each symbol
        self.executors = {symbol: TradingExecutor(trading_client, symbol) for symbol in symbols}
            
        # Initialize the application and bot
        self.application = Application.builder().token(self.bot_token).build()
        self._bot = None  # Will be initialized in start()
        self.setup_handlers(self.application)
        
    def setup_handlers(self, application: Application):
        """Setup all command handlers"""
        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CommandHandler("status", self.status_command))
        application.add_handler(CommandHandler("position", self.position_command))
        application.add_handler(CommandHandler("balance", self.balance_command))
        application.add_handler(CommandHandler("performance", self.performance_command))
        application.add_handler(CommandHandler("indicators", self.indicators_command))
        application.add_handler(CommandHandler("plot", self.plot_command))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("signals", self.signals_command))
        application.add_handler(CommandHandler("markets", self.markets_command))
        application.add_handler(CommandHandler("symbols", self.symbols_command))
        application.add_handler(CommandHandler("open", self.open_command))
        application.add_handler(CommandHandler("close", self.close_command))

    @property
    def bot(self):
        """Lazy initialization of bot instance"""
        if self._bot is None:
            self._bot = Bot(token=self.bot_token)
        return self._bot

    async def start(self):
        """Start the Telegram bot"""
        try:
            # Initialize the application
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling()
            
            # Send startup message
            await self.send_message("🤖 Trading Bot started successfully!")
            
            # Log startup
            logger.info(f"Starting trading bot for {', '.join(self.symbols)}")
            print(f"Starting trading bot for {', '.join(self.symbols)}...")
            print("Telegram bot initialized. Use /start to begin.")
            
        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            raise

    async def stop(self):
        """Stop the Telegram bot"""
        try:
            if self._bot:
                await self._bot.close()
                self._bot = None
                
            if hasattr(self.application, 'updater'):
                await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
        except Exception as e:
            logger.error(f"Error stopping bot: {e}")
            
    async def send_message(self, message: str):
        """Send message to Telegram"""
        try:
            # Split message into chunks of 4096 characters (Telegram's limit)
            chunks = [message[i:i+4096] for i in range(0, len(message), 4096)]
            for chunk in chunks:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=chunk,
                    parse_mode='HTML',  # Enable HTML formatting
                    disable_web_page_preview=True  # Disable link previews
                )
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start the bot and show available commands"""
        commands = """
🤖 Multi-Symbol Trading Bot Commands:

📊 Status Commands:
/status [symbol] - Get current trading status (all symbols if none specified)
/position [symbol] - View current position details
/balance - Check account balance
/performance - View today's performance

📈 Analysis Commands:
/indicators [symbol] - View current indicator values
/plot [symbol] [days] - Generate strategy visualization
/signals - View latest signals for all symbols

🔧 Trading Commands:
/open <symbol> <amount> - Open a position with specified amount
/close [symbol] - Close positions (all positions if no symbol specified)

⚙️ Management Commands:
/symbols - List all trading symbols
/markets - View market hours for all symbols
/help - Show this help message
        """
        symbols_list = "\n".join([f"• {symbol}" for symbol in self.symbols])
        await update.message.reply_text(f"Trading bot started\nMonitoring the following symbols:\n{symbols_list}\n\n{commands}")

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Get current status"""
        try:
            # Check if a specific symbol was requested
            symbol = context.args[0].upper() if context.args else None
            
            if symbol and symbol not in self.symbols:
                await update.message.reply_text(f"❌ Invalid symbol: {symbol}")
                return
                
            symbols_to_check = [symbol] if symbol else self.symbols
            status_messages = []
            has_data = False
            
            for sym in symbols_to_check:
                try:
                    analysis = self.strategies[sym].analyze()
                    if not analysis:
                        status_messages.append(f"No data available for {sym}")
                        continue
                        
                    has_data = True
                    position = "LONG" if self.strategies[sym].current_position == 1 else "SHORT" if self.strategies[sym].current_position == -1 else "NEUTRAL"
                    
                    # Get position details if any
                    try:
                        pos = self.trading_client.get_open_position(sym)
                        pos_pnl = f"P&L: ${float(pos.unrealized_pl):.2f} ({float(pos.unrealized_plpc)*100:.2f}%)"
                    except:
                        pos_pnl = "No open position"
                    
                    status_messages.append(f"""
📊 {sym} Status:
Position: {position}
Current Price: ${analysis['current_price']:.2f}
{pos_pnl}

Indicators:
• Daily Composite: {analysis['daily_composite']:.4f}
  - Upper: {analysis['daily_upper_limit']:.4f}
  - Lower: {analysis['daily_lower_limit']:.4f}
• Weekly Composite: {analysis['weekly_composite']:.4f}
  - Upper: {analysis['weekly_upper_limit']:.4f}
  - Lower: {analysis['weekly_lower_limit']:.4f}

Price Changes:
• 5min: {analysis['price_change_5m']*100:.2f}%
• 1hr: {analysis['price_change_1h']*100:.2f}%

Last Update: {analysis['timestamp']}
                    """)
                except Exception as e:
                    status_messages.append(f"Error analyzing {sym}: {str(e)}")
            
            if not has_data:
                await update.message.reply_text("❌ No data available for any symbol. The market may be closed or there might be connection issues.")
                return
                
            await update.message.reply_text("\n---\n".join(status_messages))
        except Exception as e:
            await update.message.reply_text(f"❌ Error getting status: {str(e)}")

    async def position_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Get current position details"""
        try:
            # Check if a specific symbol was requested
            symbol = context.args[0].upper() if context.args else None
            
            if symbol and symbol not in self.symbols:
                await update.message.reply_text(f"❌ Invalid symbol: {symbol}")
                return
                
            symbols_to_check = [symbol] if symbol else self.symbols
            position_messages = []
            
            for sym in symbols_to_check:
                try:
                    position = self.trading_client.get_open_position(sym)
                    message = f"""
📈 {sym} Position Details:
Side: {position.side.upper()}
Quantity: {position.qty}
Entry Price: ${float(position.avg_entry_price):.2f}
Current Price: ${float(position.current_price):.2f}
Market Value: ${float(position.market_value):.2f}
Unrealized P&L: ${float(position.unrealized_pl):.2f} ({float(position.unrealized_plpc)*100:.2f}%)
                    """
                except:
                    message = f"No open position for {sym}"
                position_messages.append(message)
            
            await update.message.reply_text("\n---\n".join(position_messages))
        except Exception as e:
            await update.message.reply_text(f"❌ Error getting position: {str(e)}")

    async def balance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Check account balance"""
        try:
            account = self.trading_client.get_account()
            message = f"""
💰 Account Balance:
Cash: ${float(account.cash):.2f}
Portfolio Value: ${float(account.portfolio_value):.2f}
Buying Power: ${float(account.buying_power):.2f}
Today's P&L: ${float(account.equity) - float(account.last_equity):.2f}
            """
            await update.message.reply_text(message)
        except Exception as e:
            await update.message.reply_text(f"❌ Error getting balance: {str(e)}")

    async def performance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """View today's performance"""
        try:
            account = self.trading_client.get_account()
            today_pl = float(account.equity) - float(account.last_equity)
            today_pl_pct = (today_pl / float(account.last_equity)) * 100
            
            message = f"""
📈 Today's Performance:
P&L: ${today_pl:.2f} ({today_pl_pct:.2f}%)
Starting Equity: ${float(account.last_equity):.2f}
Current Equity: ${float(account.equity):.2f}
            """
            await update.message.reply_text(message)
        except Exception as e:
            await update.message.reply_text(f"❌ Error getting performance: {str(e)}")

    async def indicators_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """View current indicator values"""
        try:
            # Check if a specific symbol was requested
            symbol = context.args[0].upper() if context.args else None
            
            if symbol and symbol not in self.symbols:
                await update.message.reply_text(f"❌ Invalid symbol: {symbol}")
                return
                
            symbols_to_check = [symbol] if symbol else self.symbols
            indicator_messages = []
            has_data = False
            
            for sym in symbols_to_check:
                try:
                    analysis = self.strategies[sym].analyze()
                    if not analysis:
                        indicator_messages.append(f"No data available for {sym}")
                        continue
                        
                    has_data = True
                    message = f"""
📈 {sym} Indicators:

Daily Composite: {analysis['daily_composite']:.4f}
• Upper Limit: {analysis['daily_upper_limit']:.4f}
• Lower Limit: {analysis['daily_lower_limit']:.4f}

Weekly Composite: {analysis['weekly_composite']:.4f}
• Upper Limit: {analysis['weekly_upper_limit']:.4f}
• Lower Limit: {analysis['weekly_lower_limit']:.4f}

Price Changes:
• 5min: {analysis['price_change_5m']*100:.2f}%
• 1hr: {analysis['price_change_1h']*100:.2f}%

Current Price: ${analysis['current_price']:.2f}
Last Update: {analysis['timestamp']}
                    """
                    indicator_messages.append(message)
                except Exception as e:
                    indicator_messages.append(f"Error analyzing {sym}: {str(e)}")
            
            if not has_data:
                await update.message.reply_text("❌ No data available for any symbol. The market may be closed or there might be connection issues.")
                return
                
            await update.message.reply_text("\n---\n".join(indicator_messages))
        except Exception as e:
            await update.message.reply_text(f"❌ Error getting indicators: {str(e)}")

    async def plot_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Generate and send strategy visualization plots."""
        try:
            # Parse arguments
            args = context.args if context.args else []
            
            # Handle specific symbol request
            if len(args) >= 2:
                symbol = args[0].upper()
                days = int(args[1])
                if symbol not in self.symbols:
                    await update.message.reply_text(f"❌ Invalid symbol: {symbol}\nAvailable symbols: {', '.join(self.symbols)}")
                    return
                symbols_to_plot = [symbol]
            else:
                # Default: all symbols
                symbols_to_plot = self.symbols
                days = int(args[0]) if args else 5
            
            if days <= 0 or days > 30:
                await update.message.reply_text("❌ Days must be between 1 and 30")
                return
            
            await update.message.reply_text(f"📊 Generating plots for the last {days} days...")
            
            # Generate and send plot for each symbol
            for symbol in symbols_to_plot:
                try:
                    buf, stats = create_strategy_plot(symbol, days)
                    
                    stats_message = f"""
📈 {symbol} Statistics ({days} days):
• Trading Days: {stats['trading_days']}
• Price Change: {stats['price_change']:.2f}%
• Buy Signals: {stats['buy_signals']}
• Sell Signals: {stats['sell_signals']}
                    """
                    
                    await update.message.reply_document(
                        document=buf,
                        filename=f"{symbol}_strategy_{days}d.png",
                        caption=stats_message
                    )
                except Exception as e:
                    logger.error(f"Error plotting {symbol}: {str(e)}")
                    await update.message.reply_text(f"❌ Could not generate plot for {symbol}: {str(e)}")
                    continue
                    
        except ValueError as e:
            await update.message.reply_text(f"❌ Invalid input: {str(e)}")
        except Exception as e:
            logger.error(f"Plot command error: {str(e)}")
            await update.message.reply_text(f"❌ Error: {str(e)}")

    async def signals_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """View latest signals for all symbols"""
        try:
            # Check if a specific symbol was requested
            symbol = context.args[0].upper() if context.args else None
            
            if symbol and symbol not in self.symbols:
                await update.message.reply_text(f"❌ Invalid symbol: {symbol}")
                return
                
            symbols_to_check = [symbol] if symbol else self.symbols
            signal_messages = []
            has_data = False
            
            # Process symbols in chunks of 3
            for i in range(0, len(symbols_to_check), 3):
                chunk_messages = []
                chunk_symbols = symbols_to_check[i:i+3]
                
                for sym in chunk_symbols:
                    try:
                        analysis = self.strategies[sym].analyze()
                        if not analysis:
                            chunk_messages.append(f"No data available for {sym}")
                            continue
                            
                        has_data = True
                        # Get signal strength and direction
                        signal_strength = abs(analysis['daily_composite'])
                        signal_direction = "BUY" if analysis['daily_composite'] > 0 else "SELL"
                        
                        # Format last signal time
                        last_signal_str = "No signals generated yet"
                        if analysis.get('last_signal_time'):
                            last_signal_time = analysis['last_signal_time']
                            if isinstance(last_signal_time, str):
                                last_signal_time = pd.to_datetime(last_signal_time)
                            last_signal_str = f"Last Signal: {last_signal_time.strftime('%H:%M')} ({signal_direction})"
                        
                        # Check if signal crosses thresholds
                        daily_signal = (
                            "STRONG BUY" if analysis['daily_composite'] > analysis['daily_upper_limit']
                            else "STRONG SELL" if analysis['daily_composite'] < analysis['daily_lower_limit']
                            else "WEAK " + signal_direction if signal_strength > 0.5
                            else "NEUTRAL"
                        )
                        
                        weekly_signal = (
                            "STRONG BUY" if analysis['weekly_composite'] > analysis['weekly_upper_limit']
                            else "STRONG SELL" if analysis['weekly_composite'] < analysis['weekly_lower_limit']
                            else "WEAK BUY" if analysis['weekly_composite'] > 0
                            else "WEAK SELL" if analysis['weekly_composite'] < 0
                            else "NEUTRAL"
                        )
                        
                        message = f"""
📊 {sym} Signals:
Daily Signal: {daily_signal}
• Composite: {analysis['daily_composite']:.4f}
• Strength: {signal_strength:.2f}
Weekly Signal: {weekly_signal}
• Composite: {analysis['weekly_composite']:.4f}
Price: ${analysis['current_price']:.2f}
{last_signal_str}"""
                        chunk_messages.append(message)
                    except Exception as e:
                        chunk_messages.append(f"Error analyzing {sym}: {str(e)}")
                
                if chunk_messages:
                    await update.message.reply_text("\n---\n".join(chunk_messages))
            
            if not has_data:
                await update.message.reply_text("❌ No signals available. The market may be closed or there might be connection issues.")
                
        except Exception as e:
            await update.message.reply_text(f"❌ Error getting signals: {str(e)}")

    async def markets_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """View market hours for all symbols"""
        try:
            market_info = []
            
            for symbol in self.symbols:
                config = TRADING_SYMBOLS[symbol]
                market_info.append(f"""
{symbol} ({config['market']}):
• Trading Hours: {config['market_hours']['start']} - {config['market_hours']['end']}
• Timezone: {config['market_hours']['timezone']}
                """)
            
            await update.message.reply_text("🕒 Market Hours:\n" + "\n".join(market_info))
        except Exception as e:
            await update.message.reply_text(f"❌ Error getting market hours: {str(e)}")

    async def symbols_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """List all trading symbols"""
        try:
            symbols_info = []
            
            for symbol in self.symbols:
                config = TRADING_SYMBOLS[symbol]
                symbols_info.append(f"""
{symbol}:
• Market: {config['market']}
• Interval: {config['interval']}
                """)
            
            await update.message.reply_text("📈 Trading Symbols:\n" + "\n".join(symbols_info))
        except Exception as e:
            await update.message.reply_text(f"❌ Error listing symbols: {str(e)}")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show help message"""
        await self.start_command(update, context)

    async def open_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Open a new position with specified amount"""
        try:
            if not context.args or len(context.args) != 2:
                await update.message.reply_text(
                    "❌ Usage: /open <symbol> <amount>\n"
                    "Example: /open SPY 1000 (to open $1000 position in SPY)"
                )
                return
            
            symbol = context.args[0].upper()
            try:
                amount = float(context.args[1])
            except ValueError:
                await update.message.reply_text("❌ Amount must be a number")
                return
            
            if symbol not in self.symbols:
                await update.message.reply_text(f"❌ Invalid symbol: {symbol}")
                return
            
            if amount <= 0:
                await update.message.reply_text("❌ Amount must be positive")
                return
            
            # Get current price from strategy
            analysis = self.strategies[symbol].analyze()
            if not analysis:
                await update.message.reply_text(f"❌ Unable to get current price for {symbol}")
                return
            
            current_price = analysis['current_price']
            
            # Execute the trade using the appropriate executor
            await self.executors[symbol].open_position(amount, current_price, self.send_message)
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error opening position: {str(e)}")

    async def close_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Close positions"""
        try:
            # If no symbol specified, close all positions
            symbol = context.args[0].upper() if context.args else None
            
            if symbol and symbol not in self.symbols:
                await update.message.reply_text(f"❌ Invalid symbol: {symbol}")
                return
            
            symbols_to_close = [symbol] if symbol else self.symbols
            success_count = 0
            
            for sym in symbols_to_close:
                try:
                    if await self.executors[sym].close_position(self.send_message):
                        success_count += 1
                except Exception as e:
                    await update.message.reply_text(f"❌ Error closing {sym} position: {str(e)}")
            
            if success_count > 0:
                message = f"Successfully closed {success_count} position(s)"
                if symbol:
                    message += f" for {symbol}"
                await update.message.reply_text(message)
            elif not symbol:  # No positions were closed when trying to close all
                await update.message.reply_text("No open positions to close")
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error closing positions: {str(e)}")
