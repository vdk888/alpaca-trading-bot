import os
from datetime import datetime
import logging
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes
from strategy import TradingStrategy
from alpaca.trading.client import TradingClient
from visualization import create_strategy_plot

logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self, trading_client: TradingClient, strategy: TradingStrategy, symbol: str):
        self.trading_client = trading_client
        self.strategy = strategy
        self.symbol = symbol
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('CHAT_ID')
        
        if not self.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN not found in environment variables")
        if not self.chat_id:
            raise ValueError("CHAT_ID not found in environment variables")
            
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
            logger.info(f"Starting trading bot for {self.symbol}")
            print(f"Starting trading bot for {self.symbol}...")
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
            await self.bot.send_message(chat_id=self.chat_id, text=message)
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start the bot and show available commands"""
        commands = """
🤖 Trading Bot Commands:
/start - Show this help message
/status - Get current trading status
/position - View current position details
/balance - Check account balance
/performance - View today's performance
/plot [symbol] [days] - Generate strategy visualization
/indicators - View current indicator values
/help - Show this help message
        """
        await update.message.reply_text(f"Trading bot started\nTrading {self.symbol} on 5-minute timeframe\n\n{commands}")

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Get current status"""
        try:
            analysis = self.strategy.analyze()
            position = "LONG" if self.strategy.current_position == 1 else "SHORT" if self.strategy.current_position == -1 else "NEUTRAL"
            
            message = f"""
📊 Status for {self.symbol}:
Position: {position}
Current Price: ${analysis['current_price']:.2f}
Daily Composite: {analysis['daily_composite']:.4f}
Weekly Composite: {analysis['weekly_composite']:.4f}
Last Update: {analysis['timestamp']}
            """
            await update.message.reply_text(message)
        except Exception as e:
            await update.message.reply_text(f"❌ Error getting status: {str(e)}")

    async def position_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Get current position details"""
        try:
            try:
                position = self.trading_client.get_open_position(self.symbol)
                message = f"""
📈 Position Details for {self.symbol}:
Side: {position.side.upper()}
Quantity: {position.qty}
Entry Price: ${float(position.avg_entry_price):.2f}
Current Price: ${float(position.current_price):.2f}
Market Value: ${float(position.market_value):.2f}
Unrealized P&L: ${float(position.unrealized_pl):.2f} ({float(position.unrealized_plpc)*100:.2f}%)
                """
            except:
                message = f"No open position for {self.symbol}"
            await update.message.reply_text(message)
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
            analysis = self.strategy.analyze()
            message = f"""
📈 {self.symbol} Indicators:

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
            await update.message.reply_text(message)
        except Exception as e:
            await update.message.reply_text(f"❌ Error getting indicators: {str(e)}")

    async def plot_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Generate and send a strategy visualization plot."""
        try:
            # Parse arguments
            args = context.args
            symbol = args[0].upper() if len(args) > 0 else 'SPY'
            days = int(args[1]) if len(args) > 1 else 5
            
            # Validate days
            if days <= 0 or days > 30:
                await update.message.reply_text("❌ Days must be between 1 and 30")
                return
            
            # Send "generating" message
            status_message = await update.message.reply_text(
                f"📊 Generating visualization for {symbol} (last {days} days)..."
            )
            
            try:
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
            except Exception as e:
                logger.error(f"Plot generation error for {symbol}: {str(e)}")
                if hasattr(e, '__traceback__'):
                    import traceback
                    logger.error(traceback.format_exc())
                await update.message.reply_text(f"❌ Error generating plot: {str(e)}")
            finally:
                # Always try to delete the "generating" message
                try:
                    await status_message.delete()
                except Exception:
                    pass
                
        except ValueError as e:
            await update.message.reply_text(f"❌ Invalid input: {str(e)}")
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {str(e)}")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show help message"""
        await self.start_command(update, context)
