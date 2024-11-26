import os
from datetime import datetime
import logging
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes
from strategy import TradingStrategy
from alpaca.trading.client import TradingClient

logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self, trading_client: TradingClient, strategy: TradingStrategy, symbol: str):
        self.trading_client = trading_client
        self.strategy = strategy
        self.symbol = symbol
        self.bot_token = os.getenv('BOT_KEY')
        self.chat_id = os.getenv('CHAT_ID')
        self.bot = Bot(token=self.bot_token)
        
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
/settings - View current trading settings
/pause - Pause trading
/resume - Resume trading
/risk - View risk metrics
/orders - View recent orders
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
📊 Technical Indicators for {self.symbol}:
MACD:
  - Line: {analysis['macd_line']:.4f}
  - Signal: {analysis['macd_signal']:.4f}
  - Histogram: {analysis['macd_hist']:.4f}

RSI: {analysis['rsi']:.2f}

Stochastic:
  - %K: {analysis['stoch_k']:.2f}
  - %D: {analysis['stoch_d']:.2f}

Composite Scores:
  - Daily: {analysis['daily_composite']:.4f}
  - Weekly: {analysis['weekly_composite']:.4f}
            """
            await update.message.reply_text(message)
        except Exception as e:
            await update.message.reply_text(f"❌ Error getting indicators: {str(e)}")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show help message"""
        await self.start_command(update, context)

    def setup_handlers(self, application: Application):
        """Setup all command handlers"""
        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CommandHandler("status", self.status_command))
        application.add_handler(CommandHandler("position", self.position_command))
        application.add_handler(CommandHandler("balance", self.balance_command))
        application.add_handler(CommandHandler("performance", self.performance_command))
        application.add_handler(CommandHandler("indicators", self.indicators_command))
        application.add_handler(CommandHandler("help", self.help_command))
