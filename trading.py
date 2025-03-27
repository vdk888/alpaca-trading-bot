import logging
import numpy as np
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from fetch import is_market_open
from config import TRADING_SYMBOLS, default_interval_yahoo, PER_SYMBOL_CAPITAL_MULTIPLIER, calculate_capital_multiplier, lookback_days_param
import pytz
from datetime import datetime, timedelta
from utils import get_api_symbol, get_display_symbol
from fetch import fetch_historical_data

logger = logging.getLogger(__name__)

class TradingExecutor:
    def __init__(self, trading_client: TradingClient, symbol: str):
        self.trading_client = trading_client
        self.symbol = symbol
        self.is_active = True
        self.config = TRADING_SYMBOLS[symbol]
        
    def _check_market_hours(self) -> bool:
        """Check if market is open for this symbol"""
        market_hours = self.config['market_hours']
        market_tz = pytz.timezone(market_hours['timezone'])
        now = datetime.now(market_tz)
        
        # For 24/7 markets like Forex
        if market_hours['start'] == '00:00' and market_hours['end'] == '23:59':
            return True
            
        # Check weekday
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
            
        # Parse market hours
        start_time = datetime.strptime(market_hours['start'], '%H:%M').time()
        end_time = datetime.strptime(market_hours['end'], '%H:%M').time()
        current_time = now.time()
        
        return start_time <= current_time <= end_time

    def get_position(self):
        """Get current position details"""
        try:
            return self.trading_client.get_open_position(get_api_symbol(self.symbol))
        except Exception as e:
            if "no position" in str(e).lower():
                return None
            raise

    def calculate_position_size(self, current_price: float, risk_percent: float = 0.02) -> float:
        """
        Calculate position size based on account equity and risk management
        
        Args:
            current_price: Current price of the asset
            risk_percent: Maximum risk per trade as percentage of equity (default: 2%)
        """
        try:
            account = self.trading_client.get_account()
            equity = float(account.equity)
            
            # Get current position value if any
            try:
                position = self.trading_client.get_open_position(get_api_symbol(self.symbol))
                current_position_value = float(position.market_value)
            except Exception:
                current_position_value = 0
                
            # Calculate remaining available capital (PER_SYMBOL_CAPITAL_MULTIPLIER% of equity - current position value)
            symbols = list(TRADING_SYMBOLS.keys())
            capital_multiplier = calculate_capital_multiplier(lookback_days_param/2)
            max_total_position = equity / len(symbols) * capital_multiplier  # capital_multiplier% of total capital
            available_capital = max_total_position - current_position_value
            
            if available_capital <= 0:
                logger.info(f"Maximum position size reached for {get_display_symbol(self.symbol)} ({self.config['name']}) ({max_total_position/equity}% of capital)")
                return 0
            
            # Calculate quantity based on available capital and risk
            qty = min(available_capital, equity * risk_percent) / current_price
            
            # Round down to nearest whole number for stocks, keep decimals for crypto
            if self.config['market'] == 'CRYPTO':
                qty = round(qty, 8)  # Round to 8 decimal places for crypto
            else:
                qty = int(qty)  # Round down to whole number for stocks
            
            # Ensure minimum position size
            min_qty = 1 if self.config['market'] != 'CRYPTO' else 0.0001
            if qty < min_qty:
                qty = min_qty
                
            return qty
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0

    def calculate_shares_from_amount(self, amount: float, current_price: float) -> float:
        """Calculate number of shares based on dollar amount"""
        shares = amount / current_price
        if self.config['market'] == 'CRYPTO':
            shares = round(shares, 8)  # Round to 8 decimal places for crypto
        else:
            shares = int(shares)  # Round down to nearest whole share for stocks
        return shares

    def calculate_performance_ranking(self, current_price: float, lookback_days: int = lookback_days_param) -> tuple[float, float]:
        """Calculate performance ranking using backtest results."""
        try:
            logger.info(f"Calculating performance ranking for {self.symbol} at price {current_price:.2f} over {lookback_days} days")
            
            # Try to load best_params.json for strategy performance data
            try:
                params_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_params.json")
                with open(params_file, "r") as f:
                    best_params_data = json.load(f)
            except Exception as e:
                logger.error(f"Could not load best_params.json: {e}")
                best_params_data = {}

            performance_dict = {}

            # Calculate performance for each symbol using backtest results
            for sym, config in TRADING_SYMBOLS.items():
                try:
                    logger.info(f"Processing symbol: {sym}")
                    
                    # Run a quick backtest to get recent performance
                    backtest_days = min(lookback_days, 30)  # Limit to 30 days for speed
                    params = best_params_data.get(sym, {}).get('best_params', get_default_params())
                    
                    from backtest_individual import run_backtest
                    result = run_backtest(symbol=sym,
                                       days=backtest_days,
                                       params=params,
                                       is_simulating=True,
                                       lookback_days_param=lookback_days)
                    
                    # Use total return as performance metric
                    performance = result['stats']['total_return']
                    performance_dict[sym] = performance
                    
                    logger.info(f"{sym} Performance: {performance:.2f}% (Backtest)")

                except Exception as e:
                    logger.error(f"Error calculating performance for {sym}: {str(e)}")
                    continue

            # Calculate percentile ranking
            if performance_dict:
                performances = list(performance_dict.values())
                current_perf = performance_dict.get(self.symbol)
                if current_perf is not None:
                    rank = sum(p <= current_perf for p in performances) / len(performances)
                    logger.info(f"Performance rank for {self.symbol}: {rank:.2f} (based on {len(performances)} symbols)")
                    return rank, current_perf
                else:
                    logger.warning(f"No performance data found for current symbol {self.symbol}")
            else:
                logger.warning("No performance data available for any symbols")

            return 0.0, 0.0  # Default to worst rank if calculation fails

        except Exception as e:
            logger.error(f"Error in performance ranking calculation: {str(e)}", exc_info=True)
            return 0.0, 0.0

    async def execute_trade(self, action: str, analysis: dict, notify_callback=None) -> bool:
        """
        Execute trade on Alpaca
        
        Args:
            action: "BUY" or "SELL"
            analysis: Analysis dict containing current_price and other metrics
            notify_callback: Optional callback for notifications
            
        Returns:
            bool: True if trade was executed successfully
        """
        capital_multiplier = calculate_capital_multiplier(lookback_days_param/2)
        if not self.is_active:
            if notify_callback:
                await notify_callback("Trading is currently paused. Use /resume to resume trading.")
            return False

        try:
            # Check market hours
            if not self._check_market_hours():
                message = f"Market is closed for {get_display_symbol(self.symbol)} ({self.config['name']})"
                logger.warning(message)
                if notify_callback:
                    await notify_callback(message)
                return False
            
            # For buy orders, calculate new position size
            if action == "BUY":
                # Calculate performance ranking
                rank, performance = self.calculate_performance_ranking(analysis['current_price'])
                
                # Calculate buy percentage (linear function)
                # rank 1 (best) = 50% buy
                # rank 0 (worst) = 0% buy
                def calculate_buy_percentage(rank: float) -> float:
                    """
                    Calculate buy percentage based on rank.
                    rank: 1 is best performer, 0 is worst performer
                    Returns: float between 0.0 and 0.5 representing buy percentage
                    """
                    # If in bottom third, buy 0%
                    if rank < 0.33:
                        return 0.0
                    
                    # For top two-thirds, use inverted wave function
                    x = (rank - 0.33) / 0.67  # Normalize to 0-1 range for top two-thirds
                    wave = 0.02 * np.sin(2 * np.pi * x)  # Small oscillation
                    linear = 0.48 * x  # Linear increase from 0.0 to 0.48
                    return max(0.0, min(0.5, linear + wave))  # Clamp between 0.0 and 0.5
                
                buy_percentage = calculate_buy_percentage(rank)
                
                # Calculate account equity
                account = self.trading_client.get_account()
                equity = float(account.equity)
                
                # Calculate position size based on buy percentage
                max_position_size = self.calculate_position_size(analysis['current_price'])
                new_qty = max_position_size * buy_percentage
              
                
                # Round based on market type
                if self.config['market'] == 'CRYPTO':
                    new_qty = round(new_qty, 8)  # Round to 8 decimal places for crypto
                else:
                    new_qty = int(new_qty)  # Round down to whole number for stocks
                
                # Ensure minimum position size
                min_qty = 1 if self.config['market'] != 'CRYPTO' else 0.0001
                if new_qty < min_qty:
                    new_qty = min_qty
                
                # Get total position value (existing + new)
                try:
                    position = self.trading_client.get_open_position(get_api_symbol(self.symbol))
                    existing_position_value = float(position.market_value)
                except Exception:
                    existing_position_value = 0
                    
                new_position_value = new_qty * analysis['current_price']
                total_position_value = existing_position_value + new_position_value
                exposure_percentage = (total_position_value / equity) * 100
                
                # Notify that order is being sent
                notional_value = round(new_qty * analysis['current_price'], 2) if self.config['market'] == 'CRYPTO' else new_qty * analysis['current_price']
                sending_message = f"""🔄 Sending BUY Order for {get_display_symbol(self.symbol)} ({self.config['name']}):
• Performance Rank: {rank:.2f}
• Buy Percentage: {buy_percentage*100:.1f}%
• Capital Multiplier: {capital_multiplier:.2f}
• Quantity: {new_qty}
• Target Price: ${analysis['current_price']:.2f}
• Order Value: ${notional_value:.2f}
• Total Position Value: ${total_position_value:.2f}
• Total Account Exposure: {exposure_percentage:.2f}%"""
                logger.info(sending_message)
                if notify_callback:
                    await notify_callback(sending_message)
                
                # Check if new_qty is <= 0 after sending the message
                if new_qty <= 0:
                    error_message = f"❌ Maximum position size reached or invalid size calculated for {get_display_symbol(self.symbol)} ({self.config['name']})"
                    logger.info(error_message)
                    if notify_callback:
                        await notify_callback(error_message)
                    return False

                # Submit buy order (only if new_qty is > 0)
                # Submit buy order
                order = MarketOrderRequest(
                    symbol=get_api_symbol(self.symbol),
                    notional=round(analysis['current_price'] * new_qty, 2) if self.config['market'] == 'CRYPTO' else None,
                    qty=None if self.config['market'] == 'CRYPTO' else new_qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC if self.config['market'] == 'CRYPTO' else TimeInForce.DAY
                )
                
                submitted_order = self.trading_client.submit_order(order)
                
                # Create detailed order confirmation message
                message = f"""✅ BUY Order Executed for {get_display_symbol(self.symbol)} ({self.config['name']}):
• Performance Rank: {rank:.2f}
• Buy Percentage: {buy_percentage*100:.1f}%
• Capital Multiplier: {capital_multiplier:.2f}
• Quantity: {new_qty}
• Price: ${analysis['current_price']:.2f}
• Order Value: ${(new_qty * analysis['current_price']):.2f}
• Total Position Value: ${total_position_value:.2f}
• Total Account Exposure: {exposure_percentage:.2f}%
• Daily Signal: {analysis['daily_composite']:.4f}
• Weekly Signal: {analysis['weekly_composite']:.4f}
• Order ID: {submitted_order.id}"""
                
                logger.info(message)
                if notify_callback:
                    await notify_callback(message)
                
                return True
                
            # For sell orders, get current position
            else:
                try:
                    position = self.trading_client.get_open_position(get_api_symbol(self.symbol))
                    total_qty = abs(float(position.qty))
                    avg_entry_price = float(position.avg_entry_price)
                    
                    # Calculate performance ranking and sell percentage
                    rank, performance = self.calculate_performance_ranking(analysis['current_price'])
                    
                    # Calculate sell percentage (linear function)
                    # rank 1 (best) = 10% sell
                    # rank 0 (worst) = 100% sell
                    sell_percentage = 1.0 - (0.9 * rank)
                    qty_to_sell = total_qty * sell_percentage
                    
                    # Round based on market type
                    if self.config['market'] == 'CRYPTO':
                        # For crypto, ensure we don't exceed available balance by rounding down
                        qty_to_sell = float(str(total_qty * sell_percentage).rstrip('0'))  # Remove trailing zeros
                        if qty_to_sell > total_qty:
                            qty_to_sell = total_qty
                    else:
                        qty_to_sell = int(qty_to_sell)
                    
                    # Calculate performance metrics
                    profit_loss = (analysis['current_price'] - avg_entry_price) * qty_to_sell
                    profit_loss_percentage = ((analysis['current_price'] / avg_entry_price) - 1) * 100
                    
                    # Notify that order is being sent
                    sending_message = f"""🔄 Sending SELL Order for {get_display_symbol(self.symbol)} ({self.config['name']}):
• Performance Rank: {rank:.2f}
• Capital Multiplier: {capital_multiplier:.2f}
• Sell Percentage: {sell_percentage*100:.1f}%
• Quantity to Sell: {qty_to_sell} of {total_qty}
• Target Price: ${analysis['current_price']:.2f}
• Estimated Value: ${(qty_to_sell * analysis['current_price']):.2f}"""
                    logger.info(sending_message)
                    if notify_callback:
                        await notify_callback(sending_message)
                    
                    # Submit sell order
                    order = MarketOrderRequest(
                        symbol=get_api_symbol(self.symbol),
                        qty=qty_to_sell,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.GTC if self.config['market'] == 'CRYPTO' else TimeInForce.DAY
                    )
                    
                    submitted_order = self.trading_client.submit_order(order)
                    
                    # Create detailed order confirmation message
                    message = f"""✅ SELL Order Executed for {get_display_symbol(self.symbol)} ({self.config['name']}):
• Performance Rank: {rank:.2f}
• Capital Multiplier: {capital_multiplier:.2f}
• Sell Percentage: {sell_percentage*100:.1f}%
• Quantity Sold: {qty_to_sell} of {total_qty}
• Price: ${analysis['current_price']:.2f}
• Total Value: ${(qty_to_sell * analysis['current_price']):.2f}
• P&L: ${profit_loss:.2f} ({profit_loss_percentage:+.2f}%)
• Daily Signal: {analysis['daily_composite']:.4f}
• Weekly Signal: {analysis['weekly_composite']:.4f}
• Order ID: {submitted_order.id}"""
                    
                    logger.info(message)
                    if notify_callback:
                        await notify_callback(message)
                    
                    return True

                except Exception as e:
                    if "no position" in str(e).lower():
                        message = f"No position to sell for {get_display_symbol(self.symbol)} ({self.config['name']})"
                        logger.info(message)
                        if notify_callback:
                            await notify_callback(message)
                        return False
                    raise

        except Exception as e:
            error_msg = f"Error executing trade for {get_display_symbol(self.symbol)} ({self.config['name']}): {str(e)}"
            logger.error(error_msg)
            if notify_callback:
                await notify_callback(f"❌ {error_msg}")
            return False

    async def open_position(self, amount: float, current_price: float, notify_callback=None) -> bool:
        """
        Open a new position with specified dollar amount
        
        Args:
            amount: Dollar amount to invest
            current_price: Current price of the asset
            notify_callback: Optional callback for notifications
        """
        try:
            if not self._check_market_hours():
                message = f"Market is closed for {get_display_symbol(self.symbol)} ({self.config['name']})"
                logger.warning(message)
                if notify_callback:
                    await notify_callback(message)
                return False
            
            # Calculate shares based on amount
            shares = self.calculate_shares_from_amount(amount, current_price)
            
            if shares <= 0:
                message = f"Invalid position size calculated for {get_display_symbol(self.symbol)} ({self.config['name']})"
                logger.error(message)
                if notify_callback:
                    await notify_callback(message)
                return False
            
            # Submit buy order
            order = MarketOrderRequest(
                symbol=get_api_symbol(self.symbol),
                qty=shares,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC if self.config['market'] == 'CRYPTO' else TimeInForce.DAY
            )
            
            # Submit the order and get confirmation
            submitted_order = self.trading_client.submit_order(order)
            
            # Initial order message
            message = f"""🔄 Opening position: BUY {shares} {get_display_symbol(self.symbol)} (${amount:.2f}) at ${current_price:.2f}
Order ID: {submitted_order.id}"""
            logger.info(message)
            if notify_callback:
                await notify_callback(message)
            
            # Wait briefly for order to be processed
            import asyncio
            await asyncio.sleep(2)
            
            # Get order status
            order_status = self.trading_client.get_order_by_id(submitted_order.id)
            
            # Create confirmation message
            if order_status.status == 'filled':
                filled_price = float(order_status.filled_avg_price)
                filled_qty = float(order_status.filled_qty)
                total_value = filled_price * filled_qty
                
                confirmation = f"""✅ Order Executed Successfully:
• Symbol: {get_display_symbol(self.symbol)} ({self.config['name']})
• Quantity: {filled_qty}
• Price: ${filled_price:.2f}
• Total Value: ${total_value:.2f}
• Order ID: {order_status.id}"""
                logger.info(confirmation)
                if notify_callback:
                    await notify_callback(confirmation)
            else:
                status_msg = f"Order Status: {order_status.status}"
                logger.info(status_msg)
                if notify_callback:
                    await notify_callback(status_msg)
            
            return True
            
        except Exception as e:
            error_msg = f"Error opening position for {get_display_symbol(self.symbol)} ({self.config['name']}): {str(e)}"
            logger.error(error_msg)
            if notify_callback:
                await notify_callback(f"❌ {error_msg}")
            return False

    async def close_position(self, notify_callback=None) -> bool:
        """
        Close entire position for this symbol
        """
        try:
            if not self._check_market_hours():
                message = f"Market is closed for {get_display_symbol(self.symbol)} ({self.config['name']})"
                logger.warning(message)
                if notify_callback:
                    await notify_callback(message)
                return False
            
            # Get current position
            try:
                position = self.trading_client.get_open_position(get_api_symbol(self.symbol))
                shares = abs(float(position.qty))
                
                # Submit sell order
                order = MarketOrderRequest(
                    symbol=get_api_symbol(self.symbol),
                    qty=shares,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC if self.config['market'] == 'CRYPTO' else TimeInForce.DAY
                )
                
                self.trading_client.submit_order(order)
                
                message = f"Closing position: SELL {shares} {get_display_symbol(self.symbol)} ({self.config['name']}) at market price"
                logger.info(message)
                if notify_callback:
                    await notify_callback(message)
                
                return True
                
            except Exception as e:
                if "position does not exist" in str(e).lower() or "no position" in str(e).lower():
                    message = f"No open position for {get_display_symbol(self.symbol)} ({self.config['name']})"
                    logger.info(message)
                    if notify_callback:
                        await notify_callback(message)
                    return False
                raise
                
        except Exception as e:
            error_msg = f"Error closing position for {get_display_symbol(self.symbol)} ({self.config['name']}): {str(e)}"
            logger.error(error_msg)
            if notify_callback:
                await notify_callback(f"❌ {error_msg}")
            return False

    def pause_trading(self) -> str:
        """Pause trading"""
        self.is_active = False
        return "Trading paused. Use /resume to resume trading."

    def resume_trading(self) -> str:
        """Resume trading"""
        self.is_active = True
        return "Trading resumed. Bot will now execute trades."
