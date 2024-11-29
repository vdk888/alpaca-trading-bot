import logging
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from fetch import is_market_open
from config import TRADING_SYMBOLS
import pytz
from datetime import datetime

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
            return self.trading_client.get_open_position(self.symbol)
        except Exception as e:
            if "no position" in str(e).lower():
                return None
            raise

    def calculate_position_size(self, current_price: float, risk_percent: float = 0.02) -> int:
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
                position = self.trading_client.get_open_position(self.symbol)
                current_position_value = float(position.market_value)
            except Exception:
                current_position_value = 0
                
            # Calculate remaining available capital (10% of equity - current position value)
            max_total_position = equity * 0.10  # 10% of total capital
            available_capital = max_total_position - current_position_value
            
            if available_capital <= 0:
                logger.info(f"Maximum position size reached for {self.symbol} (10% of capital)")
                return 0
            
            # Calculate quantity based on available capital and risk
            qty = min(available_capital, equity * risk_percent) / current_price
            
            # Round down to nearest whole number for stocks
            if self.config['market'] != 'FX':
                qty = int(qty)
            
            # Ensure minimum position size
            min_qty = 1 if self.config['market'] != 'FX' else 0.01
            if qty < min_qty:
                qty = min_qty
                
            return qty
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0

    def calculate_shares_from_amount(self, amount: float, current_price: float) -> float:
        """Calculate number of shares based on dollar amount"""
        shares = amount / current_price
        if self.config['market'] != 'FX':
            shares = int(shares)  # Round down to nearest whole share for stocks
        return shares

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
        if not self.is_active:
            if notify_callback:
                await notify_callback("Trading is currently paused. Use /resume to resume trading.")
            return False

        try:
            # Check market hours
            if not self._check_market_hours():
                message = f"Market is closed for {self.symbol}"
                logger.warning(message)
                if notify_callback:
                    await notify_callback(message)
                return False
            
            # For buy orders, calculate new position size
            if action == "BUY":
                new_qty = self.calculate_position_size(analysis['current_price'])
                
                if new_qty <= 0:
                    message = f"Maximum position size reached or invalid size calculated for {self.symbol}"
                    logger.info(message)
                    if notify_callback:
                        await notify_callback(message)
                    return False
                
                # Notify that order is being sent
                sending_message = f"""🔄 Sending BUY Order for {self.symbol}:
• Quantity: {new_qty}
• Target Price: ${analysis['current_price']:.2f}
• Estimated Value: ${(new_qty * analysis['current_price']):.2f}"""
                logger.info(sending_message)
                if notify_callback:
                    await notify_callback(sending_message)
                
                # Submit buy order
                order = MarketOrderRequest(
                    symbol=self.symbol,
                    qty=new_qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                
                submitted_order = self.trading_client.submit_order(order)
                
                # Create detailed order confirmation message
                message = f"""✅ BUY Order Executed for {self.symbol}:
• Quantity: {new_qty}
• Price: ${analysis['current_price']:.2f}
• Total Value: ${(new_qty * analysis['current_price']):.2f}
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
                    position = self.trading_client.get_open_position(self.symbol)
                    qty = abs(float(position.qty))
                    
                    # Notify that order is being sent
                    sending_message = f"""🔄 Sending SELL Order for {self.symbol}:
• Quantity: {qty}
• Target Price: ${analysis['current_price']:.2f}
• Estimated Value: ${(qty * analysis['current_price']):.2f}"""
                    logger.info(sending_message)
                    if notify_callback:
                        await notify_callback(sending_message)
                    
                    # Submit sell order
                    order = MarketOrderRequest(
                        symbol=self.symbol,
                        qty=qty,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY
                    )
                    
                    submitted_order = self.trading_client.submit_order(order)
                    
                    # Create detailed order confirmation message
                    message = f"""✅ SELL Order Executed for {self.symbol}:
• Quantity: {qty}
• Price: ${analysis['current_price']:.2f}
• Total Value: ${(qty * analysis['current_price']):.2f}
• Daily Signal: {analysis['daily_composite']:.4f}
• Weekly Signal: {analysis['weekly_composite']:.4f}
• Order ID: {submitted_order.id}"""
                    
                    logger.info(message)
                    if notify_callback:
                        await notify_callback(message)
                    
                    return True
                    
                except Exception as e:
                    if "no position" in str(e).lower():
                        message = f"No position to sell for {self.symbol}"
                        logger.info(message)
                        if notify_callback:
                            await notify_callback(message)
                        return False
                    raise

        except Exception as e:
            error_msg = f"Error executing trade for {self.symbol}: {str(e)}"
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
                message = f"Market is closed for {self.symbol}"
                logger.warning(message)
                if notify_callback:
                    await notify_callback(message)
                return False
            
            # Calculate shares based on amount
            shares = self.calculate_shares_from_amount(amount, current_price)
            
            if shares <= 0:
                message = f"Invalid position size calculated for {self.symbol}"
                logger.error(message)
                if notify_callback:
                    await notify_callback(message)
                return False
            
            # Submit buy order
            order = MarketOrderRequest(
                symbol=self.symbol,
                qty=shares,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            
            self.trading_client.submit_order(order)
            
            message = f"Opening position: BUY {shares} {self.symbol} (${amount:.2f}) at ${current_price:.2f}"
            logger.info(message)
            if notify_callback:
                await notify_callback(message)
            
            return True
            
        except Exception as e:
            error_msg = f"Error opening position for {self.symbol}: {str(e)}"
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
                message = f"Market is closed for {self.symbol}"
                logger.warning(message)
                if notify_callback:
                    await notify_callback(message)
                return False
            
            # Get current position
            try:
                position = self.trading_client.get_open_position(self.symbol)
                shares = abs(float(position.qty))
                
                # Submit sell order
                order = MarketOrderRequest(
                    symbol=self.symbol,
                    qty=shares,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                )
                
                self.trading_client.submit_order(order)
                
                message = f"Closing position: SELL {shares} {self.symbol} at market price"
                logger.info(message)
                if notify_callback:
                    await notify_callback(message)
                
                return True
                
            except Exception as e:
                if "no position" in str(e).lower():
                    message = f"No position to close for {self.symbol}"
                    logger.info(message)
                    if notify_callback:
                        await notify_callback(message)
                    return False
                raise
                
        except Exception as e:
            error_msg = f"Error closing position for {self.symbol}: {str(e)}"
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
