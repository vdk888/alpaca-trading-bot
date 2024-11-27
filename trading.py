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
            
            # Calculate maximum position value based on risk
            max_position_value = equity * risk_percent
            
            # Calculate quantity
            qty = max_position_value / current_price
            
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
            
            # Get current position
            current_position = self.get_position()
            
            # Handle existing position
            if current_position:
                current_side = current_position.side
                current_qty = abs(float(current_position.qty))
                
                # If we're already in the desired position, do nothing
                if (action == "BUY" and current_side == "long") or (action == "SELL" and current_side == "short"):
                    message = f"Already in {action} position for {self.symbol}"
                    logger.info(message)
                    if notify_callback:
                        await notify_callback(message)
                    return True
                
                # Close existing position
                close_order = MarketOrderRequest(
                    symbol=self.symbol,
                    qty=current_qty,
                    side=OrderSide.SELL if current_side == "long" else OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                self.trading_client.submit_order(close_order)
                logger.info(f"Closed existing {current_side} position of {current_qty} {self.symbol}")
            
            # Calculate new position size
            new_qty = self.calculate_position_size(analysis['current_price'])
            
            if new_qty <= 0:
                message = f"Invalid position size calculated for {self.symbol}"
                logger.error(message)
                if notify_callback:
                    await notify_callback(message)
                return False
            
            # Submit new order
            order = MarketOrderRequest(
                symbol=self.symbol,
                qty=new_qty,
                side=OrderSide.BUY if action == "BUY" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            
            self.trading_client.submit_order(order)
            
            message = f"{action} {new_qty} {self.symbol} at ${analysis['current_price']:.2f}"
            logger.info(message)
            if notify_callback:
                await notify_callback(message)
            
            return True

        except Exception as e:
            error_msg = f"Error executing trade for {self.symbol}: {str(e)}"
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
