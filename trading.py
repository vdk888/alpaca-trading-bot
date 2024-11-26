import logging
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

logger = logging.getLogger(__name__)

class TradingExecutor:
    def __init__(self, trading_client: TradingClient, symbol: str):
        self.trading_client = trading_client
        self.symbol = symbol
        self.is_active = True

    def get_position_size(self):
        """Get current position size"""
        try:
            position = self.trading_client.get_open_position(self.symbol)
            return abs(float(position.qty)) if position else 0
        except:
            return 0

    def calculate_position_size(self, current_price: float, risk_percent: float = 0.95):
        """Calculate position size based on buying power and risk"""
        account = self.trading_client.get_account()
        buying_power = float(account.buying_power)
        return int(buying_power * risk_percent / current_price)

    async def execute_trade(self, action: str, analysis: dict, notify_callback=None):
        """Execute trade on Alpaca"""
        if not self.is_active:
            if notify_callback:
                await notify_callback("Trading is currently paused. Use /resume to resume trading.")
            return

        try:
            qty = self.get_position_size()
            
            if action in ["BUY", "SELL"]:
                # Close existing position if any
                if qty > 0:
                    close_order = MarketOrderRequest(
                        symbol=self.symbol,
                        qty=qty,
                        side=OrderSide.SELL if position.side == 'long' else OrderSide.BUY,
                        time_in_force=TimeInForce.DAY
                    )
                    self.trading_client.submit_order(close_order)
                
                # Calculate new position size
                new_qty = self.calculate_position_size(analysis['current_price'])
                
                if new_qty > 0:
                    order = MarketOrderRequest(
                        symbol=self.symbol,
                        qty=new_qty,
                        side=OrderSide.BUY if action == "BUY" else OrderSide.SELL,
                        time_in_force=TimeInForce.DAY
                    )
                    self.trading_client.submit_order(order)
                    
                    message = f"{action} {new_qty} shares of {self.symbol} at ${analysis['current_price']:.2f}"
                    logger.info(message)
                    if notify_callback:
                        await notify_callback(message)

        except Exception as e:
            error_msg = f"Error executing trade: {str(e)}"
            logger.error(error_msg)
            if notify_callback:
                await notify_callback(f"❌ {error_msg}")

    def pause_trading(self):
        """Pause trading"""
        self.is_active = False
        return "Trading paused. Use /resume to resume trading."

    def resume_trading(self):
        """Resume trading"""
        self.is_active = True
        return "Trading resumed. Bot will now execute trades."
