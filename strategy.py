from typing import Dict, Optional
import pandas as pd
from indicators import generate_signals, get_default_params
from fetch import get_latest_data, fetch_historical_data
import logging
import pytz

logger = logging.getLogger(__name__)

class TradingStrategy:
    def __init__(self, symbol: str, interval: str = '5m', params: Optional[Dict] = None):
        self.symbol = symbol
        self.interval = interval
        self.params = params if params is not None else get_default_params()
        self.current_position = 0  # -1: short, 0: neutral, 1: long
        self.data = None
        self.last_update = None
        self.last_signal_time = None
        self._initialize_data()
        
    def _initialize_data(self) -> None:
        """Initialize historical data for the strategy"""
        try:
            # Fetch 3 days of historical data
            self.data = fetch_historical_data(self.symbol, self.interval, days=3)
            self.last_update = pd.Timestamp.now(tz=pytz.UTC)
            logger.info(f"Initialized data for {self.symbol}: {len(self.data)} bars")
        except Exception as e:
            logger.error(f"Error initializing data for {self.symbol}: {str(e)}")
            self.data = None
        
    def _update_data(self) -> None:
        """Update data if needed"""
        now = pd.Timestamp.now(tz=pytz.UTC)
        
        # If no data or last update was more than 5 minutes ago
        if self.data is None or (now - self.last_update).total_seconds() > 300:
            try:
                # Get latest data
                new_data = get_latest_data(self.symbol, self.interval)
                
                if new_data is not None and not new_data.empty:
                    self.data = new_data
                    self.last_update = now
                    logger.info(f"Updated data for {self.symbol}: {len(self.data)} bars")
                else:
                    logger.warning(f"No new data available for {self.symbol}")
                    
            except Exception as e:
                logger.error(f"Error updating data for {self.symbol}: {str(e)}")
                # Keep using old data if update fails
                if self.data is None:
                    raise ValueError(f"No data available for {self.symbol}")
        
    def analyze(self) -> Dict:
        """
        Analyze current market data and generate trading signals
        
        Returns:
            Dict containing signal and analysis data
        """
        try:
            # Update data
            self._update_data()
            
            if self.data is None or self.data.empty:
                raise ValueError(f"No data available for {self.symbol}")
            
            # Generate signals using our indicators
            signals, daily_data, weekly_data = generate_signals(self.data, self.params)
            
            if signals.empty:
                raise ValueError(f"No signals generated for {self.symbol}")
            
            # Get the latest signal
            latest_signal = signals.iloc[-1]
            
            # Update last signal time if we have a new signal
            if latest_signal['signal'] != 0:
                self.last_signal_time = self.data.index[-1]
                # Log the signal details
                signal_type = "BUY" if latest_signal['signal'] == 1 else "SELL"
                logger.info(f" New {signal_type} Signal for {self.symbol}:")
                logger.info(f"• Daily Composite: {latest_signal['daily_composite']:.4f}")
                logger.info(f"• Weekly Composite: {latest_signal['weekly_composite']:.4f}")
                logger.info(f"• Current Price: ${self.data['close'].iloc[-1]:.2f}")
            
            # Calculate price changes
            price_change_5m = self.data['close'].pct_change().iloc[-1]
            price_change_1h = self.data['close'].pct_change(12).iloc[-1]  # 12 5-minute bars = 1 hour
            
            # Prepare the analysis result
            analysis = {
                'signal': latest_signal['signal'],
                'daily_composite': latest_signal['daily_composite'],
                'daily_upper_limit': latest_signal['daily_up_lim'],
                'daily_lower_limit': latest_signal['daily_down_lim'],
                'weekly_composite': latest_signal['weekly_composite'],
                'weekly_upper_limit': latest_signal['weekly_up_lim'],
                'weekly_lower_limit': latest_signal['weekly_down_lim'],
                'current_price': self.data['close'].iloc[-1],
                'price_change_5m': price_change_5m,
                'price_change_1h': price_change_1h,
                'timestamp': self.data.index[-1],
                'position': self.current_position,
                'data_points': len(self.data),
                'weekly_bars': len(self.data.resample('35min').last()),
                'last_signal_time': self.last_signal_time
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing {self.symbol}: {str(e)}")
            return None
            
    def should_trade(self, analysis: Dict) -> tuple[bool, Optional[str]]:
        """
        Determine if we should trade based on the analysis
        
        Args:
            analysis: Dict containing analysis data
            
        Returns:
            tuple(bool, str): (should_trade, action)
            where action is "BUY" or "SELL" if should_trade is True
        """
        if not analysis:
            return False, None
            
        signal = analysis['signal']
        
        # No signal
        if signal == 0:
            return False, None
            
        # Long signal - always allow buying
        if signal == 1:
            return True, "BUY"
                
        # Short signal - will close entire position
        elif signal == -1:
            return True, "SELL"
                
        return False, None

    def update_position(self, new_position: int) -> None:
        """Update the current position"""
        self.current_position = new_position
