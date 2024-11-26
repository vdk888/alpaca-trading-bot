from typing import Dict, Optional
import pandas as pd
from indicators import generate_signals, get_default_params
from fetch import get_latest_data

class TradingStrategy:
    def __init__(self, symbol: str, interval: str = '5m', params: Optional[Dict] = None):
        self.symbol = symbol
        self.interval = interval
        self.params = params if params is not None else get_default_params()
        self.current_position = 0  # -1: short, 0: neutral, 1: long
        
    def analyze(self) -> Dict:
        """
        Analyze current market data and generate trading signals
        
        Returns:
            Dict containing signal and analysis data
        """
        # Get latest data
        df = get_latest_data(self.symbol, self.interval)
        
        # Generate signals using our indicators
        signals, daily_data, weekly_data = generate_signals(df, self.params)
        
        # Get the latest signal
        latest_signal = signals.iloc[-1]
        
        # Prepare the analysis result
        analysis = {
            'signal': latest_signal['Signal'],
            'daily_composite': latest_signal['Daily_Composite'],
            'weekly_composite': latest_signal['Weekly_Composite'],
            'current_price': df['close'].iloc[-1],
            'timestamp': df.index[-1],
            'position': self.current_position
        }
        
        return analysis
    
    def should_trade(self, analysis: Dict) -> tuple[bool, str]:
        """
        Determine if we should make a trade based on the analysis
        
        Returns:
            (bool, str): Whether to trade and the reason
        """
        signal = analysis['signal']
        
        # No position
        if self.current_position == 0:
            if signal == 1:
                return True, "BUY"
            if signal == -1:
                return True, "SELL"
        
        # Long position
        elif self.current_position == 1:
            if signal == -1:
                return True, "SELL"
        
        # Short position
        elif self.current_position == -1:
            if signal == 1:
                return True, "BUY"
        
        return False, "HOLD"
    
    def update_position(self, action: str):
        """Update the current position based on the trade action"""
        if action == "BUY":
            self.current_position = 1
        elif action == "SELL":
            self.current_position = -1
        elif action == "CLOSE":
            self.current_position = 0
