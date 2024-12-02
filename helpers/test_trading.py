import unittest
from unittest.mock import Mock, patch
from datetime import datetime
import pytz
import asyncio
from trading import TradingExecutor
from config import TRADING_SYMBOLS

class MockPosition:
    def __init__(self, symbol, qty, side):
        self.symbol = symbol
        self.qty = qty
        self.side = side

class MockAccount:
    def __init__(self, equity):
        self.equity = str(equity)

class TestTradingExecutor(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_client = Mock()
        self.symbols = list(TRADING_SYMBOLS.keys())
        
    def test_market_hours(self):
        """Test market hours checking for different symbols"""
        for symbol in self.symbols:
            executor = TradingExecutor(self.mock_client, symbol)
            
            # Test during market hours
            market_hours = TRADING_SYMBOLS[symbol]['market_hours']
            tz = pytz.timezone(market_hours['timezone'])
            
            # Test forex 24/7 market
            if market_hours['start'] == '00:00' and market_hours['end'] == '23:59':
                with patch('trading.datetime') as mock_datetime:
                    mock_dt = datetime(2024, 1, 3, 14, 30)  # Wednesday 2:30 PM
                    mock_datetime.now.return_value = tz.localize(mock_dt)
                    mock_datetime.strptime = datetime.strptime
                    self.assertTrue(executor._check_market_hours())
            else:
                # Test during market hours
                with patch('trading.datetime') as mock_datetime:
                    mock_dt = datetime(2024, 1, 3, 14, 30)  # Wednesday 2:30 PM
                    mock_datetime.now.return_value = tz.localize(mock_dt)
                    mock_datetime.strptime = datetime.strptime
                    self.assertTrue(executor._check_market_hours())
                
                # Test outside market hours
                with patch('trading.datetime') as mock_datetime:
                    mock_dt = datetime(2024, 1, 3, 3, 0)  # Wednesday 3:00 AM
                    mock_datetime.now.return_value = tz.localize(mock_dt)
                    mock_datetime.strptime = datetime.strptime
                    self.assertFalse(executor._check_market_hours())
                
                # Test weekend
                with patch('trading.datetime') as mock_datetime:
                    mock_dt = datetime(2024, 1, 6, 14, 30)  # Saturday 2:30 PM
                    mock_datetime.now.return_value = tz.localize(mock_dt)
                    mock_datetime.strptime = datetime.strptime
                    self.assertFalse(executor._check_market_hours())

    def test_position_size_calculation(self):
        """Test position size calculations"""
        executor = TradingExecutor(self.mock_client, self.symbols[0])
        
        # Mock account with $100,000 equity
        self.mock_client.get_account.return_value = MockAccount(100000)
        
        # Test stock position size (2% risk)
        size = executor.calculate_position_size(current_price=100.0)
        self.assertEqual(size, 20)  # $100,000 * 0.02 / $100 = 20 shares
        
        # Test minimum position size
        size = executor.calculate_position_size(current_price=10000.0)
        self.assertEqual(size, 1)  # Should return minimum 1 share
        
        # Test forex position size
        executor.config['market'] = 'FX'
        size = executor.calculate_position_size(current_price=1.10)
        self.assertGreaterEqual(size, 0.01)  # Should be at least minimum forex size

    async def test_trade_execution(self):
        """Test trade execution logic"""
        executor = TradingExecutor(self.mock_client, self.symbols[0])
        
        # Mock account and market hours
        self.mock_client.get_account.return_value = MockAccount(100000)
        
        with patch.object(executor, '_check_market_hours', return_value=True):
            # Test new position (no existing position)
            self.mock_client.get_open_position.side_effect = Exception("no position")
            success = await executor.execute_trade("BUY", {'current_price': 100.0})
            self.assertTrue(success)
            self.mock_client.submit_order.assert_called_once()
            
            # Reset mock
            self.mock_client.submit_order.reset_mock()
            self.mock_client.get_open_position.side_effect = None
            
            # Test closing existing position and opening new one
            self.mock_client.get_open_position.return_value = MockPosition(self.symbols[0], 10, "long")
            success = await executor.execute_trade("SELL", {'current_price': 100.0})
            self.assertTrue(success)
            self.assertEqual(self.mock_client.submit_order.call_count, 2)  # Close old + open new
            
            # Test market closed
            with patch.object(executor, '_check_market_hours', return_value=False):
                success = await executor.execute_trade("BUY", {'current_price': 100.0})
                self.assertFalse(success)

if __name__ == '__main__':
    unittest.main()
