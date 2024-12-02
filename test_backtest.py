import logging
from backtest import run_backtest

# Set up logging
logging.basicConfig(level=logging.INFO)

try:
    # Try running backtest for QQQ
    result = run_backtest('QQQ', days=5)
    print("Backtest successful!")
    print(f"Total trades: {len(result['trades'])}")
    print(f"Final return: {result['stats']['total_return']:.2f}%")
except Exception as e:
    print(f"Error running backtest: {str(e)}")
