from config import TRADING_SYMBOLS

def get_api_symbol(symbol: str) -> str:
    """Convert a symbol to its API format.
    
    Args:
        symbol (str): The symbol from config (e.g., 'LTC/USD')
        
    Returns:
        str: The symbol in API format (e.g., 'LTCUSD' for crypto)
    """
    if symbol not in TRADING_SYMBOLS:
        return symbol
    return symbol.replace('/', '') if TRADING_SYMBOLS[symbol]['market'] == 'CRYPTO' else symbol

def get_display_symbol(api_symbol: str) -> str:
    """Convert an API symbol back to display format.
    
    Args:
        api_symbol (str): The symbol in API format (e.g., 'LTCUSD')
        
    Returns:
        str: The symbol in display format (e.g., 'LTC/USD' for crypto)
    """
    # First try direct match
    if api_symbol in TRADING_SYMBOLS:
        return api_symbol
        
    # Try to find corresponding crypto symbol
    for symbol in TRADING_SYMBOLS:
        if TRADING_SYMBOLS[symbol]['market'] == 'CRYPTO' and symbol.replace('/', '') == api_symbol:
            return symbol
            
    return api_symbol  # Return as-is if no match found
