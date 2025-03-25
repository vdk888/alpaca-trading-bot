from flask import Blueprint, request, jsonify
import logging
import base64
from helpers.alpaca_service import AlpacaService
from config import symbols, TRADING_SYMBOLS
from utils import get_api_symbol
from portfolio import create_portfolio_plot


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Blueprint for account API endpoints
account_api = Blueprint('account_api', __name__)

# Get trading client from AlpacaService
alpaca_service = AlpacaService()
trading_client = alpaca_service.client

# Store executors for each symbol (will be initialized from dashboard.py)
executors = {}

@account_api.route('/api/orders')
def get_orders():
    """View past orders from Alpaca account"""
    logger.info("API call: /api/orders")
    symbol = request.args.get('symbol', None)
    limit_str = request.args.get('limit', '10')
    limit = None if limit_str == 'all' else int(limit_str)
    
    if symbol and symbol not in symbols:
        return jsonify({"error": f"Invalid symbol: {symbol}"}), 400
    
    try:
        # Initialize AlpacaService
        alpaca_service = AlpacaService()
        logger.info(f"Getting orders with limit: {limit}")
        
        # Get all orders using get_recent_trades method
        orders = alpaca_service.get_recent_trades(limit=limit if limit else None)
        logger.info(f"Retrieved {len(orders)} orders")
        
        # Filter by symbol if specified
        if symbol:
            api_symbol = get_api_symbol(symbol)
            logger.info(f"Filtering orders for symbol: {api_symbol}")
            orders = [order for order in orders if order['symbol'].lower() == api_symbol.lower()]
            
            if not orders:
                logger.warning(f"No orders found for symbol: {symbol}")
                return jsonify({
                    "success": True,
                    "orders": []
                })
        
        orders_data = []
        for order in orders:
            try:
                # Convert datetime objects to strings if needed
                submitted_at = order['submitted_at'].strftime('%Y-%m-%d %H:%M:%S') if order['submitted_at'] else None
                filled_at = order['filled_at'].strftime('%Y-%m-%d %H:%M:%S') if order['filled_at'] else None
                
                orders_data.append({
                    "id": order.get('id', 'N/A'),
                    "symbol": order['symbol'],
                    "side": order['side'],
                    "type": order['type'],
                    "qty": order['qty'],
                    "filled_qty": order['filled_qty'],
                    "filled_avg_price": order['filled_avg_price'],
                    "status": order['status'],
                    "created_at": submitted_at,
                    "updated_at": filled_at
                })
            except Exception as e:
                logger.error(f"Error processing order: {str(e)}")
                # Continue processing other orders
                continue
        
        logger.info(f"Returning {len(orders_data)} processed orders")
        return jsonify({
            "success": True,
            "orders": orders_data
        })
    except Exception as e:
        logger.error(f"Error getting orders: {str(e)}")
        return jsonify({"error": f"Error getting orders: {str(e)}"}), 500

@account_api.route('/api/position')
def get_position():
    """Get current position details"""
    logger.info("API call: /api/position")
    symbol = request.args.get('symbol', None)
    
    if symbol and symbol not in symbols:
        return jsonify({"error": f"Invalid symbol: {symbol}"}), 400
        
    symbols_to_check = [symbol] if symbol else symbols
    position_data = {}
    
    for sym in symbols_to_check:
        try:
            position = trading_client.get_open_position(get_api_symbol(sym))
            # Get account equity for exposure calculation
            account = trading_client.get_account()
            equity = float(account.equity)
            market_value = float(position.market_value)
            exposure_percentage = (market_value / equity) * 100
            
            position_data[sym] = {
                "side": position.side.upper(),
                "quantity": position.qty,
                "entry_price": float(position.avg_entry_price),
                "current_price": float(position.current_price),
                "market_value": market_value,
                "exposure": exposure_percentage,
                "unrealized_pl": float(position.unrealized_pl),
                "unrealized_plpc": float(position.unrealized_plpc)*100,
                "name": TRADING_SYMBOLS[sym]['name']
            }
        except Exception as e:
            position_data[sym] = {"message": f"No open position for {sym} ({TRADING_SYMBOLS[sym]['name']})"}
    
    # Add summary of all positions if not looking at a specific symbol
    if not symbol:
        try:
            account = trading_client.get_account()
            equity = float(account.equity)
            total_market_value = 0
            total_pnl = 0
            positions_summary = []
            
            # Calculate totals and collect position details
            for sym in symbols:
                try:
                    position = trading_client.get_open_position(get_api_symbol(sym))
                    market_value = float(position.market_value)
                    total_market_value += market_value
                    total_pnl += float(position.unrealized_pl)
                    positions_summary.append({
                        'symbol': sym,
                        'market_value': market_value,
                        'side': position.side.upper(),
                        'qty': position.qty,
                        'pnl': float(position.unrealized_pl)
                    })
                except Exception:
                    continue
            
            if total_market_value > 0:
                total_exposure = (total_market_value / equity) * 100
                
                # Sort positions by market value
                positions_summary.sort(key=lambda x: abs(x['market_value']), reverse=True)
                
                # Build positions text with weights
                positions_weighted = []
                for pos in positions_summary:
                    weight = (abs(pos['market_value']) / total_market_value) * 100
                    pnl_pct = (pos['pnl'] / abs(pos['market_value'])) * 100 if pos['market_value'] != 0 else 0
                    positions_weighted.append({
                        "symbol": pos['symbol'],
                        "side": pos['side'],
                        "qty": pos['qty'],
                        "weight": weight,
                        "pnl": pos['pnl'],
                        "pnl_pct": pnl_pct
                    })
                
                position_data["summary"] = {
                    "total_position_value": total_market_value,
                    "total_exposure": total_exposure,
                    "cash_balance": float(account.cash),
                    "portfolio_value": float(account.portfolio_value),
                    "total_unrealized_pl": total_pnl,
                    "positions": positions_weighted
                }
        except Exception as e:
            position_data["summary_error"] = f"Error generating position summary: {str(e)}"
    
    return jsonify(position_data)

@account_api.route('/api/account/portfolio')
def get_account_portfolio():
    """Get portfolio history graph"""
    logger.info("API call: /api/portfolio")
    timeframe = request.args.get('interval', '1D')  # This is passed as 'interval' from frontend
    period = request.args.get('timeframe', '1M')    # This is passed as 'timeframe' from frontend
    
    try:
        # Initialize AlpacaService
        alpaca_service = AlpacaService()
        logger.info(f"Getting portfolio history with timeframe={timeframe}, period={period}")
        
        # Get portfolio history
        history = alpaca_service.get_portfolio_history(timeframe=timeframe, period=period)
        
        # Create plot - pass only the history parameter as that's what the function expects
        buf = create_portfolio_plot(history)
        buf.seek(0)
        plot_url = base64.b64encode(buf.read()).decode()
        
        logger.info("Portfolio plot created successfully")
        return jsonify({
            "success": True,
            "plot": plot_url,
            "timeframe": timeframe,
            "period": period
        })
    except Exception as e:
        logger.error(f"Error getting portfolio history: {str(e)}")
        return jsonify({"error": f"Error getting portfolio history: {str(e)}"}), 500

@account_api.route('/api/open-position', methods=['POST'])
def open_position():
    """Open a new position with specified amount"""
    logger.info("API call: /api/open-position")
    data = request.json
    symbol = data.get('symbol')
    amount = data.get('amount')
    
    if not symbol or not amount:
        return jsonify({"error": "Symbol and amount are required"}), 400
    
    if symbol not in symbols:
        return jsonify({"error": f"Invalid symbol: {symbol}"}), 400
    
    try:
        amount = float(amount)
    except ValueError:
        return jsonify({"error": "Amount must be a number"}), 400
    
    if amount <= 0:
        return jsonify({"error": "Amount must be positive"}), 400
    
    try:
        executor = executors[symbol]
        result = executor.open_position(amount)
        
        return jsonify({
            "success": True,
            "message": f"Position opened for {symbol}",
            "order_id": result.id if result else None
        })
    except Exception as e:
        return jsonify({"error": f"Error opening position: {str(e)}"}), 500

@account_api.route('/api/close-position', methods=['POST'])
def close_position():
    """Close positions"""
    logger.info("API call: /api/close-position")
    data = request.json
    symbol = data.get('symbol')
    
    if symbol and symbol not in symbols:
        return jsonify({"error": f"Invalid symbol: {symbol}"}), 400
    
    symbols_to_close = [symbol] if symbol else symbols
    results = {}
    
    for sym in symbols_to_close:
        try:
            executor = executors[sym]
            result = executor.close_position()
            
            results[sym] = {
                "success": True,
                "message": f"Position closed for {sym}",
                "order_id": result.id if result else None
            }
        except Exception as e:
            results[sym] = {
                "success": False,
                "error": f"Error closing position: {str(e)}"
            }
    
    return jsonify(results)

# Function to set executors from dashboard.py
def set_executors(executor_dict):
    global executors
    executors = executor_dict
