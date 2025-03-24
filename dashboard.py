from flask import Blueprint, render_template, jsonify, request, send_file
import os
from datetime import datetime
import logging
from strategy import TradingStrategy
from alpaca.trading.client import TradingClient
from visualization import create_strategy_plot, create_multi_symbol_plot
from config import TRADING_SYMBOLS, default_backtest_interval, PER_SYMBOL_CAPITAL_MULTIPLIER, lookback_days_param
from trading import TradingExecutor
from backtest import run_portfolio_backtest, create_portfolio_backtest_plot, create_portfolio_with_prices_plot
from backtest_individual import run_backtest, create_backtest_plot
from portfolio import get_portfolio_history, create_portfolio_plot
import pandas as pd
import pytz
from utils import get_api_symbol, get_display_symbol
import io
import base64
import matplotlib.pyplot as plt
import json
from fetch import is_market_open, get_latest_data
from helpers.alpaca_service import AlpacaService
from datetime import timedelta

# Create blueprint instead of Flask app
dashboard = Blueprint('dashboard', __name__, template_folder='templates')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dashboard.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize trading client and strategies
trading_client = TradingClient(
    os.getenv('ALPACA_API_KEY'),
    os.getenv('ALPACA_SECRET_KEY')
)

# Initialize trading executors for each symbol
symbols = list(TRADING_SYMBOLS.keys())
executors = {symbol: TradingExecutor(trading_client, symbol) for symbol in symbols}
strategies = {symbol: TradingStrategy(symbol) for symbol in symbols}

def get_best_params(symbol):
    """Get best parameters for a symbol from Object Storage"""
    try:
        # Try local file as fallback
        params_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_params.json")
        try:
            if os.path.exists(params_file):
                with open(params_file, "r") as f:
                    best_params_data = json.load(f)
                if symbol in best_params_data:
                    return best_params_data[symbol]['best_params']
                else:
                    return "Using default parameters"
            else:
                logger.warning(f"Parameters file not found at {params_file}")
                return "Using default parameters"
        except FileNotFoundError:
            logger.warning(f"Parameters file not found at {params_file}")
            return "Using default parameters"
    except Exception as e:
        logger.error(f"Error reading from best_params.json: {e}")
        return "Using default parameters"

@dashboard.route('/')
def index():
    """Render dashboard template"""
    logger.info("Rendering dashboard template")
    return render_template('dashboard.html', symbols=symbols, lookback_days=int(lookback_days_param))

@dashboard.route('/backtest')
def backtest_page():
    """Render backtest template"""
    logger.info("Rendering backtest template")
    return render_template('backtest.html', symbols=symbols)

@dashboard.route('/positions')
def positions_page():
    """Render positions template"""
    logger.info("Rendering positions template")
    return render_template('positions.html', symbols=symbols)

@dashboard.route('/orders')
def orders_page():
    """Render orders template"""
    logger.info("Rendering orders template")
    return render_template('orders.html', symbols=symbols)

@dashboard.route('/api/status')
def get_status():
    """Get current trading status for all symbols"""
    logger.info("API call: /api/status")
    symbol = request.args.get('symbol', None)
    
    if symbol and symbol not in symbols:
        return jsonify({"error": f"Invalid symbol: {symbol}"}), 400
        
    symbols_to_check = [symbol] if symbol else symbols
    status_data = {}
    
    for sym in symbols_to_check:
        try:
            strategy = strategies[sym]
            executor = executors[sym]
            analysis = strategy.analyze()
            
            if not analysis:
                status_data[sym] = {"error": "No data available"}
                continue
                
            position = "LONG" if strategy.current_position == 1 else "SHORT" if strategy.current_position == -1 else "NEUTRAL"
            
            # Get best parameters
            params = get_best_params(sym)
            
            # Get position details if any
            try:
                pos = trading_client.get_open_position(get_api_symbol(sym))
                pos_pnl = f"${float(pos.unrealized_pl):.2f} ({float(pos.unrealized_plpc)*100:.2f}%)"
            except:
                pos_pnl = "No open position"
            
            status_data[sym] = {
                "position": position,
                "current_price": analysis['current_price'],
                "pos_pnl": pos_pnl,
                "daily_composite": analysis['daily_composite'],
                "daily_upper_limit": analysis['daily_upper_limit'],
                "daily_lower_limit": analysis['daily_lower_limit'],
                "weekly_composite": analysis['weekly_composite'],
                "weekly_upper_limit": analysis['weekly_upper_limit'],
                "weekly_lower_limit": analysis['weekly_lower_limit'],
                "price_change_5m": analysis['price_change_5m']*100,
                "price_change_1h": analysis['price_change_1h']*100,
                "params": params,
                "name": TRADING_SYMBOLS[sym]['name']
            }
        except Exception as e:
            status_data[sym] = {"error": f"Error analyzing {sym}: {str(e)}"}
    
    return jsonify(status_data)

@dashboard.route('/api/position')
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

@dashboard.route('/api/balance')
def get_balance():
    """Get account balance"""
    logger.info("API call: /api/balance")
    try:
        account = trading_client.get_account()
        return jsonify({
            'cash': float(account.cash),
            'portfolio_value': float(account.portfolio_value),
            'buying_power': float(account.buying_power),
            'today_pl': float(account.equity) - float(account.last_equity)
        })
    except Exception as e:
        return jsonify({"error": f"Error getting balance: {str(e)}"}), 500

@dashboard.route('/api/performance')
def get_performance():
    """View today's performance"""
    logger.info("API call: /api/performance")
    try:
        account = trading_client.get_account()
        today_pl = float(account.equity) - float(account.last_equity)
        today_pl_pct = (today_pl / float(account.last_equity)) * 100
        
        return jsonify({
            'today_pl': today_pl,
            'today_pl_pct': today_pl_pct,
            'starting_equity': float(account.last_equity),
            'current_equity': float(account.equity)
        })
    except Exception as e:
        return jsonify({"error": f"Error getting performance: {str(e)}"}), 500

@dashboard.route('/api/indicators')
def get_indicators():
    """View current indicator values"""
    logger.info("API call: /api/indicators")
    symbol = request.args.get('symbol', None)
    
    if symbol and symbol not in symbols:
        return jsonify({"error": f"Invalid symbol: {symbol}"}), 400
        
    symbols_to_check = [symbol] if symbol else symbols
    indicators_data = {}
    
    for sym in symbols_to_check:
        try:
            analysis = strategies[sym].analyze()
            if not analysis:
                indicators_data[sym] = {"error": "No data available"}
                continue
                
            # Get best parameters
            params = get_best_params(sym)
            
            indicators_data[sym] = {
                "daily_composite": analysis['daily_composite'],
                "daily_upper_limit": analysis['daily_upper_limit'],
                "daily_lower_limit": analysis['daily_lower_limit'],
                "weekly_composite": analysis['weekly_composite'],
                "weekly_upper_limit": analysis['weekly_upper_limit'],
                "weekly_lower_limit": analysis['weekly_lower_limit'],
                "price_change_5m": analysis['price_change_5m']*100,
                "price_change_1h": analysis['price_change_1h']*100,
                "params": params,
                "name": TRADING_SYMBOLS[sym]['name']
            }
        except Exception as e:
            indicators_data[sym] = {"error": f"Error analyzing {sym}: {str(e)}"}
    
    return jsonify(indicators_data)

@dashboard.route('/api/plot')
def get_plot():
    """Generate strategy visualization"""
    logger.info("API call: /api/plot")
    symbol = request.args.get('symbol', None)
    
    if not symbol or symbol not in symbols:
        return jsonify({"error": f"Invalid or missing symbol parameter"}), 400
        
    days = request.args.get('days', default=5, type=int)
    
    if days <= 0 or days > default_backtest_interval:
        return jsonify({"error": f"Days must be between 1 and {default_backtest_interval}"}), 400
    
    try:
        # Get best parameters
        params = get_best_params(symbol)
        
        # Create plot
        buf, stats = create_strategy_plot(symbol, days)
        
        # Convert plot to base64 image
        buf.seek(0)
        plot_url = base64.b64encode(buf.read()).decode()
        
        return jsonify({
            'plot': plot_url,
            'stats': {
                'symbol': symbol,
                'name': TRADING_SYMBOLS[symbol]['name'],
                'days': days,
                'params': params,
                'trading_days': stats['trading_days'],
                'price_change': stats['price_change'],
                'buy_signals': stats['buy_signals'],
                'sell_signals': stats['sell_signals']
            }
        })
    except Exception as e:
        logger.error(f"Error generating plot: {str(e)}")
        return jsonify({"error": f"Error generating plot: {str(e)}"}), 500

@dashboard.route('/api/signals')
def get_signals():
    """View latest signals for all symbols"""
    logger.info("API call: /api/signals")
    symbol = request.args.get('symbol', None)
    
    if symbol and symbol not in symbols:
        return jsonify({"error": f"Invalid symbol: {symbol}"}), 400
        
    symbols_to_check = [symbol] if symbol else symbols
    signals_data = {}
    
    for sym in symbols_to_check:
        try:
            analysis = strategies[sym].analyze()
            if not analysis:
                signals_data[sym] = {"error": "No data available"}
                continue
                
            # Get signal strength and direction
            signal_strength = abs(analysis['daily_composite'])
            strength_emoji = "🔥" if signal_strength > 0.8 else "💪" if signal_strength > 0.5 else "👍"
            
            # Format time since last signal with signal type
            last_signal_info = None
            if analysis.get('last_signal_time') is not None:
                now = pd.Timestamp.now(tz=pytz.UTC)
                last_time = analysis['last_signal_time']
                time_diff = now - last_time
                hours = int(time_diff.total_seconds() / 3600)
                minutes = int((time_diff.total_seconds() % 3600) / 60)
                # Get the signal type from the stored composite value
                signal_type = "BUY" if analysis['daily_composite'] > 0 else "SELL"
                last_signal_info = {
                    "type": signal_type,
                    "strength": strength_emoji,
                    "time": last_time.strftime('%Y-%m-%d %H:%M'),
                    "hours_ago": hours,
                    "minutes_ago": minutes
                }
            
            # Classify signals
            signal_direction = "BUY" if analysis['daily_composite'] > 0 else "SELL"
            daily_signal = (
                "STRONG BUY" if analysis['daily_composite'] > analysis['daily_upper_limit']
                else "STRONG SELL" if analysis['daily_composite'] < analysis['daily_lower_limit']
                else "WEAK " + signal_direction if signal_strength > 0.5
                else "NEUTRAL"
            )
            
            weekly_signal = (
                "STRONG BUY" if analysis['weekly_composite'] > analysis['weekly_upper_limit']
                else "STRONG SELL" if analysis['weekly_composite'] < analysis['weekly_lower_limit']
                else "WEAK BUY" if analysis['weekly_composite'] > 0
                else "WEAK SELL" if analysis['weekly_composite'] < 0
                else "NEUTRAL"
            )
            
            # Get best parameters
            params = get_best_params(sym)
            
            signals_data[sym] = {
                "last_signal": last_signal_info,
                "daily_signal": daily_signal,
                "daily_composite": analysis['daily_composite'],
                "weekly_signal": weekly_signal,
                "weekly_composite": analysis['weekly_composite'],
                "params": params,
                "name": TRADING_SYMBOLS[sym]['name']
            }
        except Exception as e:
            signals_data[sym] = {"error": f"Error analyzing {sym}: {str(e)}"}
    
    return jsonify(signals_data)

@dashboard.route('/api/markets')
def get_markets():
    """View market hours for all symbols"""
    logger.info("API call: /api/markets")
    market_data = {}
    
    for symbol in symbols:
        try:
            is_open = is_market_open(symbol)
            market_data[symbol] = {
                "is_open": is_open,
                "name": TRADING_SYMBOLS[symbol]['name'],
                "exchange": TRADING_SYMBOLS[symbol].get('exchange', 'Unknown')
            }
        except Exception as e:
            market_data[symbol] = {"error": f"Error checking market status: {str(e)}"}
    
    return jsonify(market_data)

@dashboard.route('/api/symbols')
def get_symbols():
    """List all trading symbols"""
    logger.info("API call: /api/symbols")
    symbol_data = []
    
    for symbol in symbols:
        symbol_data.append({
            "name": TRADING_SYMBOLS[symbol]['name'],
            "exchange": TRADING_SYMBOLS[symbol].get('exchange', 'Unknown'),
            "api_symbol": get_api_symbol(symbol),
            "display_symbol": get_display_symbol(symbol),
            "symbol": symbol  # Add the original symbol key
        })
    
    return jsonify(symbol_data)

@dashboard.route('/api/position/open', methods=['POST'])
def open_position():
    """Open a new position with specified amount"""
    logger.info("API call: /api/position/open")
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

@dashboard.route('/api/position/close', methods=['POST'])
def close_position():
    """Close positions"""
    logger.info("API call: /api/position/close")
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

@dashboard.route('/api/backtest', methods=['POST'])
def run_backtest_api():
    """Run backtest simulation"""
    logger.info("API call: /api/backtest")
    data = request.json
    symbol = data.get('symbol')
    days = data.get('days', default_backtest_interval)
    
    logger.info(f"Backtest request: symbol={symbol}, days={days}")
    
    try:
        days = int(days)
    except ValueError:
        logger.error(f"Invalid days value: {days}")
        return jsonify({"error": "Days must be a number"}), 400
    
    if days <= 0 or days > default_backtest_interval:
        logger.error(f"Days out of range: {days}")
        return jsonify({"error": f"Days must be between 1 and {default_backtest_interval}"}), 400
    
    if symbol == "portfolio":
        try:
            logger.info("Running portfolio backtest")
            # Run portfolio backtest directly
            results = run_portfolio_backtest(symbols, days)
            
            logger.info("Creating portfolio backtest plot")
            # Create plot
            buf = create_portfolio_backtest_plot(results)
            buf.seek(0)
            plot_url = base64.b64encode(buf.read()).decode()
            
            # Extract key metrics for the frontend
            metrics = {
                "total_return_pct": results['metrics']['total_return'],
                "max_drawdown_pct": results['metrics']['max_drawdown'],
                "final_value": results['metrics']['final_value'],
                "initial_capital": results['metrics']['initial_capital'],
                "trading_costs": results['metrics']['trading_costs']
            }
            
            # Calculate annualized return if we have enough data
            if len(results['data']) > 1:
                days_elapsed = (results['data'].index[-1] - results['data'].index[0]).days
                if days_elapsed > 0:
                    annualized_return = ((1 + results['metrics']['total_return']/100) ** (365/days_elapsed) - 1) * 100
                    metrics["annualized_return_pct"] = annualized_return
                else:
                    metrics["annualized_return_pct"] = results['metrics']['total_return']
            else:
                metrics["annualized_return_pct"] = 0
                
            # Calculate Sharpe ratio if available
            if 'sharpe_ratio' in results['metrics']:
                metrics["sharpe_ratio"] = results['metrics']['sharpe_ratio']
            else:
                metrics["sharpe_ratio"] = 0
            
            logger.info("Portfolio backtest completed successfully")
            return jsonify({
                "success": True,
                "plot": plot_url,
                "results": metrics
            })
        except Exception as e:
            logger.error(f"Error running portfolio backtest: {str(e)}", exc_info=True)
            return jsonify({"error": f"Error running portfolio backtest: {str(e)}"}), 500
    else:
        if symbol and symbol not in symbols:
            logger.error(f"Invalid symbol: {symbol}")
            return jsonify({"error": f"Invalid symbol: {symbol}"}), 400
            
        symbols_to_backtest = [symbol] if symbol else symbols
        results = {}
        
        for sym in symbols_to_backtest:
            try:
                logger.info(f"Running backtest for {sym}")
                # Run backtest
                result = run_backtest(sym, days)
                
                logger.info(f"Creating backtest plot for {sym}")
                # Create plot - this returns a tuple of (buffer, stats)
                buf, stats = create_backtest_plot(result)
                buf.seek(0)
                plot_url = base64.b64encode(buf.read()).decode()
                
                # Extract key metrics for the frontend
                metrics = {
                    "total_return_pct": stats['total_return'],
                    "buy_hold_return_pct": stats.get('buy_hold_return', 0),
                    "max_drawdown_pct": stats['max_drawdown'],
                    "win_rate": stats['win_rate'],
                    "total_trades": stats['total_trades'],
                    "sharpe_ratio": stats.get('sharpe_ratio', 0),
                    "trading_costs": stats.get('trading_costs', 0)
                }
                
                results[sym] = {
                    "success": True,
                    "plot": plot_url,
                    "result": metrics
                }
                logger.info(f"Backtest for {sym} completed successfully")
            except Exception as e:
                logger.error(f"Error running backtest for {sym}: {str(e)}", exc_info=True)
                results[sym] = {
                    "success": False,
                    "error": f"Error running backtest: {str(e)}"
                }
        
        return jsonify(results)

@dashboard.route('/api/backtest/info')
def get_backtest_info():
    """Get backtest info without running a full simulation"""
    logger.info("API call: /api/backtest/info")
    symbol = request.args.get('symbol', None)
    days = request.args.get('days', default=default_backtest_interval, type=int)
    
    if symbol and symbol not in symbols:
        return jsonify({"error": f"Invalid symbol: {symbol}"}), 400
        
    if days <= 0 or days > default_backtest_interval:
        return jsonify({"error": f"Days must be between 1 and {default_backtest_interval}"}), 400
    
    try:
        # Get best parameters
        params = get_best_params(symbol) if symbol else "Using default parameters"
        
        return jsonify({
            'symbol': symbol,
            'days': days,
            'params': params,
            'message': f"Ready to run backtest for {symbol if symbol else 'all symbols'} over {days} days"
        })
    except Exception as e:
        logger.error(f"Error getting backtest info: {str(e)}")
        return jsonify({"error": f"Error getting backtest info: {str(e)}"}), 500

@dashboard.route('/api/portfolio')
def get_portfolio():
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

@dashboard.route('/api/rank')
def get_rank():
    """Display performance ranking of all assets"""
    logger.info("API call: /api/rank")
    days = request.args.get('days', 7, type=int)
    logger.info(f"Performance ranking for the last {days} days")
    
    try:
        # Get performance data for all symbols
        performance_data = []
        
        for symbol in symbols:
            try:
                executor = executors[symbol]
                strategy = strategies[symbol]
                analysis = strategy.analyze()
                
                if not analysis or 'current_price' not in analysis:
                    continue
                
                # Calculate performance ranking like in the Telegram bot
                rank, return_pct = executor.calculate_performance_ranking(analysis['current_price'], lookback_days=days)
                
                # Add to performance data
                performance_data.append({
                    "symbol": symbol,
                    "return_pct": return_pct,
                    "rank": rank
                })
            except Exception as e:
                logger.error(f"Error processing {symbol} for ranking: {str(e)}")
                continue
        
        # Sort by return percentage (best to worst)
        performance_data.sort(key=lambda x: x['return_pct'], reverse=True)
        
        return jsonify({
            "success": True,
            "performance": performance_data,
            "days": days
        })
    except Exception as e:
        logger.error(f"Error getting performance ranking: {str(e)}", exc_info=True)
        return jsonify({"error": f"Error getting performance ranking: {str(e)}"}), 500

@dashboard.route('/api/orders')
def get_orders():
    """View past orders from Alpaca account"""
    logger.info("API call: /api/orders")
    symbol = request.args.get('symbol', None)
    limit = request.args.get('limit', 10, type=int)
    
    if symbol and symbol not in symbols:
        return jsonify({"error": f"Invalid symbol: {symbol}"}), 400
    
    try:
        # Initialize AlpacaService
        alpaca_service = AlpacaService()
        logger.info(f"Getting orders with limit: {limit}")
        
        # Get all orders using get_recent_trades method
        orders = alpaca_service.get_recent_trades(limit=limit)
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

@dashboard.route('/api/price-data')
def get_price_data():
    """Get price data for a specific symbol"""
    logger.info("API call: /api/price-data")
    symbol = request.args.get('symbol', None)
    days = request.args.get('days', int(lookback_days_param), type=int)
    logger.info(f"Fetching price data for {symbol} over {days} days")
    
    if not symbol:
        return jsonify({"error": "Symbol parameter is required"}), 400
        
    if symbol not in symbols:
        return jsonify({"error": f"Invalid symbol: {symbol}"}), 400
    
    try:
        from fetch import get_latest_data
        from indicators import generate_signals, get_default_params
        import pytz
        from datetime import timedelta
        import pandas as pd
        
        # Fetch price data for the symbol
        df = get_latest_data(symbol, days=days)
        
        # Limit to the requested number of days
        if days:
            # Convert index to datetime if it's not already
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
                
            # Filter for the last N days
            cutoff_date = datetime.now(pytz.UTC) - timedelta(days=days)
            df = df[df.index >= cutoff_date]
            logger.info(f"Filtered data for {symbol} from {cutoff_date} to now, resulting in {len(df)} data points")
        
        # Get parameters for indicators
        try:
            # Try to get best parameters for the symbol
            best_params = get_best_params(symbol)
            if isinstance(best_params, str):  # If it returns a message string instead of params
                params = get_default_params()
            else:
                params = best_params
        except Exception as e:
            logger.warning(f"Error getting parameters for {symbol}: {str(e)}. Using default parameters.")
            params = get_default_params()
            
        # Calculate indicators
        try:
            signal_data = generate_signals(df, params)
            daily_data = signal_data[1]  # Daily composite data
            weekly_data = signal_data[2]  # Weekly composite data
            logger.info(f"Generated indicators for {symbol}: {len(daily_data)} daily points, {len(weekly_data)} weekly points")
        except Exception as e:
            logger.error(f"Error generating indicators for {symbol}: {str(e)}")
            raise
        
        # Format the data for Chart.js
        price_data = {
            'labels': [idx.strftime('%Y-%m-%d %H:%M') for idx in df.index],
            'symbol': symbol,
            'name': TRADING_SYMBOLS[symbol]['name'],
            'days': days,  # Include days in the response
            
            # OHLC data for candlestick chart
            'ohlc': [
                {
                    'x': idx.strftime('%Y-%m-%d %H:%M'),
                    'o': float(row['open']),
                    'h': float(row['high']),
                    'l': float(row['low']),
                    'c': float(row['close']),
                    'v': float(row['volume'])
                } for idx, row in df.iterrows()
            ],
            
            # Add daily indicator data
            'daily_composite': daily_data['Composite'].tolist(),
            'daily_up_lim': daily_data['Up_Lim'].tolist(),
            'daily_down_lim': daily_data['Down_Lim'].tolist(),
            'daily_up_lim_2std': daily_data['Up_Lim_2STD'].tolist(),
            'daily_down_lim_2std': daily_data['Down_Lim_2STD'].tolist(),
            
            # Add weekly indicator data
            'weekly_composite': weekly_data['Composite'].tolist(),
            'weekly_up_lim': weekly_data['Up_Lim'].tolist(),
            'weekly_down_lim': weekly_data['Down_Lim'].tolist(),
            'weekly_up_lim_2std': weekly_data['Up_Lim_2STD'].tolist(),
            'weekly_down_lim_2std': weekly_data['Down_Lim_2STD'].tolist()
        }
        
        return jsonify(price_data)
        
    except Exception as e:
        logger.error(f"Error fetching price data for {symbol}: {str(e)}")
        return jsonify({"error": f"Error fetching price data: {str(e)}"}), 500

if __name__ == '__main__':
    app = dashboard
    app.run(debug=True)
