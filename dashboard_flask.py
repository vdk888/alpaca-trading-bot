import os
import io
import json
from flask import Flask, render_template, jsonify, request, send_file
from alpaca.trading.client import TradingClient
from strategy import TradingStrategy
from config import TRADING_SYMBOLS
from telegram_bot import TradingBot
from utils import get_api_symbol, get_display_symbol
from visualization import create_strategy_plot, create_multi_symbol_plot, generate_signals
from portfolio import get_portfolio_history, create_portfolio_plot
from backtest import run_portfolio_backtest, create_portfolio_backtest_plot
from backtest_individual import run_backtest, create_backtest_plot
import pandas as pd
import base64
from datetime import datetime, timedelta

# Initialize Flask app
app = Flask(__name__)

# Initialize Alpaca client
trading_client = TradingClient(
    api_key=os.getenv('ALPACA_API_KEY'),
    secret_key=os.getenv('ALPACA_SECRET_KEY'),
    paper=True  # Set to False for live trading
)

# Initialize trading strategies for all symbols
symbols = list(TRADING_SYMBOLS.keys())
strategies = {symbol: TradingStrategy(symbol) for symbol in symbols}

# Initialize TradingBot (reusing the existing class)
trading_bot = TradingBot(trading_client, strategies, symbols)

@app.route('/')
def dashboard():
    """Render the main dashboard page"""
    return render_template('dashboard.html', symbols=symbols)

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current trading status for all symbols or a specific symbol"""
    symbol = request.args.get('symbol')
    
    if symbol and symbol not in symbols:
        return jsonify({"error": f"Invalid symbol: {symbol}"}), 400
        
    symbols_to_check = [symbol] if symbol else symbols
    status_data = []
    
    for sym in symbols_to_check:
        try:
            analysis = strategies[sym].analyze()
            if not analysis:
                status_data.append({"symbol": sym, "error": "No data available"})
                continue
                
            position = "LONG" if strategies[sym].current_position == 1 else "SHORT" if strategies[sym].current_position == -1 else "NEUTRAL"
            
            # Get best parameters
            params = trading_bot.get_best_params(sym)
            
            # Get position details if any
            try:
                pos = trading_client.get_open_position(get_api_symbol(sym))
                pos_pnl = {"value": float(pos.unrealized_pl), "percent": float(pos.unrealized_plpc)*100}
            except:
                pos_pnl = {"value": 0, "percent": 0}
            
            status_data.append({
                "symbol": sym,
                "name": TRADING_SYMBOLS[sym]['name'],
                "position": position,
                "current_price": analysis['current_price'],
                "pos_pnl": pos_pnl,
                "indicators": {
                    "daily_composite": analysis['daily_composite'],
                    "daily_upper_limit": analysis['daily_upper_limit'],
                    "daily_lower_limit": analysis['daily_lower_limit'],
                    "weekly_composite": analysis['weekly_composite'],
                    "weekly_upper_limit": analysis['weekly_upper_limit'],
                    "weekly_lower_limit": analysis['weekly_lower_limit']
                },
                "price_changes": {
                    "5min": analysis['price_change_5m']*100,
                    "1hr": analysis['price_change_1h']*100
                },
                "params": params
            })
        except Exception as e:
            status_data.append({"symbol": sym, "error": str(e)})
    
    return jsonify(status_data)

@app.route('/api/balance', methods=['GET'])
def get_balance():
    """Get account balance information"""
    try:
        account = trading_client.get_account()
        today_pl = float(account.equity) - float(account.last_equity)
        today_pl_pct = (today_pl / float(account.last_equity)) * 100
        
        return jsonify({
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "buying_power": float(account.buying_power),
            "today_pl": today_pl,
            "today_pl_pct": today_pl_pct,
            "last_equity": float(account.last_equity),
            "equity": float(account.equity)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/positions', methods=['GET'])
def get_positions():
    """Get current position details for all symbols or a specific symbol"""
    symbol = request.args.get('symbol')
    
    if symbol and symbol not in symbols:
        return jsonify({"error": f"Invalid symbol: {symbol}"}), 400
        
    symbols_to_check = [symbol] if symbol else symbols
    positions_data = []
    
    for sym in symbols_to_check:
        try:
            position = trading_client.get_open_position(get_api_symbol(sym))
            account = trading_client.get_account()
            equity = float(account.equity)
            market_value = float(position.market_value)
            exposure_percentage = (market_value / equity) * 100
            
            positions_data.append({
                "symbol": sym,
                "name": TRADING_SYMBOLS[sym]['name'],
                "side": position.side.upper(),
                "qty": position.qty,
                "entry_price": float(position.avg_entry_price),
                "current_price": float(position.current_price),
                "market_value": market_value,
                "exposure_percentage": exposure_percentage,
                "unrealized_pl": float(position.unrealized_pl),
                "unrealized_plpc": float(position.unrealized_plpc)*100
            })
        except Exception as e:
            # No position for this symbol
            pass
    
    # Add summary if no specific symbol was requested
    if not symbol:
        try:
            account = trading_client.get_account()
            equity = float(account.equity)
            total_market_value = 0
            total_pnl = 0
            
            for pos in positions_data:
                total_market_value += pos["market_value"]
                total_pnl += pos["unrealized_pl"]
            
            if total_market_value > 0:
                total_exposure = (total_market_value / equity) * 100
                
                # Sort positions by market value
                positions_data.sort(key=lambda x: abs(x['market_value']), reverse=True)
                
                # Add weights to positions
                for pos in positions_data:
                    pos["weight"] = (abs(pos["market_value"]) / total_market_value) * 100
                
                return jsonify({
                    "positions": positions_data,
                    "summary": {
                        "total_market_value": total_market_value,
                        "total_exposure": total_exposure,
                        "cash_balance": float(account.cash),
                        "portfolio_value": float(account.portfolio_value),
                        "total_unrealized_pnl": total_pnl
                    }
                })
        except Exception as e:
            pass
    
    return jsonify(positions_data)

@app.route('/api/plot/<symbol>', methods=['GET'])
def get_plot(symbol):
    """Generate and return a strategy plot for a symbol"""
    if symbol not in symbols:
        return jsonify({"error": f"Invalid symbol: {symbol}"}), 400
    
    days = request.args.get('days', default=30, type=int)
    
    try:
        buf, stats = create_strategy_plot(symbol, days)
        
        # Convert plot to base64 for embedding in HTML
        buf.seek(0)
        plot_data = base64.b64encode(buf.read()).decode('utf-8')
        
        return jsonify({
            "plot": plot_data,
            "stats": stats
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/signals', methods=['GET'])
def get_signals():
    """Get latest signals for all symbols or a specific symbol"""
    symbol = request.args.get('symbol')
    
    if symbol and symbol not in symbols:
        return jsonify({"error": f"Invalid symbol: {symbol}"}), 400
        
    symbols_to_check = [symbol] if symbol else symbols
    signals_data = []
    
    for sym in symbols_to_check:
        try:
            analysis = strategies[sym].analyze()
            if not analysis:
                signals_data.append({"symbol": sym, "error": "No data available"})
                continue
                
            # Get signal strength and direction
            signal_strength = abs(analysis['daily_composite'])
            signal_direction = "BUY" if analysis['daily_composite'] > 0 else "SELL"
            
            # Format time since last signal
            last_signal_info = None
            if analysis.get('last_signal_time') is not None:
                now = pd.Timestamp.now(tz=analysis['last_signal_time'].tzinfo)
                last_time = analysis['last_signal_time']
                time_diff = now - last_time
                hours = int(time_diff.total_seconds() / 3600)
                minutes = int((time_diff.total_seconds() % 3600) / 60)
                
                last_signal_info = {
                    "time": last_time.strftime('%Y-%m-%d %H:%M'),
                    "hours_ago": hours,
                    "minutes_ago": minutes,
                    "type": "BUY" if analysis['daily_composite'] > 0 else "SELL"
                }
            
            # Classify signals
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
            
            signals_data.append({
                "symbol": sym,
                "name": TRADING_SYMBOLS[sym]['name'],
                "signal_strength": signal_strength,
                "signal_direction": signal_direction,
                "daily_signal": daily_signal,
                "weekly_signal": weekly_signal,
                "last_signal": last_signal_info,
                "current_price": analysis['current_price'],
                "daily_composite": analysis['daily_composite'],
                "weekly_composite": analysis['weekly_composite']
            })
        except Exception as e:
            signals_data.append({"symbol": sym, "error": str(e)})
    
    return jsonify(signals_data)

@app.route('/api/backtest', methods=['POST'])
def run_backtest_api():
    """Run backtest simulation"""
    data = request.json
    symbol = data.get('symbol')
    days = data.get('days', 30)
    
    if symbol and symbol not in symbols:
        return jsonify({"error": f"Invalid symbol: {symbol}"}), 400
    
    symbols_to_backtest = [symbol] if symbol else symbols
    
    try:
        results = {}
        for sym in symbols_to_backtest:
            backtest_result, stats = run_backtest(sym, days)
            
            # Generate plot
            plot_buf = create_backtest_plot(backtest_result, sym, days)
            plot_buf.seek(0)
            plot_data = base64.b64encode(plot_buf.read()).decode('utf-8')
            
            results[sym] = {
                "stats": stats,
                "plot": plot_data
            }
        
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/portfolio', methods=['GET'])
def get_portfolio():
    """Get portfolio history"""
    interval = request.args.get('interval', '1D')
    timeframe = request.args.get('timeframe', '1M')
    
    try:
        # Get portfolio history
        portfolio_data = get_portfolio_history(trading_client, interval, timeframe)
        
        # Create plot
        plot_buf = create_portfolio_plot(portfolio_data, interval, timeframe)
        plot_buf.seek(0)
        plot_data = base64.b64encode(plot_buf.read()).decode('utf-8')
        
        return jsonify({
            "plot": plot_data,
            "data": portfolio_data.to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/open', methods=['POST'])
def open_position():
    """Open a new position"""
    data = request.json
    symbol = data.get('symbol')
    amount = data.get('amount')
    
    if not symbol or symbol not in symbols:
        return jsonify({"error": f"Invalid symbol: {symbol}"}), 400
    
    if not amount or float(amount) <= 0:
        return jsonify({"error": "Invalid amount"}), 400
    
    try:
        # Get current price
        analysis = strategies[symbol].analyze()
        if not analysis:
            return jsonify({"error": f"No data available for {symbol}"}), 400
        
        current_price = analysis['current_price']
        
        # Calculate shares from amount
        executor = trading_bot.executors[symbol]
        shares = executor.calculate_shares_from_amount(float(amount), current_price)
        
        # Execute trade
        result = trading_client.submit_order(
            symbol=get_api_symbol(symbol),
            qty=shares,
            side='buy',
            type='market',
            time_in_force='day'
        )
        
        return jsonify({
            "success": True,
            "message": f"Successfully opened position for {symbol}",
            "order_id": result.id,
            "symbol": symbol,
            "shares": shares,
            "price": current_price,
            "amount": float(amount)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/close', methods=['POST'])
def close_position():
    """Close positions"""
    data = request.json
    symbol = data.get('symbol')
    
    if symbol and symbol not in symbols:
        return jsonify({"error": f"Invalid symbol: {symbol}"}), 400
    
    symbols_to_close = [symbol] if symbol else symbols
    results = []
    
    for sym in symbols_to_close:
        try:
            # Check if position exists
            try:
                position = trading_client.get_open_position(get_api_symbol(sym))
            except Exception:
                results.append({"symbol": sym, "message": "No open position"})
                continue
            
            # Close position
            result = trading_client.close_position(get_api_symbol(sym))
            
            results.append({
                "symbol": sym,
                "success": True,
                "message": f"Successfully closed position for {sym}",
                "order_id": result.id
            })
        except Exception as e:
            results.append({"symbol": sym, "error": str(e)})
    
    return jsonify(results)

@app.route('/api/orders', methods=['GET'])
def get_orders():
    """Get past orders"""
    symbol = request.args.get('symbol')
    limit = request.args.get('limit', default=20, type=int)
    
    if symbol and symbol not in symbols:
        return jsonify({"error": f"Invalid symbol: {symbol}"}), 400
    
    try:
        if symbol:
            orders = trading_client.get_orders(
                symbol=get_api_symbol(symbol),
                limit=limit,
                nested=True
            )
        else:
            orders = trading_client.get_orders(
                limit=limit,
                nested=True
            )
        
        orders_data = []
        for order in orders:
            orders_data.append({
                "id": order.id,
                "symbol": get_display_symbol(order.symbol),
                "side": order.side,
                "qty": order.qty,
                "filled_qty": order.filled_qty,
                "type": order.type,
                "status": order.status,
                "submitted_at": order.submitted_at.strftime('%Y-%m-%d %H:%M:%S'),
                "filled_at": order.filled_at.strftime('%Y-%m-%d %H:%M:%S') if order.filled_at else None,
                "filled_avg_price": order.filled_avg_price
            })
        
        return jsonify(orders_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Data store for dashboard
class DataStore:
    def __init__(self):
        self.portfolio_data = {}
        
    def update_portfolio(self, data):
        self.portfolio_data = data
        
    def get_portfolio(self):
        return self.portfolio_data

# Initialize data store
data_store = DataStore()

def run_flask_server():
    """Run the Flask server with modern parameters"""
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)