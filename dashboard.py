from flask import Blueprint, render_template, jsonify, request, send_file
import os
from datetime import datetime
import logging
from strategy import TradingStrategy
from alpaca.trading.client import TradingClient
from visualization import create_strategy_plot, create_multi_symbol_plot
from config import TRADING_SYMBOLS, default_backtest_interval, PER_SYMBOL_CAPITAL_MULTIPLIER, lookback_days_param, DEFAULT_INTERVAL, BARS_PER_DAY
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
from flask import make_response
# Import the account_api blueprint
from account_api import account_api, set_executors

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

# Pass executors to account_api
set_executors(executors)

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
    logger.info("Rendering dashboard")

    # Convert interval to milliseconds for JavaScript
    interval_ms = 60000  # Default to 1 minute
    if DEFAULT_INTERVAL == '1h':
        interval_ms = 60 * 60 * 1000  # 1 hour in milliseconds
    elif DEFAULT_INTERVAL in BARS_PER_DAY:
        # Calculate milliseconds based on the interval
        minutes_per_bar = 24 * 60 / BARS_PER_DAY[DEFAULT_INTERVAL]
        interval_ms = int(minutes_per_bar * 60 * 1000)

    logger.info(f"Using update interval: {DEFAULT_INTERVAL} ({interval_ms}ms)")
    return render_template('dashboard.html', symbols=symbols, lookback_days=int(lookback_days_param), interval_ms=interval_ms)

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

from services.cache_service import CacheService
cache_service = CacheService()

def get_cache_key(endpoint: str, **params) -> str:
    """Generate standardized cache key"""
    key = f"dashboard:{endpoint}"
    if params:
        param_str = "_".join(f"{k}:{v}" for k, v in sorted(params.items()))
        key += f":{param_str}"
    return key

@dashboard.route('/api/status') 
def get_status():
    """Get current trading status for all symbols"""
    logger.info("API call: /api/status")
    symbol = request.args.get('symbol', None)
    
    cache_key = get_cache_key('status', symbol=symbol if symbol else 'all')
    cached_data = cache_service.get(cache_key)
    if cached_data and cache_service.is_fresh(cache_key):
        logger.info(f"Returning cached status data for {symbol if symbol else 'all symbols'}")
        return jsonify(cached_data)

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

    # Cache the result before returning
    cache_service.set_with_ttl(cache_key, status_data)
    return jsonify(status_data)

@dashboard.route('/api/balance')
def get_balance():
    """Get account balance"""
    logger.info("API call: /api/balance")
    
    cache_key = get_cache_key('balance')
    cached_data = cache_service.get(cache_key)
    if cached_data and cache_service.is_fresh(cache_key, max_age_hours=1):
        logger.info("Returning cached balance data")
        return jsonify(cached_data)
        
    try:
        account = trading_client.get_account()
        balance_data = {
            'cash': float(account.cash),
            'portfolio_value': float(account.portfolio_value),
            'buying_power': float(account.buying_power),
            'today_pl': float(account.equity) - float(account.last_equity)
        }
        cache_service.set_with_ttl(cache_key, balance_data)
        return jsonify(balance_data)
    except Exception as e:
        return jsonify({"error": f"Error getting balance: {str(e)}"}), 500

@dashboard.route('/api/performance')
def get_performance():
    """View today's performance"""
    logger.info("API call: /api/performance")
    
    cache_key = get_cache_key('performance')
    cached_data = cache_service.get(cache_key)
    if cached_data and cache_service.is_fresh(cache_key, max_age_hours=1):
        logger.info("Returning cached performance data")
        return jsonify(cached_data)
        
    try:
        account = trading_client.get_account()
        today_pl = float(account.equity) - float(account.last_equity)
        today_pl_pct = (today_pl / float(account.last_equity)) * 100

        performance_data = {
            'today_pl': today_pl,
            'today_pl_pct': today_pl_pct,
            'starting_equity': float(account.last_equity),
            'current_equity': float(account.equity)
        }
        cache_service.set_with_ttl(cache_key, performance_data, ttl_hours=1)
        return jsonify(performance_data)
    except Exception as e:
        return jsonify({"error": f"Error getting performance: {str(e)}"}), 500

@dashboard.route('/api/markets')
def get_markets():
    """View market hours for all symbols"""
    logger.info("API call: /api/markets")
    
    cache_key = get_cache_key('markets')
    cached_data = cache_service.get(cache_key)
    if cached_data and cache_service.is_fresh(cache_key, max_age_hours=1):
        logger.info("Returning cached market hours data")
        return jsonify(cached_data)
        
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
    
    cache_service.set_with_ttl(cache_key, market_data, ttl_hours=1)
    return jsonify(market_data)

@dashboard.route('/api/symbols')
def get_symbols():
    """List all trading symbols"""
    logger.info("API call: /api/symbols")
    
    cache_key = get_cache_key('symbols')
    cached_data = cache_service.get(cache_key)
    if cached_data and cache_service.is_fresh(cache_key, max_age_hours=24):
        logger.info("Returning cached symbols data")
        return jsonify(cached_data)
        
    symbol_data = []

    for symbol in symbols:
        symbol_data.append({
            "name": TRADING_SYMBOLS[symbol]['name'],
            "exchange": TRADING_SYMBOLS[symbol].get('exchange', 'Unknown'),
            "api_symbol": get_api_symbol(symbol),
            "display_symbol": get_display_symbol(symbol),
            "symbol": symbol  # Add the original symbol key
        })
    
    cache_service.set_with_ttl(cache_key, symbol_data, ttl_hours=24)
    return jsonify(symbol_data)

@dashboard.route('/api/backtest', methods=['GET']) #Changed to accept GET requests
def run_backtest_api():
    """Run backtest simulation"""
    logger.info("API call: /api/backtest")
    symbol = request.args.get('symbol', None)
    days = request.args.get('days', default=default_backtest_interval)

    logger.info(f"Backtest request: symbol={symbol}, days={days}")
    
    # Generate cache key
    cache_key = get_cache_key('backtest', symbol=symbol if symbol else 'portfolio', days=days)
    
    # Try to get from cache
    cached_data = cache_service.get(cache_key)
    if cached_data and cache_service.is_fresh(cache_key, max_age_hours=4):
        logger.info(f"Returning cached backtest data for {symbol if symbol else 'portfolio'}")
        return jsonify(cached_data)

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
                result = run_backtest(sym, days=days)

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

        # Cache the results before returning
        cache_service.set_with_ttl(cache_key, results, ttl_hours=4)
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

@dashboard.route('/api/rank')
def get_rank():
    """Display performance ranking of all assets"""
    logger.info("API call: /api/rank")
    days = request.args.get('days', 7, type=int)
    
    cache_key = get_cache_key('rank', days=days)
    cached_data = cache_service.get(cache_key)
    if cached_data and cache_service.is_fresh(cache_key, max_age_hours=1):
        logger.info("Returning cached ranking data")
        return jsonify(cached_data)
        
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

        rank_data = {
            "success": True,
            "performance": performance_data,
            "days": days
        }
        cache_service.set_with_ttl(cache_key, rank_data)
        return jsonify(rank_data)
    except Exception as e:
        logger.error(f"Error getting performance ranking: {str(e)}", exc_info=True)
        return jsonify({"error": f"Error getting performance ranking: {str(e)}"}), 500

@dashboard.route('/api/capital-multiplier')
def get_capital_multiplier():
    """Calculate and return the capital multiplier history"""
    logger.info("API call: /api/capital-multiplier")
    
    days = request.args.get('days', type=int)
    if days is None or days <= 0:
        days = int(lookback_days_param)
        
    cache_key = get_cache_key('capital_multiplier', days=days)
    cached_data = cache_service.get(cache_key)
    if cached_data and cache_service.is_fresh(cache_key, max_age_hours=1):
        logger.info("Returning cached capital multiplier data")
        return jsonify(cached_data)

    try:
        from config import calculate_capital_multiplier
        import yfinance as yf
        import numpy as np
        import pandas as pd
        from datetime import datetime, timedelta
        import pytz

        # Get the days parameter, default to lookback_days_param if not provided
        days = request.args.get('days', type=int)
        if days is None or days <= 0:
            days = int(lookback_days_param)

        logger.info(f"Calculating capital multiplier history for the last {days} days")

        # Get end and start dates - use more days than requested to have enough data for calculations
        end_date = datetime.now(pytz.UTC)
        start_date = end_date - timedelta(days=days * 2)  # Double the days to have enough data for calculations

        # Collect daily performance data for all assets
        symbol_data = {}

        for symbol, config in TRADING_SYMBOLS.items():
            try:
                # Get the yfinance symbol
                yf_symbol = config['yfinance']
                if '/' in yf_symbol:
                    yf_symbol = yf_symbol.replace('/', '-')

                # Fetch historical data
                ticker = yf.Ticker(yf_symbol)
                data = ticker.history(start=start_date, end=end_date, interval='1d')

                if len(data) >= 10:  # Need enough data for meaningful calculation
                    # Calculate daily returns
                    data['return'] = data['Close'].pct_change() * 100  # as percentage
                    symbol_data[symbol] = data
            except Exception as e:
                logger.error(f"Error processing {symbol} for capital multiplier: {str(e)}")
                continue

        if not symbol_data:
            return jsonify({"error": "No data available for capital multiplier calculation"}), 500

        # Get the common date range for all symbols
        # First, convert all DatetimeIndex to date objects for consistent comparison
        all_dates = {}
        for symbol, data in symbol_data.items():
            all_dates[symbol] = [d.date() if hasattr(d, 'date') else d for d in data.index]

        # Find common dates across all symbols
        common_dates = set(all_dates[list(all_dates.keys())[0]])
        for symbol in all_dates:
            common_dates = common_dates.intersection(set(all_dates[symbol]))

        # Sort the common dates
        common_dates = sorted(list(common_dates))

        # We need at least some dates for calculation
        if len(common_dates) < 10:
            return jsonify({"error": f"Not enough common data points: {len(common_dates)}"}), 500

        # Calculate capital multiplier for each date
        multiplier_history = []
        std_history = []
        dates = []

        # We need a few days of data before we can start calculating
        window_size = 7  # For moving average calculation
        min_data_points = 10  # Minimum data points needed

        # Start from a point where we have enough historical data
        for i in range(min_data_points, len(common_dates)):
            current_date = common_dates[i]

            # Collect daily performances for this date range
            daily_performances = []

            for symbol, data in symbol_data.items():
                # Convert index to date objects for comparison
                date_indices = [d.date() if hasattr(d, 'date') else d for d in data.index]

                # Find data up to current date
                valid_indices = [j for j, date in enumerate(date_indices) if date <= current_date]
                if len(valid_indices) >= min_data_points:
                    # Get returns and add to collection
                    returns = data['return'].iloc[valid_indices].dropna().values
                    if len(returns) >= min_data_points:
                        daily_performances.append(returns[-min_data_points:])

            if not daily_performances or len(daily_performances) < 3:
                continue

            # Calculate average daily performance across all assets
            min_length = min(len(perfs) for perfs in daily_performances)
            if min_length < min_data_points:
                continue

            aligned_performances = [perfs[-min_length:] for perfs in daily_performances]
            daily_avg_performance = np.mean(aligned_performances, axis=0)

            # Calculate moving average
            window = min(window_size, len(daily_avg_performance)//2)
            if window < 3:
                continue

            ma = np.convolve(daily_avg_performance, np.ones(window)/window, mode='valid')

            if len(ma) < 2:
                continue

            # Get current performance (average of last 3 days) and MA
            current_perf = np.mean(daily_avg_performance[-3:])
            current_ma = ma[-1]

            # Calculate differences between performance and MA
            diffs = daily_avg_performance[-len(ma):] - ma

            # Calculate standard deviation of these differences
            std_diff = np.std(diffs)
            if std_diff == 0:
                std_diff = 0.1  # Avoid division by zero

            # Calculate current difference
            current_diff = current_perf - current_ma

            # Normalize the difference with bounds at -2std and 2std
            normalized_diff = max(min(current_diff / (2 * std_diff), 2), -2)

            # Apply sigmoid function to get a value between 0 and 1
            sigmoid = 1 / (1 + np.exp(-normalized_diff))

            # Scale to range [0.5, 3.0]
            multiplier = 0.5 + 2.5 * sigmoid

            # Add to history
            if isinstance(current_date, datetime):
                date_str = current_date.strftime('%Y-%m-%d')
            else:
                date_str = current_date.strftime('%Y-%m-%d')

            dates.append(date_str)
            multiplier_history.append(round(multiplier, 2))
            std_history.append(round(std_diff, 2))

        # Limit to the requested number of days
        if days and len(dates) > days:
            dates = dates[-days:]
            multiplier_history = multiplier_history[-days:]
            std_history = std_history[-days:]

        # Calculate the current capital multiplier using the same parameters as trading implementation
        current_multiplier = calculate_capital_multiplier(lookback_days_param/2)

        multiplier_data = {
            "success": True,
            "dates": dates,
            "multiplier": multiplier_history,
            "std": std_history,
            "current_multiplier": round(current_multiplier, 2),
            "lookback_days": lookback_days_param/2  # Include the actual lookback days used in trading
        }
        cache_service.set_with_ttl(cache_key, multiplier_data, ttl_hours=4)
        return jsonify(multiplier_data)
    except Exception as e:
        logger.error(f"Error calculating capital multiplier history: {str(e)}", exc_info=True)
        return jsonify({"error": f"Error calculating capital multiplier history: {str(e)}"}), 500

@dashboard.route('/api/price-data')
def get_price_data():
    """Get price data for a specific symbol using backtest output"""
    logger.info("API call: /api/price-data")
    symbol = request.args.get('symbol', None)
    days = request.args.get('days', int(lookback_days_param), type=int)
    
    # Generate cache key with days parameter
    cache_key = get_cache_key('price_data', symbol=symbol, days=days)
    
    # Try to get from cache first
    cached_data = cache_service.get(cache_key)
    if cached_data and cache_service.is_fresh(cache_key, max_age_hours=1):  # Cache for 4 hours
        logger.info(f"Returning cached price data for {symbol}")
        return jsonify(cached_data)
        
    logger.info(f"Fetching fresh price data for {symbol} over {days} days")

    if not symbol:
        return jsonify({"error": "Symbol parameter is required"}), 400

    if symbol not in symbols:
        return jsonify({"error": f"Invalid symbol: {symbol}"}), 400

    try:
        # Run backtest for the symbol
        logger.info(f"Running backtest for {symbol}")
        backtest_result = run_backtest(symbol, days=days)

        # Extract data from backtest result
        data = backtest_result['data']
        signals = backtest_result['signals']
        daily_data = backtest_result['daily_data']
        weekly_data = backtest_result['weekly_data']
        portfolio_values = backtest_result['portfolio_value']
        shares = backtest_result['shares']

        # Calculate allocation percentages
        allocation_percentages = []
        for i in range(len(shares)):
            if i < len(data.index):
                position_value = shares[i] * data['close'].iloc[i]
                if portfolio_values[i] > 0:
                    allocation_pct = (position_value / portfolio_values[i]) * 100
                else:
                    allocation_pct = 0
                allocation_percentages.append(allocation_pct)

        # Format dates
        dates = [idx.strftime('%Y-%m-%d %H:%M') for idx in data.index]

        # Format data for Chart.js
        price_data = {
            'labels': dates,
            'symbol': symbol,
            'name': TRADING_SYMBOLS[symbol]['name'],
            'days': days,

            # OHLC data for candlestick chart
            'ohlc': [
                {
                    'x': idx.strftime('%Y-%m-%d %H:%M'),
                    'o': float(row['open']),
                    'h': float(row['high']),
                    'l': float(row['low']),
                    'c': float(row['close']),
                    'v': float(row['volume'])
                } for idx, row in data.iterrows()
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
            'weekly_down_lim_2std': weekly_data['Down_Lim_2STD'].tolist(),

            # Add backtest portfolio value data
            'portfolio_values': portfolio_values,
            'portfolio_dates': dates,
            'allocation_percentages': allocation_percentages,
            'shares_owned': shares,

            # Add buy/sell signals data
            'signals': signals['signal'].tolist(),
            'buy_signals': [
                {
                    'time': idx.strftime('%Y-%m-%d %H:%M'),
                    'price': float(data.loc[idx, 'close'])
                } for idx in signals.index if signals.loc[idx, 'signal'] == 1
            ],
            'sell_signals': [
                {
                    'time': idx.strftime('%Y-%m-%d %H:%M'),
                    'price': float(data.loc[idx, 'close'])
                } for idx in signals.index if signals.loc[idx, 'signal'] == -1
            ]
        }

        # Store in cache with 4 hour TTL before returning
        cache_service.set_with_ttl(cache_key, price_data, ttl_hours=4)
        logger.info(f"Cached price data for {symbol} with TTL of 4 hours")
        return jsonify(price_data)

    except Exception as e:
        logger.error(f"Error getting price data for {symbol}: {str(e)}")
        return jsonify({"error": f"Error getting price data: {str(e)}"}), 500

@dashboard.route('/api/portfolio')
def get_portfolio_data():
    """Get portfolio backtest data for the dashboard"""
    logger.info("API call: /api/portfolio")
    
    try:
        # Get days parameter and generate cache key
        days = request.args.get('days', default=30, type=int)
        cache_key = get_cache_key('portfolio', days=days)
        
        # Try to get from cache
        cached_data = cache_service.get(cache_key)
        if cached_data and cache_service.is_fresh(cache_key, max_age_hours=1):
            logger.info("Returning cached portfolio data")
            return jsonify(cached_data)
        try:
            # Get days parameter, default to 30 days
            days = request.args.get('days', default=30, type=int)

            # Validate days
            if days <= 0 or days > default_backtest_interval:
                return jsonify({"error": f"Days must be between 1 and {default_backtest_interval}"}), 400

            # Ensure days is an integer
            days = int(days)
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid 'days' parameter.  Must be an integer."}), 400

        # Run portfolio backtest
        result = run_portfolio_backtest(symbols, days)

        # Get the portfolio data
        data = result['data']

        # Convert index to string format for JSON serialization
        data_dict = data.reset_index().to_dict(orient='records')

        # Calculate benchmark (equal-weight portfolio) - same as in create_portfolio_backtest_plot
        price_columns = [col for col in data.columns if col.endswith('_price')]
        initial_prices = data[price_columns].iloc[0]

        # Calculate returns for each asset
        asset_returns = data[price_columns].div(initial_prices) - 1

        # Equal-weight benchmark return
        benchmark_return = asset_returns.mean(axis=1)
        initial_capital = result['metrics']['initial_capital']
        benchmark_value = (1 + benchmark_return) * initial_capital

        # Add benchmark to the response
        benchmark_data = []
        for i, (idx, val) in enumerate(zip(data.index, benchmark_value)):
            benchmark_data.append({
                'timestamp': idx.strftime('%Y-%m-%dT%H:%M:%S'),
                'value': float(val)
            })

        # Calculate allocations for each symbol
        symbol_values = [col for col in data.columns if col.endswith('_value') 
                        and not col.startswith('total')]
        symbols_list = [col.replace('_value', '') for col in symbol_values]

        # Include both position values and cash in total
        total_portfolio = data[symbol_values].sum(axis=1) + data['total_cash']
        allocations = {}

        # Add cash allocation
        cash_allocation = (data['total_cash'] / total_portfolio * 100).fillna(0)
        allocations['Cash'] = cash_allocation.tolist()

        # Add symbol allocations
        for symbol in symbols_list:
            allocation = (data[f'{symbol}_value'] / total_portfolio * 100).fillna(0)
            allocations[symbol] = allocation.tolist()

        return jsonify({
            'portfolio_data': data_dict,
            'benchmark_data': benchmark_data,
            'allocations': allocations,
            'timestamps': [idx.strftime('%Y-%m-%dT%H:%M:%S') for idx in data.index],
            'metrics': result['metrics']
        })
        
        # Cache results before returning
        cache_service.set_with_ttl(cache_key, portfolio_data, ttl_hours=1)
        return jsonify(portfolio_data)
    except Exception as e:
        logger.error(f"Error generating portfolio data: {str(e)}")
        return jsonify({"error": f"Error generating portfolio data: {str(e)}"}), 500

@dashboard.route('/api/download-symbol-data', methods=['GET'])
def download_symbol_data():
    """
    API endpoint to download all data for a specific symbol as CSV
    """
    try:
        # Get parameters from request
        symbol = request.args.get('symbol', 'BTCUSD')
        days = int(request.args.get('days', 30))

        logging.info(f"Preparing CSV download for {symbol} with {days} days of data")

        # Get price data directly from the API function
        price_data_response = get_price_data()

        # Check if the response is a tuple (error response)
        if isinstance(price_data_response, tuple):
            return price_data_response

        # Get the JSON data from the response
        price_data = price_data_response.get_json()

        if not price_data:
            return jsonify({"error": f"No data available for {symbol}"}), 404

        # Create a DataFrame with all the data
        df = pd.DataFrame()

        # Add timestamp and OHLCV data
        timestamps = [item['x'] for item in price_data['ohlc']]
        df['timestamp'] = timestamps
        df['open'] = [item['o'] for item in price_data['ohlc']]
        df['high'] = [item['h'] for item in price_data['ohlc']]
        df['low'] = [item['l'] for item in price_data['ohlc']]
        df['close'] = [item['c'] for item in price_data['ohlc']]
        df['volume'] = [item['v'] for item in price_data['ohlc']]

        # Add indicator data
        if 'daily_composite' in price_data and len(price_data['daily_composite']) == len(timestamps):
            df['daily_composite'] = price_data['daily_composite']
            df['daily_upper_limit'] = price_data['daily_up_lim']
            df['daily_lower_limit'] = price_data['daily_down_lim']
            # Add 2STD limits if available
            if 'daily_up_lim_2std' in price_data and len(price_data['daily_up_lim_2std']) == len(timestamps):
                df['daily_upper_limit_2std'] = price_data['daily_up_lim_2std']
                df['daily_lower_limit_2std'] = price_data['daily_down_lim_2std']

        if 'weekly_composite' in price_data and len(price_data['weekly_composite']) == len(timestamps):
            df['weekly_composite'] = price_data['weekly_composite']
            df['weekly_upper_limit'] = price_data['weekly_up_lim']
            df['weekly_lower_limit'] = price_data['weekly_down_lim']
            # Add 2STD limits if available
            if 'weekly_up_lim_2std' in price_data and len(price_data['weekly_up_lim_2std']) == len(timestamps):
                df['weekly_upper_limit_2std'] = price_data['weekly_up_lim_2std']
                df['weekly_lower_limit_2std'] = price_data['weekly_down_lim_2std']

        # Add portfolio value if available
        if 'portfolio_values' in price_data and len(price_data['portfolio_values']) == len(timestamps):
            df['portfolio_value'] = price_data['portfolio_values']

        # Add shares owned if available
        if 'shares_owned' in price_data and len(price_data['shares_owned']) == len(timestamps):
            df['shares_owned'] = price_data['shares_owned']

        # Add allocation percentages if available
        if 'allocation_percentages' in price_data and len(price_data['allocation_percentages']) == len(timestamps):
            df['allocation_percentage'] = price_data['allocation_percentages']

        # Add signals data
        df['signal'] = 0  # Default: no signal
        df['signal_price'] = None  # Default: no signal price

        # Process buy signals
        if 'buy_signals' in price_data and price_data['buy_signals']:
            for signal in price_data['buy_signals']:
                signal_time = signal['time']
                signal_idx = df[df['timestamp'] == signal_time].index
                if len(signal_idx) > 0:
                    df.loc[signal_idx, 'signal'] = 1  # Buy signal
                    df.loc[signal_idx, 'signal_price'] = signal['price']

        # Process sell signals
        if 'sell_signals' in price_data and price_data['sell_signals']:
            for signal in price_data['sell_signals']:
                signal_time = signal['time']
                signal_idx = df[df['timestamp'] == signal_time].index
                if len(signal_idx) > 0:
                    df.loc[signal_idx, 'signal'] = -1  # Sell signal
                    df.loc[signal_idx, 'signal_price'] = signal['price']

        # Generate CSV
        csv_data = df.to_csv(index=False)

        # Create response with CSV data
        response = make_response(csv_data)
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = f'attachment; filename={symbol}_data_{days}days.csv'

        logging.info(f"CSV download prepared for {symbol} with {len(df)} rows")

        return response

    except Exception as e:
        logging.error(f"Error generating CSV download: {str(e)}")
        return jsonify({"error": str(e)}), 500

def register_blueprints(app):
    """Register all blueprints with the Flask app"""
    # Register the dashboard blueprint
    app.register_blueprint(dashboard, url_prefix='/dashboard')

    # Register the account_api blueprint with the same url_prefix
    app.register_blueprint(account_api, url_prefix='/dashboard')

if __name__ == '__main__':
    from flask import Flask
    app = Flask(__name__)
    register_blueprints(app)
    app.run(debug=True)