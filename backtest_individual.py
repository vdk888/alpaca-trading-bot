import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import pytz
from indicators import generate_signals, get_default_params
from config import TRADING_SYMBOLS, TRADING_COSTS, DEFAULT_RISK_PERCENT, DEFAULT_INTERVAL, DEFAULT_INTERVAL_WEEKLY, default_interval_yahoo, default_backtest_interval
import matplotlib.pyplot as plt
import io
import matplotlib.dates as mdates
from matplotlib.dates import HourLocator, num2date
import json
from itertools import product

from datetime import datetime, timedelta
import json


def is_market_hours(timestamp, market_hours):
    """Check if given timestamp is within market hours"""
    if timestamp.tz is None:
        timestamp = timestamp.tz_localize('UTC')

    # Convert to market timezone
    market_tz = pytz.timezone(market_hours['timezone'])
    local_time = timestamp.astimezone(market_tz)

    # Parse market hours
    market_start = pd.Timestamp(
        f"{local_time.date()} {market_hours['start']}").tz_localize(market_tz)
    market_end = pd.Timestamp(
        f"{local_time.date()} {market_hours['end']}").tz_localize(market_tz)

    return market_start <= local_time <= market_end


param_grid = {
    'percent_increase_buy': [0.02],
    'percent_decrease_sell': [0.02],
    'sell_down_lim': [2.0],
    'sell_rolling_std': [20],
    'buy_up_lim': [-2.0],
    'buy_rolling_std': [20],
    'macd_fast': [12],
    'macd_slow': [26],
    'macd_signal': [9],
    'rsi_period': [14],
    'stochastic_k_period': [14],
    'stochastic_d_period': [3],
    'fractal_window': [50, 100, 150],
    'fractal_lags': [[5, 10, 20], [10, 20, 40], [15, 30, 60]],
    'reactivity': [0.8, 0.9, 1.0, 1.1, 1.2],
    'weights': [
        {
            'weekly_macd_weight': 0.25,
            'weekly_rsi_weight': 0.25,
            'weekly_stoch_weight': 0.25,
            'weekly_complexity_weight': 0.25,
            'macd_weight': 0.4,
            'rsi_weight': 0.3,
            'stoch_weight': 0.2,
            'complexity_weight': 0.1
        },
        {
            'weekly_macd_weight': 0.2,
            'weekly_rsi_weight': 0.4,
            'weekly_stoch_weight': 0.2,
            'weekly_complexity_weight': 0.2,
            'macd_weight': 0.3,
            'rsi_weight': 0.4,
            'stoch_weight': 0.2,
            'complexity_weight': 0.1
        },
        {
            'weekly_macd_weight': 0.3,
            'weekly_rsi_weight': 0.2,
            'weekly_stoch_weight': 0.3,
            'weekly_complexity_weight': 0.2,
            'macd_weight': 0.2,
            'rsi_weight': 0.3,
            'stoch_weight': 0.4,
            'complexity_weight': 0.1
        },
        {
            'weekly_macd_weight': 0.4,
            'weekly_rsi_weight': 0.3,
            'weekly_stoch_weight': 0.2,
            'weekly_complexity_weight': 0.1,
            'macd_weight': 0.1,
            'rsi_weight': 0.4,
            'stoch_weight': 0.3,
            'complexity_weight': 0.2
        },
    ]
}


def find_best_params(symbol: str,
                     param_grid: dict,
                     days: int = 5,
                     output_file: str = "best_params.json") -> dict:
    """Find the best parameter set by running a backtest for each combination."""
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]

    # Load existing data to check the last update date
    try:
        with open(output_file, "r") as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        existing_data = {}

    # Check if the current symbol exists in the JSON data
    last_update_date = None
    if symbol in existing_data:
        last_update_date_str = existing_data[symbol].get('date')
        if last_update_date_str:
            last_update_date = datetime.strptime(last_update_date_str,
                                                 "%Y-%m-%d")

    # Determine if we need to run simulations
    if last_update_date:
        if datetime.now() - last_update_date < timedelta(weeks=1):
            print(
                f"Using existing best parameters for {symbol} (last updated on {last_update_date_str})."
            )
            return existing_data[symbol][
                'best_params']  # Return existing best params

    # Proceed with simulations if no existing data or it's older than a week
    param_combinations = [
        dict(zip(param_names, values)) for values in product(*param_values)
    ]

    best_params = None
    best_performance = float('-inf')
    best_metrics = {}
    performances = []  # List to store all performance metrics

    for params in param_combinations:
        # Update default parameters with the current combination
        default_params = get_default_params()
        default_params.update(params)

        # Access weights directly from default_params
        weight_combination = default_params['weights']
        default_params.update(weight_combination)

        # Run a single backtest with the current parameter set
        result = run_backtest(symbol,
                              days=days,
                              params=default_params,
                              is_simulating=True)
        performance = result['stats'][
            'sharpe_ratio']  # Use total return as the performance metric
        win_rate = result['stats']['win_rate']  # Example metric
        max_drawdown = result['stats']['max_drawdown']  # Example metric
        total_return = result['stats']['total_return']  # Example metric
        sharpe_ratio = result['stats']['sharpe_ratio']  # Example metric

        # Store performance for later analysis
        performances.append(performance)

        print(
            f"Params: {params}, Target (Sharpe Ratio): {performance:.2f}%, Win Rate: {win_rate:.2f}%, Max Drawdown: {max_drawdown:.2f}%, Total Return: {total_return:.2f}%, Sharpe Ratio: {sharpe_ratio:.2f}"
        )

        # Update best parameters if current is better
        if performance > best_performance:
            best_performance = performance
            best_params = params
            best_metrics = {
                'performance': performance,
                'win_rate': win_rate,
                'max_drawdown': max_drawdown,
            }

    # Calculate max, min, and average performance
    max_performance = max(performances)
    min_performance = min(performances)
    avg_performance = sum(performances) / len(performances)

    # Save best parameters and metrics to JSON
    if output_file:
        existing_data[symbol] = {
            'best_params': best_params,
            'metrics': best_metrics,
            'performance_summary': {
                'max_performance': max_performance,
                'min_performance': min_performance,
                'avg_performance': avg_performance,
            },
            'date': datetime.now().strftime("%Y-%m-%d")  # Add current date
        }

        # Write updated data back to JSON
        with open(output_file, "w") as f:
            json.dump(existing_data, f, indent=4)

    print(f"Best params and metrics for {symbol} saved to {output_file}")
    return best_params


def run_backtest(symbol: str,
                 days: int = 5,
                 params: dict = None,
                 is_simulating: bool = False) -> dict:
    """Run a single backtest simulation for a given symbol and parameter set."""

    # Fetch price data for all symbols
    end_date = datetime.now(pytz.UTC)
    start_date = end_date - timedelta(days=default_backtest_interval)  # 30 days of data

    print(f"\nFetching data for {symbol} from {start_date} to {end_date}")

    prices_dataset = {}
    for sym, config in TRADING_SYMBOLS.items():
        yf_symbol = config['yfinance']
        if '/' in yf_symbol:
            yf_symbol = yf_symbol.replace('/', '-')

        print(f"Fetching {yf_symbol} data...")
        ticker = yf.Ticker(yf_symbol)
        data = ticker.history(start=start_date,
                              end=end_date,
                              interval=default_interval_yahoo,
                              actions=False)

        print(f"Retrieved {len(data)} data points for {sym}")

        if len(data) == 0:
            print(
                f"Warning: No data available for {sym} in the specified date range"
            )
            print(f"Start date: {start_date}, End date: {end_date}")
            print(f"Symbol config: {config}")
            continue

        # Localize timezone if needed
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC')

        # Filter for market hours
        market_hours_data = data[data.index.map(
            lambda x: is_market_hours(x, config['market_hours']))]
        print(
            f"After market hours filtering: {len(market_hours_data)} data points for {sym}"
        )

        if len(market_hours_data) == 0:
            print(
                f"Warning: No data available for {sym} after market hours filtering"
            )
            print(f"Market hours config: {config['market_hours']}")
            continue

        prices_dataset[sym] = market_hours_data

    # Load the best parameters from JSON based on the symbol
    try:
        with open("best_params.json", "r") as f:
            best_params_data = json.load(f)
    except FileNotFoundError:
        print("Best parameters file not found. Using default parameters.")
        best_params_data = {}

    if is_simulating == False:
        if symbol in best_params_data:
            # Use the best parameters for this symbol
            params = best_params_data[symbol]['best_params']
            print(f"Using best parameters for {symbol}: {params}")
        else:
            print(
                f"No best parameters found for {symbol}. Using default parameters."
            )
            params = get_default_params()  # Fallback to default parameters

    # Get symbol configuration
    symbol_config = TRADING_SYMBOLS[symbol]
    yf_symbol = symbol_config['yfinance']

    # Handle crypto symbols with forward slashes
    if '/' in yf_symbol:
        yf_symbol = yf_symbol.replace('/', '-')

    # Calculate date range
    end_date = datetime.now(pytz.UTC)
    # Round down end_date to the nearest minute to avoid potential future timestamps
    end_date = end_date.replace(second=0, microsecond=0)
    start_date = end_date - timedelta(days=days + 2)  # Add buffer days

    # Fetch historical data
    ticker = yf.Ticker(yf_symbol)
    data = ticker.history(start=start_date,
                          end=end_date,
                          interval=symbol_config.get('interval', DEFAULT_INTERVAL),
                          actions=False)

    if len(data) == 0:
        raise ValueError(
            f"No data available for {symbol} in the specified date range")

    # Localize timezone if needed
    if data.index.tz is None:
        data.index = data.index.tz_localize('UTC')

    # Filter for market hours
    data = data[data.index.map(
        lambda x: is_market_hours(x, symbol_config['market_hours']))]
    data.columns = data.columns.str.lower()

    # Generate signals using the provided parameters
    signals, daily_data, weekly_data = generate_signals(data, params)

    # Initialize portfolio tracking
    initial_capital = 100000  # $100k initial capital
    position = 0  # Current position in shares
    cash = initial_capital
    portfolio_value = [initial_capital]  # Start with initial capital
    shares = [0]  # Start with no shares
    trades = []  # Track individual trades
    total_position_value = 0  # Track total position value for position sizing

    # Get trading costs based on market type
    market_type = symbol_config['market']
    costs = TRADING_COSTS.get(market_type, TRADING_COSTS['DEFAULT'])
    trading_fee = costs['trading_fee']
    spread = costs['spread']

    # Simulate trading
    for i in range(len(data)):
        current_price = data['close'].iloc[i]
        current_time = data.index[i]

        # Update total position value
        total_position_value = position * current_price

        if i > 0:  # Skip first bar for signal processing
            signal = signals['signal'].iloc[i]

            # Process signals
            if signal == 1:  # Buy signal
                # Calculate maximum position value (100% of initial capital)
                max_position_value = initial_capital

                # If total position is less than max, allow adding 20% more
                if total_position_value < max_position_value:
                    # Calculate position size as 20% of initial capital
                    capital_to_use = initial_capital * 0.20
                    shares_to_buy = capital_to_use / current_price

                    # Round based on market type
                    if symbol_config['market'] == 'CRYPTO':
                        shares_to_buy = round(
                            shares_to_buy,
                            8)  # Round to 8 decimal places for crypto
                    else:
                        shares_to_buy = int(
                            shares_to_buy
                        )  # Round down to whole shares for stocks

                    # Ensure minimum position size
                    min_qty = 1 if symbol_config[
                        'market'] != 'CRYPTO' else 0.0001
                    if shares_to_buy < min_qty:
                        shares_to_buy = min_qty

                    # Check if adding this position would exceed max position value
                    new_total_value = total_position_value + (shares_to_buy *
                                                              current_price)
                    if new_total_value > max_position_value:
                        # Adjust shares to not exceed max position
                        shares_to_buy = (max_position_value -
                                         total_position_value) / current_price
                        if symbol_config['market'] == 'CRYPTO':
                            shares_to_buy = round(shares_to_buy, 8)
                        else:
                            shares_to_buy = int(shares_to_buy)

                    cost = shares_to_buy * current_price
                    # Apply trading costs to buy (full spread + fee)
                    total_cost = cost * (1 + spread + trading_fee)

                    if total_cost <= cash and shares_to_buy > 0:  # Check if we have enough cash and shares to buy
                        position += shares_to_buy  # Add to existing position
                        cash -= total_cost
                        trades.append({
                            'time': current_time,
                            'type': 'buy',
                            'price': current_price,
                            'shares': shares_to_buy,
                            'value': cost,
                            'total_cost': total_cost,
                            'trading_costs': total_cost - cost,
                            'total_position': position
                        })
                        print(f"\n{'='*80}")
                        print(
                            f"BUY SIGNAL detected for {symbol} at {current_time}"
                        )
                        print(
                            f"Current position: {position:.8f} shares at ${current_price:.2f}"
                        )
                        print(
                            f"Buying {shares_to_buy:.8f} shares at ${current_price:.2f}"
                        )
                        print(f"Remaining cash: ${cash:.2f}")

            elif signal == -1 and position > 0:  # Sell signal
                print(f"\n{'='*80}")
                print(f"SELL SIGNAL detected for {symbol} at {current_time}")
                print(
                    f"Current position: {position:.8f} shares at ${current_price:.2f}"
                )

                # Calculate performance ranking
                perf_rankings = calculate_performance_ranking(
                    prices_dataset, current_time)

                if perf_rankings is not None and symbol in perf_rankings.index:
                    # Get percentile rank (0 to 1)
                    rank = perf_rankings.loc[symbol, 'rank']
                    performance = perf_rankings.loc[symbol, 'performance']

                    # Calculate sell percentage using new dynamic formula
                    def calculate_sell_percentage(rank: int,
                                                  total_assets: int) -> float:
                        """
                        Calculate sell percentage based on rank and total number of assets.
                        rank: 1 is best performer, total_assets is worst performer
                        Returns: float between 0.1 and 1.0 representing sell percentage
                        """
                        # Calculate cutoff points
                        bottom_third = int(total_assets * (1 / 3))
                        top_two_thirds = total_assets - bottom_third

                        # If in bottom third, sell 100%
                        if rank > top_two_thirds:
                            return 1.0

                        # For top two-thirds, use a wave function
                        x = (rank - 1) / (top_two_thirds - 1)
                        wave = 0.1 * np.sin(
                            2 * np.pi * x)  # Oscillation between -0.1 and 0.1
                        linear = 0.3 + 0.7 * x  # Linear increase from 0.3 to 1.0
                        return max(0.1, min(1.0, linear +
                                            wave))  # Clamp between 0.1 and 1.0

                    # Get total number of assets
                    total_assets = len(perf_rankings)

                    # Calculate sell percentage using new dynamic formula
                    rank = 1 + sum(
                        1 for other_metric in perf_rankings['rank'].values
                        if other_metric > rank)
                    sell_percentage = calculate_sell_percentage(
                        rank, total_assets)

                    # Calculate shares to sell
                    shares_to_sell = position * sell_percentage

                    # Round based on market type
                    if symbol_config['market'] == 'CRYPTO':
                        shares_to_sell = round(shares_to_sell, 8)
                    else:
                        shares_to_sell = int(shares_to_sell)

                    print(f"\nSell Decision:")
                    print(f"Performance: {performance:.2f}%")
                    print(f"Rank: {rank:.2f}")
                    print(f"Sell Percentage: {sell_percentage*100:.1f}%")
                    print(
                        f"Shares to sell: {shares_to_sell:.8f} out of {position:.8f}"
                    )

                    if shares_to_sell > 0:
                        # Calculate sale value with trading costs (half spread + fee since we're already at bid)
                        gross_sale_value = shares_to_sell * current_price
                        trading_costs = gross_sale_value * (trading_fee +
                                                            spread / 2)
                        net_sale_value = gross_sale_value - trading_costs

                        cash += net_sale_value
                        position -= shares_to_sell
                        trades.append({
                            'time': current_time,
                            'type': 'sell',
                            'price': current_price,
                            'shares': shares_to_sell,
                            'gross_value': gross_sale_value,
                            'value': net_sale_value,
                            'trading_costs': trading_costs,
                            'total_position': position,
                            'performance_rank': rank,
                            'sell_percentage': sell_percentage * 100
                        })
                        print(
                            f"Trade executed: Sold {shares_to_sell:.8f} shares at ${current_price:.2f}"
                        )
                        print(
                            f"Gross value: ${gross_sale_value:.2f}, Trading costs: ${trading_costs:.2f}"
                        )
                        print(f"Net value: ${net_sale_value:.2f}")
                        print(f"Remaining position: {position:.8f} shares")
                else:
                    # Fallback to selling entire position if we can't calculate ranking
                    gross_sale_value = position * current_price
                    trading_costs = gross_sale_value * (trading_fee +
                                                        spread / 2)
                    net_sale_value = gross_sale_value - trading_costs
                    cash += net_sale_value
                    trades.append({
                        'time': current_time,
                        'type': 'sell',
                        'price': current_price,
                        'shares': position,
                        'gross_value': gross_sale_value,
                        'value': net_sale_value,
                        'trading_costs': trading_costs,
                        'total_position': 0,
                        'performance_rank': None,
                        'sell_percentage': 100
                    })
                    print(
                        "\nWARNING: Could not calculate rankings, selling entire position"
                    )
                    print(
                        f"Sold all {position:.8f} shares at ${current_price:.2f}"
                    )
                    position = 0

        # Update portfolio value and shares owned after processing any trades
        current_value = cash + (position * current_price)
        portfolio_value.append(current_value)
        shares.append(position)

    # Calculate final portfolio value
    final_value = cash + (position * data['close'].iloc[-1])
    total_return = ((final_value - initial_capital) / initial_capital) * 100

    # Calculate performance metrics
    if trades:
        trades_df = pd.DataFrame(trades)
        if len(trades_df) > 0:
            # Separate buy and sell trades
            buy_trades = trades_df[trades_df['type'] == 'buy']
            sell_trades = trades_df[trades_df['type'] == 'sell']

            # Calculate profits only if we have matching buy/sell pairs
            if len(buy_trades) > 0 and len(sell_trades) > 0:
                # Take the minimum length to ensure we only compare complete trades
                min_trades = min(len(buy_trades), len(sell_trades))
                profits = sell_trades[
                    'value'].iloc[:min_trades].values - buy_trades[
                        'total_cost'].iloc[:min_trades].values
                win_rate = (len(profits[profits > 0]) /
                            len(profits)) * 100 if len(profits) > 0 else 0
            else:
                win_rate = 0

        # Calculate max drawdown
        portfolio_series = pd.Series(portfolio_value)
        rolling_max = portfolio_series.expanding().max()
        drawdowns = (portfolio_series - rolling_max) / rolling_max * 100
        max_drawdown = abs(drawdowns.min())

        # Calculate Sharpe Ratio
        # Convert portfolio values to returns
        returns = pd.Series(portfolio_value).pct_change().dropna()

        # Annualize the risk-free rate (2%)
        annual_rf_rate = 0.02
        # Convert to 5-minute rate (assuming 252 trading days, 24/7 trading for crypto)
        period_rf_rate = (1 + annual_rf_rate)**(1 / (252 * 24 * 12)) - 1

        # Calculate excess returns
        excess_returns = returns - period_rf_rate

        # Calculate annualized Sharpe Ratio
        # For 5-minute data, multiply by sqrt(252 * 24 * 12) to annualize
        if len(returns) > 1:  # Need at least 2 points for std calculation
            annualization_factor = np.sqrt(252 * 24 * 12)  # For 5-minute data
            sharpe_ratio = annualization_factor * (excess_returns.mean() /
                                                   excess_returns.std())
        else:
            sharpe_ratio = 0
    else:
        win_rate = 0
        max_drawdown = 0
        sharpe_ratio = 0

    return {
        'symbol': symbol,
        'data': data,
        'signals': signals,
        'daily_data': daily_data,
        'weekly_data': weekly_data,
        'portfolio_value': portfolio_value,
        'shares': shares,
        'trades': trades,
        'stats': {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'params_used': params
        }
    }


def calculate_performance_ranking(prices_dataset, current_time):
    """Calculate performance ranking of all symbols over the last 5 days."""
    performance_dict = {}
    lookback_days = 5
    lookback_time = current_time - pd.Timedelta(days=lookback_days)

    print(f"\n{'='*80}")
    print(f"Calculating rankings at {current_time}")
    print(f"Looking back to {lookback_time}")
    print(
        f"{'Symbol':<10} {'Start Price':>12} {'End Price':>12} {'Performance':>12}"
    )
    print(f"{'-'*10:<10} {'-'*12:>12} {'-'*12:>12} {'-'*12:>12}")

    for symbol, data in prices_dataset.items():
        try:
            # Get data up until current time and within lookback period
            mask = (data.index <= current_time) & (data.index >= lookback_time)
            symbol_data = data[mask]

            if len(symbol_data
                   ) >= 2:  # Need at least 2 points to calculate performance
                # Make column names lowercase if they aren't already
                symbol_data.columns = symbol_data.columns.str.lower()

                if 'close' not in symbol_data.columns:
                    print(
                        f"Warning: 'close' column not found for {symbol}. Available columns: {symbol_data.columns.tolist()}"
                    )
                    continue

                start_price = symbol_data['close'].iloc[0]
                end_price = symbol_data['close'].iloc[-1]
                performance = ((end_price - start_price) / start_price) * 100
                performance_dict[symbol] = performance

                print(
                    f"{symbol:<10} {start_price:>12.2f} {end_price:>12.2f} {performance:>12.2f}%"
                )
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue

    # Convert to DataFrame and calculate rankings
    if performance_dict:
        perf_df = pd.DataFrame.from_dict(performance_dict,
                                         orient='index',
                                         columns=['performance'])
        perf_df['rank'] = perf_df['performance'].rank(
            pct=True)  # Percentile ranking

        print("\nFinal Rankings:")
        print(f"{'Symbol':<10} {'Performance':>12} {'Rank':>8}")
        print(f"{'-'*10:<10} {'-'*12:>12} {'-'*8:>8}")
        for idx in perf_df.index:
            print(
                f"{idx:<10} {perf_df.loc[idx, 'performance']:>12.2f}% {perf_df.loc[idx, 'rank']:>8.2f}"
            )

        return perf_df
    return None


def split_into_sessions(data):
    """Split data into continuous market sessions"""
    sessions = []
    current_session = []

    for idx, row in data.iterrows():
        if not current_session or (idx - current_session[-1].name
                                   ).total_seconds() <= 300:  # 5 minutes
            current_session.append(row)
        else:
            sessions.append(pd.DataFrame(current_session))
            current_session = [row]

    if current_session:
        sessions.append(pd.DataFrame(current_session))

    return sessions


def create_backtest_plot(backtest_result: dict) -> tuple:
    """Create visualization of backtest results"""
    data = backtest_result['data']
    signals = backtest_result['signals']
    daily_data = backtest_result['daily_data']
    weekly_data = backtest_result['weekly_data']
    portfolio_value = backtest_result['portfolio_value']
    shares = backtest_result['shares']
    stats = backtest_result['stats']

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 20))
    gs = plt.GridSpec(4, 1, height_ratios=[3, 1.5, 1.5, 3], hspace=0.3)

    # Plot 1: Price and Signals
    ax1 = plt.subplot(gs[0])
    ax1_volume = ax1.twinx()

    # Split data into sessions
    sessions = split_into_sessions(data)

    # Plot each session separately
    all_timestamps = []
    session_boundaries = []
    last_timestamp = None
    shifted_data = pd.DataFrame()
    session_start_times = []

    # Plot each session
    for i, session in enumerate(sessions):
        session_df = session.copy()

        if last_timestamp is not None:
            # Add a small gap between sessions
            gap = pd.Timedelta(minutes=5)
            time_shift = (last_timestamp + gap) - session_df.index[0]
            session_df.index = session_df.index + time_shift

        # Store original and shifted start times
        session_start_times.append((session_df.index[0], session.index[0]))

        # Plot price
        ax1.plot(session_df.index,
                 session_df['close'],
                 color='blue',
                 alpha=0.7)

        # Plot volume
        volume_data = session_df['volume'].rolling(window=5).mean()
        ax1_volume.fill_between(session_df.index,
                                volume_data,
                                color='gray',
                                alpha=0.3)

        all_timestamps.extend(session_df.index)
        session_boundaries.append(session_df.index[0])
        last_timestamp = session_df.index[-1]
        shifted_data = pd.concat([shifted_data, session_df])

    # Create timestamp mapping for signals
    original_to_shifted = {}
    for orig_session, shifted_session in zip(sessions, session_boundaries):
        time_diff = shifted_session - orig_session.index[0]
        for orig_time in orig_session.index:
            original_to_shifted[orig_time] = orig_time + time_diff

    # Plot signals with correct timestamps
    buy_signals = signals[signals['signal'] == 1]
    sell_signals = signals[signals['signal'] == -1]

    for signals_df, color, marker, va, offset in [
        (buy_signals, 'green', '^', 'bottom', 10),
        (sell_signals, 'red', 'v', 'top', -10)
    ]:
        if len(signals_df) > 0:
            signals_df = signals_df.copy()
            signals_df['close'] = data.loc[signals_df.index, 'close']
            shifted_indices = [
                original_to_shifted[idx] for idx in signals_df.index
            ]
            ax1.scatter(shifted_indices,
                        signals_df['close'],
                        color=color,
                        marker=marker,
                        s=100)

            for idx, shifted_idx in zip(signals_df.index, shifted_indices):
                ax1.annotate(f'${signals_df.loc[idx, "close"]:.2f}',
                             (shifted_idx, signals_df.loc[idx, 'close']),
                             xytext=(0, offset),
                             textcoords='offset points',
                             ha='center',
                             va=va,
                             color=color)

    # Format x-axis
    def format_date(x, p):
        try:
            x_ts = pd.Timestamp(num2date(x, tz=pytz.UTC))

            # Find the closest session start time
            for shifted_time, original_time in session_start_times:
                if abs((x_ts - shifted_time).total_seconds()) < 300:
                    return original_time.strftime('%Y-%m-%d\n%H:%M')

            # For other times, find the corresponding original time
            for shifted_time, original_time in session_start_times:
                if x_ts >= shifted_time:
                    last_session_start = shifted_time
                    last_original_start = original_time
                    break
            else:
                return ''

            time_since_session_start = x_ts - last_session_start
            original_time = last_original_start + time_since_session_start
            return original_time.strftime('%H:%M')

        except Exception:
            return ''

    ax1.xaxis.set_major_locator(HourLocator(interval=1))
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(format_date))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    ax1.set_title('Price Action with Trading Signals')
    ax1.set_ylabel('Price')
    ax1_volume.set_ylabel('Volume')
    ax1.legend(['Price', 'Buy Signal', 'Sell Signal'])

    # Plot 2: Daily Composite (reduced height)
    ax2 = plt.subplot(gs[1])
    sessions_daily = split_into_sessions(daily_data)
    last_timestamp = None

    for session_data in sessions_daily:
        if last_timestamp is not None:
            gap = pd.Timedelta(minutes=5)
            session_data.index = session_data.index.shift(
                -1, freq=(session_data.index[0] - (last_timestamp + gap)))

        ax2.plot(session_data.index, session_data['Composite'], color='blue')
        ax2.plot(session_data.index,
                 session_data['Up_Lim'],
                 '--',
                 color='green',
                 alpha=0.6)
        ax2.plot(session_data.index,
                 session_data['Down_Lim'],
                 '--',
                 color='red',
                 alpha=0.6)
        ax2.fill_between(session_data.index,
                         session_data['Up_Lim'],
                         session_data['Down_Lim'],
                         color='gray',
                         alpha=0.1)
        last_timestamp = session_data.index[-1]

    ax2.set_title('Daily Composite Indicator')
    ax2.legend(['Daily Composite', 'Upper Limit', 'Lower Limit'])
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(format_date))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 3: Weekly Composite (reduced height)
    ax3 = plt.subplot(gs[2])
    sessions_weekly = split_into_sessions(weekly_data)
    last_timestamp = None

    for session_data in sessions_weekly:
        if last_timestamp is not None:
            gap = pd.Timedelta(minutes=5)
            session_data.index = session_data.index.shift(
                -1, freq=(session_data.index[0] - (last_timestamp + gap)))

        ax3.plot(session_data.index, session_data['Composite'], color='purple')
        ax3.plot(session_data.index,
                 session_data['Up_Lim'],
                 '--',
                 color='green',
                 alpha=0.6)
        ax3.plot(session_data.index,
                 session_data['Down_Lim'],
                 '--',
                 color='red',
                 alpha=0.6)
        ax3.fill_between(session_data.index,
                         session_data['Up_Lim'],
                         session_data['Down_Lim'],
                         color='gray',
                         alpha=0.1)
        last_timestamp = session_data.index[-1]

    ax3.set_title('Weekly Composite Indicator')
    ax3.legend(['Weekly Composite', 'Upper Limit', 'Lower Limit'])
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(format_date))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 4: Portfolio Performance and Position Size
    ax4 = plt.subplot(gs[3])
    ax4_shares = ax4.twinx()

    # Create a DataFrame with portfolio data
    portfolio_df = pd.DataFrame(
        {
            'value': portfolio_value[1:],  # Skip initial value
            'shares': shares[1:]  # Skip initial shares
        },
        index=data.index)

    # Split portfolio data into sessions
    sessions_portfolio = split_into_sessions(portfolio_df)
    last_timestamp = None

    for session_data in sessions_portfolio:
        if last_timestamp is not None:
            gap = pd.Timedelta(minutes=5)
            session_data.index = session_data.index.shift(
                -1, freq=(session_data.index[0] - (last_timestamp + gap)))

        ax4.plot(session_data.index, session_data['value'], color='green')
        ax4_shares.plot(session_data.index,
                        session_data['shares'],
                        color='blue',
                        alpha=0.5)
        last_timestamp = session_data.index[-1]

    ax4.set_ylabel('Portfolio Value ($)')
    ax4_shares.set_ylabel('Shares Owned')
    ax4.set_title('Portfolio Performance and Position Size')

    # Add both legends
    ax4_shares.legend(['Portfolio Value', 'Shares Owned'], loc='upper left')

    ax4.xaxis.set_major_formatter(plt.FuncFormatter(format_date))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    # Save plot to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    buf.seek(0)

    return buf, backtest_result['stats']


if __name__ == "__main__":
    # Define the parameter grid

    param_grid = {
        'percent_increase_buy': [0.02],
        'percent_decrease_sell': [0.02],
        'sell_down_lim': [2.0],
        'sell_rolling_std': [20],
        'buy_up_lim': [-2.0],
        'buy_rolling_std': [20],
        'macd_fast': [12],
        'macd_slow': [26],
        'macd_signal': [9],
        'rsi_period': [14],
        'stochastic_k_period': [14],
        'stochastic_d_period': [3],
        'fractal_window': [50, 100, 150],
        'fractal_lags': [[5, 10, 20], [10, 20, 40], [15, 30, 60]],
        'reactivity': [0.8, 0.9, 1.0, 1.1, 1.2],
        'weights': [
            {
                'weekly_macd_weight': 0.25,
                'weekly_rsi_weight': 0.25,
                'weekly_stoch_weight': 0.25,
                'weekly_complexity_weight': 0.25,
                'macd_weight': 0.4,
                'rsi_weight': 0.3,
                'stoch_weight': 0.2,
                'complexity_weight': 0.1
            },
            {
                'weekly_macd_weight': 0.2,
                'weekly_rsi_weight': 0.4,
                'weekly_stoch_weight': 0.2,
                'weekly_complexity_weight': 0.2,
                'macd_weight': 0.3,
                'rsi_weight': 0.4,
                'stoch_weight': 0.2,
                'complexity_weight': 0.1
            },
            {
                'weekly_macd_weight': 0.3,
                'weekly_rsi_weight': 0.2,
                'weekly_stoch_weight': 0.3,
                'weekly_complexity_weight': 0.2,
                'macd_weight': 0.2,
                'rsi_weight': 0.3,
                'stoch_weight': 0.4,
                'complexity_weight': 0.1
            },
            {
                'weekly_macd_weight': 0.4,
                'weekly_rsi_weight': 0.3,
                'weekly_stoch_weight': 0.2,
                'weekly_complexity_weight': 0.1,
                'macd_weight': 0.1,
                'rsi_weight': 0.4,
                'stoch_weight': 0.3,
                'complexity_weight': 0.2
            },
        ]
    }

    # Find the best parameters
    best_params = find_best_params(symbol="SPY",
                                   param_grid=param_grid,
                                   days=10)
    print(f"Optimal Parameters: {best_params}")

    # Run the final backtest with the best parameters
    final_result = run_backtest(symbol="SPY",
                                days=10,
                                params=best_params,
                                is_simulating=False)
    print(f"Final Backtest Results: {final_result}")
