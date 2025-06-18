import collections
import requests
import time
import os
from datetime import datetime, timedelta, UTC # Import UTC for timezone-aware datetime
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import csv
from art import text2art #pip install art

# Import the CoinGecko API client
try:
    from pycoingecko import CoinGeckoAPI
except ImportError:
    print("Error: 'pycoingecko' library not found.")
    print("Please install it using: pip install pycoingecko numpy scikit-learn")
    exit()

# --- Configuration ---
DEFAULT_VS_CURRENCY = "usd"
CHART_DATA_DAYS = 30 # Data from CoinGecko, 30 days gives 4-hourly data points.
PREDICTION_TIMEFRAME_HOURS = 4 # Define how many hours into the future the ML model will predict.

UPDATE_INTERVAL_SECONDS = 60

# SMA Windows
SHORT_SMA_WINDOW = 10
LONG_SMA_WINDOW = 30

# RSI Window
RSI_WINDOW = 14

# MACD Windows
FAST_EMA_WINDOW = 12
SLOW_EMA_WINDOW = 26
SIGNAL_EMA_WINDOW = 9

# Bollinger Bands Window
BB_WINDOW = 20
BB_STD_DEV_MULTIPLIER = 2

# Stochastic Oscillator Windows
STOCH_K_WINDOW = 14
STOCH_D_WINDOW = 3

# Average True Range Window
ATR_WINDOW = 14

# Ichimoku Cloud Windows
TENKAN_SEN_WINDOW = 9
KIJUN_SEN_WINDOW = 26
SENKOU_SPAN_B_WINDOW = 52
CHIKOU_SPAN_LAG = 26

# List of popular meme coin IDs to prioritize when searching trending
MEME_COIN_IDS = ["dogecoin", "shiba-inu", "pepe", "bonk", "floki"]
# List of known stablecoin IDs to handle ML predictions specially
STABLECOIN_IDS = ["tether", "usd-coin", "binance-usd", "dai"]

# --- Helper Function for Clearing Terminal ---
def clear_terminal():
    """Clears the terminal screen for cleaner live updates."""
    os.system('cls' if os.name == 'nt' else 'clear')

# --- Data Fetching from CoinGecko API ---
def get_coin_chart_data(crypto_id, vs_currency, days):
    """
    Fetches historical OHLC (Open, High, Low, Close) prices for a given cryptocurrency from CoinGecko API.
    Returns a list of (timestamp, open, high, low, close) tuples.
    Granularity based on 'days' parameter for ohlc endpoint:
    - 1 day: 30-minutely data.
    - 7, 14, 30 days: 4-hourly data.
    - 90, 180, 365, max: daily data.
    """
    cg = CoinGeckoAPI()
    try:
        ohlc_data = cg.get_coin_ohlc_by_id(id=crypto_id, vs_currency=vs_currency, days=days)
        
        if not ohlc_data:
            return []

        formatted_ohlc = []
        for item in ohlc_data:
            timestamp = datetime.fromtimestamp(item[0] / 1000)
            open_price = item[1]
            high_price = item[2]
            low_price = item[3]
            close_price = item[4]
            formatted_ohlc.append((timestamp, open_price, high_price, low_price, close_price))
        return formatted_ohlc

    except requests.exceptions.RequestException as e:
        print(f"API Request Error for {crypto_id}: {e}")
        return []
    except Exception as e:
        print(f"General Error fetching OHLC data for {crypto_id}: {e}")
        return []

def get_top_popular_coins(vs_currency, count=10):
    """Fetches the top N cryptocurrencies by market cap."""
    cg = CoinGeckoAPI()
    try:
        top_coins = cg.get_coins_markets(vs_currency=vs_currency, order='market_cap_desc', per_page=count)
        return [coin['id'] for coin in top_coins]
    except Exception as e:
        print(f"Error fetching top popular coins: {e}")
        return []

def get_trending_coins(count=5):
    """Fetches trending coins and attempts to prioritize meme coins."""
    cg = CoinGeckoAPI()
    try:
        trending_data = cg.get_search_trending()
        trending_coin_ids = [coin['item']['id'] for coin in trending_data.get('coins', [])]
        
        selected_meme_coins = []
        for meme_id in MEME_COIN_IDS:
            if meme_id in trending_coin_ids:
                selected_meme_coins.append(meme_id)
                if len(selected_meme_coins) >= count:
                    return selected_meme_coins[:count]
        
        remaining_slots = count - len(selected_meme_coins)
        for trend_id in trending_coin_ids:
            if trend_id not in selected_meme_coins:
                selected_meme_coins.append(trend_id)
            if len(selected_meme_coins) >= count:
                break
        
        return selected_meme_coins[:count]
    except Exception as e:
        print(f"Error fetching trending coins: {e}")
        return []

# --- Technical Indicator Calculations ---
def calculate_sma(prices, window):
    """
    Calculates the Simple Moving Average (SMA) for a given list of prices.
    Returns a list of SMA values, with NaN for periods where an SMA cannot be calculated.
    """
    if window <= 0:
        raise ValueError("SMA window must be greater than zero.")

    prices_np = np.array(prices, dtype=float)
    sma_values = np.full(len(prices_np), np.nan) # Use NaN for None equivalent

    for i in range(window - 1, len(prices_np)):
        sma_values[i] = np.mean(prices_np[i - (window - 1) : i + 1])
    
    return list(sma_values) # Convert back to list for consistency

def calculate_rsi(prices, window):
    """
    Calculates the Relative Strength Index (RSI) for a given list of prices.
    Returns a list of RSI values, with NaN for periods where an RSI cannot be calculated.
    """
    if window <= 0:
        raise ValueError("RSI window must be greater than zero.")
    
    prices_np = np.array(prices, dtype=float)
    rsi_values = np.full(len(prices_np), np.nan)

    if len(prices_np) < window + 1:
        return list(rsi_values)

    # Calculate price changes
    price_changes = np.diff(prices_np)
    gains = np.where(price_changes > 0, price_changes, 0)
    losses = np.where(price_changes < 0, np.abs(price_changes), 0)

    # Initial average gain and loss
    avg_gain = np.mean(gains[:window])
    avg_loss = np.mean(losses[:window])

    if avg_loss == 0:
        rs = np.inf
    else:
        rs = avg_gain / avg_loss
    
    rsi_values[window] = 100 - (100 / (1 + rs))

    # Subsequent calculations
    for i in range(window + 1, len(prices_np)):
        avg_gain = ((avg_gain * (window - 1)) + gains[i-1]) / window # gains/losses array is one shorter
        avg_loss = ((avg_loss * (window - 1)) + losses[i-1]) / window

        if avg_loss == 0:
            rs = np.inf
        else:
            rs = avg_gain / avg_loss # Corrected bug: was avg_gain / rs
        
        rsi_values[i] = 100 - (100 / (1 + rs))
            
    return list(rsi_values)

def _calculate_ema(prices, window):
    """
    Helper function to calculate Exponential Moving Average (EMA).
    Returns a list of EMA values. Uses NaN for initial periods.
    """
    if window <= 0:
        raise ValueError("EMA window must be greater than zero.")

    prices_np = np.array(prices, dtype=float)
    ema_values = np.full(len(prices_np), np.nan)

    if len(prices_np) < window:
        return list(ema_values)

    # First EMA is SMA
    ema_current = np.mean(prices_np[0:window])
    ema_values[window-1] = ema_current

    multiplier = 2 / (window + 1)
    for i in range(window, len(prices_np)):
        ema_current = (prices_np[i] - ema_current) * multiplier + ema_current
        ema_values[i] = ema_current
    return list(ema_values)

def calculate_macd(prices, fast_window, slow_window, signal_window):
    """
    Calculates the Moving Average Convergence Divergence (MACD) indicator.
    Returns MACD Line, Signal Line, and MACD Histogram. Uses NaN for initial periods.
    """
    if len(prices) < max(fast_window, slow_window):
        return ([np.nan]*len(prices), [np.nan]*len(prices), [np.nan]*len(prices))

    fast_ema = _calculate_ema(prices, fast_window)
    slow_ema = _calculate_ema(prices, slow_window)

    macd_line = [
        (fast_ema[i] - slow_ema[i]) if not np.isnan(fast_ema[i]) and not np.isnan(slow_ema[i]) else np.nan
        for i in range(len(prices))
    ]
    
    # Calculate Signal Line (EMA of MACD Line)
    valid_macd_line_for_signal = [val for val in macd_line if not np.isnan(val)]
    
    signal_line_raw = _calculate_ema(valid_macd_line_for_signal, signal_window)
    
    # Pad signal_line with NaN at the beginning to align with macd_line
    signal_line = [np.nan] * (len(prices) - len(signal_line_raw)) + signal_line_raw

    macd_histogram = [
        (macd_line[i] - signal_line[i]) if not np.isnan(macd_line[i]) and not np.isnan(signal_line[i]) else np.nan
        for i in range(len(prices))
    ]
    
    return list(macd_line), list(signal_line), list(macd_histogram)

def calculate_bollinger_bands(prices, window, std_dev_multiplier):
    """
    Calculates Bollinger Bands (Middle, Upper, Lower bands).
    Returns three lists: middle_band, upper_band, lower_band. Uses NaN for initial periods.
    """
    if window <= 0:
        raise ValueError("Bollinger Band window must be greater than zero.")

    prices_np = np.array(prices, dtype=float)
    middle_band = calculate_sma(prices, window)
    upper_band = np.full(len(prices_np), np.nan)
    lower_band = np.full(len(prices_np), np.nan)

    for i in range(window - 1, len(prices_np)):
        current_window_prices = prices_np[i - (window - 1) : i + 1]
        
        # Check for NaN in the window to avoid errors
        if np.any(np.isnan(current_window_prices)):
            continue

        std_dev = np.std(current_window_prices) # Use numpy's std dev

        if not np.isnan(middle_band[i]):
            upper_band[i] = middle_band[i] + (std_dev * std_dev_multiplier)
            lower_band[i] = middle_band[i] - (std_dev * std_dev_multiplier)
            
    return list(middle_band), list(upper_band), list(lower_band)

def calculate_stochastic_oscillator(high_prices, low_prices, close_prices, k_window, d_window):
    """
    Calculates the Stochastic Oscillator (%K and %D lines).
    Returns two lists: %K values and %D values. Uses NaN for initial periods.
    """
    if k_window <= 0 or d_window <= 0:
        raise ValueError("Stochastic window periods must be greater than zero.")

    high_prices_np = np.array(high_prices, dtype=float)
    low_prices_np = np.array(low_prices, dtype=float)
    close_prices_np = np.array(close_prices, dtype=float)

    percent_k = np.full(len(close_prices_np), np.nan)
    
    # %K calculation
    for i in range(k_window - 1, len(close_prices_np)):
        window_high = np.max(high_prices_np[i - (k_window - 1) : i + 1])
        window_low = np.min(low_prices_np[i - (k_window - 1) : i + 1])
        current_close = close_prices_np[i]

        if (window_high - window_low) != 0:
            percent_k[i] = ((current_close - window_low) / (window_high - window_low)) * 100
        else:
            percent_k[i] = 0.0 # Price hasn't moved within the window

    # %D calculation (SMA of %K)
    # Filter out NaN values from %K for SMA calculation
    valid_percent_k_for_d = [val for val in percent_k if not np.isnan(val)]
    
    percent_d = np.full(len(close_prices_np), np.nan)

    if len(valid_percent_k_for_d) < d_window:
        return list(percent_k), list(percent_d)
        
    raw_percent_d = calculate_sma(valid_percent_k_for_d, d_window)
    # Pad %D with NaN at the beginning to align with %K
    percent_d_start_index = len(close_prices_np) - len(raw_percent_d)
    percent_d[percent_d_start_index:] = raw_percent_d

    return list(percent_k), list(percent_d)

def calculate_atr(high_prices, low_prices, close_prices, window):
    """
    Calculates the Average True Range (ATR).
    Returns a list of ATR values. Uses NaN for initial periods.
    """
    if window <= 0:
        raise ValueError("ATR window must be greater than zero.")

    high_prices_np = np.array(high_prices, dtype=float)
    low_prices_np = np.array(low_prices, dtype=float)
    close_prices_np = np.array(close_prices, dtype=float)

    true_ranges = np.full(len(close_prices_np), np.nan)
    atr_values = np.full(len(close_prices_np), np.nan)

    # Calculate True Ranges
    for i in range(len(close_prices_np)):
        if i == 0:
            # For the very first data point, TR is High - Low
            true_ranges[i] = high_prices_np[i] - low_prices_np[i]
        else:
            tr1 = high_prices_np[i] - low_prices_np[i]
            tr2 = np.abs(high_prices_np[i] - close_prices_np[i-1])
            tr3 = np.abs(low_prices_np[i] - close_prices_np[i-1])
            true_ranges[i] = np.max([tr1, tr2, tr3])
            
    # Calculate ATR (SMA of True Ranges, then EMA-like smoothing)
    valid_true_ranges_for_atr = [tr for tr in true_ranges if not np.isnan(tr)]

    if len(valid_true_ranges_for_atr) < window:
        return list(atr_values) # Not enough data for initial ATR

    # First ATR is SMA of the first 'window' True Ranges
    atr_current = np.mean(valid_true_ranges_for_atr[0:window])
    atr_values[window-1] = atr_current

    # Subsequent ATR calculations (using a simplified EMA-like smoothing)
    for i in range(window, len(valid_true_ranges_for_atr)):
        atr_current = ((atr_current * (window - 1)) + valid_true_ranges_for_atr[i]) / window
        atr_values[i] = atr_current
            
    return list(atr_values)


def calculate_ichimoku_cloud(high_prices, low_prices, close_prices, 
                             tenkan_sen_window, kijun_sen_window, senkou_span_b_window, chikou_span_lag):
    """
    Calculates the Ichimoku Cloud components.
    Returns lists for Tenkan-sen, Kijun-sen, Senkou Span A, Senkou Span B, Chikou Span (for plotting/historical context).
    Uses NaN for initial periods.
    """
    high_prices_np = np.array(high_prices, dtype=float)
    low_prices_np = np.array(low_prices, dtype=float)
    close_prices_np = np.array(close_prices, dtype=float)

    tenkan_sen = np.full(len(close_prices_np), np.nan)
    kijun_sen = np.full(len(close_prices_np), np.nan)
    senkou_span_a = np.full(len(close_prices_np), np.nan)
    senkou_span_b = np.full(len(close_prices_np), np.nan)
    chikou_span_plot = np.full(len(close_prices_np), np.nan)

    # Calculate Tenkan-sen (Conversion Line)
    for i in range(tenkan_sen_window - 1, len(close_prices_np)):
        high_period = np.max(high_prices_np[i - (tenkan_sen_window - 1) : i + 1])
        low_period = np.min(low_prices_np[i - (tenkan_sen_window - 1) : i + 1])
        tenkan_sen[i] = (high_period + low_period) / 2

    # Calculate Kijun-sen (Base Line)
    for i in range(kijun_sen_window - 1, len(close_prices_np)):
        high_period = np.max(high_prices_np[i - (kijun_sen_window - 1) : i + 1])
        low_period = np.min(low_prices_np[i - (kijun_sen_window - 1) : i + 1])
        kijun_sen[i] = (high_period + low_period) / 2

    # Calculate Senkou Span A (Leading Span A) - (Tenkan-sen + Kijun-sen) / 2, plotted forward
    for i in range(len(close_prices_np)):
        if not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
            future_index = i + CHIKOU_SPAN_LAG # Senkou spans are plotted 26 periods ahead
            if future_index < len(close_prices_np):
                senkou_span_a[future_index] = (tenkan_sen[i] + kijun_sen[i]) / 2

    # Calculate Senkou Span B (Leading Span B) - (52-period high + 52-period low) / 2, plotted forward
    for i in range(senkou_span_b_window - 1, len(close_prices_np)):
        high_period = np.max(high_prices_np[i - (senkou_span_b_window - 1) : i + 1])
        low_period = np.min(low_prices_np[i - (senkou_span_b_window - 1) : i + 1])
        senkou_span_b_val = (high_period + low_period) / 2
        
        future_index = i + CHIKOU_SPAN_LAG # Senkou spans are plotted 26 periods ahead
        if future_index < len(close_prices_np):
            senkou_span_b[future_index] = senkou_span_b_val

    # Calculate Chikou Span (Lagging Span) - Current closing price, plotted backward
    for i in range(len(close_prices_np)):
        lagged_index = i - chikou_span_lag
        if lagged_index >= 0:
            chikou_span_plot[lagged_index] = close_prices_np[i]

    return list(tenkan_sen), list(kijun_sen), list(senkou_span_a), list(senkou_span_b), list(chikou_span_plot)


def generate_trading_signals(ohlc_data_with_dates, short_sma_window, long_sma_window, rsi_window, 
                             fast_ema_window, slow_ema_window, signal_ema_window,
                             bb_window, bb_std_dev_multiplier, 
                             stoch_k_window, stoch_d_window, atr_window,
                             tenkan_sen_window, kijun_sen_window, senkou_span_b_window, chikou_span_lag,
                             is_stablecoin, prediction_period_offset): # Added prediction_period_offset
    """
    Generates trading signals (BUY, SELL, HOLD) based on SMA crossovers and calculates RSI, MACD, Bollinger Bands, Stochastic, ATR, and Ichimoku Cloud.
    Includes Machine Learning prediction with a specified future timeframe.
    """
    # Extract individual price lists from OHLC data
    timestamps, open_prices, high_prices, low_prices, close_prices = zip(*ohlc_data_with_dates)
    close_prices_list = list(close_prices)
    high_prices_list = list(high_prices)
    low_prices_list = list(low_prices)

    # Determine minimum data points needed for all indicators to be calculated
    min_data_points_for_all_indicators = max(
        long_sma_window,
        rsi_window + 1,
        slow_ema_window + signal_ema_window, # For MACD, roughly max EMA window + signal window's EMA window
        bb_window,
        stoch_k_window + stoch_d_window, # For Stochastic
        atr_window + 1,
        senkou_span_b_window + chikou_span_lag
    )

    # For ML, we need data up to (min_data_points_for_all_indicators + prediction_period_offset)
    # The target `y` will be `close_prices_list[i + prediction_period_offset]`
    # So, we need at least `min_data_points_for_all_indicators + prediction_period_offset` data points in total.
    if len(close_prices_list) < min_data_points_for_all_indicators + prediction_period_offset:
        # Return Nones for all output lists if not enough data
        return ([np.nan]*len(ohlc_data_with_dates), [np.nan]*len(ohlc_data_with_dates), ["N/A"]*len(ohlc_data_with_dates),
                [np.nan]*len(ohlc_data_with_dates), [np.nan]*len(ohlc_data_with_dates), [np.nan]*len(ohlc_data_with_dates),
                [np.nan]*len(ohlc_data_with_dates), [np.nan]*len(ohlc_data_with_dates), [np.nan]*len(ohlc_data_with_dates),
                [np.nan]*len(ohlc_data_with_dates), [np.nan]*len(ohlc_data_with_dates), [np.nan]*len(ohlc_data_with_dates),
                [np.nan]*len(ohlc_data_with_dates), [np.nan]*len(ohlc_data_with_dates), [np.nan]*len(ohlc_data_with_dates),
                [np.nan]*len(ohlc_data_with_dates), [np.nan]*len(ohlc_data_with_dates), [np.nan]*len(ohlc_data_with_dates),
                np.nan) # predicted_next_close

    # Calculate all indicators
    short_smas = calculate_sma(close_prices_list, short_sma_window)
    long_smas = calculate_sma(close_prices_list, long_sma_window)
    rsi_values = calculate_rsi(close_prices_list, rsi_window)
    macd_line, signal_line, macd_histogram = calculate_macd(close_prices_list, fast_ema_window, slow_ema_window, signal_ema_window)
    bb_middle, bb_upper, bb_lower = calculate_bollinger_bands(close_prices_list, bb_window, bb_std_dev_multiplier)
    stoch_k, stoch_d = calculate_stochastic_oscillator(high_prices_list, low_prices_list, close_prices_list, stoch_k_window, stoch_d_window)
    atr_values = calculate_atr(high_prices_list, low_prices_list, close_prices_list, atr_window)
    tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span_plot = calculate_ichimoku_cloud(
        high_prices_list, low_prices_list, close_prices_list,
        tenkan_sen_window, kijun_sen_window, senkou_span_b_window, chikou_span_lag
    )

    # Determine SMA crossover signal
    signals = []
    for i in range(len(ohlc_data_with_dates)):
        current_signal = "HOLD" # Default signal

        if i > 0 and not np.isnan(short_smas[i]) and not np.isnan(long_smas[i]):
            previous_short_sma = short_smas[i-1]
            previous_long_sma = long_smas[i-1]

            if not np.isnan(previous_short_sma) and not np.isnan(previous_long_sma):
                if previous_short_sma <= previous_long_sma and \
                   short_smas[i] > long_smas[i]:
                    current_signal = "BUY"
                elif previous_short_sma >= previous_long_sma and \
                     short_smas[i] < long_smas[i]:
                    current_signal = "SELL"
        signals.append(current_signal)
    
    # --- Machine Learning Prediction ---
    predicted_next_close = np.nan

    if is_stablecoin:
        # For stablecoins, predict stability around their peg (e.g., 1.0 USD)
        predicted_next_close = 1.0 # Or latest_close for current peg
    else:
        # We need enough data to create features and a target for training
        ml_min_data_points = min_data_points_for_all_indicators # Adjusted for proper feature generation
        
        # Ensure that we have enough data points to create features AND a future target
        if len(close_prices_list) > ml_min_data_points + prediction_period_offset:
            X = [] # Features
            y = [] # Target (next close price)

            # Iterate through the data to create feature-target pairs
            # The loop goes up to len(close_prices_list) - prediction_period_offset
            # because we need a target for prediction_period_offset periods ahead.
            start_index_for_ml = min_data_points_for_all_indicators - 1 # Adjusted start index for earliest valid indicator set

            for i in range(start_index_for_ml, len(close_prices_list) - prediction_period_offset): 
                # Ensure all indicators are not NaN for this period and target is valid
                if all(not np.isnan(val) for val in [
                    close_prices_list[i], short_smas[i], long_smas[i], rsi_values[i], macd_line[i], signal_line[i], 
                    macd_histogram[i], bb_middle[i], bb_upper[i], bb_lower[i], 
                    stoch_k[i], stoch_d[i], atr_values[i], tenkan_sen[i], kijun_sen[i],
                    senkou_span_a[i], senkou_span_b[i]
                ]) and not np.isnan(close_prices_list[i + prediction_period_offset]):
                    features_row = [
                        close_prices_list[i],
                        short_smas[i],
                        long_smas[i],
                        rsi_values[i],
                        macd_line[i],
                        signal_line[i],
                        macd_histogram[i],
                        bb_middle[i],
                        bb_upper[i],
                        bb_lower[i],
                        stoch_k[i],
                        stoch_d[i],
                        atr_values[i],
                        tenkan_sen[i],
                        kijun_sen[i],
                        senkou_span_a[i],
                        senkou_span_b[i]
                    ]
                    X.append(features_row)
                    y.append(close_prices_list[i + prediction_period_offset]) # Target is 'prediction_period_offset' periods ahead

            if X: # Only train if we have data
                X = np.array(X)
                y = np.array(y)

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                model = LinearRegression()
                model.fit(X_scaled, y)

                # Predict the next close price using the latest available features
                # Ensure latest features are not NaN
                if all(not np.isnan(val) for val in [
                    close_prices_list[-1], short_smas[-1], long_smas[-1], rsi_values[-1], macd_line[-1], signal_line[-1], 
                    macd_histogram[-1], bb_middle[-1], bb_upper[-1], bb_lower[-1], 
                    stoch_k[-1], stoch_d[-1], atr_values[-1], tenkan_sen[-1], kijun_sen[-1],
                    senkou_span_a[-1], senkou_span_b[-1]
                ]):
                    latest_features = np.array([
                        close_prices_list[-1],
                        short_smas[-1],
                        long_smas[-1],
                        rsi_values[-1],
                        macd_line[-1],
                        signal_line[-1],
                        macd_histogram[-1],
                        bb_middle[-1],
                        bb_upper[-1],
                        bb_lower[-1],
                        stoch_k[-1],
                        stoch_d[-1],
                        atr_values[-1],
                        tenkan_sen[-1],
                        kijun_sen[-1],
                        senkou_span_a[-1],
                        senkou_span_b[-1]
                    ]).reshape(1, -1)

                    latest_features_scaled = scaler.transform(latest_features)
                    predicted_next_close = model.predict(latest_features_scaled)[0]

    return (short_smas, long_smas, signals, rsi_values, 
            macd_line, signal_line, macd_histogram, 
            bb_middle, bb_upper, bb_lower,
            stoch_k, stoch_d, atr_values,
            tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span_plot,
            predicted_next_close)

def calculate_confidence_score(overall_sentiment_score):
    """
    Calculates a heuristic confidence score (0-100%) based on the overall sentiment.
    Assumes overall_sentiment_score ranges approximately from -5 to +5.
    A neutral sentiment (0) maps to 50% confidence.
    Strong bullish/bearish maps to higher confidence.
    """
    # Define the min and max possible sentiment scores for normalization
    # These are rough estimates based on the current scoring logic.
    MIN_SENTIMENT = -5.5
    MAX_SENTIMENT = 5.5

    if overall_sentiment_score >= 0:
        # Scale positive sentiment from 0 to 0.5 to 50% to 100%
        # (sentiment_score - 0) / (MAX_SENTIMENT - 0) * 0.5 + 0.5
        confidence = (overall_sentiment_score / MAX_SENTIMENT) * 50 + 50
    else:
        # Scale negative sentiment from -0.5 to 0 to 50% to 100% (in terms of conviction)
        # abs(sentiment_score - 0) / abs(MIN_SENTIMENT - 0) * 0.5 + 0.5
        confidence = (abs(overall_sentiment_score) / abs(MIN_SENTIMENT)) * 50 + 50
    
    # Cap confidence between 0 and 100
    confidence = max(0, min(100, confidence))
    return confidence


# --- Recommendation Logic ---
def get_recommendation(latest_open, latest_high, latest_low, latest_close, 
                         latest_short_sma, latest_long_sma, latest_signal, 
                         latest_rsi, latest_macd_line, latest_signal_line, latest_macd_histogram,
                         latest_bb_middle, latest_bb_upper, latest_bb_lower,
                         latest_stoch_k, latest_stoch_d, latest_atr,
                         latest_tenkan_sen, latest_kijun_sen, latest_senkou_span_a, latest_senkou_span_b, latest_chikou_span,
                         predicted_next_close, 
                         crypto_id, vs_currency, current_utc_datetime,
                         prediction_timeframe_description, overall_confidence_percent): # Added prediction_timeframe_description and overall_confidence_percent
    """
    Generates a sophisticated recommendation based on SMA, RSI, MACD, Bollinger Bands, Stochastic, ATR, and Ichimoku Cloud,
    and a basic ML price prediction. Includes time frame for signal execution and confidence.
    """
    recommendation_parts = []
    
    # Convert current UTC time to EST
    est_offset = timedelta(hours=-5)
    current_est_datetime = current_utc_datetime + est_offset
    est_time_str = current_est_datetime.strftime('%I:%M %p EST')

    # Check for sufficient data for all indicators (using np.isnan for NaN values)
    if any(np.isnan(val) for val in [latest_close, latest_short_sma, latest_long_sma, latest_rsi, 
                                      latest_macd_line, latest_signal_line, latest_macd_histogram,
                                      latest_bb_middle, latest_bb_upper, latest_bb_lower,
                                      latest_stoch_k, latest_stoch_d, latest_atr,
                                      latest_tenkan_sen, latest_kijun_sen, latest_senkou_span_a, latest_senkou_span_b, latest_chikou_span]):
        return f"Not enough data for a comprehensive recommendation as of {est_time_str}."

    price_str = f"{latest_close:.4f} {vs_currency.upper()}"
    short_sma_str = f"{latest_short_sma:.4f}"
    long_sma_str = f"{latest_long_sma:.4f}"
    rsi_str = f"{latest_rsi:.2f}"
    macd_line_str = f"{latest_macd_line:.4f}"
    signal_line_str = f"{latest_signal_line:.4f}"
    macd_hist_str = f"{latest_macd_histogram:.4f}"
    bb_upper_str = f"{latest_bb_upper:.4f}"
    bb_lower_str = f"{latest_bb_lower:.4f}"
    stoch_k_str = f"{latest_stoch_k:.2f}"
    stoch_d_str = f"{latest_stoch_d:.2f}"
    atr_str = f"{latest_atr:.4f}"
    tenkan_str = f"{latest_tenkan_sen:.4f}"
    kijun_str = f"{latest_kijun_sen:.4f}"
    senkou_a_str = f"{latest_senkou_span_a:.4f}"
    senkou_b_str = f"{latest_senkou_span_b:.4f}"
    chikou_str = f"{latest_chikou_span:.4f}"


    # --- SMA Analysis ---
    if latest_short_sma > latest_long_sma:
        recommendation_parts.append(f"SMA Trend: Bullish (Short SMA {short_sma_str} > Long SMA {long_sma_str}).")
    elif latest_short_sma < latest_long_sma:
        recommendation_parts.append(f"SMA Trend: Bearish (Short SMA {short_sma_str} < Long SMA {long_sma_str}).")
    else:
        recommendation_parts.append(f"SMA Trend: Sideways/Unclear.")
    
    # --- RSI Analysis ---
    if latest_rsi >= 70:
        recommendation_parts.append(f"RSI ({rsi_str}): Overbought. Potential bearish reversal.")
    elif latest_rsi <= 30:
        recommendation_parts.append(f"RSI ({rsi_str}): Oversold. Potential for rebound.")
    else:
        recommendation_parts.append(f"RSI ({rsi_str}): Neutral. No strong overbought/oversold condition.")

    # --- MACD Analysis ---
    if latest_macd_line > latest_signal_line and latest_macd_histogram > 0:
        recommendation_parts.append(f"MACD: Bullish momentum building (MACD Line {macd_line_str} > Signal Line {signal_line_str}, Histogram positive {macd_hist_str}).")
    elif latest_macd_line < latest_signal_line and latest_macd_histogram < 0:
        recommendation_parts.append(f"MACD: Bearish momentum building (MACD Line {macd_line_str} < Signal Line {signal_line_str}, Histogram negative {macd_hist_str}).")
    else:
        recommendation_parts.append(f"MACD: Trend uncertainty or weakening momentum (MACD Line {macd_line_str}, Signal Line {signal_line_str}, Histogram {macd_hist_str}).")

    # --- Bollinger Bands Analysis ---
    if latest_close > latest_bb_upper:
        recommendation_parts.append(f"Bollinger Bands: Price ({price_str}) is above Upper Band ({bb_upper_str}). Overextension, likely pullback.")
    elif latest_close < latest_bb_lower:
        recommendation_parts.append(f"Bollinger Bands: Price ({price_str}) is below Lower Band ({bb_lower_str}). Underextension, likely rebound.")
    else:
        recommendation_parts.append(f"Bollinger Bands: Price ({price_str}) is within bands ({bb_lower_str}-{bb_upper_str}).")

    # --- Stochastic Oscillator Analysis ---
    # Prioritize overbought/oversold first as per prompt's analysis
    if latest_stoch_k >= 80 or latest_stoch_d >= 80:
        recommendation_parts.append(f"Stochastic ({stoch_k_str}/{stoch_d_str}): Overbought. Potential bearish reversal.")
    elif latest_stoch_k <= 20 or latest_stoch_d <= 20:
        recommendation_parts.append(f"Stochastic ({stoch_k_str}/{stoch_d_str}): Oversold. Potential bullish reversal.")
    elif latest_stoch_k > latest_stoch_d: # Bullish crossover in neutral zone
        recommendation_parts.append(f"Stochastic ({stoch_k_str}/{stoch_d_str}): Bullish crossover. Momentum shifting up.")
    elif latest_stoch_k < latest_stoch_d: # Bearish crossover in neutral zone
        recommendation_parts.append(f"Stochastic ({stoch_k_str}/{stoch_d_str}): Bearish crossover. Momentum shifting down.")
    else:
        recommendation_parts.append(f"Stochastic ({stoch_k_str}/{stoch_d_str}): Neutral territory or unclear momentum.")


    # --- ATR Analysis ---
    recommendation_parts.append(f"ATR ({ATR_WINDOW} periods): {atr_str} {vs_currency.upper()}. Current volatility measure.")

    # --- Ichimoku Cloud Analysis ---
    ichimoku_summary = []
    
    # Price vs Tenkan/Kijun
    if latest_close > latest_tenkan_sen and latest_close > latest_kijun_sen:
        ichimoku_summary.append("Price is above Tenkan/Kijun (Bullish).")
    elif latest_close < latest_tenkan_sen and latest_close < latest_kijun_sen:
        ichimoku_summary.append("Price is below Tenkan/Kijun (Bearish).")
    else:
        ichimoku_summary.append("Price is between Tenkan/Kijun (Neutral/Choppy).")

    # Tenkan-sen vs Kijun-sen Crossover
    if latest_tenkan_sen > latest_kijun_sen:
        ichimoku_summary.append("Tenkan-sen > Kijun-sen (Bullish momentum).")
    else:
        ichimoku_summary.append("Tenkan-sen < Kijun-sen (Bearish momentum).")

    # Price vs Cloud (Senkou Span A and B) - Using the *projected* cloud values
    current_cloud_upper = max(latest_senkou_span_a, latest_senkou_span_b)
    current_cloud_lower = min(latest_senkou_span_a, latest_senkou_span_b)

    if latest_close > current_cloud_upper:
        ichimoku_summary.append("Price is above the Cloud (Strong Bullish).")
    elif latest_close < current_cloud_lower:
        ichimoku_summary.append("Price is below the Cloud (Strong Bearish).")
    else:
        ichimoku_summary.append("Price is within the Cloud (Neutral/Consolidation).")

    # Cloud formation (twist and thickness)
    if latest_senkou_span_a > latest_senkou_span_b:
        ichimoku_summary.append("Bullish Cloud (Senkou A > Senkou B).")
    else:
        ichimoku_summary.append("Bearish Cloud (Senkou A < Senkou B).")

    # Chikou Span (Lagging Span) vs Lagged Price
    if not np.isnan(latest_chikou_span):
        if latest_close > latest_chikou_span:
            ichimoku_summary.append("Chikou Span is above lagged price (Bullish).")
        elif latest_close < latest_chikou_span:
            ichimoku_summary.append("Chikou Span is below lagged price (Bearish).")
        else:
            ichimoku_summary.append("Chikou Span is near lagged price (Neutral).")
    else:
        ichimoku_summary.append("Chikou Span: Not available (insufficient historical data).")
            
    recommendation_parts.append(f"Ichimoku Cloud: " + "; ".join(ichimoku_summary))

    # --- ML Prediction Analysis ---
    if not np.isnan(predicted_next_close):
        # Handle stablecoins specifically for ML prediction display
        if crypto_id in STABLECOIN_IDS:
            recommendation_parts.append(f"ML Prediction: Next close predicted to be stable around {predicted_next_close:.4f} {vs_currency.upper()} (Stablecoin behavior).")
        else:
            prediction_diff = ((predicted_next_close - latest_close) / latest_close) * 100
            prediction_str = f"{predicted_next_close:.4f} {vs_currency.upper()}"
            if prediction_diff > 0.5:
                recommendation_parts.append(f"ML Prediction for {prediction_timeframe_description}: Next close predicted to rise by {prediction_diff:.2f}% to {prediction_str}.")
            elif prediction_diff < -0.5:
                recommendation_parts.append(f"ML Prediction for {prediction_timeframe_description}: Next close predicted to fall by {abs(prediction_diff):.2f}% to {prediction_str}.")
            else:
                recommendation_parts.append(f"ML Prediction for {prediction_timeframe_description}: Next close predicted to be stable around {prediction_str} ({prediction_diff:.2f}% change).")
    else:
        recommendation_parts.append("ML Prediction: Not enough data for prediction or prediction failed.")


    # --- Overall Recommendation based on combined signals ---
    overall_sentiment_score = 0 # +1 for bullish, -1 for bearish, 0 for neutral

    # Simplified sentiment scoring, focusing on strong signals
    # SMA Trend
    if latest_short_sma > latest_long_sma: overall_sentiment_score += 1
    elif latest_short_sma < latest_long_sma: overall_sentiment_score -= 1

    # RSI
    if latest_rsi >= 70: overall_sentiment_score -= 1 # Overbought is bearish
    elif latest_rsi <= 30: overall_sentiment_score += 1 # Oversold is bullish

    # MACD
    if latest_macd_line > latest_signal_line and latest_macd_histogram > 0: overall_sentiment_score += 1
    elif latest_macd_line < latest_signal_line and latest_macd_histogram < 0: overall_sentiment_score -= 1

    # Bollinger Bands
    if latest_close < latest_bb_lower: overall_sentiment_score += 0.5 # Underextension is bullish bounce potential
    elif latest_close > latest_bb_upper: overall_sentiment_score -= 0.5 # Overextension is bearish pullback potential

    # Stochastic
    if (latest_stoch_k >= 80 or latest_stoch_d >= 80) : overall_sentiment_score -= 1 # Overbought
    elif (latest_stoch_k <= 20 or latest_stoch_d <= 20): overall_sentiment_score += 1 # Oversold
    elif latest_stoch_k > latest_stoch_d and latest_stoch_d < 80: overall_sentiment_score += 0.5 # Bullish crossover (not extreme)
    elif latest_stoch_k < latest_stoch_d and latest_stoch_d > 20: overall_sentiment_score -= 0.5 # Bearish crossover (not extreme)

    # Ichimoku Cloud - stronger weighting for cloud position
    if latest_close > max(latest_senkou_span_a, latest_senkou_span_b): overall_sentiment_score += 1.5 # Strong bullish cloud position
    elif latest_close < min(latest_senkou_span_a, latest_senkou_span_b): overall_sentiment_score -= 1.5 # Strong bearish cloud position
    
    # Internal Ichimoku momentum
    if latest_tenkan_sen > latest_kijun_sen: overall_sentiment_score += 0.5
    else: overall_sentiment_score -= 0.5 # Even if neutral cloud, Tenkan-Kijun still indicates momentum

    if not np.isnan(latest_chikou_span):
        if latest_close > latest_chikou_span: overall_sentiment_score += 0.5
        elif latest_close < latest_chikou_span: overall_sentiment_score -= 0.5

    # ML Prediction (if not stablecoin and valid prediction)
    if not np.isnan(predicted_next_close) and crypto_id not in STABLECOIN_IDS:
        prediction_diff = ((predicted_next_close - latest_close) / latest_close) * 100
        if prediction_diff > 0.5: overall_sentiment_score += 1
        elif prediction_diff < -0.5: overall_sentiment_score -= 1

    final_recommendation_prefix = ""
    final_recommendation_color = ""

    # Determine overall recommendation based on sentiment score and add time frame/confidence
    if overall_sentiment_score >= 3: # Strong buy threshold adjusted
        final_recommendation_prefix = "Strong BUY"
        final_recommendation_color = "\033[92m"
        recommendation_parts.append(f"Consider executing a BUY signal within the next {prediction_timeframe_description}.")
    elif overall_sentiment_score >= 1.5: # Moderate buy threshold adjusted
        final_recommendation_prefix = "BUY"
        final_recommendation_color = "\033[92m"
        recommendation_parts.append(f"Consider executing a BUY signal within the next {prediction_timeframe_description}.")
    elif overall_sentiment_score <= -3: # Strong sell threshold adjusted
        final_recommendation_prefix = "Strong SELL"
        final_recommendation_color = "\033[91m"
        recommendation_parts.append(f"Consider executing a SELL signal within the next {prediction_timeframe_description}. Act swiftly to limit potential losses.")
    elif overall_sentiment_score <= -1.5: # Moderate sell threshold adjusted
        final_recommendation_prefix = "SELL"
        final_recommendation_color = "\033[91m"
        recommendation_parts.append(f"Consider executing a SELL signal within the next {prediction_timeframe_description}.")
    else: # Neutral
        final_recommendation_prefix = "NEUTRAL / HOLD"
        final_recommendation_color = "\033[93m"
        recommendation_parts.append(f"The market for {crypto_id.capitalize()} is currently neutral or lacks strong direction for the next {prediction_timeframe_description}. Consider holding your position or waiting for clearer signals.")

    full_recommendation = (
        f"{final_recommendation_color}{final_recommendation_prefix} for {crypto_id.capitalize()} (As of {est_time_str}) with {overall_confidence_percent:.1f}% Confidence:\033[0m\n"
        + "  " + "\n  ".join(recommendation_parts)
    )

    return full_recommendation

def save_report_to_csv(report_data, vs_currency):
    """
    Saves the aggregated report data to a CSV file.
    """
    if not report_data:
        print("No data to save to CSV.")
        return

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"crypto_report_{timestamp_str}.csv"

    fieldnames = [
        "Coin ID", "VS Currency", "Latest Date/Time", "Latest Price (Close)",
        f"Short SMA ({SHORT_SMA_WINDOW} periods)",
        f"Long SMA ({LONG_SMA_WINDOW} periods)",
        f"RSI ({RSI_WINDOW} periods)",
        f"MACD Line ({FAST_EMA_WINDOW},{SLOW_EMA_WINDOW})",
        f"Signal Line ({SIGNAL_EMA_WINDOW})",
        "MACD Histogram",
        f"Bollinger Bands Middle ({BB_WINDOW},{BB_STD_DEV_MULTIPLIER} StdDev)",
        f"Bollinger Bands Upper ({BB_WINDOW},{BB_STD_DEV_MULTIPLIER} StdDev)",
        f"Bollinger Bands Lower ({BB_WINDOW},{BB_STD_DEV_MULTIPLIER} StdDev)",
        f"Stochastic %K ({STOCH_K_WINDOW},{STOCH_D_WINDOW})",
        f"Stochastic %D ({STOCH_K_WINDOW},{STOCH_D_WINDOW})",
        f"ATR ({ATR_WINDOW} periods)",
        f"Ichimoku Tenkan-sen ({TENKAN_SEN_WINDOW})",
        f"Ichimoku Kijun-sen ({KIJUN_SEN_WINDOW})",
        "Ichimoku Senkou Span A",
        f"Ichimoku Senkou Span B ({SENKOU_SPAN_B_WINDOW})",
        f"Ichimoku Chikou Span (Lag {CHIKOU_SPAN_LAG})",
        "ML Predicted Next Close",
        "Confidence (%)", # Added Confidence to CSV
        "Overall Recommendation"
    ]

    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in report_data:
                # Replace None/NaN with "N/A" for CSV output
                cleaned_row = {k: ("N/A" if (isinstance(v, float) and np.isnan(v)) or v is None else v) for k, v in row.items()}
                writer.writerow(cleaned_row)
        print(f"\nReport successfully saved to {filename}")
    except IOError as e:
        print(f"Error saving report to CSV: {e}")

# --- Main Application Execution ---
def main():
    """
    Main function to run the cryptocurrency signal generator with live updates.
    """
    print(text2art("CoinSight", font="poison"))
    print("--- Starting Cryptocurrency Signal Generator ---")
    print(f"Fetching data and updating every {UPDATE_INTERVAL_SECONDS} seconds...")
    print("Press Ctrl+C to stop the application.\n")

    vs_currency = input(f"Enter currency to compare against (e.g., 'usd', 'eur', default: {DEFAULT_VS_CURRENCY}): ").strip().lower()
    if not vs_currency:
        vs_currency = DEFAULT_VS_CURRENCY

    while True:
        clear_terminal()
        
        # Use datetime.now(UTC) to replace deprecated utcnow()
        current_utc_datetime = datetime.now(UTC)
        
        print("\n" + "="*80)
        print(f"--- Live Crypto Signal Report ({current_utc_datetime.strftime('%Y-%m-%d %H:%M:%S %Z')} UTC) ---") # Changed format to show timezone
        print("="*80)

        monitored_coin_ids = []
        all_reports_data = []

        print("\n--- Fetching Top 10 Popular Coins ---")
        top_popular_coins = get_top_popular_coins(vs_currency, count=10)
        monitored_coin_ids.extend(top_popular_coins)
        
        print("\n--- Fetching Top 5 Trending / Meme Coins ---")
        trending_meme_coins = get_trending_coins(count=5)
        for coin_id in trending_meme_coins:
            if coin_id not in monitored_coin_ids:
                monitored_coin_ids.append(coin_id)
        
        monitored_coin_ids = list(dict.fromkeys(monitored_coin_ids))
        
        if not monitored_coin_ids:
            print("No coins to monitor. Please check API connection or configuration.")
            time.sleep(UPDATE_INTERVAL_SECONDS)
            continue

        print(f"\n--- Analyzing {len(monitored_coin_ids)} Coins ---")
        print("-" * 80)

        # Determine the data granularity based on CHART_DATA_DAYS
        # 30 days gives 4-hourly data points.
        data_granularity_hours = 4 
        
        # Calculate the number of data points to look ahead for prediction
        # If PREDICTION_TIMEFRAME_HOURS is 4 and data_granularity_hours is 4, then prediction_period_offset = 1
        if data_granularity_hours > 0:
            prediction_period_offset = int(PREDICTION_TIMEFRAME_HOURS / data_granularity_hours)
        else:
            prediction_period_offset = 1 # Default to 1 period ahead if granularity is very fine (should not happen with fixed 4 hours)

        prediction_timeframe_description = f"{PREDICTION_TIMEFRAME_HOURS} hours ahead"


        for coin_id in monitored_coin_ids:
            print(f"\nAnalysing: {coin_id.capitalize()} ({coin_id.upper()})")
            
            ohlc_data_with_dates = get_coin_chart_data(coin_id, vs_currency, CHART_DATA_DAYS)
            
            if ohlc_data_with_dates:
                timestamps, open_prices, high_prices, low_prices, close_prices = zip(*ohlc_data_with_dates)
                open_prices_list = list(open_prices)
                high_prices_list = list(high_prices)
                low_prices_list = list(low_prices)
                close_prices_list = list(close_prices)
            else:
                open_prices_list, high_prices_list, low_prices_list, close_prices_list = [], [], [], []
            
            # Determine minimum data points needed for all indicators to be calculated
            min_data_points_needed = max(
                LONG_SMA_WINDOW,
                RSI_WINDOW + 1,
                SLOW_EMA_WINDOW + SIGNAL_EMA_WINDOW, 
                BB_WINDOW,
                STOCH_K_WINDOW + STOCH_D_WINDOW,
                ATR_WINDOW + 1,
                SENKOU_SPAN_B_WINDOW + CHIKOU_SPAN_LAG
            )
            # Add required periods for ML prediction target
            min_data_points_needed_for_ml = min_data_points_needed + prediction_period_offset

            is_stablecoin = coin_id in STABLECOIN_IDS
            
            if not ohlc_data_with_dates or len(ohlc_data_with_dates) < min_data_points_needed_for_ml:
                print(f"  Not sufficient data ({len(ohlc_data_with_dates)} points) for {coin_id.capitalize()} to calculate all indicators and make a {prediction_timeframe_description} prediction (requires at least {min_data_points_needed_for_ml} points). Skipping.")
                print("-" * 70)
                # Store a minimal record for CSV even if skipped
                all_reports_data.append({
                    "Coin ID": coin_id,
                    "VS Currency": vs_currency.upper(),
                    "Latest Date/Time": "N/A",
                    "Latest Price (Close)": "N/A",
                    f"Short SMA ({SHORT_SMA_WINDOW} periods)": "N/A",
                    f"Long SMA ({LONG_SMA_WINDOW} periods)": "N/A",
                    f"RSI ({RSI_WINDOW} periods)": "N/A",
                    f"MACD Line ({FAST_EMA_WINDOW},{SLOW_EMA_WINDOW})": "N/A",
                    f"Signal Line ({SIGNAL_EMA_WINDOW})": "N/A",
                    "MACD Histogram": "N/A",
                    f"Bollinger Bands Middle ({BB_WINDOW},{BB_STD_DEV_MULTIPLIER} StdDev)": "N/A",
                    f"Bollinger Bands Upper ({BB_WINDOW},{BB_STD_DEV_MULTIPLIER} StdDev)": "N/A",
                    f"Bollinger Bands Lower ({BB_WINDOW},{BB_STD_DEV_MULTIPLIER} StdDev)": "N/A",
                    f"Stochastic %K ({STOCH_K_WINDOW},{STOCH_D_WINDOW})": "N/A",
                    f"Stochastic %D ({STOCH_K_WINDOW},{STOCH_D_WINDOW})": "N/A",
                    f"ATR ({ATR_WINDOW} periods)": "N/A",
                    f"Ichimoku Tenkan-sen ({TENKAN_SEN_WINDOW})": "N/A",
                    f"Ichimoku Kijun-sen ({KIJUN_SEN_WINDOW})": "N/A",
                    "Ichimoku Senkou Span A": "N/A",
                    f"Ichimoku Senkou Span B ({SENKOU_SPAN_B_WINDOW})": "N/A",
                    f"Ichimoku Chikou Span (Lag {CHIKOU_SPAN_LAG})": "N/A",
                    "ML Predicted Next Close": "N/A",
                    "Confidence (%)": "N/A", # Added Confidence
                    "Overall Recommendation": f"Not enough data for a comprehensive recommendation as of {current_utc_datetime.strftime('%I:%M %p EST')}."
                })
                continue

            # Generate all signals and indicator values
            (short_smas, long_smas, signals, rsi_values, 
             macd_line, signal_line, macd_histogram, 
             bb_middle, bb_upper, bb_lower,
             stoch_k, stoch_d, atr_values,
             tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span_plot,
             predicted_next_close) = generate_trading_signals(
                ohlc_data_with_dates, SHORT_SMA_WINDOW, LONG_SMA_WINDOW, RSI_WINDOW,
                FAST_EMA_WINDOW, SLOW_EMA_WINDOW, SIGNAL_EMA_WINDOW,
                BB_WINDOW, BB_STD_DEV_MULTIPLIER,
                STOCH_K_WINDOW, STOCH_D_WINDOW, ATR_WINDOW,
                TENKAN_SEN_WINDOW, KIJUN_SEN_WINDOW, SENKOU_SPAN_B_WINDOW, CHIKOU_SPAN_LAG,
                is_stablecoin,
                prediction_period_offset # Pass prediction_period_offset
            )

            # Get the latest data point and indicator values for current analysis
            latest_date_obj, latest_open, latest_high, latest_low, latest_close = ohlc_data_with_dates[-1]
            
            # Use .get() with default NaN to handle cases where indicators might be None/NaN due to insufficient data
            latest_short_sma = short_smas[-1] if len(short_smas) > 0 else np.nan
            latest_long_sma = long_smas[-1] if len(long_smas) > 0 else np.nan
            latest_signal = signals[-1] if len(signals) > 0 else "N/A" # Signals can be "N/A" or "HOLD"
            latest_rsi = rsi_values[-1] if len(rsi_values) > 0 else np.nan
            latest_macd_line = macd_line[-1] if len(macd_line) > 0 else np.nan
            latest_signal_line = signal_line[-1] if len(signal_line) > 0 else np.nan
            latest_macd_histogram = macd_histogram[-1] if len(macd_histogram) > 0 else np.nan
            latest_bb_middle = bb_middle[-1] if len(bb_middle) > 0 else np.nan
            latest_bb_upper = bb_upper[-1] if len(bb_upper) > 0 else np.nan
            latest_bb_lower = bb_lower[-1] if len(bb_lower) > 0 else np.nan
            latest_stoch_k = stoch_k[-1] if len(stoch_k) > 0 else np.nan
            latest_stoch_d = stoch_d[-1] if len(stoch_d) > 0 else np.nan
            latest_atr = atr_values[-1] if len(atr_values) > 0 else np.nan
            latest_tenkan_sen = tenkan_sen[-1] if len(tenkan_sen) > 0 else np.nan
            latest_kijun_sen = kijun_sen[-1] if len(kijun_sen) > 0 else np.nan
            latest_senkou_span_a = senkou_span_a[-1] if len(senkou_span_a) > 0 else np.nan
            latest_senkou_span_b = senkou_span_b[-1] if len(senkou_span_b) > 0 else np.nan
            
            # Correctly get the Chikou Span value corresponding to the current (latest) price
            # It's the close price from CHIKOU_SPAN_LAG periods ago
            if len(close_prices_list) > CHIKOU_SPAN_LAG:
                latest_chikou_span = close_prices_list[-1 - CHIKOU_SPAN_LAG]
            else:
                latest_chikou_span = np.nan # Not enough historical data for the lagged Chikou Span

            # Calculate the overall sentiment score (same as original logic)
            overall_sentiment_score = 0
            if latest_short_sma > latest_long_sma: overall_sentiment_score += 1
            elif latest_short_sma < latest_long_sma: overall_sentiment_score -= 1
            if latest_rsi >= 70: overall_sentiment_score -= 1
            elif latest_rsi <= 30: overall_sentiment_score += 1
            if latest_macd_line > latest_signal_line and latest_macd_histogram > 0: overall_sentiment_score += 1
            elif latest_macd_line < latest_signal_line and latest_macd_histogram < 0: overall_sentiment_score -= 1
            if latest_close < latest_bb_lower: overall_sentiment_score += 0.5
            elif latest_close > latest_bb_upper: overall_sentiment_score -= 0.5
            if (latest_stoch_k >= 80 or latest_stoch_d >= 80) : overall_sentiment_score -= 1
            elif (latest_stoch_k <= 20 or latest_stoch_d <= 20): overall_sentiment_score += 1
            elif latest_stoch_k > latest_stoch_d and latest_stoch_d < 80: overall_sentiment_score += 0.5
            elif latest_stoch_k < latest_stoch_d and latest_stoch_d > 20: overall_sentiment_score -= 0.5
            if latest_close > max(latest_senkou_span_a, latest_senkou_span_b): overall_sentiment_score += 1.5
            elif latest_close < min(latest_senkou_span_a, latest_senkou_span_b): overall_sentiment_score -= 1.5
            if latest_tenkan_sen > latest_kijun_sen: overall_sentiment_score += 0.5
            else: overall_sentiment_score -= 0.5
            if not np.isnan(latest_chikou_span):
                if latest_close > latest_chikou_span: overall_sentiment_score += 0.5
                elif latest_close < latest_chikou_span: overall_sentiment_score -= 0.5
            if not np.isnan(predicted_next_close) and coin_id not in STABLECOIN_IDS:
                prediction_diff = ((predicted_next_close - latest_close) / latest_close) * 100
                if prediction_diff > 0.5: overall_sentiment_score += 1
                elif prediction_diff < -0.5: overall_sentiment_score -= 1

            # Calculate confidence score
            confidence_percent = calculate_confidence_score(overall_sentiment_score)


            # Get comprehensive recommendation
            recommendation_text = get_recommendation(
                latest_open, latest_high, latest_low, latest_close, 
                latest_short_sma, latest_long_sma, latest_signal, 
                latest_rsi, latest_macd_line, latest_signal_line, latest_macd_histogram,
                latest_bb_middle, latest_bb_upper, latest_bb_lower,
                latest_stoch_k, latest_stoch_d, latest_atr,
                latest_tenkan_sen, latest_kijun_sen, latest_senkou_span_a, latest_senkou_span_b, latest_chikou_span,
                predicted_next_close, 
                coin_id, vs_currency, current_utc_datetime,
                prediction_timeframe_description, # Pass timeframe description
                confidence_percent # Pass confidence percentage
            )
            
            print(f"  Latest Date/Time: {latest_date_obj.strftime('%Y-%m-%d %H:%M')}")
            print(f"  Latest Price (Close): {latest_close:.4f} {vs_currency.upper()}")
            # Print indicator values only if they are not NaN
            if not np.isnan(latest_short_sma): print(f"  Short SMA ({SHORT_SMA_WINDOW} periods): {latest_short_sma:.4f} {vs_currency.upper()}")
            if not np.isnan(latest_long_sma): print(f"  Long SMA ({LONG_SMA_WINDOW} periods): {latest_long_sma:.4f} {vs_currency.upper()}")
            if not np.isnan(latest_rsi): print(f"  RSI ({RSI_WINDOW} periods): {latest_rsi:.2f}")
            if not np.isnan(latest_macd_line): print(f"  MACD Line ({FAST_EMA_WINDOW},{SLOW_EMA_WINDOW}): {latest_macd_line:.4f}")
            if not np.isnan(latest_signal_line): print(f"  Signal Line ({SIGNAL_EMA_WINDOW}): {latest_signal_line:.4f}")
            if not np.isnan(latest_macd_histogram): print(f"  MACD Histogram: {latest_macd_histogram:.4f}")
            if not np.isnan(latest_bb_middle):
                print(f"  Bollinger Bands ({BB_WINDOW},{BB_STD_DEV_MULTIPLIER} StdDev):")
                print(f"    Middle Band: {latest_bb_middle:.4f} {vs_currency.upper()}")
                print(f"    Upper Band: {latest_bb_upper:.4f} {vs_currency.upper()}")
                print(f"    Lower Band: {latest_bb_lower:.4f} {vs_currency.upper()}")
            if not np.isnan(latest_stoch_k): print(f"  Stochastic Oscillator ({STOCH_K_WINDOW},{STOCH_D_WINDOW}): %K={latest_stoch_k:.2f}, %D={latest_stoch_d:.2f}")
            if not np.isnan(latest_atr): print(f"  Average True Range ({ATR_WINDOW} periods): {latest_atr:.4f} {vs_currency.upper()}")
            if not np.isnan(latest_tenkan_sen):
                print(f"  Ichimoku Cloud:")
                print(f"    Tenkan-sen ({TENKAN_SEN_WINDOW}): {latest_tenkan_sen:.4f}")
                print(f"    Kijun-sen ({KIJUN_SEN_WINDOW}): {latest_kijun_sen:.4f}")
                print(f"    Senkou Span A: {latest_senkou_span_a:.4f}")
                print(f"    Senkou Span B ({SENKOU_SPAN_B_WINDOW}): {latest_senkou_span_b:.4f}")
                print(f"    Chikou Span (Lag {CHIKOU_SPAN_LAG}): {latest_chikou_span:.4f}")
            
            if not np.isnan(predicted_next_close):
                print(f"  ML Predicted Next Close ({prediction_timeframe_description}): {predicted_next_close:.4f} {vs_currency.upper()}")
            else:
                print(f"  ML Predicted Next Close: N/A (Not enough data for prediction)")
            
            print(f"  Recommendation: \n{recommendation_text}")
            print("-" * 70)

            # Store data for CSV
            all_reports_data.append({
                "Coin ID": coin_id,
                "VS Currency": vs_currency.upper(),
                "Latest Date/Time": latest_date_obj.strftime('%Y-%m-%d %H:%M'),
                "Latest Price (Close)": f"{latest_close:.4f}",
                f"Short SMA ({SHORT_SMA_WINDOW} periods)": f"{latest_short_sma:.4f}",
                f"Long SMA ({LONG_SMA_WINDOW} periods)": f"{latest_long_sma:.4f}",
                f"RSI ({RSI_WINDOW} periods)": f"{latest_rsi:.2f}",
                f"MACD Line ({FAST_EMA_WINDOW},{SLOW_EMA_WINDOW})": f"{latest_macd_line:.4f}",
                f"Signal Line ({SIGNAL_EMA_WINDOW})": f"{latest_signal_line:.4f}",
                "MACD Histogram": f"{latest_macd_histogram:.4f}",
                f"Bollinger Bands Middle ({BB_WINDOW},{BB_STD_DEV_MULTIPLIER} StdDev)": f"{latest_bb_middle:.4f}",
                f"Bollinger Bands Upper ({BB_WINDOW},{BB_STD_DEV_MULTIPLIER} StdDev)": f"{latest_bb_upper:.4f}",
                f"Bollinger Bands Lower ({BB_WINDOW},{BB_STD_DEV_MULTIPLIER} StdDev)": f"{latest_bb_lower:.4f}",
                f"Stochastic %K ({STOCH_K_WINDOW},{STOCH_D_WINDOW})": f"{latest_stoch_k:.2f}",
                f"Stochastic %D ({STOCH_K_WINDOW},{STOCH_D_WINDOW})": f"{latest_stoch_d:.2f}",
                f"ATR ({ATR_WINDOW} periods)": f"{latest_atr:.4f}",
                f"Ichimoku Tenkan-sen ({TENKAN_SEN_WINDOW})": f"{latest_tenkan_sen:.4f}",
                f"Ichimoku Kijun-sen ({KIJUN_SEN_WINDOW})": f"{latest_kijun_sen:.4f}",
                "Ichimoku Senkou Span A": f"{latest_senkou_span_a:.4f}",
                f"Ichimoku Senkou Span B ({SENKOU_SPAN_B_WINDOW})": f"{latest_senkou_span_b:.4f}",
                f"Ichimoku Chikou Span (Lag {CHIKOU_SPAN_LAG})": f"{latest_chikou_span:.4f}" if not np.isnan(latest_chikou_span) else "N/A",
                "ML Predicted Next Close": f"{predicted_next_close:.4f}" if not np.isnan(predicted_next_close) else "N/A",
                "Confidence (%)": f"{confidence_percent:.1f}", # Added Confidence to CSV
                "Overall Recommendation": recommendation_text.replace('\033[92m', '').replace('\033[91m', '').replace('\033[93m', '').replace('\033[0m', '').replace('\n', ' ')
            })


        print("\n" + "="*80)
        print(f"Next update in {UPDATE_INTERVAL_SECONDS} seconds...")
        print(f"Disclaimer: This is for informational purposes only and not financial advice. Markets are highly volatile. ML predictions are experimental and should not be solely relied upon for trading decisions.")
        print("="*80)

        save_option = input("\nDo you want to save this report to a CSV file? (yes/no): ").strip().lower()
        if save_option == 'yes':
            save_report_to_csv(all_reports_data, vs_currency)
        else:
            print("Report not saved.")

        time.sleep(UPDATE_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
