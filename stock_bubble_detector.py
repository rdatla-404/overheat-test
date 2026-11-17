# =============================================================
# Bubble scoring is now driven by an Isolation Forest anomaly
# detection model (see train_bubble_model.py) instead of hand-
# tuned point thresholds. The model is trained offline across a
# broad universe of tickers and loaded here for inference.

import os
import time
import logging
from typing import Optional

import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bubble_model.pkl')

# Feature order must match train_bubble_model.py exactly.
FEATURE_NAMES = ['rsi', 'ma_ratio', 'volatility', 'accel', 'max_drawdown']


# -------------------------------------------------------------
# SECTION 1: DATA LOADING
# -------------------------------------------------------------

def _clean_prices(data):
    """
    Drop rows Yahoo sometimes returns with a blank/invalid Close (typically today's
    unfinished row). Without this the latest price, 200-day average and features are NaN.
    """
    if data is None or data.empty or 'Close' not in data.columns:
        return None
    data = data.copy()
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data = data[np.isfinite(data['Close']) & (data['Close'] > 0)]
    data = data[~data.index.duplicated(keep='last')].sort_index()
    return data if not data.empty else None


def load_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
    max_retries: int = 3,
    retry_backoff_seconds: float = 2.0,
) -> Optional[pd.DataFrame]:
    """
    Download historical daily price data from Yahoo Finance.

    Retries with exponential backoff on transient failures (Yahoo
    Finance rate-limits or hiccups often enough that a bare single
    attempt silently drops tickers, which matters most during
    training when we're hitting it 60+ times in a row).

    Parameters:
        ticker                 : str   -- Stock symbol, e.g. 'AAPL'
        start_date              : str   -- Start date in 'YYYY-MM-DD' format
        end_date                : str   -- End date in 'YYYY-MM-DD' format
        max_retries              : int   -- Attempts before giving up (default 3)
        retry_backoff_seconds    : float -- Base delay between retries, doubles
                                             each attempt (default 2.0s)

    Returns:
        pandas DataFrame with columns [Open, High, Low, Close, Volume],
        or None if the download failed or the ticker was invalid.
    """
    ticker = ticker.upper().strip()
    logger.info(f"Downloading data for {ticker} ...")

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(start=start_date, end=end_date)
            data = _clean_prices(data)

            if data is None or data.empty:
                logger.warning(
                    f"No data returned for '{ticker}'. "
                    "Check that the ticker symbol is correct."
                )
                return None

            logger.info(
                f"OK -- {ticker}: {len(data)} trading days loaded "
                f"({start_date} to {end_date})"
            )
            return data

        except Exception as error:
            last_error = error
            if attempt < max_retries:
                delay = retry_backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    f"Attempt {attempt}/{max_retries} failed for {ticker} "
                    f"({error}). Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)

    logger.error(f"Giving up on {ticker} after {max_retries} attempts: {last_error}")
    return None


# -------------------------------------------------------------
# SECTION 2: METRIC CALCULATIONS
# -------------------------------------------------------------

def get_current_price(data: pd.DataFrame) -> float:
    """
    Return the most recent closing price.

    Parameters:
        data : DataFrame -- Stock price data from load_stock_data()

    Returns:
        float -- Latest closing price rounded to 2 decimal places.
    """
    return round(float(data['Close'].iloc[-1]), 2)


def get_total_return(data: pd.DataFrame) -> float:
    """
    Calculate the percentage gain/loss from first to last price.

    Formula: ((end_price - start_price) / start_price) * 100

    Example: Bought at $100, now $130 -> total return = +30.0%

    Returns:
        float -- Total return as a percentage (e.g. 30.0 means +30%).
    """
    if len(data) < 2:
        return 0.0

    start_price = float(data['Close'].iloc[0])
    end_price   = float(data['Close'].iloc[-1])
    return round(((end_price - start_price) / start_price) * 100, 2)


def get_daily_returns(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the percentage price change for each trading day.

    Example: Price goes $100 -> $103, daily return = +3.0%

    Returns:
        pandas Series of daily return values (decimal, not percent).
    """
    return data['Close'].pct_change().dropna()


def get_volatility(data: pd.DataFrame) -> float:
    """
    Calculate annualized volatility -- a standard measure of price risk.

    High volatility means prices swing wildly (risky).
    Low volatility means prices are stable.

    We annualize by multiplying the daily standard deviation by
    sqrt(252), because there are 252 trading days per year.

    Returns:
        float -- Annualized volatility as a percentage (e.g. 35.0 = 35%).
    """
    daily_returns = get_daily_returns(data)

    if len(daily_returns) == 0:
        return 0.0

    # std() = standard deviation -- measures how spread out returns are
    daily_std  = daily_returns.std()
    annual_std = daily_std * (252 ** 0.5)
    return round(float(annual_std * 100), 2)


def get_moving_average(data: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculate a simple moving average over a rolling window of days.

    A moving average smooths out short-term noise to reveal the trend.
    Example: The 50-day MA is the average of the last 50 closing prices,
    recalculated each day.

    Parameters:
        data   : DataFrame -- Stock price data
        window : int       -- Number of days in the rolling window (e.g. 50)

    Returns:
        pandas Series of moving average values.
    """
    return data['Close'].rolling(window=window).mean()


def get_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI).

    RSI is a momentum indicator scaled from 0 to 100:
        RSI > 70  -- Overbought (prices rose too fast, potential bubble)
        RSI < 30  -- Oversold  (prices fell too fast, potential bargain)
        RSI 40-60 -- Normal range

    Parameters:
        data   : DataFrame -- Stock price data
        period : int       -- Lookback period in days (default 14)

    Returns:
        pandas Series of RSI values.
    """
    price_change = data['Close'].diff()

    # Separate positive days (gains) from negative days (losses)
    gains  = price_change.copy()
    losses = price_change.copy()
    gains[gains < 0]   = 0        # Zero out losses in the gains series
    losses[losses > 0] = 0        # Zero out gains in the losses series
    losses = abs(losses)          # Make losses positive for the formula

    # Rolling average of gains and losses over the lookback period
    avg_gain = gains.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = losses.ewm(com=period - 1, min_periods=period).mean()

    # Guard against division by zero
    avg_loss = avg_loss.replace(0, 0.0001)

    # RSI formula
    rs  = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def get_price_to_200ma_ratio(data: pd.DataFrame) -> Optional[pd.Series]:
    """
    Calculate how far the current price is above the 200-day moving average.

    This ratio is a widely used bubble indicator:
        1.0  -- Price equals the 200-day MA (normal)
        1.30 -- Price is 30% above the 200-day MA (warning)
        1.50 -- Price is 50% above the 200-day MA (danger)

    During the dot-com bubble, many tech stocks were 3x-5x above their
    200-day MAs just before they collapsed.

    Returns:
        pandas Series of ratio values, or None if not enough data.
    """
    if len(data) < 200:
        logger.info("Less than 200 days of data -- MA ratio unavailable.")
        return None

    ma_200 = get_moving_average(data, 200)
    return data['Close'] / ma_200


def get_max_drawdown(data: pd.DataFrame) -> float:
    """
    Calculate the maximum drawdown -- the largest peak-to-trough decline.

    Example: Stock peaks at $200 then falls to $120 -> drawdown = -40%.

    This tells you the worst-case loss you would have suffered if you
    bought at the peak and sold at the trough.

    Returns:
        float -- Maximum drawdown as a negative percentage (e.g. -40.0).
    """
    prices      = data['Close']
    rolling_max = prices.cummax()              # Highest price seen so far
    drawdown    = (prices - rolling_max) / rolling_max * 100
    return round(float(drawdown.min()), 2)     # Most negative value


# -------------------------------------------------------------
# SECTION 3: BUBBLE DETECTION (Isolation Forest anomaly model)
# -------------------------------------------------------------
# Rather than hand-picked point thresholds, the bubble score now
# comes from an Isolation Forest trained on feature snapshots
# pulled from a broad universe of tickers (train_bubble_model.py).
# The model flags feature combinations that are statistically
# unusual versus typical market behavior. A directional gate
# below ensures we only escalate risk when the anomaly points
# toward "overheating" (elevated RSI/MA ratio/acceleration), not
# toward a crash or otherwise-unusual-but-cheap stock.

_MODEL_BUNDLE = None  # lazy-loaded cache


def get_model_info():
    """
    Return metadata about the currently loaded model, or None if no
    trained model exists yet. Used by the Streamlit app to show
    model status in the sidebar.
    """
    bundle = _load_model_bundle()
    if bundle is None:
        return None
    return {
        'trained_on': bundle.get('trained_on', 'unknown'),
        'n_training_rows': bundle.get('n_training_rows', 'unknown'),
        'ticker_universe_size': bundle.get('ticker_universe_size', 'unknown'),
    }


def reload_model() -> Optional[dict]:
    """
    Force a fresh read of bubble_model.pkl from disk, discarding any
    cached bundle. Call this after retraining so the new model takes
    effect without restarting the process (e.g. after the app's
    "Train / retrain model" button finishes).

    Returns:
        The freshly loaded model bundle dict, or None if the file
        doesn't exist.
    """
    global _MODEL_BUNDLE
    _MODEL_BUNDLE = None
    return _load_model_bundle()


def _load_model_bundle():
    """
    Load the trained model bundle from disk (cached after first call).

    Returns:
        dict with keys: model, scaler, feature_names, score_low, score_high
        or None if bubble_model.pkl has not been trained yet.
    """
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is not None:
        return _MODEL_BUNDLE

    if not os.path.exists(MODEL_PATH):
        return None

    _MODEL_BUNDLE = joblib.load(MODEL_PATH)
    return _MODEL_BUNDLE


def compute_feature_vector(data: pd.DataFrame) -> Optional[dict]:
    """
    Build the continuous feature snapshot fed to the ML model.

    Unlike the old rule-based checks, these are raw continuous
    values -- no thresholds baked in here. The model learns what
    "normal" combinations of these look like from training data.

    Features:
        rsi          -- current 14-day RSI (0-100)
        ma_ratio     -- current price / 200-day moving average
        volatility   -- annualized volatility over the trailing 90 days
        accel        -- (last 30-day return %) - (prior 30-day return %)
        max_drawdown -- max drawdown (%) over the trailing 252 days

    Parameters:
        data : DataFrame -- Stock price data (needs >= 200 trading days)

    Returns:
        dict of {feature_name: float}, or None if there isn't enough
        history to compute all five features.
    """
    if len(data) < 200:
        return None

    ma_ratio_series = get_price_to_200ma_ratio(data)
    if ma_ratio_series is None or pd.isna(ma_ratio_series.iloc[-1]):
        return None

    rsi_val = float(get_rsi(data).iloc[-1])
    ma_ratio_val = float(ma_ratio_series.iloc[-1])
    vol_val = get_volatility(data.tail(90))
    dd_val = get_max_drawdown(data.tail(252))

    prices = data['Close']
    if len(data) >= 60:
        recent = prices.tail(30)
        prior  = prices.tail(60).head(30)
        recent_return = ((float(recent.iloc[-1]) - float(recent.iloc[0]))
                          / float(recent.iloc[0])) * 100
        prior_return  = ((float(prior.iloc[-1]) - float(prior.iloc[0]))
                          / float(prior.iloc[0])) * 100
        accel_val = recent_return - prior_return
    else:
        accel_val = 0.0

    if any(pd.isna(v) for v in (rsi_val, ma_ratio_val, vol_val, dd_val, accel_val)):
        return None

    return {
        'rsi': rsi_val,
        'ma_ratio': ma_ratio_val,
        'volatility': vol_val,
        'accel': accel_val,
        'max_drawdown': dd_val,
    }


def assign_risk_level(bubble_score: float) -> str:
    """
    Convert a numeric bubble score (0-100) to a text risk label.

    Parameters:
        bubble_score : float -- Overall bubble score

    Returns:
        str -- Human-readable risk level.
    """
    if bubble_score >= 70:
        return "CRITICAL BUBBLE RISK"
    elif bubble_score >= 50:
        return "HIGH BUBBLE RISK"
    elif bubble_score >= 30:
        return "MODERATE BUBBLE RISK"
    elif bubble_score >= 10:
        return "LOW BUBBLE RISK"
    else:
        return "HEALTHY -- No Bubble Detected"


def run_bubble_analysis(ticker: str, data: pd.DataFrame) -> dict:
    """
    Score bubble risk using the trained Isolation Forest model.

    Steps:
        1. Compute the 5-feature snapshot for this ticker today.
        2. Scale it using the training-set scaler.
        3. Run it through the Isolation Forest to get an anomaly score.
        4. Normalize that score to 0-100 using the training set's
           score distribution (1st/99th percentile).
        5. Apply a directional gate: only let the score run high if
           the anomaly is specifically in the "overheating" direction
           (elevated RSI, MA ratio, acceleration) rather than a crash
           or an unrelated kind of statistical outlier.

    Parameters:
        ticker : str       -- Stock symbol (used for display only)
        data   : DataFrame -- Stock price data from load_stock_data()

    Returns:
        dict with keys: ticker, bubble_score, risk_level,
                        warnings, sub_scores (z-scores per feature)
    """
    logger.info(f"Running ML bubble analysis on {ticker} ...")

    bundle = _load_model_bundle()
    if bundle is None:
        logger.warning("bubble_model.pkl not found. Run train_bubble_model.py first.")
        return {
            'ticker': ticker,
            'bubble_score': None,
            'risk_level': "MODEL NOT TRAINED",
            'warnings': ["No trained model found -- run train_bubble_model.py."],
            'sub_scores': {},
        }

    features = compute_feature_vector(data)
    if features is None:
        logger.info("Not enough history to compute the full feature set.")
        return {
            'ticker': ticker,
            'bubble_score': None,
            'risk_level': "INSUFFICIENT DATA",
            'warnings': ["Need at least 200 trading days of history."],
            'sub_scores': {},
        }

    model, scaler = bundle['model'], bundle['scaler']
    score_low, score_high = bundle['score_low'], bundle['score_high']

    x = np.array([[features[name] for name in FEATURE_NAMES]])
    x_scaled = scaler.transform(x)

    raw_score = float(model.score_samples(x_scaled)[0])
    # score_samples: higher = more normal, lower = more anomalous.
    # Flip and rescale to 0-100 using the training distribution's range.
    bubble_score = 100 * (score_high - raw_score) / (score_high - score_low)
    bubble_score = float(np.clip(bubble_score, 0, 100))

    # Per-feature z-scores relative to the training population --
    # used both for the directional gate and for human-readable warnings.
    z = {
        name: (features[name] - scaler.mean_[i]) / scaler.scale_[i]
        for i, name in enumerate(FEATURE_NAMES)
    }

    # Directional gate: "bubble" means overheating, not just "unusual."
    # Drawdown is excluded here -- a large recent drawdown argues against
    # a live bubble, it doesn't confirm one.
    direction_signal = z['rsi'] + z['ma_ratio'] + z['accel']
    # (The old cap that forced scores to <= 15 when direction_signal <= 0 was removed:
    # it pinned most stocks at 15 and made ranking worse in backtests.)

    risk_level = assign_risk_level(bubble_score)

    warnings_out = []
    if z['rsi'] > 2 and direction_signal > 0:
        warnings_out.append(
            f"RSI unusually elevated: {features['rsi']:.1f} "
            f"(z-score {z['rsi']:.2f} vs. training population)"
        )
    if z['ma_ratio'] > 2 and direction_signal > 0:
        pct_above = (features['ma_ratio'] - 1) * 100
        warnings_out.append(
            f"Price unusually extended above 200-day MA: {pct_above:.1f}% above "
            f"(z-score {z['ma_ratio']:.2f})"
        )
    if z['accel'] > 2 and direction_signal > 0:
        warnings_out.append(
            f"Unusually parabolic move: 30-day return accelerated by "
            f"{features['accel']:.1f} percentage points vs. the prior 30 days "
            f"(z-score {z['accel']:.2f})"
        )
    if z['volatility'] > 2:
        warnings_out.append(
            f"Volatility unusually high: {features['volatility']:.1f}% annualized "
            f"(z-score {z['volatility']:.2f})"
        )
    if not warnings_out:
        warnings_out.append("No feature stood out as statistically unusual.")

    return {
        'ticker': ticker,
        'bubble_score': round(bubble_score, 1),
        'risk_level': risk_level,
        'warnings': warnings_out,
        'sub_scores': {name: round(z[name], 2) for name in FEATURE_NAMES},
    }


# -------------------------------------------------------------
# SECTION 4: AVOIDANCE STRATEGIES
# -------------------------------------------------------------

def generate_strategies(ticker, current_price, bubble_score, ma_200_value=None):
    """
    Return a list of actionable strategies based on the bubble score.

    Each strategy is a dict with:
        priority : str -- 'URGENT', 'HIGH', 'MEDIUM', 'LOW', or 'INFO'
        action   : str -- Short action headline
        detail   : str -- Longer explanation

    Parameters:
        ticker        : str   -- Stock ticker symbol
        current_price : float -- Most recent closing price
        bubble_score  : int   -- Overall bubble score (0-100)

    Returns:
        list of strategy dicts.
    """
    strategies = []

    if bubble_score >= 70:
        strategies = [
            {
                'priority': 'URGENT',
                'action':   'REDUCE POSITION BY 50-75%',
                'detail':   (f"Sell 50-75% of your {ticker} shares now to lock "
                             f"in gains. Current price: ${current_price}.")
            },
            {
                'priority': 'URGENT',
                'action':   f'SET STOP-LOSS AT ${current_price * 0.90:.2f}',
                'detail':   (f"Place an automatic sell order at "
                             f"${current_price * 0.90:.2f} (10% below current price). "
                             f"This caps your worst-case downside.")
            },
            {
                'priority': 'HIGH',
                'action':   'HEDGE WITH PUT OPTIONS',
                'detail':   (f"Buy put options on {ticker} as insurance. "
                             f"A put gains value if the stock drops -- "
                             f"a standard hedging technique at risk desks.")
            },
            {
                'priority': 'HIGH',
                'action':   'ROTATE INTO DEFENSIVE SECTORS',
                'detail':   ("Move proceeds into utilities (XLU), healthcare (XLV), "
                             "or consumer staples (XLP) -- sectors that historically "
                             "hold up better during market crashes.")
            },
            {
                'priority': 'MEDIUM',
                'action':   f'PLAN RE-ENTRY NEAR ${ma_200_value:.2f}' if ma_200_value else 'WAIT FOR 200-DAY MA TO FORM',
                'detail':   (f"Bubbles often mean-revert to the 200-day MA. "
                            f"Current 200-day MA: ${ma_200_value:.2f}." if ma_200_value
                             else "Insufficient history to calculate 200-day MA re-entry level.")
            },
            {
                'priority': 'MEDIUM',
                'action':   'RAISE CASH RESERVE TO 20%',
                'detail':   ("Holding 20% cash lets you buy quality assets cheaply "
                             "after the bubble bursts.")
            },
        ]

    elif bubble_score >= 50:
        strategies = [
            {
                'priority': 'HIGH',
                'action':   'TRIM POSITION BY 25-40%',
                'detail':   (f"Sell 25-40% of {ticker} to reduce exposure and "
                             f"book partial profits at ${current_price}.")
            },
            {
                'priority': 'HIGH',
                'action':   'SET 15% TRAILING STOP-LOSS',
                'detail':   (f"A trailing stop follows the price upward, triggering "
                             f"a sell if it drops 15% from peak. "
                             f"Currently that would be ${current_price * 0.85:.2f}.")
            },
            {
                'priority': 'MEDIUM',
                'action':   'CAP POSITION AT 10% OF PORTFOLIO',
                'detail':   ("No single stock should exceed 10% of your total "
                             "portfolio. Concentration in a bubble stock magnifies loss.")
            },
            {
                'priority': 'MEDIUM',
                'action':   'MONITOR RSI WEEKLY',
                'detail':   ("If RSI crosses 80, escalate to the CRITICAL protocol. "
                             "Set a weekly reminder to rerun this analysis.")
            },
            {
                'priority': 'LOW',
                'action':   'DO NOT ADD TO POSITION',
                'detail':   (f"Avoid buying more {ticker} until the bubble score "
                             f"drops below 25. Never average up into a bubble.")
            },
        ]

    elif bubble_score >= 30:
        strategies = [
            {
                'priority': 'MEDIUM',
                'action':   'MONITOR WEEKLY',
                'detail':   (f"{ticker} shows early warning signs. Not critical yet, "
                             f"but requires weekly review.")
            },
            {
                'priority': 'MEDIUM',
                'action':   f'MENTAL STOP-LOSS AT ${current_price * 0.88:.2f}',
                'detail':   (f"If {ticker} falls below ${current_price * 0.88:.2f} "
                             f"(12% below current), treat it as a sell signal.")
            },
            {
                'priority': 'LOW',
                'action':   'HOLD -- DO NOT ADD YET',
                'detail':   ("Hold your current position but do not add more until "
                             "the bubble score falls back below 20.")
            },
            {
                'priority': 'LOW',
                'action':   'KEEP PORTFOLIO DIVERSIFIED',
                'detail':   ("Hold stocks across at least 5 different sectors so that "
                             "one sector crash does not devastate the whole portfolio.")
            },
        ]

    else:
        strategies = [
            {
                'priority': 'LOW',
                'action':   'HOLD OR ACCUMULATE',
                'detail':   (f"{ticker} shows no significant bubble signals. "
                             f"Normal buy-and-hold or dollar-cost averaging applies.")
            },
            {
                'priority': 'LOW',
                'action':   'DOLLAR-COST AVERAGE (DCA)',
                'detail':   ("Invest a fixed amount on a regular schedule regardless "
                             "of price. This reduces the impact of bad timing.")
            },
            {
                'priority': 'INFO',
                'action':   'RERUN THIS ANALYSIS MONTHLY',
                'detail':   ("Conditions change. Schedule a monthly check to catch "
                             "developing bubble signals early.")
            },
        ]

    return strategies


# -------------------------------------------------------------
# SECTION 5: PRINTING / REPORTING
# -------------------------------------------------------------

def print_stock_summary(ticker, data):
    """
    Print a clean summary of key metrics for a single stock.

    Parameters:
        ticker : str       -- Stock symbol
        data   : DataFrame -- Stock price data
    """
    price    = get_current_price(data)
    ret      = get_total_return(data)
    vol      = get_volatility(data)
    max_dd   = get_max_drawdown(data)
    rsi_val  = float(get_rsi(data).iloc[-1])

    ratio_series = get_price_to_200ma_ratio(data)
    ratio_str = (f"{float(ratio_series.iloc[-1]):.2f}x"
                 if ratio_series is not None else "N/A (< 200 days)")

    print(f"\n  {'=' * 45}")
    print(f"  STOCK SUMMARY: {ticker}")
    print(f"  {'=' * 45}")
    print(f"  Current Price       : ${price}")
    print(f"  Total Return        : {ret:+.1f}%")
    print(f"  Annualized Volatility: {vol:.1f}%")
    print(f"  Max Drawdown        : {max_dd:.1f}%")
    print(f"  Current RSI (14-day): {rsi_val:.1f}")
    print(f"  Price / 200-day MA  : {ratio_str}")
    print(f"  {'=' * 45}")


def print_bubble_report(result):
    """
    Print the full bubble analysis report to the console.

    Parameters:
        result : dict -- Output from run_bubble_analysis()
    """
    print(f"\n  {'=' * 50}")
    print(f"  BUBBLE REPORT: {result['ticker']}")
    print(f"  {'=' * 50}")

    if result['bubble_score'] is None:
        print(f"  Status: {result['risk_level']}")
        for warning in result['warnings']:
            print(f"    [!]  {warning}")
        print(f"  {'=' * 50}")
        return

    print(f"  Bubble Score : {result['bubble_score']} / 100  (ML anomaly score)")
    print(f"  Risk Level   : {result['risk_level']}")
    print(f"\n  Feature Z-Scores (vs. training population, |z| > 2 is unusual):")

    labels = {
        'rsi': 'RSI',
        'ma_ratio': 'Price / 200-day MA',
        'volatility': 'Volatility',
        'accel': 'Acceleration',
        'max_drawdown': 'Max Drawdown',
    }
    for name, z_val in result['sub_scores'].items():
        bar_len = int(min(abs(z_val), 4) / 4 * 20)
        bar = ('#' * bar_len).ljust(20, '-')
        sign = '+' if z_val >= 0 else '-'
        print(f"    {labels.get(name, name):<20} z={z_val:>+6.2f}  [{bar}]")

    print(f"\n  Warning Flags ({len(result['warnings'])} found):")
    for warning in result['warnings']:
        print(f"    [!]  {warning}")

    print(f"  {'=' * 50}")


def print_strategies(ticker, bubble_score, strategies):
    """
    Print all recommended avoidance strategies to the console.

    Parameters:
        ticker       : str  -- Stock symbol
        bubble_score : int  -- Overall bubble score
        strategies   : list -- Output from generate_strategies()
    """
    priority_labels = {
        'URGENT': '[URGENT]',
        'HIGH':   '[HIGH  ]',
        'MEDIUM': '[MEDIUM]',
        'LOW':    '[LOW   ]',
        'INFO':   '[INFO  ]',
    }

    print(f"\n  {'=' * 50}")
    print(f"  AVOIDANCE STRATEGIES: {ticker}")
    print(f"  Bubble Score: {bubble_score} / 100")
    print(f"  {'=' * 50}")

    for i, strat in enumerate(strategies, start=1):
        label = priority_labels.get(strat['priority'], '[     ]')
        print(f"\n  {i}. {label} {strat['action']}")

        # Word-wrap the detail text at 60 characters
        words = strat['detail'].split()
        line  = "     "
        for word in words:
            if len(line) + len(word) > 62:
                print(line)
                line = "     " + word + " "
            else:
                line += word + " "
        if line.strip():
            print(line)

    print(f"\n  {'=' * 50}")


# -------------------------------------------------------------
# SECTION 6: CHARTING
# -------------------------------------------------------------

def plot_stock_detail(ticker, data, show=True, save=True):
    """
    Produce a two-panel chart for one stock:
        Top panel    : Closing price with 50-day and 200-day moving averages.
                       Red shading marks where price is >30% above 200-day MA.
        Bottom panel : RSI with overbought (70) and oversold (30) zones.

    By default this saves a .png and displays it (CLI usage). Pass
    show=False, save=False to just get the Figure object back for
    embedding elsewhere (e.g. the Streamlit app calls it this way).

    Parameters:
        ticker : str       -- Stock symbol (used for title and filename)
        data   : DataFrame -- Stock price data
        show   : bool      -- Call plt.show() (default True)
        save   : bool      -- Save a .png to disk (default True)

    Returns:
        matplotlib.figure.Figure
    """
    # gridspec_kw sets the height ratio: price panel is 3x taller than RSI panel
    fig, (ax_price, ax_rsi) = plt.subplots(
        2, 1, figsize=(14, 9),
        gridspec_kw={'height_ratios': [3, 1]},
        sharex=True
    )
    fig.suptitle(f'{ticker} -- Price and RSI Analysis',
                 fontsize=16, fontweight='bold')

    # ---- Top panel: price and moving averages ----
    prices = data['Close']
    ma_50  = get_moving_average(data, 50)
    ma_200 = get_moving_average(data, 200)

    ax_price.plot(prices.index, prices, color='#1565C0',
                  linewidth=1.2, label='Close Price')
    ax_price.plot(ma_50.index, ma_50, color='#FF9800',
                  linewidth=1.5, linestyle='--', label='50-Day MA')
    ax_price.plot(ma_200.index, ma_200, color='#F44336',
                  linewidth=1.8, linestyle='-.', label='200-Day MA')

    # Shade zone where price is more than 30% above the 200-day MA.
    # This is a display convention only -- actual scoring is model-driven.
    CHART_MA_RATIO_FLAG = 1.30
    ratio = get_price_to_200ma_ratio(data)
    if ratio is not None:
        bubble_zone = ratio > CHART_MA_RATIO_FLAG
        ax_price.fill_between(
            prices.index, prices, ma_200,
            where=(bubble_zone & prices.notna() & ma_200.notna()),
            alpha=0.12, color='red',
            label=f'>{int((CHART_MA_RATIO_FLAG - 1)*100)}% above 200MA'
        )

    ax_price.set_ylabel('Price (USD)', fontsize=12)
    ax_price.legend(loc='upper left', fontsize=10)

    # ---- Bottom panel: RSI ----
    rsi = get_rsi(data)

    ax_rsi.plot(rsi.index, rsi, color='#7B1FA2',
                linewidth=1.5, label='RSI (14-day)')
    ax_rsi.axhline(70, color='#F44336', linestyle='--',
                   linewidth=1.2, label='Overbought (70)')
    ax_rsi.axhline(30, color='#388E3C', linestyle='--',
                   linewidth=1.2, label='Oversold (30)')
    ax_rsi.axhline(50, color='gray', linestyle=':', linewidth=0.8)

    ax_rsi.fill_between(rsi.index, 70, rsi, where=(rsi > 70),
                        alpha=0.25, color='#F44336')
    ax_rsi.fill_between(rsi.index, 30, rsi, where=(rsi < 30),
                        alpha=0.25, color='#388E3C')

    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel('RSI', fontsize=12)
    ax_rsi.set_xlabel('Date', fontsize=12)
    ax_rsi.legend(loc='upper left', fontsize=9)

    ax_rsi.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax_rsi.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax_rsi.xaxis.get_majorticklabels(), rotation=45, ha='right')

    filename = f'{ticker}_analysis.png'
    plt.tight_layout()
    if save:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n  Chart saved: {filename}")
    if show:
        plt.show()

    return fig


# -------------------------------------------------------------
# SECTION 7: USER INPUT AND MAIN LOOP
# -------------------------------------------------------------

def get_user_input():
    """
    Prompt the user to enter a ticker symbol and analysis period.

    Validates that:
        - The ticker is not blank.
        - The number of years is a positive integer.

    Returns:
        ticker     : str -- Uppercased stock symbol
        years_back : int -- Number of years of history to pull
    """
    print("\n  -----------------------------------------------")
    print("  STOCK BUBBLE ANALYZER -- INPUT")
    print("  -----------------------------------------------")

    # Get ticker symbol
    while True:
        ticker = input("  Enter stock ticker symbol (e.g. AAPL, TSLA, NVDA): ").strip()
        if ticker:
            ticker = ticker.upper()
            break
        print("  Ticker cannot be blank. Please try again.")

    # Get analysis period
    while True:
        years_input = input("  How many years of data to analyze? (1-5, default 2): ").strip()

        if years_input == "":
            years_back = 2
            break

        if years_input.isdigit():
            years_back = int(years_input)
            if 1 <= years_back <= 5:
                break
            else:
                print("  Please enter a number between 1 and 5.")
        else:
            print("  Invalid input. Please enter a whole number.")

    return ticker, years_back


def ask_another():
    """
    Ask the user if they want to analyze another stock.

    Returns:
        bool -- True if the user wants to continue, False to exit.
    """
    while True:
        answer = input("\n  Analyze another stock? (yes / no): ").strip().lower()
        if answer in ('yes', 'y'):
            return True
        elif answer in ('no', 'n'):
            return False
        else:
            print("  Please type 'yes' or 'no'.")


def main():
    """
    Main entry point. Runs an interactive loop that:
        1. Prompts the user for a ticker and date range.
        2. Downloads the stock data.
        3. Prints a metric summary.
        4. Runs the bubble analysis and prints the report.
        5. Prints avoidance strategies scaled to the risk level.
        6. Plots the price/RSI chart.
        7. Asks whether to analyze another stock.
    """
    print("\n" + "=" * 55)
    print("    STOCK BUBBLE DETECTOR")
    print("    Powered by Yahoo Finance and Python")
    print("=" * 55)

    while True:
        # Step 1: Collect user input
        ticker, years_back = get_user_input()

        # Step 2: Calculate the date range
        end_date   = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today()
                      - timedelta(days=365 * years_back)).strftime('%Y-%m-%d')

        print(f"\n  Fetching {years_back} year(s) of data "
              f"({start_date} to {end_date}) ...")

        # Step 3: Download data
        data = load_stock_data(ticker, start_date, end_date)

        if data is None:
            print("  Could not load data. Please check the ticker and try again.")
            if not ask_another():
                break
            continue

        # Step 4: Print metric summary
        print_stock_summary(ticker, data)

        # Step 5: Run bubble analysis and print report
        result = run_bubble_analysis(ticker, data)
        print_bubble_report(result)

        # Step 6: Generate and print avoidance strategies
        if result['bubble_score'] is not None:
            current_price = get_current_price(data)
            ma_200_value  = float(get_moving_average(data, 200).iloc[-1]) if len(data) >= 200 else None
            strategies    = generate_strategies(ticker, current_price, result['bubble_score'], ma_200_value)
            print_strategies(ticker, result['bubble_score'], strategies)

        # Step 7: Plot the detail chart
        show_chart = input("\n  Show price and RSI chart? (yes / no): ").strip().lower()
        if show_chart in ('yes', 'y'):
            plot_stock_detail(ticker, data)

        # Step 8: Continue or exit
        if not ask_another():
            break

    print("\n  Analysis session ended. Goodbye.")
    print("=" * 55)


# Run the program when executed directly: python stock_analyzer.py
if __name__ == "__main__":
    main()
