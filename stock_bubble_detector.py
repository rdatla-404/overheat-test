# =============================================================
# Overheat Score v2: a crash-risk model trained on outcomes.
# The score comes from a small logistic regression (see
# crash_model.py / train_bubble_model.py) that estimates the chance
# a stock falls 20%+ within ~6 months, using volatility-adjusted
# features. Each trained model carries its own held-out validation
# result so the app can say how far to trust it.

import json
import math
import os
import re
import sys
import time
import logging
from typing import Optional

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings

# Only silence the noisy FutureWarnings from third-party libraries; other warnings
# (which can point at real problems) stay visible.
warnings.filterwarnings('ignore', category=FutureWarning)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# The trained model is stored as plain JSON (numbers and text only), NOT a pickle:
# loading a pickle can run arbitrary code, so a tampered or downloaded model file
# could take over the machine. JSON cannot, and the loader below validates every field.
MODEL_PATH = os.environ.get(
    'OVERHEAT_MODEL_PATH',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bubble_model.json'),
)

# Raw features (v1) and volatility-adjusted features (v2). The shipped model uses
# RISK_MODEL_FEATURES; the full set is computed for analysis and backtests.
BASE_FEATURE_NAMES = ['rsi', 'ma_ratio', 'volatility', 'accel', 'max_drawdown']
FEATURE_NAMES = BASE_FEATURE_NAMES + ['ext_vol', 'accel_vol', 'mom_vol', 'dist_high']
RISK_MODEL_FEATURES = ['volatility', 'rsi', 'ext_vol', 'accel_vol', 'mom_vol', 'dist_high']


# -------------------------------------------------------------
# INPUT VALIDATION (everything typed by a user goes through these)
# -------------------------------------------------------------

# Plain equity / ETF symbols only: letters, digits, '.' and '-', up to 10 characters
# (e.g. AAPL, BRK-B, BF.B). This keeps odd strings out of the Yahoo request, log lines,
# cache file names and the web page.
_TICKER_RE = re.compile(r'[A-Z0-9][A-Z0-9.\-]{0,9}')


def validate_ticker(raw) -> Optional[str]:
    """Return a cleaned, upper-case ticker, or None if it is not a plain symbol."""
    if not isinstance(raw, str):
        return None
    t = raw.strip().upper()
    return t if _TICKER_RE.fullmatch(t) else None


def _parse_date(value) -> Optional[datetime]:
    try:
        return datetime.strptime(value, '%Y-%m-%d')
    except (TypeError, ValueError):
        return None


def _clean_prices(data) -> Optional[pd.DataFrame]:
    """Reject empty or corrupt downloads: need a numeric, finite, positive Close column."""
    if data is None or len(data) == 0 or 'Close' not in data.columns:
        return None
    df = data.copy()
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df[np.isfinite(df['Close'].astype(float)) & (df['Close'] > 0)]
    df = df[~df.index.duplicated(keep='last')].sort_index()
    return df if len(df) else None


# -------------------------------------------------------------
# SECTION 1: DATA LOADING
# -------------------------------------------------------------

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
    clean_ticker = validate_ticker(ticker)
    if clean_ticker is None:
        logger.warning("Rejected an invalid ticker symbol.")
        return None
    ticker = clean_ticker
    d0, d1 = _parse_date(start_date), _parse_date(end_date)
    if d0 is None or d1 is None or d0 >= d1 or d0.year < 1970:
        logger.warning("Rejected an invalid date range.")
        return None
    max_retries = max(1, min(int(max_retries), 5))
    logger.info(f"Downloading data for {ticker} ...")

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            ticker_obj = yf.Ticker(ticker)
            data = _clean_prices(ticker_obj.history(start=start_date, end=end_date, timeout=15))

            if data is None:
                logger.warning(
                    f"No usable data returned for '{ticker}'. "
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
# SECTION 3: CRASH-RISK MODEL (supervised, volatility-adjusted)
# -------------------------------------------------------------
# What changed in v2 and why: the original Isolation Forest scored how UNUSUAL a
# stock looked. Out-of-time tests showed that ranked later crashes worse than plain
# volatility, and that RSI or price-over-200MA alone had no skill. v2 trains a small,
# interpretable logistic regression directly on the outcome people care about
# ("did this stock fall 20%+ within ~6 months?") using volatility-adjusted features,
# with labels that were fully known at training time. Every trained model carries its
# own held-out validation result, and this module says plainly when the model is not
# better than a volatility screen.

_MODEL_BUNDLE = None  # lazy-loaded cache

PLAIN_NAMES = {
    'volatility': 'Price swings (volatility)',
    'rsi': 'Momentum (RSI)',
    'ext_vol': 'Stretch above 200-day average, relative to own volatility',
    'accel_vol': 'Speed-up of recent gains, relative to own volatility',
    'mom_vol': '6-month return, relative to own volatility',
    'dist_high': 'Distance below 52-week high',
}


def get_model_info():
    """
    Metadata about the loaded model, or None if no trained model exists.
    Used by the Streamlit app's sidebar.
    """
    bundle = _load_model_bundle()
    if bundle is None:
        return None
    return {
        'trained_on': bundle.get('trained_on', 'unknown'),
        'n_training_rows': bundle.get('n_training_rows', 'unknown'),
        'ticker_universe_size': bundle.get('ticker_universe_size', 'unknown'),
        'trained_through': bundle.get('trained_through', 'unknown'),
        'base_rate': bundle.get('base_rate'),
        'threshold': bundle.get('threshold'),
        'horizon': bundle.get('horizon'),
        'validation': bundle.get('validation'),
        'validation_status': validation_status(bundle),
        'validation_note': validation_note(bundle),
    }


def reload_model() -> Optional[dict]:
    """Force a fresh read of bubble_model.pkl (call after retraining)."""
    global _MODEL_BUNDLE
    _MODEL_BUNDLE = None
    return _load_model_bundle()


MODEL_SCHEMA = 2
MAX_MODEL_BYTES = 1_000_000


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _float_list(v, n, name):
    if not isinstance(v, list) or len(v) != n or not all(_finite(a) for a in v):
        raise ValueError(f"model field '{name}' is invalid")
    return [float(a) for a in v]


def _short_str(v, name, limit=40):
    if not isinstance(v, str) or len(v) > limit:
        raise ValueError(f"model field '{name}' is invalid")
    return v


def _parse_validation(v):
    if v is None:
        return None
    if not isinstance(v, dict):
        raise ValueError("model field 'validation' is invalid")
    out = {k: _short_str(v.get(k), k) for k in ('cutoff', 'holdout_start', 'holdout_end')}
    for k in ('n_train', 'n_holdout'):
        if not isinstance(v.get(k), int) or isinstance(v.get(k), bool) or v[k] < 0:
            raise ValueError(f"model field '{k}' is invalid")
        out[k] = int(v[k])
    for k in ('auc_model', 'auc_volatility'):
        if not _finite(v.get(k)):
            raise ValueError(f"model field '{k}' is invalid")
        out[k] = float(v[k])
    for k in ('holdout_base_rate', 'auc_rsi', 'auc_ma_ratio'):
        out[k] = float(v[k]) if _finite(v.get(k)) else None
    ci = v.get('ci_vs_volatility')
    out['ci_vs_volatility'] = _float_list(ci, 3, 'ci_vs_volatility') if ci is not None else None
    return out


def parse_bundle(raw) -> dict:
    """
    Validate a model dict that came from an UNTRUSTED source (a file on disk) and return
    a clean copy built only from plain numbers and short strings. Raises ValueError on
    anything unexpected, so a corrupt or tampered file is rejected instead of used.
    """
    if not isinstance(raw, dict) or raw.get('schema') != MODEL_SCHEMA:
        raise ValueError("unsupported model file version")
    cols = raw.get('cols')
    if (not isinstance(cols, list) or not 0 < len(cols) <= 20 or len(set(cols)) != len(cols)
            or not all(isinstance(c, str) and c in FEATURE_NAMES for c in cols)):
        raise ValueError("model field 'cols' is invalid")
    n = len(cols)
    w = raw.get('weights')
    if not isinstance(w, dict):
        raise ValueError("model field 'weights' is invalid")
    mean, scale, coef = (_float_list(w.get(k), n, k) for k in ('mean', 'scale', 'coef'))
    if any(s <= 0 for s in scale) or not _finite(w.get('intercept')):
        raise ValueError("model weights are invalid")

    base_rate, p50, p99, thr = (raw.get(k) for k in ('base_rate', 'score_p50', 'score_p99', 'threshold'))
    if not all(_finite(x) for x in (base_rate, p50, p99, thr)):
        raise ValueError("model scalars are invalid")
    if not (0 < base_rate < 1 and 0 < thr < 1 and p50 < p99):
        raise ValueError("model scalars are out of range")
    horizon = raw.get('horizon')
    if not isinstance(horizon, int) or isinstance(horizon, bool) or not 1 <= horizon <= 2000:
        raise ValueError("model field 'horizon' is invalid")
    counts = {}
    for k in ('n_training_rows', 'ticker_universe_size'):
        if not isinstance(raw.get(k), int) or isinstance(raw.get(k), bool) or raw[k] < 0:
            raise ValueError(f"model field '{k}' is invalid")
        counts[k] = int(raw[k])
    stats_in = raw.get('feature_stats', {})
    if not isinstance(stats_in, dict) or len(stats_in) > 30:
        raise ValueError("model field 'feature_stats' is invalid")
    stats = {}
    for name, s in stats_in.items():
        if name not in FEATURE_NAMES or not isinstance(s, dict) or not all(_finite(s.get(k)) for k in ('mean', 'std', 'median')):
            raise ValueError("model field 'feature_stats' is invalid")
        stats[name] = {k: float(s[k]) for k in ('mean', 'std', 'median')}

    return {
        'schema': MODEL_SCHEMA, 'cols': list(cols),
        'weights': {'mean': mean, 'scale': scale, 'coef': coef, 'intercept': float(w['intercept'])},
        'feature_stats': stats, 'score_p50': float(p50), 'score_p99': float(p99),
        'base_rate': float(base_rate), 'threshold': float(thr), 'horizon': horizon,
        'trained_on': _short_str(raw.get('trained_on'), 'trained_on'),
        'trained_through': _short_str(raw.get('trained_through'), 'trained_through'),
        'validation': _parse_validation(raw.get('validation')),
        **counts,
    }


def _to_jsonable(obj):
    """Convert numpy values to plain Python and NaN/inf to None so the file is valid JSON."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return _to_jsonable(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def save_bundle(bundle: dict, path: Optional[str] = None) -> str:
    """Validate and write the model as JSON (atomically, so a crash cannot leave half a file)."""
    path = path or MODEL_PATH
    clean = parse_bundle(_to_jsonable(bundle))
    tmp = f"{path}.tmp"
    with open(tmp, 'w', encoding='utf-8') as fh:
        fh.write(json.dumps(clean, indent=1, allow_nan=False))
    os.replace(tmp, path)
    return path


def _reject_constant(name):
    raise ValueError(f"invalid JSON constant {name}")


def _load_model_bundle():
    """
    Load the trained model from bubble_model.json (cached after first call).
    Returns None if it is missing, malformed, or fails validation. Old pickle models
    (bubble_model.pkl) are deliberately NOT loaded: unpickling can run arbitrary code.
    """
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is not None:
        return _MODEL_BUNDLE
    if not os.path.exists(MODEL_PATH):
        legacy = os.path.splitext(MODEL_PATH)[0] + '.pkl'
        if os.path.exists(legacy):
            logger.warning("Found an old pickle model that is ignored for safety. "
                           "Retrain with train_bubble_model.py to create bubble_model.json.")
        return None
    try:
        if os.path.getsize(MODEL_PATH) > MAX_MODEL_BYTES:
            raise ValueError("model file is too large")
        with open(MODEL_PATH, 'r', encoding='utf-8') as fh:
            raw = json.load(fh, parse_constant=_reject_constant)
        _MODEL_BUNDLE = parse_bundle(raw)
    except (OSError, ValueError) as error:       # JSONDecodeError is a ValueError
        logger.warning(f"Model file rejected ({error}). Retrain with train_bubble_model.py.")
        return None
    return _MODEL_BUNDLE


def validation_status(bundle) -> str:
    """'validated' | 'not_better_than_volatility' | 'worse_than_volatility' | 'unvalidated'."""
    v = (bundle or {}).get('validation') or {}
    ci = v.get('ci_vs_volatility')
    if not ci:
        return 'unvalidated'
    if ci[1] > 0:
        return 'validated'
    if ci[2] < 0:
        return 'worse_than_volatility'
    return 'not_better_than_volatility'


def validation_note(bundle) -> str:
    """One plain-English sentence on how the model did on data it never trained on."""
    status = validation_status(bundle)
    v = (bundle or {}).get('validation') or {}
    if status == 'unvalidated':
        return ("This model has no held-out validation result. Treat its output as unproven "
                "and retrain with train_bubble_model.py.")
    a_m, a_v = v['auc_model'], v['auc_volatility']
    span = f"{v['holdout_start']} to {v['holdout_end']}"
    if status == 'validated':
        return (f"On held-out data ({span}) the model ranked later crashes better than plain "
                f"volatility (AUC {a_m:.2f} vs {a_v:.2f}). One market period only.")
    if status == 'worse_than_volatility':
        return (f"On held-out data ({span}) the model did WORSE than plain volatility "
                f"(AUC {a_m:.2f} vs {a_v:.2f}). Treat the score as a rough volatility screen.")
    return (f"On held-out data ({span}) the model was not demonstrably better than plain "
            f"volatility (AUC {a_m:.2f} vs {a_v:.2f}). Treat the score as a volatility-driven "
            f"risk screen, not a bubble detector.")


def compute_feature_vector(data: pd.DataFrame) -> Optional[dict]:
    """
    Build the feature snapshot used by the model. Uses only data up to the last row.

    Base features (raw values):
        rsi, ma_ratio, volatility (90d, annualized %), accel, max_drawdown (252d, %)
    Volatility-adjusted features (a 30% stretch means little for a 60%-volatility
    stock and a lot for a 15%-volatility one):
        ext_vol    -- (price/200MA - 1) in percent, per point of volatility
        accel_vol  -- acceleration per point of volatility
        mom_vol    -- 126-day return (percent) per point of volatility
        dist_high  -- price / 252-day high - 1  (0 at a 52-week high)

    Returns dict of {feature_name: float}, or None if history is too short (< 200 rows).
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
        prior = prices.tail(60).head(30)
        recent_return = ((float(recent.iloc[-1]) - float(recent.iloc[0]))
                         / float(recent.iloc[0])) * 100
        prior_return = ((float(prior.iloc[-1]) - float(prior.iloc[0]))
                        / float(prior.iloc[0])) * 100
        accel_val = recent_return - prior_return
    else:
        accel_val = 0.0

    if not all(np.isfinite(v) for v in (rsi_val, ma_ratio_val, vol_val, dd_val, accel_val)):
        return None

    c = prices.values.astype(float)
    vol_floor = max(float(vol_val), 1.0)
    mom = (c[-1] / c[-127] - 1) * 100 if len(c) > 126 else 0.0
    return {
        'rsi': float(rsi_val),
        'ma_ratio': float(ma_ratio_val),
        'volatility': float(vol_val),
        'accel': float(accel_val),
        'max_drawdown': float(dd_val),
        'ext_vol': float((ma_ratio_val - 1) * 100 / vol_floor),
        'accel_vol': float(accel_val / vol_floor),
        'mom_vol': float(mom / vol_floor),
        'dist_high': float(c[-1] / c[-252:].max() - 1),
    }


def _features_at(c, rsi, ma200, i):
    """Feature dict for row position i, from precomputed arrays. Mirrors compute_feature_vector."""
    if i + 1 < 200 or i >= len(c):
        return None
    ma = ma200[i]
    if not np.isfinite(ma) or ma == 0:
        return None
    w90 = c[max(0, i - 89): i + 1]
    r = w90[1:] / w90[:-1] - 1
    if len(r) < 2:
        return None
    vol = round(float(r.std(ddof=1) * (252 ** 0.5) * 100), 2)
    w252 = c[max(0, i - 251): i + 1]
    peak = np.maximum.accumulate(w252)
    dd = round(float(((w252 - peak) / peak * 100).min()), 2)
    recent, prior = c[i - 29: i + 1], c[i - 59: i - 29]
    accel = ((recent[-1] - recent[0]) / recent[0] * 100) - ((prior[-1] - prior[0]) / prior[0] * 100)
    ma_ratio = c[i] / ma
    vol_floor = max(vol, 1.0)
    mom = (c[i] / c[i - 126] - 1) * 100
    feats = {
        'rsi': float(rsi[i]), 'ma_ratio': float(ma_ratio), 'volatility': float(vol),
        'accel': float(accel), 'max_drawdown': float(dd),
        'ext_vol': float((ma_ratio - 1) * 100 / vol_floor), 'accel_vol': float(accel / vol_floor),
        'mom_vol': float(mom / vol_floor), 'dist_high': float(c[i] / w252.max() - 1),
    }
    return feats if all(np.isfinite(v) for v in feats.values()) else None


def compute_feature_series(data: pd.DataFrame, indices) -> list:
    """
    Features at many row positions in one pass. For each position i this returns the same
    values as compute_feature_vector(data.iloc[:i + 1]) (a test checks that), but it
    computes the RSI and moving average once instead of once per snapshot, which makes
    building the training table many times faster. Uses only data up to row i.
    """
    close = data['Close'].astype(float)
    c = close.values
    rsi = get_rsi(data).values
    ma200 = close.rolling(200).mean().values
    return [_features_at(c, rsi, ma200, int(i)) for i in indices]


def assign_risk_level(bubble_score: float) -> str:
    """
    Convert a 0-100 risk score to a text label. The score is RELATIVE: 0 means typical
    (median) risk or lower in the training population, 100 means riskier than 99% of
    training snapshots.
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


def _predict(bundle, feats):
    """
    Return (crash probability, per-feature contribution to risk vs. a typical stock).
    Plain logistic-regression arithmetic on the stored weights; no pickled objects.
    """
    w = bundle['weights']
    x = np.array([feats[c] for c in bundle['cols']], dtype=float)
    contrib = np.asarray(w['coef']) * ((x - np.asarray(w['mean'])) / np.asarray(w['scale']))
    logit = float(w['intercept'] + contrib.sum())
    p = 1.0 / (1.0 + math.exp(-max(min(logit, 35.0), -35.0)))
    return p, contrib


def probability_to_score(bundle, p: float) -> float:
    """0 at the training median probability, 100 at the 99th percentile."""
    lo, hi = bundle['score_p50'], bundle['score_p99']
    return float(np.clip(100 * (p - lo) / (hi - lo), 0, 100))


def _describe(name, f, stats):
    med = stats.get(name, {}).get('median')
    if name == 'volatility':
        typ = f" (typical: {med:.0f}%)" if med is not None else ""
        return f"Price swings are {f['volatility']:.0f}% annualized{typ}"
    if name == 'rsi':
        return f"RSI is {f['rsi']:.0f} (above 70 means it was bought up fast)"
    if name == 'ext_vol':
        return (f"Price is {(f['ma_ratio'] - 1) * 100:+.0f}% from its 200-day average "
                f"(judged against its own volatility)")
    if name == 'accel_vol':
        word = "more" if f['accel'] >= 0 else "less"
        return (f"The last 30 days gained {abs(f['accel']):.0f} percentage points {word} than the "
                f"30 days before (judged against its own volatility)")
    if name == 'mom_vol':
        return "Its 6-month return is large or small relative to how much it normally swings"
    if name == 'dist_high':
        return f"Price is {abs(f['dist_high']) * 100:.0f}% below its 52-week high"
    return name


def explain_drivers(bundle, feats, contrib, top=3):
    """Plain-English list of what pushes this stock's risk up or down vs. a typical stock."""
    stats = bundle.get('feature_stats', {})
    order = np.argsort(-np.abs(contrib))[:top]
    drivers = []
    for j in order:
        name = bundle['cols'][j]
        if abs(contrib[j]) < 0.15:
            continue
        drivers.append({
            'feature': name, 'label': PLAIN_NAMES.get(name, name),
            'contribution': round(float(contrib[j]), 2),
            'direction': 'raises risk' if contrib[j] > 0 else 'lowers risk',
            'text': _describe(name, feats, stats),
        })
    return drivers


VERDICT_LEVELS = [
    (70, "AVOID NEW MONEY", "red"),
    (50, "CAUTION", "orange"),
    (30, "WATCH", "yellow"),
    (10, "OK: TYPICAL RISK", "green"),
    (0, "OK: BELOW-TYPICAL RISK", "green"),
]


def build_verdict(score: float, p: float, base_rate: float, threshold=0.20, horizon=126) -> dict:
    """A decision-oriented summary: what the model says and how big the risk is vs. typical."""
    ratio = p / base_rate if base_rate else float('nan')
    months = max(1, round(horizon / 21))
    for lower, headline, color in VERDICT_LEVELS:
        if score >= lower:
            break
    detail = (f"The model estimates a {p:.0%} chance of a {int(threshold * 100)}%+ drop within "
              f"about {months} months, versus {base_rate:.0%} for a typical stock in its training "
              f"data ({ratio:.1f}x).")
    return {'headline': headline, 'color': color, 'detail': detail, 'ratio': ratio}


def run_bubble_analysis(ticker: str, data: pd.DataFrame) -> dict:
    """
    Score a stock with the trained crash-risk model.

    Returns a dict with:
        ticker, bubble_score (0-100, relative), risk_level, warnings, sub_scores
        crash_probability, base_rate, risk_ratio, verdict {headline, color, detail, ratio}
        drivers [{feature, label, contribution, direction, text}]
        model_status, model_note   (how the model did on held-out data)
    """
    logger.info(f"Running crash-risk analysis on {ticker} ...")
    empty = {'ticker': ticker, 'bubble_score': None, 'crash_probability': None,
             'base_rate': None, 'risk_ratio': None, 'verdict': None, 'drivers': [],
             'sub_scores': {}, 'model_status': 'none', 'model_note': ''}

    bundle = _load_model_bundle()
    if bundle is None:
        return {**empty, 'risk_level': "MODEL NOT TRAINED",
                'warnings': ["No trained model found -- run train_bubble_model.py "
                             "(or use the app's Train button)."]}

    feats = compute_feature_vector(data)
    if feats is None:
        return {**empty, 'risk_level': "INSUFFICIENT DATA",
                'warnings': ["Need at least 200 trading days of history."]}

    p, contrib = _predict(bundle, feats)
    score = probability_to_score(bundle, p)
    verdict = build_verdict(score, p, bundle['base_rate'], bundle.get('threshold', 0.20),
                            bundle.get('horizon', 126))
    drivers = explain_drivers(bundle, feats, contrib)
    warnings_out = [f"{d['text']} -- {d['direction']}" for d in drivers if d['contribution'] > 0.25]
    if not warnings_out:
        warnings_out.append("No single factor stands out as pushing risk above a typical stock.")

    return {
        'ticker': ticker,
        'bubble_score': round(score, 1),
        'risk_level': assign_risk_level(score),
        'warnings': warnings_out,
        'sub_scores': {bundle['cols'][j]: round(float(contrib[j]), 2) for j in range(len(contrib))},
        'crash_probability': p,
        'base_rate': bundle['base_rate'],
        'risk_ratio': verdict['ratio'],
        'verdict': verdict,
        'drivers': drivers,
        'model_status': validation_status(bundle),
        'model_note': validation_note(bundle),
    }


def score_history(data: pd.DataFrame, step: int = 21, bundle: Optional[dict] = None):
    """
    Re-score the stock at past dates, each time using only the data available then.
    Returns a DataFrame indexed by date with columns [score, probability], or None.
    Lets you see whether the model was already worried before past drops.
    """
    bundle = bundle or _load_model_bundle()
    if bundle is None or len(data) < 251:
        return None
    idxs = list(range(250, len(data), step))
    if (len(data) - 1) not in idxs:
        idxs.append(len(data) - 1)
    rows = []
    for i, f in zip(idxs, compute_feature_series(data, idxs)):
        if f is None:
            continue
        p, _ = _predict(bundle, f)
        rows.append((data.index[i], probability_to_score(bundle, p), p))
    if not rows:
        return None
    return pd.DataFrame(rows, columns=['date', 'score', 'probability']).set_index('date')


MAX_SCAN_TICKERS = 50


def scan_tickers(tickers, years: int = 2, progress_callback=None, loader=None) -> pd.DataFrame:
    """
    Score a watchlist or portfolio. Returns a DataFrame sorted riskiest first with
    columns: ticker, score, verdict, crash_prob, vs_typical, main_driver.
    `loader` lets a caller supply a cached download function (the web app does).
    At most MAX_SCAN_TICKERS unique valid symbols are scored; invalid ones are listed as such.
    """
    loader = loader or load_stock_data
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365 * max(years, 2))).strftime('%Y-%m-%d')
    rows, seen = [], set()
    tickers = list(tickers)[:MAX_SCAN_TICKERS * 2]
    for n, raw in enumerate(tickers, start=1):
        t = validate_ticker(raw)
        if t is None:
            rows.append({'ticker': str(raw)[:12], 'score': np.nan, 'verdict': 'Invalid symbol',
                         'crash_prob': np.nan, 'vs_typical': np.nan, 'main_driver': ''})
            continue
        if t in seen or len(seen) >= MAX_SCAN_TICKERS:
            continue
        seen.add(t)
        data = loader(t, start_date, end_date)
        if data is None:
            rows.append({'ticker': t, 'score': np.nan, 'verdict': 'No data', 'crash_prob': np.nan,
                         'vs_typical': np.nan, 'main_driver': ''})
        else:
            r = run_bubble_analysis(t, data)
            if r['bubble_score'] is None:
                rows.append({'ticker': t, 'score': np.nan, 'verdict': r['risk_level'],
                             'crash_prob': np.nan, 'vs_typical': np.nan, 'main_driver': ''})
            else:
                top = next((d['text'] for d in r['drivers'] if d['contribution'] > 0), '')
                rows.append({'ticker': t, 'score': r['bubble_score'],
                             'verdict': r['verdict']['headline'], 'crash_prob': r['crash_probability'],
                             'vs_typical': r['risk_ratio'], 'main_driver': top})
        if progress_callback:
            progress_callback(n, len(tickers), t)
    return pd.DataFrame(rows).sort_values('score', ascending=False, na_position='last').reset_index(drop=True)


# -------------------------------------------------------------
# SECTION 4: RISK-AWARE GUIDANCE
# -------------------------------------------------------------

def generate_strategies(ticker, current_price, bubble_score, ma_200_value=None):
    """
    Return plain guidance matched to the risk level. These are general risk-management
    habits, NOT predictions or trade instructions, and they have not been tested for
    profitability. Each strategy is a dict: priority, action, detail.
    """
    strategies = []
    if bubble_score >= 70:
        strategies = [
            {'priority': 'URGENT', 'action': 'DO NOT ADD NEW MONEY',
             'detail': (f"{ticker} is in the model's highest-risk range. Adding here means "
                        "taking a much larger-than-typical chance of a deep drop.")},
            {'priority': 'HIGH', 'action': 'DECIDE YOUR EXIT BEFORE THE DROP',
             'detail': (f"If you hold it, pick the loss you can accept now (for example a stop "
                        f"near ${current_price * 0.90:.2f}, 10% below today's ${current_price}) "
                        "so you are not deciding in a panic.")},
            {'priority': 'HIGH', 'action': 'KEEP THE POSITION SMALL',
             'detail': ("A common rule is to cap any single risky stock at a small share of your "
                        "portfolio (for example 5%) so one drop cannot do real damage.")},
            {'priority': 'MEDIUM', 'action': 'CONSIDER A LOWER-RISK WAY TO HOLD THE THEME',
             'detail': ("If you want the same sector, a diversified fund or ETF usually swings "
                        "less than a single stock.")},
            {'priority': 'MEDIUM',
             'action': (f'REFERENCE LEVEL: ${ma_200_value:.2f} (200-DAY AVERAGE)' if ma_200_value
                        else 'WAIT FOR A 200-DAY AVERAGE TO FORM'),
             'detail': "Prices that run far above their long-term average often pull back toward it, "
                       "but not always, and not on a schedule."},
        ]
    elif bubble_score >= 50:
        strategies = [
            {'priority': 'HIGH', 'action': 'AVOID ADDING MORE',
             'detail': f"Risk for {ticker} is well above typical. Do not average up into it."},
            {'priority': 'HIGH', 'action': 'SET A LOSS LIMIT',
             'detail': (f"Decide in advance what fall you will accept (for example a trailing stop "
                        f"15% below the recent high, about ${current_price * 0.85:.2f} from today).")},
            {'priority': 'MEDIUM', 'action': 'CHECK POSITION SIZE',
             'detail': "No single stock should be large enough that a 30% drop hurts your plans."},
            {'priority': 'MEDIUM', 'action': 'RE-CHECK WEEKLY',
             'detail': "Risk can change fast in volatile stocks. Re-run this when the price moves a lot."},
        ]
    elif bubble_score >= 30:
        strategies = [
            {'priority': 'MEDIUM', 'action': 'WATCH, DO NOT RUSH IN',
             'detail': f"{ticker} shows somewhat elevated risk. Fine to hold; be careful adding."},
            {'priority': 'LOW', 'action': 'MENTAL LOSS LIMIT',
             'detail': (f"Know the price at which you would reassess (for example ${current_price * 0.88:.2f}, "
                        "12% below today).")},
            {'priority': 'LOW', 'action': 'KEEP PORTFOLIO DIVERSIFIED',
             'detail': "Spread holdings across several sectors so one crash does not dominate."},
        ]
    else:
        strategies = [
            {'priority': 'LOW', 'action': 'RISK LOOKS TYPICAL OR BELOW',
             'detail': (f"{ticker} is not flagged. That does not mean it cannot fall: "
                        "no stock is crash-proof, and the model is imperfect.")},
            {'priority': 'LOW', 'action': 'DOLLAR-COST AVERAGE',
             'detail': "Investing a fixed amount on a schedule reduces the impact of bad timing."},
            {'priority': 'INFO', 'action': 'RE-CHECK MONTHLY',
             'detail': "Conditions change. A monthly check catches developing risk early."},
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
    Print the full crash-risk report to the console.

    Parameters:
        result : dict -- Output from run_bubble_analysis()
    """
    print(f"\n  {'=' * 60}")
    print(f"  OVERHEAT REPORT: {result['ticker']}")
    print(f"  {'=' * 60}")

    if result['bubble_score'] is None:
        print(f"  Status: {result['risk_level']}")
        for warning in result['warnings']:
            print(f"    [!]  {warning}")
        print(f"  {'=' * 60}")
        return

    v = result['verdict']
    print(f"  VERDICT      : {v['headline']}")
    print(f"  Risk score   : {result['bubble_score']} / 100  (relative: 0 = typical, 100 = top 1%)")
    print(f"  {v['detail']}")

    print("\n  What drives it (vs. a typical stock):")
    if result['drivers']:
        for d in result['drivers']:
            arrow = '+' if d['contribution'] > 0 else '-'
            print(f"    [{arrow}] {d['text']}  ({d['direction']})")
    else:
        print("    Nothing stands out.")

    print(f"\n  Model check  : {result['model_note']}")
    print(f"  {'=' * 60}")


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


def plot_score_history(ticker, data, history, show=True, save=False):
    """
    Two panels: price (colored by the model's risk level at each date) and the
    risk score over time, with the risk-band lines. `history` comes from score_history().
    """
    colors = ['#388E3C', '#689F38', '#FBC02D', '#F57C00', '#D32F2F']
    cuts = [10, 30, 50, 70]

    def color_for(score):
        return colors[sum(score >= c for c in cuts)]

    fig, (ax_p, ax_s) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                     gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle(f'{ticker} -- price and the model\'s risk score over time',
                 fontsize=15, fontweight='bold')
    px = data['Close']
    ax_p.plot(px.index, px, color='#9E9E9E', linewidth=1.0)
    for (d0, r0), (d1, _) in zip(history.iloc[:-1].iterrows(), history.iloc[1:].iterrows()):
        seg = px[(px.index >= d0) & (px.index <= d1)]
        ax_p.plot(seg.index, seg, color=color_for(r0['score']), linewidth=2.2)
    ax_p.set_ylabel('Price (USD)', fontsize=12)

    ax_s.plot(history.index, history['score'], color='#1F3A5F', linewidth=1.8, marker='o', markersize=3)
    for c, col in zip(cuts, colors[1:]):
        ax_s.axhline(c, color=col, linestyle='--', linewidth=1)
    ax_s.set_ylim(0, 100)
    ax_s.set_ylabel('Risk score', fontsize=12)
    ax_s.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax_s.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax_s.xaxis.get_majorticklabels(), rotation=45, ha='right')
    fig.tight_layout()
    if save:
        fig.savefig(f'{ticker}_risk_history.png', dpi=150, bbox_inches='tight')
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
    print("  OVERHEAT SCORE -- INPUT")
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
        4. Runs the crash-risk analysis and prints the report.
        5. Prints avoidance strategies scaled to the risk level.
        6. Plots the price/RSI chart.
        7. Asks whether to analyze another stock.
    """
    print("\n" + "=" * 55)
    print("    OVERHEAT SCORE")
    print("    Crash-risk check | Yahoo Finance | Python")
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
    # `python stock_bubble_detector.py scan NVDA TSLA KO` scores a watchlist.
    if len(sys.argv) > 2 and sys.argv[1].lower() == 'scan':
        table = scan_tickers(sys.argv[2:])
        pd.set_option('display.width', 200)
        pd.set_option('display.max_colwidth', 70)
        print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    else:
        main()
