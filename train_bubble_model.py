
""" It downloads several years of history for a broad universe of
 tickers, slides a window through each ticker's history to build
 dated feature snapshots, and fits an Isolation Forest to
 learn what "normal" market behavior looks like. Snapshots that
 don't fit that pattern are the anomalies stock_bubble_detector.py
 flags at inference time.
"""
import argparse
import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import sklearn

from stock_bubble_detector import (
    load_stock_data,
    compute_feature_vector,
    FEATURE_NAMES,
    MODEL_PATH,
)

TRAINING_DATA_PATH = os.path.join(
    os.path.dirname(MODEL_PATH), 'training_snapshots.csv'
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# A broad, sector-diverse universe. More tickers = a better-calibrated
# sense of "normal." Feel free to extend this list.
TICKER_UNIVERSE = [
    # Tech / large growth
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'CRM', 'ADBE',
    'ORCL', 'CSCO', 'INTC', 'AVGO', 'NFLX', 'SHOP', 'UBER', 'PLTR', 'SNOW', 'NOW',
    # Financials
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA', 'AXP', 'BLK', 'SCHW',
    # Healthcare
    'JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'LLY', 'TMO', 'ABT', 'CVS', 'BMY',
    # Consumer / industrial / energy / staples (defensive + cyclical mix)
    'WMT', 'PG', 'KO', 'PEP', 'COST', 'HD', 'MCD', 'NKE', 'DIS', 'SBUX',
    'XOM', 'CVX', 'CAT', 'BA', 'GE', 'HON', 'UPS', 'LMT', 'DE', 'MMM',
    'XLU', 'XLV', 'XLP',  # sector ETFs help anchor "defensive/normal" behavior
]

DEFAULT_YEARS_OF_HISTORY = 6

# The longest lookback any single feature uses is 252 trading days
# (max_drawdown). Spacing snapshots at least this far apart means
# consecutive snapshots for the same ticker don't share the same
# underlying price window -- i.e. they're close to statistically
# independent observations, not near-duplicates of each other.
#
# Trade-off, stated plainly: this cuts snapshot COUNT drastically
# (~6 per ticker over 6 years instead of ~250) in exchange for each
# snapshot actually meaning something distinct. Fewer, real
# observations beat many correlated copies for a model that's
# learning "what does normal look like." If you deliberately want a
# denser, more autocorrelated training set (e.g. for a quick local
# experiment), pass a smaller --step-days -- just know you're
# reintroducing the correlation this default avoids.
NONOVERLAP_LOOKBACK_DAYS = 252
DEFAULT_SNAPSHOT_STEP_DAYS = NONOVERLAP_LOOKBACK_DAYS

MIN_HISTORY_DAYS = 250           # need >=200 for MA ratio, plus buffer
DEFAULT_CONTAMINATION = 0.05
MAX_ACCEPTABLE_SKIP_FRACTION = 0.20  # warn loudly if we lose >20% of tickers


def build_training_rows(years: int = DEFAULT_YEARS_OF_HISTORY,
                         step_days: int = DEFAULT_SNAPSHOT_STEP_DAYS,
                         tickers: list = None,
                         progress_callback=None) -> list:
    """
    Download history for the ticker universe and slide a window through
    each series to build a table of feature snapshots.

    Parameters:
        years     : int  -- Years of history to pull per ticker
        step_days : int  -- Trading days between snapshots
        tickers   : list -- Ticker universe to use (defaults to TICKER_UNIVERSE)

    Returns:
        list of dict -- each dict is one feature snapshot (one row).
    """
    tickers = tickers if tickers is not None else TICKER_UNIVERSE
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365 * years)).strftime('%Y-%m-%d')

    if step_days < NONOVERLAP_LOOKBACK_DAYS:
        logger.warning(
            f"step_days={step_days} is below the {NONOVERLAP_LOOKBACK_DAYS}-day "
            "max feature lookback -- snapshots for the same ticker will share "
            "overlapping price windows and won't be statistically independent. "
            "Fine for a quick experiment, not recommended for the model you "
            "actually rely on."
        )

    rows = []
    skipped = 0
    total = len(tickers)
    for idx, ticker in enumerate(tickers, start=1):
        data = load_stock_data(ticker, start_date, end_date)
        if data is None or len(data) < MIN_HISTORY_DAYS:
            logger.warning(f"Skipping {ticker} -- insufficient history.")
            skipped += 1
            if progress_callback:
                progress_callback(idx, total, ticker, 0, skipped=True)
            continue

        count_for_ticker = 0
        # Keep a forward 60-session window for an exploratory, chronological
        # holdout evaluation. These outcome columns are never model inputs.
        for i in range(MIN_HISTORY_DAYS, len(data) - 60, step_days):
            snapshot = data.iloc[:i + 1]
            features = compute_feature_vector(snapshot)
            if features is not None:
                future = data['Close'].iloc[i:i + 61]
                start_price = float(future.iloc[0])
                features = dict(features)
                features['_as_of'] = str(data.index[i].date())
                features['_ticker'] = ticker
                features['_forward_60d_return'] = (float(future.iloc[-1]) / start_price - 1) * 100
                features['_forward_60d_max_drawdown'] = float(
                    ((future / future.cummax()) - 1).min() * 100
                )
                rows.append(features)
                count_for_ticker += 1

        logger.info(f"{ticker}: {count_for_ticker} snapshots")
        if progress_callback:
            progress_callback(idx, total, ticker, count_for_ticker, skipped=False)

    skip_fraction = skipped / len(tickers) if tickers else 0
    if skip_fraction > MAX_ACCEPTABLE_SKIP_FRACTION:
        logger.warning(
            f"{skipped}/{len(tickers)} tickers were skipped ({skip_fraction:.0%}) -- "
            "likely rate-limiting or network issues. The trained model may be "
            "based on a much smaller, less diverse universe than intended. "
            "Consider re-running later or reducing request volume."
        )

    return rows


def train_and_save(rows: list,
                    contamination: float = DEFAULT_CONTAMINATION,
                    ticker_universe_size: int = None,
                    snapshot_step_days: int = DEFAULT_SNAPSHOT_STEP_DAYS) -> dict:
    """
    Fit the scaler + Isolation Forest on the collected feature rows
    and save everything the inference code needs to bubble_model.pkl.

    Returns:
        The saved bundle dict (also written to MODEL_PATH).
    """
    if len(rows) < 200:
        raise RuntimeError(
            f"Only collected {len(rows)} training rows -- too few to train "
            "a reliable model. Check your network access and ticker list."
        )

    ticker_universe_size = (
        ticker_universe_size if ticker_universe_size is not None else len(TICKER_UNIVERSE)
    )

    x = np.array([[row[name] for name in FEATURE_NAMES] for row in rows], dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Training data contains non-finite feature values.")

    # Chronological holdout: score later snapshots with a model fit only on
    # earlier observations. Forward outcomes are reported as associations,
    # not as proof of predictive power or bubble labels.
    dated = sorted({row.get('_as_of') for row in rows if row.get('_as_of')})
    if len(dated) >= 5:
        cutoff = dated[max(1, int(len(dated) * 0.8))]
        train_idx = [i for i, row in enumerate(rows) if row.get('_as_of') and row['_as_of'] < cutoff]
        test_idx = [i for i, row in enumerate(rows) if row.get('_as_of') and row['_as_of'] >= cutoff]
        if len(train_idx) >= 100 and len(test_idx) >= 20:
            eval_scaler = StandardScaler().fit(x[train_idx])
            eval_model = IsolationForest(n_estimators=300, contamination=contamination, random_state=42)
            eval_model.fit(eval_scaler.transform(x[train_idx]))
            anomaly = -eval_model.score_samples(eval_scaler.transform(x[test_idx]))
            eval_rows = [rows[i] for i in test_idx]
            forward = np.array([row['_forward_60d_return'] for row in eval_rows], dtype=float)
            dd = np.array([row['_forward_60d_max_drawdown'] for row in eval_rows], dtype=float)
            threshold = np.quantile(anomaly, 0.75)
            flagged = anomaly >= threshold
            logger.info(
                "Chronological holdout (%s onward, %s snapshots): top anomaly quartile "
                "mean forward 60-session return %.2f%%; mean forward max drawdown %.2f%%. "
                "Exploratory association only; survivorship bias and overlapping periods remain.",
                cutoff, len(test_idx), float(forward[flagged].mean()), float(dd[flagged].mean())
            )
        else:
            logger.warning("Not enough chronological holdout rows for evaluation; need >=100 train and >=20 holdout.")
    else:
        logger.warning("Not enough distinct snapshot dates for chronological evaluation.")

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=42,
    )
    model.fit(x_scaled)

    # Normalize inference scores against this training run's distribution.
    scores = model.score_samples(x_scaled)
    score_low, score_high = np.percentile(scores, [1, 99])

    bundle = {
        'model': model,
        'scaler': scaler,
        'feature_names': FEATURE_NAMES,
        'score_low': float(score_low),
        'score_high': float(score_high),
        'trained_on': datetime.today().strftime('%Y-%m-%d'),
        'n_training_rows': len(rows),
        'ticker_universe_size': ticker_universe_size,
        'contamination': contamination,
        'metadata': {
            'created_at_utc': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
            'data_source': 'Yahoo Finance via yfinance',
            'training_snapshot_file': os.path.basename(TRAINING_DATA_PATH),
            'training_universe': sorted({row.get('_ticker', 'unknown') for row in rows}),
            'snapshot_date_min': min(dated) if dated else None,
            'snapshot_date_max': max(dated) if dated else None,
            'feature_names': FEATURE_NAMES,
            'sklearn_version': sklearn.__version__,
            'pandas_version': pd.__version__,
            'numpy_version': np.__version__,
            'contamination': contamination,
            'snapshot_step_days': snapshot_step_days,
            'feature_rows_sha256': hashlib.sha256(json.dumps(
                [{key: row[key] for key in FEATURE_NAMES} for row in rows],
                sort_keys=True, separators=(',', ':'), allow_nan=False,
            ).encode('utf-8')).hexdigest(),
            'evaluation_horizon_sessions': 60,
            'note': 'Universe contains current constituents; survivorship bias is possible.',
        },
    }
    fd, temp_path = tempfile.mkstemp(prefix='bubble_model_', suffix='.pkl', dir=os.path.dirname(MODEL_PATH))
    os.close(fd)
    try:
        joblib.dump(bundle, temp_path)
        os.replace(temp_path, MODEL_PATH)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    fd, data_temp_path = tempfile.mkstemp(
        prefix='training_snapshots_', suffix='.csv', dir=os.path.dirname(TRAINING_DATA_PATH)
    )
    os.close(fd)
    try:
        pd.DataFrame(rows).to_csv(data_temp_path, index=False)
        os.replace(data_temp_path, TRAINING_DATA_PATH)
    finally:
        if os.path.exists(data_temp_path):
            os.remove(data_temp_path)
    logger.info(f"Saved model to {MODEL_PATH}")
    logger.info(f"Saved reproducible feature snapshots to {TRAINING_DATA_PATH}")
    logger.info(f"Trained on {len(rows)} snapshots across {ticker_universe_size} tickers.")
    return bundle


def main():
    parser = argparse.ArgumentParser(description="Train the bubble-detection Isolation Forest model.")
    parser.add_argument('--years', type=int, default=DEFAULT_YEARS_OF_HISTORY,
                         help=f"Years of history per ticker (default {DEFAULT_YEARS_OF_HISTORY})")
    parser.add_argument('--step-days', type=int, default=DEFAULT_SNAPSHOT_STEP_DAYS,
                         help=(f"Trading days between snapshots (default "
                               f"{DEFAULT_SNAPSHOT_STEP_DAYS}, matching the longest "
                               f"feature lookback so snapshots don't overlap -- values "
                               f"below {NONOVERLAP_LOOKBACK_DAYS} trade independence for "
                               f"snapshot count)"))
    parser.add_argument('--contamination', type=float, default=DEFAULT_CONTAMINATION,
                         help=f"Expected anomaly fraction for Isolation Forest (default {DEFAULT_CONTAMINATION})")
    args = parser.parse_args()

    logger.info("=" * 55)
    logger.info("TRAINING BUBBLE DETECTION MODEL")
    logger.info("=" * 55)
    logger.info(
        f"Universe: {len(TICKER_UNIVERSE)} tickers, {args.years} years each, "
        f"snapshot every {args.step_days} days, contamination={args.contamination}"
    )

    rows = build_training_rows(years=args.years, step_days=args.step_days)
    train_and_save(rows, contamination=args.contamination, snapshot_step_days=args.step_days)

    logger.info("Chronological holdout diagnostics above are exploratory, not a model certification.")


if __name__ == "__main__":
    main()
