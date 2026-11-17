"""
Train the crash-risk model and check it on data it never saw.

What this does:
  1. Downloads daily prices for a curated universe of large, liquid tickers.
  2. Builds one feature snapshot per ticker every --step-days trading days, labeled with
     what happened next (did the stock fall 20%+ within ~126 trading days?).
  3. VALIDATES: trains on rows whose outcome was known by a cutoff date (default: two
     years before today) and tests on later dates, comparing the model with plain
     volatility, RSI, and price/200-day average.
  4. Trains the final model on every row whose outcome is known and saves it, together
     with the validation result, to bubble_model.pkl.

The app and CLI read that validation result and tell you how far to trust the score.
"""
import argparse
import logging
from datetime import datetime, timedelta

import pandas as pd

import crash_model as cm
import stock_bubble_detector as sbd
from stock_bubble_detector import load_stock_data, MODEL_PATH

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# A broad, sector-diverse universe of large, liquid stocks (and three defensive sector
# ETFs). Caveat: these are companies that are large TODAY, so the training data is biased
# toward survivors. The backtest script (backtest.py) adds stocks that later fell hard
# as a test-only check, which reduces but does not remove that bias.
TICKER_UNIVERSE = [
    # Tech / large growth
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'CRM', 'ADBE',
    'ORCL', 'CSCO', 'INTC', 'AVGO', 'NFLX', 'SHOP', 'UBER', 'PLTR', 'SNOW', 'NOW',
    # Financials
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA', 'AXP', 'BLK', 'SCHW',
    # Healthcare
    'JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'LLY', 'TMO', 'ABT', 'CVS', 'BMY',
    # Consumer / industrial / energy / staples
    'WMT', 'PG', 'KO', 'PEP', 'COST', 'HD', 'MCD', 'NKE', 'DIS', 'SBUX',
    'XOM', 'CVX', 'CAT', 'BA', 'GE', 'HON', 'UPS', 'LMT', 'DE', 'MMM',
    'XLU', 'XLV', 'XLP',
]

# Dates with well-known outcomes, shown for context only. They fall INSIDE the training
# period, so they are NOT a test of anything. The held-out validation above is the test.
KNOWN_TEST_CASES = [
    ('NVDA', '2024-06-18', 'overheated', 'Pre-split run-up'),
    ('TSLA', '2021-11-01', 'overheated', 'Near 2021 all-time-high blow-off'),
    ('JNJ', '2023-06-01', 'calm', 'Low-beta defensive'),
    ('KO', '2023-09-01', 'calm', 'Low-beta defensive'),
]

DEFAULT_YEARS_OF_HISTORY = 10         # long enough to span several market regimes
DEFAULT_SNAPSHOT_STEP_DAYS = 21       # monthly snapshots (overlapping windows; see README)
DEFAULT_HOLDOUT_YEARS = 2             # validate on the most recent ~2 years
MIN_HISTORY_DAYS = cm.MIN_HISTORY_DAYS
MAX_ACCEPTABLE_SKIP_FRACTION = 0.20   # warn loudly if we lose >20% of tickers


def build_training_table(years: int = DEFAULT_YEARS_OF_HISTORY,
                         step_days: int = DEFAULT_SNAPSHOT_STEP_DAYS,
                         tickers: list = None,
                         horizon: int = cm.DEFAULT_HORIZON,
                         progress_callback=None,
                         cache_dir: str = cm.DEFAULT_CACHE_DIR) -> pd.DataFrame:
    """
    Download history for the ticker universe and build the labeled snapshot table.
    Same-day downloads are reused from `cache_dir` (plain CSV files), so re-running
    training does not hit Yahoo Finance again.

    Returns a DataFrame with one row per (ticker, snapshot date): the features known
    on that date and `fwd_worst` (the worst loss over the next `horizon` trading days;
    NaN when that future had not happened yet).
    """
    tickers = tickers if tickers is not None else TICKER_UNIVERSE
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365 * years)).strftime('%Y-%m-%d')
    cm.prune_cache(cache_dir)

    recs, skipped, total = [], 0, len(tickers)
    for idx, ticker in enumerate(tickers, start=1):
        close = cm.get_prices(ticker, start_date, end_date, cache_dir)
        if close is None or len(close) < MIN_HISTORY_DAYS + 30:
            logger.warning(f"Skipping {ticker} -- insufficient history.")
            skipped += 1
            if progress_callback:
                progress_callback(idx, total, ticker, 0, skipped=True)
            continue
        rows = cm.table_for_ticker(close, ticker, 'core', step_days, horizon)
        recs += rows
        logger.info(f"{ticker}: {len(rows)} snapshots")
        if progress_callback:
            progress_callback(idx, total, ticker, len(rows), skipped=False)

    if total and skipped / total > MAX_ACCEPTABLE_SKIP_FRACTION:
        logger.warning(
            f"{skipped}/{total} tickers were skipped ({skipped / total:.0%}) -- likely rate "
            "limiting or network issues. The model may be trained on a much smaller universe."
        )
    return cm.finish_table(recs)


def train_and_save(tab: pd.DataFrame,
                   holdout_years: int = DEFAULT_HOLDOUT_YEARS,
                   horizon: int = cm.DEFAULT_HORIZON,
                   threshold: float = cm.DEFAULT_THRESHOLD,
                   ticker_universe_size: int = None,
                   n_boot: int = 300) -> dict:
    """
    Validate on a held-out recent period, then train the final model on all known
    outcomes and save it (with the validation result) to bubble_model.pkl.
    """
    if len(tab) < 500:
        raise RuntimeError(
            f"Only {len(tab)} training rows -- too few to train a reliable model. "
            "Check your network access, ticker list, and years of history."
        )

    cutoff = datetime.today() - timedelta(days=365 * holdout_years)
    validation = cm.validate_holdout(tab, cutoff, threshold, n_boot=n_boot)
    if validation is None:
        logger.warning("Not enough data to run the held-out validation; the model will be "
                       "saved as UNVALIDATED. Use more years of history.")

    bundle = cm.make_bundle(tab, threshold, horizon, validation, ticker_universe_size)
    sbd.save_bundle(bundle, MODEL_PATH)       # validated JSON, never a pickle
    logger.info(f"Saved model to {MODEL_PATH}")
    logger.info(f"Trained on {bundle['n_training_rows']} snapshots across "
                f"{bundle['ticker_universe_size']} tickers.")
    return bundle


def print_model_card(bundle: dict) -> None:
    v = bundle.get('validation')
    print("\n" + "=" * 66)
    print("MODEL CARD")
    print("=" * 66)
    print(f"Target        : fall of {int(bundle['threshold'] * 100)}%+ within "
          f"{bundle['horizon']} trading days")
    print(f"Training rows : {bundle['n_training_rows']} "
          f"({bundle['ticker_universe_size']} tickers, outcomes known through "
          f"{bundle['trained_through']})")
    print(f"Base rate     : {bundle['base_rate']:.1%} of training snapshots hit the target")
    if v:
        print(f"\nHeld-out test : trained through {v['cutoff']}, tested {v['holdout_start']} "
              f"to {v['holdout_end']} ({v['n_holdout']} rows)")
        print(f"  AUC model       : {v['auc_model']:.3f}   (0.5 = chance)")
        print(f"  AUC volatility  : {v['auc_volatility']:.3f}   <- the bar to beat")
        print(f"  AUC RSI         : {v['auc_rsi']:.3f}")
        print(f"  AUC price/200MA : {v['auc_ma_ratio']:.3f}")
        if v['ci_vs_volatility']:
            m, lo, hi = v['ci_vs_volatility']
            print(f"  Model minus volatility: {m:+.3f}  (95% interval {lo:+.3f} to {hi:+.3f})")
    print(f"\n{sbd.validation_note(bundle)}")
    print("=" * 66 + "\n")


def show_known_cases() -> None:
    """Context only: scores for a few famous dates. These are in-sample, not a test."""
    sbd.reload_model()
    print("In-sample context (NOT a test; these dates are inside the training period):")
    print(f"{'Ticker':<8}{'Date':<13}{'Label':<12}{'Score':<8}{'Chance of 20%+ drop'}")
    for ticker, as_of, label, _ in KNOWN_TEST_CASES:
        dt = datetime.strptime(as_of, '%Y-%m-%d')
        data = load_stock_data(ticker, (dt - timedelta(days=365 * 2)).strftime('%Y-%m-%d'), as_of)
        if data is None:
            continue
        r = sbd.run_bubble_analysis(ticker, data)
        if r['bubble_score'] is None:
            continue
        print(f"{ticker:<8}{as_of:<13}{label:<12}{r['bubble_score']:<8}{r['crash_probability']:.0%}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Train and validate the crash-risk model.")
    parser.add_argument('--years', type=int, default=DEFAULT_YEARS_OF_HISTORY,
                        help=f"Years of history per ticker (default {DEFAULT_YEARS_OF_HISTORY})")
    parser.add_argument('--step-days', type=int, default=DEFAULT_SNAPSHOT_STEP_DAYS,
                        help=f"Trading days between snapshots (default {DEFAULT_SNAPSHOT_STEP_DAYS})")
    parser.add_argument('--holdout-years', type=int, default=DEFAULT_HOLDOUT_YEARS,
                        help="Years held out at the end for validation (default 2)")
    parser.add_argument('--show-known-cases', action='store_true',
                        help="Also print in-sample scores for a few famous dates")
    parser.add_argument('--cache-dir', default=cm.DEFAULT_CACHE_DIR,
                        help="Where same-day price downloads are cached (plain CSV files)")
    args = parser.parse_args()

    logger.info("=" * 55)
    logger.info("TRAINING CRASH-RISK MODEL")
    logger.info("=" * 55)
    logger.info(f"Universe: {len(TICKER_UNIVERSE)} tickers, {args.years} years, "
                f"snapshot every {args.step_days} days, holdout {args.holdout_years} years")

    tab = build_training_table(years=args.years, step_days=args.step_days, cache_dir=args.cache_dir)
    bundle = train_and_save(tab, holdout_years=args.holdout_years)
    print_model_card(bundle)
    if args.show_known_cases:
        show_known_cases()


if __name__ == "__main__":
    main()
