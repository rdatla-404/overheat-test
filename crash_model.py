"""
crash_model.py -- shared training and validation code for the crash-risk model.

The model answers one question: "Will this stock fall 20% or more (from today's close)
at some point in the next ~126 trading days (about 6 months)?"

Rules this module enforces so the answer can be trusted more than a demo:
  * A training row is used only if its outcome was fully known at the cutoff date.
  * Validation rows are all dated AFTER the cutoff (a leakage assert guards this).
  * The model is always compared with plain volatility, the bar it has to beat.
"""
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import stock_bubble_detector as sbd

logger = logging.getLogger(__name__)

MIN_HISTORY_DAYS = 250        # need >= 200 rows for the 200-day average, plus a buffer
DEFAULT_HORIZON = 126         # trading days (~6 months)
DEFAULT_THRESHOLD = 0.20      # a 20% fall counts as a "crash"


# ------------------------------------------------------------ candidates
def make_logistic():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def make_boosted():
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=100,
        l2_regularization=1.0, random_state=42)


# name -> (feature columns, model factory). 'Vol-adjusted LR' is the shipped model.
CANDIDATES = {
    'Vol-only LR': (['volatility'], make_logistic),
    'Base-5 LR': (list(sbd.BASE_FEATURE_NAMES), make_logistic),
    'Vol-adjusted LR': (list(sbd.RISK_MODEL_FEATURES), make_logistic),
    'Boosted trees (all)': (list(sbd.FEATURE_NAMES), make_boosted),
}
SHIPPED = 'Vol-adjusted LR'


# ---------------------------------------------------------------- tables
def table_for_ticker(df, ticker, group='core', step=21, horizon=DEFAULT_HORIZON):
    """One row per snapshot: features known on that date, plus the outcome that followed."""
    close, idx = df['Close'].values, df.index
    stride = max(1, round(252 / step))
    recs, k = [], 0
    positions = list(range(MIN_HISTORY_DAYS, len(df), step))
    # One pass per ticker (RSI and 200-day average computed once); each row still uses
    # only data up to its own date.
    for i, f in zip(positions, sbd.compute_feature_series(df, positions)):
        if f is None:
            continue
        rec = {'ticker': ticker, 'group': group, 'date': idx[i],
               'month': idx[i].strftime('%Y-%m'), 'k': k, 'stride': stride}
        rec.update(f)
        if i + horizon < len(df):
            path = close[i + 1:i + horizon + 1] / close[i]
            rec['fwd_worst'] = min(path.min() - 1, 0.0)       # worst loss from today's close
            rec['label_end'] = idx[i + horizon]
        else:
            rec['fwd_worst'] = np.nan
            rec['label_end'] = pd.NaT
        recs.append(rec)
        k += 1
    return recs


def finish_table(recs):
    tab = pd.DataFrame(recs)
    if len(tab):
        tab['label_end'] = pd.to_datetime(tab['label_end'])
    return tab


def build_table(prices, groups=None, step=21, horizon=DEFAULT_HORIZON):
    recs = []
    for t, df in prices.items():
        recs += table_for_ticker(df, t, (groups or {}).get(t, 'core'), step, horizon)
    return finish_table(recs)


def split_by_cutoff(tab, cutoff, thr=DEFAULT_THRESHOLD, train_groups=('core',)):
    """Train: outcome fully known by cutoff. Test: dated after cutoff. Leakage-checked."""
    cut = pd.Timestamp(cutoff)
    ok = tab['fwd_worst'].notna()
    tr = tab[tab['group'].isin(train_groups) & ok & (tab['label_end'] <= cut)]
    te = tab[ok & (tab['date'] > cut)]
    if len(tr) and len(te):
        if not tr['label_end'].max() <= cut < te['date'].min():
            raise RuntimeError("train/test leakage")   # not an assert: must survive python -O
    return tr, te


def labels(df, thr=DEFAULT_THRESHOLD):
    return (df['fwd_worst'] <= -thr).astype(int).values


# --------------------------------------------------------------- scoring
def safe_auc(y, x):
    return float(roc_auc_score(y, x)) if 0 < y.sum() < len(y) else float('nan')


def boot_diffs(y, sigs, months, ref, n_boot=300, seed=0):
    """AUC(signal) - AUC(ref), resampling calendar months (tickers move together)."""
    um = np.unique(months)
    where = {m: np.where(months == m)[0] for m in um}
    rng = np.random.default_rng(seed)
    out = {k: [] for k in sigs if k != ref}
    for _ in range(n_boot):
        idx = np.concatenate([where[m] for m in rng.choice(um, size=len(um), replace=True)])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        a_ref = roc_auc_score(yy, sigs[ref][idx])
        for k in out:
            out[k].append(roc_auc_score(yy, sigs[k][idx]) - a_ref)
    return {k: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for k, v in out.items() if len(v) >= 20}


def fit_shipped(tr, thr=DEFAULT_THRESHOLD):
    cols, mk = CANDIDATES[SHIPPED]
    y = labels(tr, thr)
    if y.min() == y.max():
        raise RuntimeError("Training labels contain a single class; need more history or tickers.")
    return mk().fit(tr[cols].values, y), cols, y


def validate_holdout(tab, cutoff, thr=DEFAULT_THRESHOLD, n_boot=300):
    """Train on data known by `cutoff`, test on later dates. Returns a model-card dict or None."""
    tr, te = split_by_cutoff(tab, cutoff, thr)
    if len(tr) < 300 or len(te) < 200:
        return None
    try:
        model, cols, _ = fit_shipped(tr, thr)
    except RuntimeError:
        return None
    te = te[te['group'] == 'core'] if (te['group'] == 'core').any() else te
    y = labels(te, thr)
    sigs = {'Volatility': te['volatility'].values,
            'Model': model.predict_proba(te[cols].values)[:, 1],
            'RSI': te['rsi'].values,
            'Price/200MA': te['ma_ratio'].values}
    ci = boot_diffs(y, sigs, te['month'].values, 'Volatility', n_boot).get('Model')
    return {
        'cutoff': str(pd.Timestamp(cutoff).date()),
        'holdout_start': str(te['date'].min().date()), 'holdout_end': str(te['date'].max().date()),
        'n_train': int(len(tr)), 'n_holdout': int(len(te)), 'holdout_base_rate': float(y.mean()),
        'auc_model': safe_auc(y, sigs['Model']), 'auc_volatility': safe_auc(y, sigs['Volatility']),
        'auc_rsi': safe_auc(y, sigs['RSI']), 'auc_ma_ratio': safe_auc(y, sigs['Price/200MA']),
        'ci_vs_volatility': list(ci) if ci else None,
    }


def make_bundle(tab, thr=DEFAULT_THRESHOLD, horizon=DEFAULT_HORIZON, validation=None,
                universe_size=None):
    """Train the shipped model on every row whose outcome is known, and package it."""
    tr = tab[(tab['group'] == 'core') & tab['fwd_worst'].notna()]
    model, cols, y = fit_shipped(tr, thr)
    p = model.predict_proba(tr[cols].values)[:, 1]
    p50, p99 = np.percentile(p, [50, 99])
    stats = {c: {'mean': float(tr[c].mean()), 'std': float(tr[c].std()), 'median': float(tr[c].median())}
             for c in cols + ['ma_ratio', 'accel']}
    scaler = model.named_steps['standardscaler']
    logistic = model.named_steps['logisticregression']
    raw = {
        'schema': sbd.MODEL_SCHEMA, 'cols': list(cols),
        # Plain numbers only: the model is stored as JSON, never as a pickle.
        'weights': {'mean': scaler.mean_, 'scale': scaler.scale_, 'coef': logistic.coef_[0],
                    'intercept': float(logistic.intercept_[0])},
        'feature_stats': stats,
        'score_p50': float(p50), 'score_p99': float(max(p99, p50 + 1e-6)),
        'base_rate': float(y.mean()), 'threshold': thr, 'horizon': horizon,
        'trained_on': datetime.today().strftime('%Y-%m-%d'),
        'trained_through': str(tr['label_end'].max().date()),
        'n_training_rows': int(len(tr)),
        'ticker_universe_size': universe_size if universe_size is not None else int(tr['ticker'].nunique()),
        'validation': validation,
    }
    try:
        return sbd.parse_bundle(sbd._to_jsonable(raw))
    except ValueError:
        # e.g. a validation AUC was NaN: ship the model as UNVALIDATED rather than fail.
        logger.warning("Validation result was unusable; the model is saved as unvalidated.")
        raw['validation'] = None
        return sbd.parse_bundle(sbd._to_jsonable(raw))


# --------------------------------------------- reference: the old v1 model
def fit_v1(rows):
    """The original Isolation Forest (kept only as a reference line in backtests)."""
    names = sbd.BASE_FEATURE_NAMES
    x = np.array([[r[n] for n in names] for r in rows])
    sc = StandardScaler()
    xs = sc.fit_transform(x)
    m = IsolationForest(n_estimators=300, contamination=0.05, random_state=42).fit(xs)
    lo, hi = np.percentile(m.score_samples(xs), [1, 99])
    return m, sc, float(lo), float(hi)


def score_v1(bundle, rows):
    m, sc, lo, hi = bundle
    names = sbd.BASE_FEATURE_NAMES
    raw = np.array([[r[n] for n in names] for r in rows])
    s = np.clip(100 * (hi - m.score_samples(sc.transform(raw))) / (hi - lo), 0, 100)
    z = (raw - sc.mean_) / sc.scale_
    gate = z[:, 0] + z[:, 1] + z[:, 3]
    return np.where(gate <= 0, np.minimum(s, 15.0), s)


# ----------------------------------------- multi-cutoff evaluation (backtest)
def evaluate_cutoff(tab, cutoff, thr=DEFAULT_THRESHOLD, n_boot=300, has_extra=False):
    tr, te = split_by_cutoff(tab, cutoff, thr)
    if len(tr) < 300 or len(te) < 200:
        return None
    ytr = labels(tr, thr)
    if ytr.min() == ytr.max():
        return None
    sigs, fitted = {'Volatility (raw)': te['volatility'].values}, {}
    sub = tr[tr['k'] % tr['stride'].iloc[0] == 0]
    if len(sub) >= 100:
        sigs['Overheat v1'] = score_v1(fit_v1(sub[sbd.BASE_FEATURE_NAMES].to_dict('records')),
                                       te[sbd.BASE_FEATURE_NAMES].to_dict('records'))
    for name, (cols, mk) in CANDIDATES.items():
        m = mk().fit(tr[cols].values, ytr)
        sigs[name] = m.predict_proba(te[cols].values)[:, 1]
        fitted[name] = m
    yte = labels(te, thr)
    subsets = [('core', (te['group'] == 'core').values)]
    if has_extra:
        subsets.append(('extra', (te['group'] == 'extra').values))
    rows = []
    for label, mask in subsets:
        for name, x in sigs.items():
            rows.append({'cutoff': str(cutoff), 'subset': label, 'signal': name,
                         'auc': safe_auc(yte[mask], x[mask]), 'n_test': int(mask.sum()),
                         'n_train': len(tr), 'test_base_rate': float(yte[mask].mean())})
    core = subsets[0][1]
    ci = boot_diffs(yte[core], {k: v[core] for k, v in sigs.items()},
                    te['month'].values[core], 'Volatility (raw)', n_boot)
    return {'rows': rows, 'ci': ci, 'sigs': sigs, 'y': yte, 'te': te, 'fitted': fitted,
            'train_base': float(ytr.mean()), 'n_train': len(tr), 'core': core}


# ------------------------------------------------------------------ data
DEFAULT_CACHE_DIR = os.environ.get(
    'OVERHEAT_CACHE_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '.price_cache'))


def _read_cached(path):
    """Read a cached price CSV (plain text, never a pickle); None if missing or corrupt."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    except (OSError, ValueError, pd.errors.ParserError):
        return None
    return sbd._clean_prices(df)


def get_prices(ticker, start, end, cache_dir=None):
    """
    Close prices for one ticker. With cache_dir, same-day downloads are reused from disk
    (the file name includes the end date, so the cache refreshes daily).
    """
    t = sbd.validate_ticker(ticker)
    if t is None:
        return None
    use_cache = bool(cache_dir) and sbd._parse_date(start) is not None and sbd._parse_date(end) is not None
    path = os.path.join(cache_dir, f"{t}_{start}_{end}.csv") if use_cache else None
    if path:
        cached = _read_cached(path)
        if cached is not None:
            return cached[['Close']]
    df = sbd.load_stock_data(t, start, end)
    if df is None or df.empty:
        return None
    df = df[['Close']].dropna().copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = path + '.tmp'
        df.to_csv(tmp)
        os.replace(tmp, path)
    return df


def prune_cache(cache_dir, keep_days=7):
    """Delete cached price files older than keep_days so the cache cannot grow forever."""
    if not cache_dir or not os.path.isdir(cache_dir):
        return 0
    cutoff = datetime.now().timestamp() - keep_days * 86400
    removed = 0
    for name in os.listdir(cache_dir):
        full = os.path.join(cache_dir, name)
        if name.endswith(('.csv', '.tmp')) and os.path.isfile(full) and os.path.getmtime(full) < cutoff:
            os.remove(full)
            removed += 1
    return removed
