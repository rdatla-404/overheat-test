# =============================================================
# These use small synthetic price series with known, hand-computable
# results rather than real market data -- the point is to catch a
# silent math bug in the formulas (e.g. a sign flip, an off-by-one
# window), not to validate anything about real markets.
# =============================================================

import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import stock_bubble_detector as sbd


def make_price_df(prices):
    """Helper: wrap a list of closing prices in the DataFrame shape
    the metric functions expect."""
    dates = pd.date_range('2020-01-01', periods=len(prices), freq='B')
    return pd.DataFrame({'Close': prices}, index=dates)


# -------------------------------------------------------------
# get_total_return
# -------------------------------------------------------------

def test_total_return_gain():
    data = make_price_df([100, 110, 120, 130])
    assert sbd.get_total_return(data) == 30.0


def test_total_return_loss():
    data = make_price_df([200, 150, 100])
    assert sbd.get_total_return(data) == -50.0


def test_total_return_single_row_is_zero():
    data = make_price_df([100])
    assert sbd.get_total_return(data) == 0.0


# -------------------------------------------------------------
# get_daily_returns
# -------------------------------------------------------------

def test_daily_returns_values():
    data = make_price_df([100, 110, 99])
    returns = sbd.get_daily_returns(data)
    assert len(returns) == 2
    assert returns.iloc[0] == pytest.approx(0.10, abs=1e-9)
    assert returns.iloc[1] == pytest.approx(-0.10, abs=1e-9)


# -------------------------------------------------------------
# get_volatility
# -------------------------------------------------------------

def test_volatility_zero_for_flat_price():
    data = make_price_df([100] * 30)
    assert sbd.get_volatility(data) == 0.0


def test_volatility_positive_for_varying_price():
    prices = [100, 105, 98, 110, 95, 108, 100, 112, 90, 115]
    data = make_price_df(prices)
    assert sbd.get_volatility(data) > 0.0


# -------------------------------------------------------------
# get_moving_average
# -------------------------------------------------------------

def test_moving_average_known_values():
    data = make_price_df([1, 2, 3, 4, 5])
    ma = sbd.get_moving_average(data, window=3)
    assert pd.isna(ma.iloc[0])
    assert pd.isna(ma.iloc[1])
    assert ma.iloc[2] == pytest.approx(2.0)
    assert ma.iloc[3] == pytest.approx(3.0)
    assert ma.iloc[4] == pytest.approx(4.0)


# -------------------------------------------------------------
# get_rsi
# -------------------------------------------------------------

def test_rsi_near_100_for_monotonic_increase():
    prices = [100 + i for i in range(40)]
    data = make_price_df(prices)
    rsi = sbd.get_rsi(data)
    assert rsi.iloc[-1] > 99.0


def test_rsi_near_zero_for_monotonic_decrease():
    prices = [200 - i for i in range(40)]
    data = make_price_df(prices)
    rsi = sbd.get_rsi(data)
    assert rsi.iloc[-1] < 1.0


def test_rsi_bounded_between_0_and_100():
    np.random.seed(0)
    prices = 100 * np.cumprod(1 + np.random.normal(0, 0.02, 60))
    data = make_price_df(prices)
    rsi = sbd.get_rsi(data)
    valid = rsi.dropna()
    assert (valid >= 0).all()
    assert (valid <= 100).all()


# -------------------------------------------------------------
# get_price_to_200ma_ratio
# -------------------------------------------------------------

def test_ma_ratio_none_when_insufficient_history():
    data = make_price_df([100] * 150)
    assert sbd.get_price_to_200ma_ratio(data) is None


def test_ma_ratio_is_one_for_flat_price():
    data = make_price_df([100] * 210)
    ratio = sbd.get_price_to_200ma_ratio(data)
    assert ratio is not None
    assert ratio.iloc[-1] == pytest.approx(1.0)


def test_ma_ratio_above_one_when_price_rising():
    # Flat for 200 days (to fill the MA window) then a jump, so price
    # sits above its own trailing 200-day average.
    prices = [100] * 200 + [150] * 10
    data = make_price_df(prices)
    ratio = sbd.get_price_to_200ma_ratio(data)
    assert ratio.iloc[-1] > 1.0


# -------------------------------------------------------------
# get_max_drawdown
# -------------------------------------------------------------

def test_max_drawdown_known_value():
    data = make_price_df([100, 120, 80, 90])
    # Peak 120 -> trough 80 => (80-120)/120*100 = -33.33...
    assert sbd.get_max_drawdown(data) == pytest.approx(-33.33, abs=0.01)


def test_max_drawdown_zero_for_monotonic_increase():
    data = make_price_df([100, 105, 110, 115])
    assert sbd.get_max_drawdown(data) == 0.0


# -------------------------------------------------------------
# compute_feature_vector
# -------------------------------------------------------------

def test_feature_vector_none_when_insufficient_history():
    data = make_price_df([100] * 100)
    assert sbd.compute_feature_vector(data) is None


def test_feature_vector_has_all_expected_keys():
    np.random.seed(1)
    prices = 100 * np.cumprod(1 + np.random.normal(0.0003, 0.015, 250))
    data = make_price_df(prices)
    features = sbd.compute_feature_vector(data)
    assert features is not None
    assert set(features.keys()) == set(sbd.FEATURE_NAMES)
    assert all(isinstance(v, float) for v in features.values())


# -------------------------------------------------------------
# assign_risk_level
# -------------------------------------------------------------

@pytest.mark.parametrize("score,expected", [
    (0, "NO STRONG OVERHEATING ANOMALY"),
    (9.9, "NO STRONG OVERHEATING ANOMALY"),
    (10, "MILD OVERHEATING ANOMALY"),
    (29.9, "MILD OVERHEATING ANOMALY"),
    (30, "ELEVATED OVERHEATING ANOMALY"),
    (49.9, "ELEVATED OVERHEATING ANOMALY"),
    (50, "HIGH OVERHEATING ANOMALY"),
    (69.9, "HIGH OVERHEATING ANOMALY"),
    (70, "VERY UNUSUAL OVERHEATING SIGNAL"),
    (100, "VERY UNUSUAL OVERHEATING SIGNAL"),
])
def test_assign_risk_level_boundaries(score, expected):
    assert sbd.assign_risk_level(score) == expected


def test_feature_vector_rejects_invalid_closes():
    prices = [100.0] * 210
    prices[-1] = np.nan
    assert sbd.compute_feature_vector(make_price_df(prices)) is None


def test_feature_vector_rejects_nonpositive_closes():
    prices = [100.0] * 210
    prices[-1] = 0.0
    assert sbd.compute_feature_vector(make_price_df(prices)) is None


def test_strategies_are_review_prompts_not_trade_orders():
    prompts = sbd.generate_strategies("XYZ", 100.0, 90.0)
    assert prompts
    assert all("sell" not in item["action"].lower() for item in prompts)
