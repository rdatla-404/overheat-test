# Tests for the crash-risk model plumbing. They use synthetic price series so they run
# offline and fast; they check that the pipeline is wired correctly (no leakage, labels
# computed from the future only, scores behave sensibly), not that it predicts real markets.

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import crash_model as cm
import stock_bubble_detector as sbd


def synth_prices(seed, n=1700, vol=0.015, start='2015-01-01'):
    rng = np.random.default_rng(seed)
    p = 100 * np.exp(np.cumsum(rng.normal(0.0004, vol, n)))
    return pd.DataFrame({'Close': p}, index=pd.bdate_range(start, periods=n))


@pytest.fixture(scope='module')
def table():
    vols = [0.008, 0.010, 0.012, 0.015, 0.018, 0.022, 0.026, 0.030]
    prices = {f"T{i}": synth_prices(i, vol=v) for i, v in enumerate(vols * 2)}
    return cm.build_table(prices, step=21, horizon=126)


def test_table_has_features_and_labels(table):
    assert set(sbd.FEATURE_NAMES) <= set(table.columns)
    assert table['fwd_worst'].notna().any() and table['fwd_worst'].isna().any()
    assert (table['fwd_worst'].dropna() <= 0).all()          # a "worst loss" is never positive


def test_label_uses_only_future_prices():
    df = synth_prices(0, n=400)
    rows = cm.table_for_ticker(df, 'X', 'core', step=21, horizon=126)
    r = rows[0]
    i = df.index.get_loc(r['date'])
    path = df['Close'].values[i + 1:i + 127] / df['Close'].values[i]
    assert r['fwd_worst'] == pytest.approx(min(path.min() - 1, 0.0))
    assert r['label_end'] == df.index[i + 126]


def test_split_has_no_leakage(table):
    cutoff = '2019-06-30'
    tr, te = cm.split_by_cutoff(table, cutoff)
    assert len(tr) > 0 and len(te) > 0
    assert tr['label_end'].max() <= pd.Timestamp(cutoff)
    assert te['date'].min() > pd.Timestamp(cutoff)


def test_split_excludes_unresolved_outcomes(table):
    tr, te = cm.split_by_cutoff(table, '2019-06-30')
    assert tr['fwd_worst'].notna().all() and te['fwd_worst'].notna().all()


def test_bundle_scores_riskier_stock_higher(table, monkeypatch):
    bundle = cm.make_bundle(table, validation=None)
    monkeypatch.setattr(sbd, '_MODEL_BUNDLE', bundle)
    calm = sbd.run_bubble_analysis('CALM', synth_prices(101, n=400, vol=0.008))
    wild = sbd.run_bubble_analysis('WILD', synth_prices(101, n=400, vol=0.035))
    assert calm['bubble_score'] is not None and wild['bubble_score'] is not None
    assert wild['crash_probability'] > calm['crash_probability']
    assert wild['bubble_score'] >= calm['bubble_score']


def test_analysis_result_structure(table, monkeypatch):
    monkeypatch.setattr(sbd, '_MODEL_BUNDLE', cm.make_bundle(table, validation=None))
    r = sbd.run_bubble_analysis('X', synth_prices(5, n=400))
    for key in ('bubble_score', 'risk_level', 'warnings', 'sub_scores', 'crash_probability',
                'base_rate', 'risk_ratio', 'verdict', 'drivers', 'model_status', 'model_note'):
        assert key in r
    assert 0 <= r['bubble_score'] <= 100
    assert 0 <= r['crash_probability'] <= 1
    assert r['model_status'] == 'unvalidated'
    assert r['verdict']['headline']


def test_insufficient_data_and_untrained(table, monkeypatch):
    monkeypatch.setattr(sbd, '_MODEL_BUNDLE', cm.make_bundle(table, validation=None))
    short = sbd.run_bubble_analysis('X', synth_prices(5, n=100))
    assert short['risk_level'] == 'INSUFFICIENT DATA' and short['bubble_score'] is None
    monkeypatch.setattr(sbd, '_MODEL_BUNDLE', None)
    monkeypatch.setattr(sbd, 'MODEL_PATH', '/nonexistent/model.pkl')
    none = sbd.run_bubble_analysis('X', synth_prices(5, n=400))
    assert none['risk_level'] == 'MODEL NOT TRAINED'


def test_score_history_uses_only_past_data(table, monkeypatch):
    bundle = cm.make_bundle(table, validation=None)
    monkeypatch.setattr(sbd, '_MODEL_BUNDLE', bundle)
    df = synth_prices(7, n=600)
    full = sbd.score_history(df, step=21)
    cut = sbd.score_history(df.iloc[:500], step=21)
    common = cut.index.intersection(full.index)
    # a date's score must not change when later data is removed
    common = [d for d in common if d != cut.index[-1]]
    assert len(common) > 3
    assert np.allclose(full.loc[common, 'score'], cut.loc[common, 'score'])


def test_validation_status_labels():
    assert sbd.validation_status({}) == 'unvalidated'
    mk = lambda ci: {'validation': {'ci_vs_volatility': ci, 'auc_model': .7, 'auc_volatility': .6,
                                    'holdout_start': 'a', 'holdout_end': 'b'}}
    assert sbd.validation_status(mk([0.1, 0.05, 0.15])) == 'validated'
    assert sbd.validation_status(mk([-0.1, -0.15, -0.05])) == 'worse_than_volatility'
    assert sbd.validation_status(mk([0.0, -0.05, 0.05])) == 'not_better_than_volatility'


def test_verdict_levels():
    assert sbd.build_verdict(85, 0.5, 0.2)['headline'] == 'AVOID NEW MONEY'
    assert sbd.build_verdict(55, 0.4, 0.2)['headline'] == 'CAUTION'
    assert sbd.build_verdict(35, 0.3, 0.2)['headline'] == 'WATCH'
    assert sbd.build_verdict(5, 0.1, 0.2)['headline'] == 'OK: BELOW-TYPICAL RISK'


def test_validate_holdout_returns_model_card(table):
    v = cm.validate_holdout(table, '2018-12-31', n_boot=30)
    assert v is not None
    for k in ('auc_model', 'auc_volatility', 'auc_rsi', 'auc_ma_ratio', 'n_holdout'):
        assert k in v
    assert 0 <= v['auc_model'] <= 1
