"""Security (model file, inputs, cache) and speed-equivalence tests."""
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crash_model as cm
import stock_bubble_detector as sbd
from test_crash_model import synth_prices


@pytest.fixture(scope='module')
def good_bundle():
    vols = [0.008, 0.012, 0.018, 0.026] * 3
    prices = {f"T{i}": synth_prices(i, vol=v) for i, v in enumerate(vols)}
    return cm.make_bundle(cm.build_table(prices, step=21, horizon=126), validation=None)


@pytest.fixture
def model_path(tmp_path, monkeypatch):
    p = tmp_path / 'bubble_model.json'
    monkeypatch.setattr(sbd, 'MODEL_PATH', str(p))
    monkeypatch.setattr(sbd, '_MODEL_BUNDLE', None)
    return p


# ---------- model file ----------
def test_save_load_roundtrip(good_bundle, model_path):
    sbd.save_bundle(good_bundle, str(model_path))
    assert sbd._load_model_bundle()['cols'] == good_bundle['cols']
    assert not os.path.exists(str(model_path) + '.tmp')


def test_pickle_never_loaded(model_path):
    legacy = model_path.with_suffix('.pkl')

    class Evil:
        def __reduce__(self):
            return (os.system, ('echo pwned > ' + str(model_path.parent / 'pwned'),))
    legacy.write_bytes(pickle.dumps(Evil()))
    assert sbd._load_model_bundle() is None
    assert not (model_path.parent / 'pwned').exists()


@pytest.mark.parametrize('payload', [
    'not json', '[]', '{}', '{"schema": 999}', 'NaN', '{"schema": 2, "cols": "x"}',
])
def test_malformed_model_rejected(model_path, payload):
    model_path.write_text(payload)
    assert sbd._load_model_bundle() is None


def test_tampered_fields_rejected(good_bundle, model_path):
    for mutate in (
        lambda b: b['weights'].update(coef=b['weights']['coef'][:-1]),
        lambda b: b.update(cols=['volatility', '__import__']),
        lambda b: b['weights']['scale'].__setitem__(0, 0.0),
        lambda b: b.update(threshold=5),
    ):
        b = json.loads(json.dumps(sbd._to_jsonable(good_bundle)))
        mutate(b)
        model_path.write_text(json.dumps(b))
        sbd._MODEL_BUNDLE = None
        assert sbd._load_model_bundle() is None


def test_oversized_model_rejected(model_path, monkeypatch):
    monkeypatch.setattr(sbd, 'MAX_MODEL_BYTES', 10)
    model_path.write_text(' ' * 100)
    assert sbd._load_model_bundle() is None


# ---------- inputs ----------
@pytest.mark.parametrize('raw,ok', [
    ('aapl', 'AAPL'), (' brk-b ', 'BRK-B'), ('BF.B', 'BF.B'),
    ('../etc/passwd', None), ('A;rm -rf', None), ('', None), ('A' * 11, None),
    ('AA PL', None), ('<script>', None), (None, None), ('-AAPL', None),
])
def test_validate_ticker(raw, ok):
    assert sbd.validate_ticker(raw) == ok


def test_load_stock_data_rejects_bad_input():
    assert sbd.load_stock_data('../x', '2020-01-01', '2021-01-01') is None
    assert sbd.load_stock_data('AAPL', '2021-01-01', '2020-01-01') is None
    assert sbd.load_stock_data('AAPL', 'garbage', '2021-01-01') is None


def test_clean_prices_drops_bad_rows():
    idx = pd.bdate_range('2020-01-01', periods=5)
    df = pd.DataFrame({'Close': [1, np.nan, -3, np.inf, 5]}, index=idx)
    out = sbd._clean_prices(df)
    assert list(out['Close']) == [1.0, 5.0]
    assert sbd._clean_prices(pd.DataFrame({'Close': [np.nan]}, index=idx[:1])) is None


def test_scan_caps_and_flags_invalid(good_bundle, monkeypatch):
    monkeypatch.setattr(sbd, '_MODEL_BUNDLE', good_bundle)
    calls = []

    def loader(t, s, e):
        calls.append(t)
        return synth_prices(1, n=400)
    syms = [f"S{i}" for i in range(80)] + ['bad;sym', 'S1']
    sbd.scan_tickers(syms, loader=loader)
    assert len(calls) == len(set(calls)) <= sbd.MAX_SCAN_TICKERS
    assert 'bad;sym'.upper() not in calls


# ---------- price cache ----------
def test_cache_is_csv_and_rejects_bad_ticker(tmp_path, monkeypatch):
    monkeypatch.setattr(sbd, 'load_stock_data',
                        lambda t, s, e: synth_prices(3, n=300))
    s = cm.get_prices('AAPL', '2020-01-01', '2021-01-01', str(tmp_path))
    assert s is not None
    files = list(tmp_path.iterdir())
    assert files and all(f.suffix == '.csv' for f in files)
    assert cm.get_prices('../../evil', '2020-01-01', '2021-01-01', str(tmp_path)) is None
    assert all(f.parent == tmp_path for f in tmp_path.rglob('*'))


# ---------- optimization equivalence ----------
def test_feature_series_matches_vector():
    data = synth_prices(7, n=900, vol=0.02)
    idx = list(range(199, 900, 37)) + [899, 50]
    fast = sbd.compute_feature_series(data, idx)
    for i, f in zip(idx, fast, strict=True):
        slow = sbd.compute_feature_vector(data.iloc[:i + 1])
        if slow is None:
            assert f is None
            continue
        assert f is not None
        for k, v in slow.items():
            assert f[k] == pytest.approx(v, rel=1e-6, abs=1e-6), (i, k)


def test_feature_series_is_faster():
    data = synth_prices(8, n=2500, vol=0.02)
    idx = list(range(250, 2500, 21))
    t0 = time.perf_counter()
    sbd.compute_feature_series(data, idx)
    fast = time.perf_counter() - t0
    t0 = time.perf_counter()
    for i in idx:
        sbd.compute_feature_vector(data.iloc[:i + 1])
    slow = time.perf_counter() - t0
    print(f"series {fast:.3f}s vs per-snapshot {slow:.3f}s ({slow / fast:.0f}x)")
    assert fast < slow
