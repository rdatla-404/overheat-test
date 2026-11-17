"""Headless app tests with Streamlit's AppTest (no network: data loader is stubbed)."""
import os
import sys

import pytest
from streamlit.testing.v1 import AppTest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import crash_model as cm
import stock_bubble_detector as sbd
from test_crash_model import synth_prices


@pytest.fixture
def model(tmp_path, monkeypatch):
    prices = {f"T{i}": synth_prices(i, vol=v) for i, v in enumerate([0.008, 0.012, 0.018, 0.026] * 3)}
    p = tmp_path / 'bubble_model.json'
    sbd.save_bundle(cm.make_bundle(cm.build_table(prices, step=21, horizon=126), validation=None), str(p))
    monkeypatch.setenv('OVERHEAT_MODEL_PATH', str(p))
    monkeypatch.setenv('OVERHEAT_PUBLIC', '1')
    return p


def _app():
    at = AppTest.from_file(os.path.join(HERE, 'app.py'), default_timeout=60)
    return at


def test_app_loads_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv('OVERHEAT_MODEL_PATH', str(tmp_path / 'none.json'))
    at = _app().run()
    assert not at.exception


def test_public_mode_hides_training(model):
    at = _app().run()
    assert not at.exception
    assert all('Train / retrain' not in b.label for b in at.sidebar.button)


def test_bad_ticker_gives_friendly_error(model):
    at = _app().run()
    ti = [t for t in at.text_input if t.label == 'Ticker symbol'][0]
    ti.set_value('A;DROP TABLE').run()
    [b for b in at.button if b.label == 'Analyze'][0].click().run()
    assert not at.exception
    assert any('plain ticker' in e.value for e in at.error)


def test_search_special_chars_do_not_crash(model):
    at = _app().run()
    s = [t for t in at.text_input if t.key == 'company_search']
    if s:
        s[0].set_value('(((.*[').run()
        assert not at.exception


def test_successful_analysis_renders(model, monkeypatch):
    monkeypatch.setattr(sbd, 'load_stock_data', lambda t, s, e: synth_prices(5, n=700, vol=0.02))
    at = _app().run()
    [b for b in at.button if b.label == 'Analyze'][0].click().run()
    assert not at.exception
    assert not at.error
    assert any('summary' in h.value for h in at.subheader)
