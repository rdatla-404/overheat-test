# Overheat Score

### A crash-risk check for stocks: what the model thinks, why, and how far to trust it

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/ML-Logistic%20Regression-orange.svg)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![Data](https://img.shields.io/badge/Data-Yahoo%20Finance-purple.svg)](https://pypi.org/project/yfinance/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Enter a ticker (or a whole watchlist). The model estimates the chance the stock falls **20% or more within about 6 months**, tells you in plain English what is driving that estimate, shows whether it was already worried before past drops, and reports how it performed on data it never trained on, compared with a plain volatility screen.

**Goal:** help people avoid putting new money into stocks that are unusually likely to fall hard, especially after a run-up.

---

## What changed in this version

The first version scored how *unusual* a stock looked (Isolation Forest) and never checked whether high scores were followed by losses. A backtest showed it ranked later drops worse than plain volatility, and that RSI and price-vs-200-day-average alone had no skill in 2023-2026. This version fixes the design:

| Before | Now |
|---|---|
| Unsupervised anomaly score, never tied to an outcome | **Supervised** model trained on the outcome that matters: did the stock fall 20%+ within ~126 trading days? |
| Raw RSI / extension features | **Volatility-adjusted** features (a 30% stretch means little for a 60%-volatility stock and a lot for a 15% one) |
| Hand-set directional gate (it lowered accuracy) | Removed; the model learns direction from the data |
| 4 hand-picked "known cases" as validation | **Held-out validation**: trained on the past, tested on the most recent years, with a leakage check |
| No baseline | Always compared with **plain volatility**; the app shows the result and says so plainly if the model does not beat it |
| Numbers only | **Verdict, reasons, risk timeline, watchlist scan** |

---

## Features

- **A decision-oriented verdict** (Avoid new money / Caution / Watch / OK) with the model's estimated chance of a 20%+ drop versus a typical stock
- **Plain-English reasons** for the score, from the model's own coefficients
- **Risk timeline**: re-scores the stock at past dates using only data available then, so you can see whether it was already flagged before past drops
- **Watchlist scan**: paste tickers, get them sorted riskiest first
- **Model check built in**: held-out AUC for the model, volatility, RSI and price/200-day average, shown in the app and CLI
- **Multi-period backtest** (`backtest.py`) across several cutoffs, with bootstrap intervals
- **Company search** from a 7,000+ symbol NASDAQ/NYSE/AMEX directory
- **Tests** for the metrics, labels, train/test split (no leakage), scoring and validation

---

## Project structure

```
Overheat-Score/
├── stock_bubble_detector.py   # data loading, features, scoring, verdicts, scan, CLI
├── crash_model.py             # shared training / validation / backtest logic
├── train_bubble_model.py      # train + validate + save the model (run this first)
├── backtest.py                # multi-cutoff out-of-time backtest
├── app.py                     # Streamlit web app
├── all_tickers.csv            # ticker directory for company-name search
├── bubble_model.json          # the trained model (created by train_bubble_model.py; safe to commit)
├── test_*.py                  # tests: metrics, leakage, scoring, security, app
├── Dockerfile, .streamlit/, .github/workflows/ci.yml   # deployment and CI
├── requirements.txt, requirements-dev.txt
└── README.md
```

File names keep the original "bubble_detector" naming so existing imports keep working.

---

## Quick start

```bash
pip install -r requirements.txt
python train_bubble_model.py          # one time, a few minutes, needs internet
```

Training prints a **model card**: the held-out result against plain volatility. Read it before trusting any score.

```bash
streamlit run app.py                                    # web app
python stock_bubble_detector.py                         # interactive CLI
python stock_bubble_detector.py scan NVDA TSLA KO JNJ   # score a watchlist
python backtest.py --extra-test-tickers                 # multi-period backtest (10-15 min)
pytest                                                  # run the tests
```

On Windows, use `python -m pip install -r requirements.txt` so packages install into the same Python that runs the scripts.

---

## How it works

1. **Features** (computed from past prices only): volatility (90-day, annualized), RSI (14-day), stretch above the 200-day average per unit of volatility, acceleration of recent gains per unit of volatility, 6-month return per unit of volatility, and distance below the 52-week high.
2. **Label**: did the stock's price fall 20% or more below that day's close at any point in the next 126 trading days?
3. **Training**: logistic regression (interpretable) on snapshots taken every 21 trading days across a curated universe. A row is used only if its outcome was fully known at training time.
4. **Validation**: the model is trained on data known by a cutoff (default: two years ago) and tested on later dates. A leakage assertion guards the split. The result is compared with plain volatility using a bootstrap over calendar months.
5. **Score (0-100)**: relative to the training population. 0 = typical (median) risk or lower; 100 = riskier than 99% of training snapshots. The "model estimate" percentage is the probability and reflects the markets the model saw.

| Score | Verdict |
|---|---|
| 0-9 | OK: below-typical risk |
| 10-29 | OK: typical risk |
| 30-49 | Watch |
| 50-69 | Caution |
| 70-100 | Avoid new money |

---

## How far to trust it

The app and CLI show one of four statuses, taken from the held-out test:

- **Validated**: the model ranked later crashes better than plain volatility (interval above zero).
- **Not better than a plain volatility screen**: treat it as a volatility-driven risk screen, not a bubble detector.
- **Worse than a plain volatility screen**.
- **Not validated**: too little data to test.

Run `python backtest.py` to see whether the result holds across several periods. A model only "works" if it beats raw volatility on most cutoffs with an interval above zero.

---

## Security and deployment

- The model is stored as **JSON and validated on load**. Pickle files are never loaded (unpickling can run arbitrary code). Never commit a `.pkl`.
- Ticker symbols and dates are validated; price data is cleaned (blank/invalid rows dropped); the price cache is plain CSV; watchlist size is capped; user text is never treated as a regex.
- Set `OVERHEAT_PUBLIC=1` on a public server to turn off in-app training and lower the watchlist cap. Train locally, commit `bubble_model.json`, then deploy.
- `docker build -t overheat . && docker run -p 8501:8501 overheat` runs the app as a non-root user. CI runs ruff, pytest, bandit and pip-audit.
- Yahoo Finance data via yfinance is unofficial; check its terms before any public or commercial use.
- Put a reverse proxy or platform rate limit in front of a public deployment; the in-app throttle is only a courtesy.

---

## Honest limitations

- **It estimates crash risk, not bubbles.** It does not know earnings, news, valuation, or when a drop will happen.
- **Volatility does most of the work** in many periods. That is why volatility is the baseline to beat, and why the app tells you when the model does not beat it.
- **Survivorship bias.** The training universe is large companies that survived to today, which flatters any model. `backtest.py --extra-test-tickers` adds stocks that later fell hard as a test-only set; they are hand-picked with hindsight and are still listed, so this reduces the problem but does not remove it.
- **Overlapping windows and correlated stocks** mean the effective sample is much smaller than the row count, so intervals are optimistic.
- **One market period is not every market.** Results can change a lot between cutoffs.
- **The suggested strategies are general risk habits**, not trade instructions, and are not tested for profit.

This tool is for learning and exploration, not financial advice.

---

## License

MIT License. Free to use, modify, and distribute for any purpose.

*Built with Python, scikit-learn, yfinance, pandas, Streamlit*
