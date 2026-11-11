# Overheat Score

### Price-behavior anomaly screening for publicly traded stocks

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/ML-Isolation%20Forest-orange.svg)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![Data](https://img.shields.io/badge/Data-Yahoo%20Finance-purple.svg)](https://pypi.org/project/yfinance/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Enter a stock ticker to see an **Overheating Anomaly Score (0–100)**. It measures how unusual recent price behavior is relative to the model's training data; it is not a probability of a bubble, crash, or loss.

---

## Why This Exists

Several major market run-ups showed a similar cluster of warning signs before conditions eventually cooled or corrected:

| Period | Year | Outcome | Notable Signal |
|--------|------|---------|-----------------|
| Dot-Com | 2000 | NASDAQ −78% | RSI > 85, parabolic price acceleration |
| Housing Crisis | 2008 | S&P 500 −57% | Extreme volatility, momentum collapse |
| Crypto Mania | 2021 | Bitcoin −77% | Price 3× above 200-day MA |
| Meme Stocks | 2021 | GameStop −90% | RSI > 90, extreme acceleration |

These are historical patterns, not a forecasting rule — this project doesn't predict *when* or *whether* a correction happens. Instead of hand-picking thresholds for those signals, it trains an **Isolation Forest** on real historical data across a diverse universe of tickers, and lets the model learn what "statistically normal" looks like on its own. A stock is flagged when its current feature combination sits far outside that learned normal — and only when the anomaly points toward overheating, not toward a crash.

---

## Features

- **ML-based screening** — Isolation Forest anomaly detection, not fixed point thresholds
- **Directional gating** — a stock cratering 40% won't score as overheated; only "overheating" anomalies count
- **Two ways to use it** — interactive CLI or a Streamlit web app
- **Company search** — look up a ticker by company name from a 7,000+ symbol NASDAQ/NYSE/AMEX directory
- **Model retraining** — retrain from the app's sidebar or via a configurable CLI script
- **Unit tests** — synthetic inputs check key metric calculations and edge cases
- **Chronological holdout diagnostics** — compares high anomaly scores with later returns and drawdowns, as exploratory evidence only

---

## Project Structure

```
overheat-score/
│
├── stock_bubble_detector.py    # Core: data loading, feature engineering, ML scoring, CLI
├── train_bubble_model.py       # Trains the Isolation Forest model (run this first)
├── app.py                      # Streamlit web app
├── all_tickers.csv             # NASDAQ/NYSE/AMEX ticker directory for company-name search
├── requirements.txt            # Dependency ranges
├── test_bubble_detector.py     # Unit tests (pytest)
└── README.md                   # This file
```

*Note: core module and file names still use the original "bubble_detector" naming internally — only the product's display name and branding changed to Overheat Score. Renaming the files themselves would require updating every import across `app.py`, `train_bubble_model.py`, and the tests, so they're left as-is intentionally.*

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model (one-time, ~5–10 minutes)
```bash
python train_bubble_model.py
```
This downloads several years of history across a curated, sector-diverse universe of tickers, builds feature snapshots, performs a chronological holdout analysis of subsequent 60-session returns/drawdowns, then fits and saves the model. The holdout output is exploratory association, not proof of predictive value. Override the defaults without touching source:
```bash
python train_bubble_model.py --years 4 --contamination 0.03 --step-days 63
```

### 3. Run it

**CLI:**
```bash
python stock_bubble_detector.py
```
```
  OVERHEAT SCORE -- INPUT
  Enter stock ticker symbol (e.g. AAPL, TSLA, NVDA): NVDA
  How many years of data to analyze? (1-5, default 2): 2

  OVERHEAT REPORT: NVDA
  Overheating anomaly score : 82.4 / 100 (not a probability)
  Signal label              : VERY UNUSUAL OVERHEATING SIGNAL

  Feature Z-Scores (vs. training population, |z| > 2 is unusual):
    RSI                  z= +2.41  [########------------]
    Price / 200-day MA   z= +3.02  [##########----------]
    Volatility           z= +1.15  [###-----------------]
    Acceleration         z= +2.88  [#########-----------]
    Max Drawdown         z= -0.62  [--------------------]
```

**Web app:**
```bash
streamlit run app.py
```

**Tests:**
```bash
pytest test_bubble_detector.py
```

---

## How Scoring Works

1. Five continuous features are computed for the ticker: **RSI (14-day)**, **price ÷ 200-day moving average**, **annualized volatility (trailing 90 days)**, **acceleration** (30-day return vs. the prior 30 days), and **max drawdown (trailing 252 days)**.
2. The trained Isolation Forest scores how statistically unusual that combination is versus its training population, producing a raw anomaly score.
3. The displayed score is scaled to **0–100** against the training set's score distribution. It is not calibrated as a probability.
4. A **directional gate** checks whether the anomaly points toward overheating (elevated RSI, MA ratio, and acceleration) rather than a crash or an unrelated kind of outlier. If it doesn't, the score is capped low regardless of how anomalous the raw signal was.

| Anomaly score | Display label | App prompt |
|-------------|-----------|-------------------|
| 0–9 | No strong overheating anomaly | No strong overheating anomaly detected |
| 10–29 | Mild overheating anomaly | Monitor |
| 30–49 | Elevated overheating anomaly | Review features and market context |
| 50–69 | High overheating anomaly | Review concentration and risk plan |
| 70–100 | Very unusual overheating signal | Review concentration and risk plan |

---

## How Training Works

`train_bubble_model.py` pulls history for a curated, sector-diverse set of ~60 large, liquid tickers (not "every ticker on the market" — see [Limitations](#honest-limitations) for why) and takes a feature snapshot from each one every **252 trading days** by default. That spacing is deliberate: it's the longest lookback window any single feature uses (max drawdown), so consecutive snapshots for the same ticker don't share overlapping price data — they're closer to independent observations rather than near-duplicates of each other. You can lower `--step-days` for a denser, faster training set, at the cost of reintroducing that correlation.

Training reports an exploratory chronological holdout: the model is fitted on earlier snapshots and the top anomaly quartile's subsequent 60-session returns and drawdowns are summarized on later snapshots. The fitted model records library versions, feature definitions, universe, date range, and a feature-row hash; the exact feature snapshots are saved to `training_snapshots.csv` beside the model. This does not establish that anomalies predict bubbles or future losses. The periods can overlap, the universe contains current listed companies only, and the method needs broader out-of-sample evaluation before any predictive claim.

---

## Honest Limitations

- **Curated ticker universe, not "all tickers."** Training on the full ~7,000-ticker market would drag the model's sense of "normal" toward thinly-traded penny stocks, making it worse at flagging overheating in stocks people actually watch. `all_tickers.csv` is a lookup directory for the app's search feature only — it's intentionally not the training set.
- **Isolation Forest flags statistical rarity, not "bubble" specifically.** The directional gate reduces false positives on crashes, but this is still anomaly detection, not a certified bubble classifier.
- **Survivorship bias.** The curated universe is based on companies that are listed today; historical snapshots do not include firms that later delisted or failed. That can make the training population look healthier than the true historical market.
- **Exploratory holdout only.** The chronological comparison is not a labeled bubble backtest and has no benchmark strategy or confidence intervals. Do not interpret it as evidence of investment performance.
- **No guarantee of predictive power.** A high score describes how unusual a stock's current behavior is relative to recent market history — it is not a forecast, and past patterns are not a guarantee of what happens next.

This tool is built for learning and exploration, not as financial advice. Its review prompts are generic, and any investment decision requires independent research and consideration of personal circumstances.

---

## Requirements

```
yfinance, pandas, numpy, matplotlib, scikit-learn, joblib, streamlit, pytest
```
`requirements.txt` gives bounded version ranges, not a fully reproducible lockfile. The training model records the actual library versions and saves its feature rows so a model fit can be audited; for repeatable fresh installs, generate and commit a lockfile in the target environment after validating the dependency set.

---

## License

MIT License — free to use, modify, and distribute for any purpose.

---

*Built with Python · scikit-learn · yfinance · pandas · Streamlit*
