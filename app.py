# app.py
# Streamlit app for Overheat Score: a crash-risk check for stocks, with a decision-oriented
# verdict, plain-English reasons, a risk timeline, a watchlist scanner, and a model check.
#
# Deployment: set OVERHEAT_PUBLIC=1 on a public server. That turns off model training (a
# long, network-heavy job that must never run on a visitor's request) and lowers the
# watchlist cap. Train locally, then deploy the resulting bubble_model.json.

from datetime import datetime, timedelta
import html
import os
import re
import time

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import stock_bubble_detector as sbd
import train_bubble_model as trainer

st.set_page_config(page_title="Overheat Score", page_icon="chart", layout="wide")

PUBLIC = os.environ.get("OVERHEAT_PUBLIC", "0").strip().lower() in ("1", "true", "yes")
MAX_SCAN = 25 if PUBLIC else 40
MIN_SECONDS_BETWEEN_ANALYSES = 2.0
MIN_SECONDS_BETWEEN_SCANS = 10.0

LEVEL_COLORS = {"red": "#D32F2F", "orange": "#F57C00", "yellow": "#F9A825", "green": "#388E3C"}
STATUS_BADGE = {
    "validated": ("Validated on held-out data", "success"),
    "not_better_than_volatility": ("Not better than a plain volatility screen", "warning"),
    "worse_than_volatility": ("Worse than a plain volatility screen", "error"),
    "unvalidated": ("Not validated", "warning"),
}


@st.cache_data(ttl=3600, show_spinner=False, max_entries=64)
def cached_load_stock_data(ticker: str, start_date: str, end_date: str):
    """Cached wrapper so widget clicks do not re-hit Yahoo Finance; expires hourly."""
    return sbd.load_stock_data(ticker, start_date, end_date)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=64)
def cached_history(ticker: str, last_date: str, n_rows: int, model_stamp: str, _data):
    """The risk timeline is slow-ish and does not change between widget clicks."""
    return sbd.score_history(_data)


@st.cache_data(show_spinner=False)
def load_ticker_directory():
    """Static NASDAQ/NYSE/AMEX directory for the company-name search helper."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "all_tickers.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["symbol", "name", "exchange", "sector", "industry", "market_cap"])
    return pd.read_csv(path)


def status_message(status: str, note: str):
    label, kind = STATUS_BADGE.get(status, ("Unknown", "warning"))
    getattr(st, kind)(f"**{label}.** {note}")


def too_soon(key: str, seconds: float) -> bool:
    """Light per-session throttle (a real rate limit belongs in a proxy; see DEPLOY.md)."""
    now = time.monotonic()
    if now - st.session_state.get(key, -1e9) < seconds:
        return True
    st.session_state[key] = now
    return False


def show_figure(fig):
    """Render a matplotlib figure and free it; otherwise a long-running server leaks memory."""
    st.pyplot(fig)
    plt.close(fig)


# -------------------------------------------------------------
# Sidebar: model status (+ training controls when running locally)
# -------------------------------------------------------------
with st.sidebar:
    st.header("Model status")
    info = sbd.get_model_info()

    if info is None:
        st.warning("No trained model found yet." if PUBLIC
                   else "No trained model found yet. Train one below (needs internet, a few minutes).")
    else:
        st.caption(f"Trained: {info['trained_on']} | outcomes known through {info['trained_through']}")
        st.caption(f"{info['n_training_rows']} snapshots, {info['ticker_universe_size']} tickers")
        status_message(info["validation_status"], info["validation_note"])
        v = info.get("validation")
        if v:
            st.caption(
                f"Held-out AUC: model {v['auc_model']:.2f} | volatility {v['auc_volatility']:.2f} "
                f"| RSI {v['auc_rsi'] if v['auc_rsi'] is not None else float('nan'):.2f} "
                f"| price/200MA {v['auc_ma_ratio'] if v['auc_ma_ratio'] is not None else float('nan'):.2f} "
                "(0.5 = chance)"
            )

    st.divider()
    if PUBLIC:
        st.caption("Training is disabled on this deployment. Retrain locally and redeploy the "
                   "model file.")
    else:
        st.caption("Training settings")
        train_years = st.slider("Years of history per ticker", 5, 12, trainer.DEFAULT_YEARS_OF_HISTORY)
        holdout_years = st.slider("Years held out for validation", 1, 4, trainer.DEFAULT_HOLDOUT_YEARS)
        step_days = st.number_input(
            "Days between snapshots", min_value=21, max_value=126,
            value=trainer.DEFAULT_SNAPSHOT_STEP_DAYS, step=21,
            help="Smaller = more (overlapping) rows. 21 is monthly.",
        )
        all_symbols = trainer.TICKER_UNIVERSE
        with st.expander("Customize ticker universe"):
            chosen_tickers = st.multiselect("Tickers to train on", options=all_symbols, default=all_symbols)
        st.caption(f"Training on {len(chosen_tickers)} tickers -- takes several minutes.")

        if st.button("Train / retrain model", width="stretch"):
            progress_bar = st.progress(0.0)
            status_box = st.empty()

            def _on_progress(idx, total, ticker, snapshots, skipped):
                progress_bar.progress(idx / total)
                verb = "skipped" if skipped else f"{snapshots} snapshots"
                status_box.info(f"[{idx}/{total}] {ticker}: {verb}")

            tab = trainer.build_training_table(
                years=train_years, step_days=int(step_days), tickers=chosen_tickers,
                progress_callback=_on_progress,
            )
            status_box.info(f"Built {len(tab)} labeled snapshots. Validating and fitting...")
            try:
                trainer.train_and_save(
                    tab, holdout_years=holdout_years, ticker_universe_size=len(chosen_tickers),
                )
                sbd.reload_model()
                progress_bar.progress(1.0)
                status_box.success("Model trained, validated and saved.")
                st.rerun()
            except RuntimeError as e:
                status_box.error(str(e))

    st.divider()
    st.caption(
        "The score comes from a small logistic regression trained on one outcome: did the "
        "stock fall 20%+ within ~6 months? Features are volatility-adjusted. It is tested on "
        "data it never saw and compared with plain volatility. Not financial advice."
    )


# -------------------------------------------------------------
# Header
# -------------------------------------------------------------
st.title("Overheat Score")
st.caption("A crash-risk check for stocks: what the model thinks, why, and how far to trust it.")

tab_one, tab_scan, tab_help = st.tabs(["Check a stock", "Scan my watchlist", "How to read this"])


# -------------------------------------------------------------
# Tab 1: single stock
# -------------------------------------------------------------
with tab_one:
    with st.expander("Don't know the ticker? Search by company name"):
        ticker_dir = load_ticker_directory()
        if ticker_dir.empty:
            st.caption("Ticker directory (all_tickers.csv) not found next to this app.")
        else:
            search_text = st.text_input("Company name contains...", value="", key="company_search",
                                        max_chars=50).strip()
            if search_text:
                # regex=False: treat the text literally (a typed "(" must not crash the app,
                # and visitors must not be able to run their own regular expressions).
                matches = ticker_dir[ticker_dir["name"].str.contains(search_text, case=False, na=False,
                                                                     regex=False)]
                st.caption(f"{len(matches)} match(es) -- copy a symbol into the field below")
                st.dataframe(matches[["symbol", "name", "exchange", "sector"]].head(50),
                             width="stretch", hide_index=True)

    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        ticker_input = st.text_input("Ticker symbol", value="NVDA", max_chars=12).upper().strip()
    with col_b:
        years = st.slider("Years of chart", min_value=1, max_value=5, value=2)
    with col_c:
        st.write("")
        st.write("")
        analyze_clicked = st.button("Analyze", type="primary", width="stretch")

    if "last_ticker" not in st.session_state:
        st.session_state.last_ticker = None

    if analyze_clicked:
        clean = sbd.validate_ticker(ticker_input)
        if clean is None:
            st.error("Enter a plain ticker symbol such as AAPL or BRK-B "
                     "(letters, digits, '.' or '-', up to 10 characters).")
        elif too_soon("last_analyze", MIN_SECONDS_BETWEEN_ANALYSES):
            st.warning("Please wait a moment between requests.")
        else:
            end_date = datetime.today().strftime("%Y-%m-%d")
            # One extra year of warm-up so the risk timeline can start at the left edge.
            start_date = (datetime.today() - timedelta(days=365 * (years + 1))).strftime("%Y-%m-%d")
            with st.spinner(f"Downloading {clean} data..."):
                data = cached_load_stock_data(clean, start_date, end_date)
            if data is None or data.empty:
                st.error(f"Could not load data for '{clean}'. Check the symbol and try again.")
            else:
                st.session_state.last_ticker = clean
                st.session_state.data = data
                st.session_state.years = years

    if st.session_state.get("last_ticker") and "data" in st.session_state:
        ticker = st.session_state.last_ticker
        data = st.session_state.data
        shown_years = st.session_state.get("years", 2)
        shown = data[data.index >= data.index[-1] - pd.Timedelta(days=365 * shown_years)]

        st.subheader(f"{ticker} summary")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Price", f"${sbd.get_current_price(data):,.2f}")
        m2.metric("Total return (period)", f"{sbd.get_total_return(shown):+.1f}%")
        m3.metric("Volatility (ann.)", f"{sbd.get_volatility(data):.1f}%")
        m4.metric("Max drawdown", f"{sbd.get_max_drawdown(shown):.1f}%")
        m5.metric("RSI (14d)", f"{float(sbd.get_rsi(data).iloc[-1]):.1f}")
        ratio = sbd.get_price_to_200ma_ratio(data)
        m6.metric("Price / 200MA", f"{float(ratio.iloc[-1]):.2f}x" if ratio is not None else "N/A")

        def draw_price_rsi_chart():
            st.subheader("Price & RSI chart")
            show_figure(sbd.plot_stock_detail(ticker, data, show=False, save=False,
                                              view_start=shown.index[0]))

        result = sbd.run_bubble_analysis(ticker, data)

        if result["bubble_score"] is None:
            st.info(result["risk_level"])
            for w in result["warnings"]:
                st.caption(w)
            draw_price_rsi_chart()
        else:
            v = result["verdict"]
            color = LEVEL_COLORS[v["color"]]
            left, right = st.columns([1, 2])
            with left:
                st.markdown(
                    f"""
                    <div style="text-align:center; padding:1.4rem; border-radius:12px;
                                background-color:{color}22; border:2px solid {color};">
                        <div style="font-size:1.6rem; font-weight:800; color:{color};">
                            {html.escape(v['headline'])}</div>
                        <div style="font-size:2.4rem; font-weight:700; color:{color};">
                            {result['bubble_score']:.0f} / 100</div>
                        <div style="font-size:0.9rem; color:#555;">relative risk score<br>
                            0 = typical, 100 = riskier than 99% of training snapshots</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with right:
                st.markdown(f"### {ticker}: {v['headline']}")
                st.write(v["detail"])
                c1, c2, c3 = st.columns(3)
                c1.metric("Model estimate", f"{result['crash_probability']:.0%}")
                c2.metric("Typical stock", f"{result['base_rate']:.0%}")
                c3.metric("Compared with typical", f"{result['risk_ratio']:.1f}x")
                status_message(result["model_status"], result["model_note"])

            st.subheader("Why")
            if result["drivers"]:
                for d in result["drivers"]:
                    icon = "UP" if d["contribution"] > 0 else "DOWN"
                    st.markdown(f"- **[{icon}] {d['label']}**: {d['text']} ({d['direction']})")
            else:
                st.caption("No single factor stands out against a typical stock.")

            draw_price_rsi_chart()

            st.subheader("What to do with this")
            strategies = sbd.generate_strategies(
                ticker, sbd.get_current_price(data), result["bubble_score"],
                float(sbd.get_moving_average(data, 200).iloc[-1]) if len(data) >= 200 else None,
            )
            for s in strategies:
                with st.container(border=True):
                    st.markdown(f"**[{s['priority']}] {s['action']}**")
                    st.caption(s["detail"])
            st.caption("General risk habits, not trade instructions and not tested for profit. "
                       "Not financial advice.")


    elif not analyze_clicked:
        st.info("Enter a ticker and click Analyze to get started.")


# -------------------------------------------------------------
# Tab 2: watchlist scan
# -------------------------------------------------------------
with tab_scan:
    st.write(f"Paste the tickers you own or are considering (up to {MAX_SCAN}), separated by commas, "
             "spaces or new lines. They are scored and sorted riskiest first.")
    raw = st.text_area("Tickers", value="NVDA, TSLA, AAPL, KO, JNJ", height=100, max_chars=500)
    if st.button("Scan", type="primary"):
        tickers = [t for t in re.split(r"[,;\s]+", raw) if t]
        tickers = list(dict.fromkeys(t.upper() for t in tickers))[:MAX_SCAN]
        if not tickers:
            st.info("Enter at least one ticker.")
        elif too_soon("last_scan", MIN_SECONDS_BETWEEN_SCANS):
            st.warning("Please wait a few seconds between scans.")
        else:
            bar = st.progress(0.0)
            table = sbd.scan_tickers(tickers, progress_callback=lambda n, tot, t: bar.progress(n / tot),
                                     loader=cached_load_stock_data)
            bar.progress(1.0)
            show = table.copy()
            show["crash_prob"] = show["crash_prob"].map(lambda x: "" if pd.isna(x) else f"{x:.0%}")
            show["vs_typical"] = show["vs_typical"].map(lambda x: "" if pd.isna(x) else f"{x:.1f}x")
            show["score"] = show["score"].map(lambda x: "" if pd.isna(x) else f"{x:.0f}")
            show = show.rename(columns={"crash_prob": "20%+ drop chance (model)",
                                        "vs_typical": "vs typical", "main_driver": "Main driver"})
            st.dataframe(show, width="stretch", hide_index=True)
            info2 = sbd.get_model_info()
            if info2:
                status_message(info2["validation_status"], info2["validation_note"])


# -------------------------------------------------------------
# Tab 3: how to read this
# -------------------------------------------------------------
with tab_help:
    st.markdown(
        """
**What it estimates.** The chance a stock falls 20% or more (from today's close) at some point
in about the next 6 months, based on how stocks with similar volatility, momentum and stretch
behaved in the training data.

**The score (0-100).** Relative to the training population: 0 means typical (median) risk or
lower; 100 means riskier than 99% of training snapshots. It ranks risk; it is not a probability.
The "model estimate" percentage is the probability, and it reflects the markets the model saw.

**How far to trust it.** The sidebar shows the model's result on data it never trained on,
compared with plain volatility. If it is not better than volatility, treat the output as a
volatility-driven risk screen.

**What it cannot do.** It does not know earnings, news, valuation, or when a drop will happen.
Training stocks are large companies that survived to today, which flatters the model. One
market period is not every market. Educational use only; not financial advice.
        """
    )

st.divider()
st.caption("Educational tool, not financial advice. Price data from Yahoo Finance via yfinance "
           "(unofficial; check its terms before any commercial use).")
