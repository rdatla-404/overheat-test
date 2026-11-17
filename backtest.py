"""
backtest.py -- does the model hold up across different market periods?

Trains at several cutoff dates (default 2020-12-31 through 2023-12-31), tests each on the
dates after it, and compares the model with plain volatility, the original v1 Isolation
Forest, and other candidate models. A model only "works" if it beats raw volatility on most
cutoffs with an interval above zero; the report says so plainly when it does not.

    python backtest.py
    python backtest.py --extra-test-tickers   # also test on stocks that later fell hard

Outputs (backtest_out/): summary.md, auc_by_cutoff.png, roc.png, results.csv.
Needs internet (Yahoo Finance); about 10-15 minutes. Educational use; not financial advice.
"""
import argparse
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

import crash_model as cm
import train_bubble_model as tbm

# Stocks that later fell hard. TEST ONLY: never used for training. They are still-listed
# survivors picked with hindsight, so expect inflated numbers on this subset.
EXTRA_TEST_TICKERS = ['ZM', 'PTON', 'COIN', 'RIVN', 'GME', 'AMC', 'CVNA', 'DOCU', 'MRNA',
                      'SNAP', 'LCID', 'HOOD', 'RBLX', 'AFRM', 'UPST', 'BYND', 'TDOC', 'ROKU',
                      'FSLY', 'SPCE']
COLORS = {'Volatility (raw)': '#6A1B9A', 'Overheat v1': '#8A1538', 'Vol-only LR': '#999999',
          'Base-5 LR': '#1F3A5F', 'Vol-adjusted LR': '#005035', 'Boosted trees (all)': '#EE7B1F'}


def md_table(df):
    try:
        import tabulate  # noqa: F401
        return df.to_markdown(index=False)
    except ImportError:
        return "```\n" + df.to_string(index=False) + "\n```"


def verdict(ci):
    return "BEATS volatility" if ci[1] > 0 else ("WORSE than volatility" if ci[2] < 0 else "tie")


def plot_auc_by_cutoff(results, path):
    df = pd.DataFrame(results)
    df = df[df.subset == 'core']
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, g in df.groupby('signal'):
        g = g.sort_values('cutoff')
        ax.plot(g['cutoff'].str[:4], g['auc'], marker='o',
                lw=3 if name == cm.SHIPPED else 1.8, color=COLORS.get(name, '#444'), label=name)
    ax.axhline(0.5, color='#999', ls='--', lw=1)
    ax.set_xlabel("Train through (test = after this year)", fontsize=12)
    ax.set_ylabel("Out-of-sample AUC (0.5 = chance)", fontsize=12)
    ax.set_title("Predicting a 20%+ drop within 6 months", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, frameon=False)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_roc(r, path):
    y, mask = r['y'][r['core']], r['core']
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, x in r['sigs'].items():
        fpr, tpr, _ = roc_curve(y, x[mask])
        ax.plot(fpr, tpr, lw=3 if name == cm.SHIPPED else 1.8, color=COLORS.get(name, '#444'),
                label=f"{name} ({roc_auc_score(y, x[mask]):.2f})")
    ax.plot([0, 1], [0, 1], color='#999', ls='--', lw=1)
    ax.set_xlabel("False alarm rate", fontsize=12)
    ax.set_ylabel("Share of 20%+ drops caught", fontsize=12)
    ax.legend(loc='lower right', fontsize=10, frameon=False)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Multi-cutoff out-of-time backtest.")
    ap.add_argument('--cutoffs', nargs='+', default=['2020-12-31', '2021-12-31', '2022-12-31', '2023-12-31'])
    ap.add_argument('--primary-cutoff', default='2022-12-31')
    ap.add_argument('--start', default='2014-01-01')
    ap.add_argument('--end', default=datetime.today().strftime('%Y-%m-%d'))
    ap.add_argument('--horizon', type=int, default=cm.DEFAULT_HORIZON)
    ap.add_argument('--crash-threshold', type=float, default=cm.DEFAULT_THRESHOLD)
    ap.add_argument('--step', type=int, default=tbm.DEFAULT_SNAPSHOT_STEP_DAYS)
    ap.add_argument('--bootstrap', type=int, default=300)
    ap.add_argument('--extra-test-tickers', action='store_true')
    ap.add_argument('--outdir', default='backtest_out')
    ap.add_argument('--cache', default=cm.DEFAULT_CACHE_DIR)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    core = list(tbm.TICKER_UNIVERSE)
    extra = [t for t in EXTRA_TEST_TICKERS if t not in core] if args.extra_test_tickers else []
    print(f"1/4 Downloading prices ({len(core)} core + {len(extra)} extra tickers)...")
    prices, groups = {}, {}
    for t in core + extra:
        df = cm.get_prices(t, args.start, args.end, args.cache)
        if df is None or len(df) < cm.MIN_HISTORY_DAYS + 30:
            print(f"  skipping {t}: no usable data")
            continue
        prices[t], groups[t] = df, ('core' if t in core else 'extra')
    n_core = sum(g == 'core' for g in groups.values())
    if n_core == 0:
        sys.exit("No core data downloaded. Check your internet connection.")

    print("2/4 Building the labeled feature table (slow step)...")
    tab = cm.build_table(prices, groups, args.step, args.horizon)
    print(f"  {len(tab)} snapshots across {tab['ticker'].nunique()} tickers")

    print("3/4 Training and testing at each cutoff...")
    results, evals = [], {}
    for c in args.cutoffs:
        r = cm.evaluate_cutoff(tab, c, args.crash_threshold, args.bootstrap, bool(extra))
        if r is None:
            print(f"  cutoff {c}: not enough data; skipped")
            continue
        evals[c] = r
        results += r['rows']
        print(f"  cutoff {c}: train {r['n_train']}, test {int(r['core'].sum())} core rows")
    if not evals:
        sys.exit("No cutoff had enough data. Try an earlier --start or other --cutoffs.")
    res = pd.DataFrame(results)
    res.to_csv(os.path.join(args.outdir, 'results.csv'), index=False)

    print("4/4 Writing report...")
    core_res = res[res.subset == 'core']
    pivot = core_res.pivot(index='signal', columns='cutoff', values='auc').round(3).reset_index()
    md = ["# Crash-risk model: out-of-time results\n",
          f"Target: a fall of {int(args.crash_threshold * 100)}%+ within {args.horizon} trading days. "
          "Training rows are used only if their outcome was known by the cutoff; tests use later dates. "
          "AUC 0.5 = chance. Core tickers only; the bar to beat is raw volatility.\n",
          "## AUC by cutoff (core tickers)\n", md_table(pivot), "",
          "## Does each model beat raw volatility? (bootstrap over months, 95% interval)\n"]
    wins = {}
    for c, r in evals.items():
        for name, ci in r['ci'].items():
            wins.setdefault(name, []).append((c, ci))
    summary = [{'model': n, 'cutoffs tested': len(wl),
                'beats volatility': sum(ci[1] > 0 for _, ci in wl),
                'worse than volatility': sum(ci[2] < 0 for _, ci in wl),
                'mean AUC gain': round(float(np.mean([ci[0] for _, ci in wl])), 3)}
               for n, wl in wins.items()]
    md.append(md_table(pd.DataFrame(summary)))
    md.append("\nPer cutoff:\n")
    for c, r in evals.items():
        for name, ci in r['ci'].items():
            md.append(f"- {c}, {name}: AUC difference {ci[0]:+.3f} (95% {ci[1]:+.3f} to {ci[2]:+.3f}) -> {verdict(ci)}")
    if extra:
        ex = res[res.subset == 'extra'].pivot(index='signal', columns='cutoff', values='auc').round(3).reset_index()
        md += ["\n## Extra 'later fell hard' tickers (hand-picked with hindsight: inflated)\n", md_table(ex)]
    md.append("\n## Read this before quoting any number\n"
              "- A win means beating raw volatility on most cutoffs with an interval above zero.\n"
              "- Tickers move together and windows overlap, so intervals are optimistic.\n"
              "- The core universe is today's large survivors; extras are hindsight-picked.\n"
              "- If results flip between cutoffs, report that. Not financial advice.\n")
    with open(os.path.join(args.outdir, 'summary.md'), 'w') as fh:
        fh.write("\n".join(md))
    pc = args.primary_cutoff if args.primary_cutoff in evals else list(evals)[-1]
    plot_auc_by_cutoff(results, os.path.join(args.outdir, 'auc_by_cutoff.png'))
    plot_roc(evals[pc], os.path.join(args.outdir, 'roc.png'))

    print("\n" + "=" * 70)
    print("AUC BY CUTOFF (core tickers; 0.5 = chance; bar to beat = Volatility (raw))")
    print(pivot.to_string(index=False))
    print()
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"\nFull report: {os.path.join(args.outdir, 'summary.md')}")


if __name__ == '__main__':
    main()
