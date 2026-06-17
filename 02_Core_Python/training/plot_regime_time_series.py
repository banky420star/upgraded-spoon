#!/usr/bin/env python3
"""Plot regime detection time series: 5-colored regime bands on XAUUSD price chart.

Loads XAUUSD data, fits the RegimeDetector (via eval_harness),
classifies every bar, and renders a two-panel figure:

  Top panel:  Price line chart with regime-colored background bands
  Bottom:     Confidence scatter coloured by regime

Use this to visually verify that the detector's labels align with
market structure (uptrends → green, downtrends → red, ranging → gray,
volatile → orange, breakout → purple).
"""
from __future__ import annotations

import os
import sys
import warnings
from itertools import groupby

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

from drl.regime_detector import REGIME_LABELS, NUM_REGIMES
from training.eval_harness import fit_regime_detector, build_regime_observations
from Python.data_feed import fetch_training_data

# ── Config ───────────────────────────────────────────────────────────

SYMBOL = "XAUUSDm"
PERIOD = "30d"
INTERVAL = "5m"
MIN_BARS = 100  # skip early bars where indicators haven't stabilised

# Colour scheme (order matches REGIME_LABELS)
REGIME_COLORS = ["#2ecc71", "#e74c3c", "#95a5a6", "#e67e22", "#9b59b6"]
#  ↑trending_up  ↑trending_down ↑ranging    ↑volatile   ↑breakout
REGIME_ALPHA = 0.18

# ── Data ────────────────────────────────────────────────────────────

print(f"Loading {SYMBOL} data…")
df = fetch_training_data(
    SYMBOL, period=PERIOD, interval=INTERVAL, strict=False, require_fresh=False
)
print(f"  {len(df)} rows  {df.index[0]}  →  {df.index[-1]}")

# ── Classify every bar ─────────────────────────────────────────────

print("Fitting RegimeDetector…")
detector = fit_regime_detector(df)

n = len(df)
regime_idx = np.zeros(n, dtype=np.int32)
confidence = np.zeros(n, dtype=np.float32)

print("Classifying bars (70-bar rolling window)…")
for i in range(70, n):
    lookback = df.iloc[max(0, i - 70) : i + 1]
    fv = detector.compute_features(lookback)
    r, c = detector.classify(fv)
    regime_idx[i] = r
    confidence[i] = c

# Distribution
valid = slice(70, n)
uniq, cnts = np.unique(regime_idx[valid], return_counts=True)
print("  Regime distribution:")
for r, c in sorted(zip(uniq, cnts)):
    print(f"    {REGIME_LABELS[r]:<16} {c:>6}  ({c / (n - 70) * 100:.1f}%)")

# ── Plot ───────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(20, 9), sharex=True,
    gridspec_kw={"height_ratios": [2, 1]},
)

times = df.index
close = df["close"].values

# ── Top panel: price + regime bands ──

# Group consecutive bars with the same regime → far fewer axvspan calls
start = MIN_BARS
groups: list[tuple[int, int, int]] = []  # (start_idx, end_idx, regime_id)
for regime_id, grp in groupby(range(start, n), key=lambda i: int(regime_idx[i])):
    pos = list(grp)
    groups.append((pos[0], pos[-1] + 1, regime_id))

for gs, ge, rid in groups:
    ge = min(ge, n - 1)
    ax1.axvspan(times[gs], times[ge], facecolor=REGIME_COLORS[rid],
                alpha=REGIME_ALPHA, lw=0)

ax1.plot(times, close, color="#1a1a2e", linewidth=0.8, label=f"{SYMBOL} Close")
ax1.set_ylabel("Price", fontsize=11)
ax1.set_title(
    f"Regime Detection Time Series — {SYMBOL} ({PERIOD} @ {INTERVAL})",
    fontsize=13, fontweight="bold",
)
ax1.grid(True, alpha=0.3)

legend_handles = [
    Patch(facecolor=REGIME_COLORS[r], alpha=0.5, label=REGIME_LABELS[r])
    for r in range(NUM_REGIMES)
]
ax1.legend(handles=legend_handles, loc="upper left", fontsize=9, title="Regime")

# ── Bottom panel: confidence per regime ──

for r in range(NUM_REGIMES):
    mask = regime_idx == r
    ax2.scatter(times[mask], confidence[mask], color=REGIME_COLORS[r],
                s=6, alpha=0.5, label=REGIME_LABELS[r], marker=".")

ax2.set_ylabel("Confidence", fontsize=11)
ax2.set_ylim(-0.02, 1.05)
ax2.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
ax2.grid(True, alpha=0.3)
ax2.legend(loc="upper left", fontsize=8, ncol=5)

ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
fig.autofmt_xdate()

plt.tight_layout()
os.makedirs("logs", exist_ok=True)
out = "logs/regime_time_series.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}")
