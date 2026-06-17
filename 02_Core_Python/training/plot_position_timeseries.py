#!/usr/bin/env python3
"""Position time series: ALL vs no_regime on real XAUUSD data.

Trains both models on real XAUUSD data, evaluates on test set,
and plots the position time series (+ price overlay) to compare
holding consistency vs whipsawing.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

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

from stable_baselines3 import PPO

from drl.regime_routed_policy import RegimeRoutedPPO, RegimeRoutedActorCriticPolicy
from drl.regime_detector import REGIME_LABELS
from training.eval_harness import (
    fit_regime_detector,
    build_regime_observations,
    make_eval_env,
    collect_positions,
    collect_metrics,
    REGIME_DIM,
)

# Real data loader
from Python.data_feed import fetch_training_data

# Ablation harness helpers
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "ablation", os.path.join(PROJECT_ROOT, "training", "run_feature_ablation.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
make_synthetic_features = _mod.make_synthetic_features

# === Config ===

SYMBOL = "XAUUSDm"
PERIOD = "30d"
INTERVAL = "5m"
TIMESTEPS = 50000
WINDOW_SIZE = 100


# === Data ===

print(f"Loading {SYMBOL} data...")
df = fetch_training_data(SYMBOL, period=PERIOD, interval=INTERVAL, strict=False, require_fresh=False)
print(f"  {len(df)} rows, {df.index[0]} to {df.index[-1]}")

obs_df = pd.DataFrame({
    "close": df["close"],
    "high": df["high"],
    "low": df["low"],
    "volume": df["volume"],
    "tick_volume": df["volume"],
})
features_raw, n_per_bar, _regime_feats = make_synthetic_features(obs_df, window_size=WINDOW_SIZE, ablation_group=None)

print("Fitting RegimeDetector on XAUUSD data...")
detector = fit_regime_detector(df)

observations = build_regime_observations(features_raw, df, detector, WINDOW_SIZE, REGIME_DIM)

split = int(len(observations) * 0.8)
train_obs = observations[:split]
test_obs = observations[split:]
print(f"Features: {len(observations)} windows, train={len(train_obs)}, test={len(test_obs)}")

test_times = df.index[WINDOW_SIZE + split:]
if len(test_times) > len(test_obs):
    test_times = test_times[:len(test_obs)]

# === Train ALL ===

policy_kw = dict(net_arch=[64, 64], num_regimes=5, regime_dim=REGIME_DIM)

print("\nTraining RegimeRoutedPPO (ALL)...")
t0 = time.time()
env_all = make_eval_env(train_obs, df=df, window_size=WINDOW_SIZE)
all_model = RegimeRoutedPPO(
    RegimeRoutedActorCriticPolicy, env_all,
    learning_rate=3e-4, n_steps=256, batch_size=64, n_epochs=10,
    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
    max_grad_norm=0.5, policy_kwargs=policy_kw, regime_loss_coef=0.05,
    verbose=0,
)
all_model.learn(total_timesteps=TIMESTEPS)
all_time = time.time() - t0
print(f"  Done in {all_time:.1f}s")

# === Train no_regime ===

print("\nTraining plain PPO (no_regime)...")
t0 = time.time()
env_nr = make_eval_env(train_obs, df=df, window_size=WINDOW_SIZE)
nr_model = PPO(
    "MlpPolicy", env_nr,
    learning_rate=3e-4, n_steps=256, batch_size=64, n_epochs=10,
    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
    max_grad_norm=0.5, policy_kwargs=dict(net_arch=[64, 64]),
    verbose=0,
)
nr_model.learn(total_timesteps=TIMESTEPS)
nr_time = time.time() - t0
print(f"  Done in {nr_time:.1f}s")

# === Evaluate: collect position time series ===

print("\nEvaluating ALL...")
all_pos, all_rews = collect_metrics(all_model, test_obs, df, WINDOW_SIZE)
print(f"  {len(all_pos)} steps")

print("Evaluating no_regime...")
nr_pos, nr_rews = collect_metrics(nr_model, test_obs, df, WINDOW_SIZE)
print(f"  {len(nr_pos)} steps")

n_steps = min(len(all_pos), len(nr_pos), len(test_times))
all_pos = all_pos[:n_steps]
nr_pos = nr_pos[:n_steps]
all_rews = all_rews[:n_steps]
nr_rews = nr_rews[:n_steps]
ts = test_times[:n_steps]

# === Metrics ===

all_changes = int((np.diff(all_pos) != 0).sum())
nr_changes = int((np.diff(nr_pos) != 0).sum())

all_turnover = float(np.mean(np.abs(np.diff(all_pos))))
nr_turnover = float(np.mean(np.abs(np.diff(nr_pos))))

all_reward = float(all_rews.sum())
nr_reward = float(nr_rews.sum())

all_winrate = float((all_rews > 0).mean()) * 100
nr_winrate = float((nr_rews > 0).mean()) * 100

all_sharpe = float(np.mean(all_rews) / (np.std(all_rews) + 1e-8))
nr_sharpe = float(np.mean(nr_rews) / (np.std(nr_rews) + 1e-8))

print("\n" + "=" * 50)
print("  Position Time Series Summary")
print("=" * 50)
for name, pos, chg, rew, wr, sh in [
    ("ALL", all_pos, all_changes, all_reward, all_winrate, all_sharpe),
    ("no_regime", nr_pos, nr_changes, nr_reward, nr_winrate, nr_sharpe),
]:
    short_pct = (pos == -1).mean() * 100
    flat_pct = (pos == 0).mean() * 100
    long_pct = (pos == 1).mean() * 100
    chg_rate = chg / len(pos) * 100
    print(f"  {name:<12} Long={long_pct:.1f}% Flat={flat_pct:.1f}% Short={short_pct:.1f}% | "
          f"Changes={chg} ({chg_rate:.1f}% of steps) | Net={np.mean(pos):+.3f} | "
          f"Reward={rew:+.4f} WinRate={wr:.1f}% Sharpe={sh:.4f}")
print(f"  Whipsaw ratio (NR/ALL): {nr_changes / max(all_changes, 1):.2f}x more changes")

# === Save CSV ===

os.makedirs("logs", exist_ok=True)
csv_path = "logs/position_metrics.csv"
import csv as _csv
with open(csv_path, "w", newline="") as f:
    w = _csv.writer(f)
    w.writerow(["model", "long_pct", "flat_pct", "short_pct", "changes",
                 "turnover", "reward", "win_rate", "sharpe"])
    for name, pos, chg, turn, rew, wr, sh in [
        ("ALL", all_pos, all_changes, all_turnover, all_reward, all_winrate, all_sharpe),
        ("no_regime", nr_pos, nr_changes, nr_turnover, nr_reward, nr_winrate, nr_sharpe),
    ]:
        short_pct = round((pos == -1).mean() * 100, 1)
        flat_pct = round((pos == 0).mean() * 100, 1)
        long_pct = round((pos == 1).mean() * 100, 1)
        w.writerow([name, long_pct, flat_pct, short_pct, chg,
                    round(turn, 4), round(rew, 4), round(wr, 1), round(sh, 4)])
print(f"  Saved: {csv_path}")

# === Plot ===

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True,
                                gridspec_kw={"height_ratios": [1.5, 1]})

close_vals = df["close"].values[WINDOW_SIZE - 1 + split: WINDOW_SIZE - 1 + split + n_steps]
ax1.plot(ts, close_vals[:len(ts)], color="#2c3e50", linewidth=0.8, label=f"{SYMBOL} Close")
ax1.set_ylabel("Price", fontsize=11)
ax1.legend(loc="upper left", fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_title(f"ALL vs no_regime Position Time Series — {SYMBOL} ({PERIOD} @ {INTERVAL})",
              fontsize=13, fontweight="bold")

ax2.step(ts, all_pos, where="post", color="#3498db", linewidth=1.5,
         alpha=0.9, label=f"ALL (RegimeRoutedPPO) — {all_changes} changes")
ax2.step(ts, nr_pos, where="post", color="#e67e22", linewidth=1.5,
         alpha=0.6, label=f"no_regime (PPO) — {nr_changes} changes")
ax2.set_ylabel("Position", fontsize=11)
ax2.set_ylim(-1.3, 1.3)
ax2.set_yticks([-1, 0, 1])
ax2.set_yticklabels(["Short", "Flat", "Long"], fontsize=10)
ax2.legend(loc="upper left", fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")

ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
fig.autofmt_xdate()

plt.tight_layout()
os.makedirs("logs", exist_ok=True)
out = "logs/position_timeseries.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}")
