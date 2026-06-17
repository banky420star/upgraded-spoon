#!/usr/bin/env python3
"""Plot per-regime action distributions for the ALL (RegimeRoutedPPO) model.

Trains a RegimeRoutedPPO model on synthetic data, evaluates on the test set,
and records per-step (position, regime_label) pairs.  Generates a 5-panel
figure showing the action distribution conditioned on the regime classifier's
current prediction.
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch as th

from drl.regime_routed_policy import RegimeRoutedPPO, RegimeRoutedActorCriticPolicy
from drl.regime_detector import REGIME_LABELS
from training.eval_harness import (
    fit_regime_detector,
    build_regime_observations,
    make_eval_env,
    REGIME_DIM,
)

# Import helpers from the ablation harness
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "ablation", os.path.join(PROJECT_ROOT, "training", "run_feature_ablation.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
make_synthetic_ohlcv = _mod.make_synthetic_ohlcv
make_synthetic_features = _mod.make_synthetic_features


# ════════════════════════════════════════════════════════════════════════
# Generate data
# ════════════════════════════════════════════════════════════════════════

print("Generating synthetic data...")
df = make_synthetic_ohlcv(5000, 4)
features_arr, n_per_bar, *_ = make_synthetic_features(df)

print("Fitting RegimeDetector...")
detector = fit_regime_detector(df)

window_size = 100
observations = build_regime_observations(features_arr, df, detector, window_size, REGIME_DIM)

split_idx = int(len(observations) * 0.8)
train_obs = observations[:split_idx]
test_obs = observations[split_idx:]

print(f"Observations: {len(observations)}  train={len(train_obs)}  test={len(test_obs)}")

# ════════════════════════════════════════════════════════════════════════
# Train ALL (RegimeRoutedPPO)
# ════════════════════════════════════════════════════════════════════════

print("\nTraining RegimeRoutedPPO (ALL) ...")
train_env = make_eval_env(train_obs, df=df, window_size=window_size)

model = RegimeRoutedPPO(
    RegimeRoutedActorCriticPolicy,
    train_env,
    learning_rate=3e-4,
    n_steps=256,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(
        net_arch=[64, 64],
        num_regimes=5,
        regime_dim=REGIME_DIM,
    ),
    regime_loss_coef=0.05,
    verbose=0,
)

t0 = time.time()
model.learn(total_timesteps=50000)
train_time = time.time() - t0
print(f"Training completed in {train_time:.1f}s")

# ════════════════════════════════════════════════════════════════════════
# Evaluate + collect per-step (position, regime_label) pairs
# ════════════════════════════════════════════════════════════════════════

print("\nEvaluating on test set with regime labels...")

vec_env = make_eval_env(test_obs, df=df, window_size=window_size)
current_obs = vec_env.reset()

positions = []
regime_indices = []

model.policy.set_training_mode(False)

while True:
    # Get regime label BEFORE predicting action (from current observation)
    obs_tensor = th.as_tensor(
        current_obs, dtype=th.float32, device=model.policy.device
    )
    with th.no_grad():
        probs = model.policy.get_regime_probs(obs_tensor)
        regime_idx = int(probs.argmax(dim=1).item())

    # Predict action
    action, _ = model.predict(current_obs, deterministic=True)
    position = float(np.sign(np.mean(action)))

    positions.append(position)
    regime_indices.append(regime_idx)

    current_obs, _, done, _ = vec_env.step(action)
    if done[0]:
        break

positions_arr = np.array(positions)
regime_indices_arr = np.array(regime_indices)

print(f"Collected {len(positions_arr)} steps")
print(
    f"Regime distribution: {dict(zip(*np.unique(regime_indices_arr, return_counts=True)))}"
)

# ════════════════════════════════════════════════════════════════════════
# Plot per-regime action distributions
# ════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)

colors_short = "#e74c3c"
colors_flat = "#95a5a6"
colors_long = "#2ecc71"

for r in range(5):
    ax = axes[r]
    mask = regime_indices_arr == r
    pos_r = positions_arr[mask]
    n_r = len(pos_r)

    if n_r == 0:
        ax.text(
            0.5,
            0.5,
            "No steps",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
            color="gray",
        )
        ax.set_title(f"{REGIME_LABELS[r]}\n(0 steps)", fontsize=10)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Short", "Flat", "Long"])
        continue

    n_short = int((pos_r == -1).sum())
    n_flat = int((pos_r == 0).sum())
    n_long = int((pos_r == 1).sum())

    bars = ax.bar(
        ["Short (-1)", "Flat (0)", "Long (+1)"],
        [n_short, n_flat, n_long],
        color=[colors_short, colors_flat, colors_long],
        edgecolor="white",
        linewidth=0.8,
    )

    for bar, count in zip(bars, [n_short, n_flat, n_long]):
        pct = count / n_r * 100
        if pct > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(1, n_r * 0.01),
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_title(
        f"{REGIME_LABELS[r]}\n({n_r} steps)", fontsize=10, fontweight="bold"
    )
    ax.set_ylim(0, max(n_short, n_flat, n_long) * 1.25)
    ax.tick_params(axis="x", labelsize=8)

axes[0].set_ylabel("Step count", fontsize=10)

fig.suptitle(
    "Per-Regime Action Distribution — RegimeRoutedPPO (ALL)",
    fontsize=13,
    fontweight="bold",
    y=1.02,
)

plt.tight_layout()
os.makedirs("logs", exist_ok=True)
out_path = "logs/regime_action_distribution.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out_path}")

# Print summary table
print("\n" + "=" * 60)
print("  Per-Regime Action Distribution Summary")
print("=" * 60)
print(f"  {'Regime':<16} {'Steps':>6} {'Short%':>8} {'Flat%':>8} {'Long%':>8}")
print("  " + "-" * 46)
for r in range(5):
    mask = regime_indices_arr == r
    n = mask.sum()
    if n > 0:
        short_pct = (positions_arr[mask] == -1).mean() * 100
        flat_pct = (positions_arr[mask] == 0).mean() * 100
        long_pct = (positions_arr[mask] == 1).mean() * 100
    else:
        short_pct = flat_pct = long_pct = 0.0
    print(
        f"  {REGIME_LABELS[r]:<16} {n:>6} "
        f"{short_pct:>7.1f}% {flat_pct:>7.1f}% {long_pct:>7.1f}%"
    )
print()
