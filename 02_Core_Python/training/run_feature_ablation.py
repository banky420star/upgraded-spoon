"""
Feature Ablation Test Harness

Trains multiple RegimeRoutedPPO agents with different feature subsets to
measure the impact of each feature group on training performance.

Feature groups tested:
  - ALL: all features (baseline)
  - NO_VOLUME: remove volume-based features
  - NO_MOMENTUM: remove momentum/RVI/Stochastic features
  - NO_VOLATILITY: remove ATR/BB/volatility features
  - NO_TREND: remove EMA/slope/trend features

Each group is trained for a configurable number of timesteps on synthetic
data. Results are logged to a CSV for comparison.

Usage:
    python training/run_feature_ablation.py --steps 20000 --trials 1
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import gymnasium as gym
import torch as th
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from drl.regime_routed_policy import RegimeRoutedPPO, RegimeRoutedActorCriticPolicy

# Disable SB3 warnings
os.environ["SB3_VERBOSE"] = "0"


# ── Feature Group Definitions ──────────────────────────────────────────

# These are indices into the feature vector that correspond to each group.
# Since we're working with synthetic data, we define logical groups.
# In production, these would be derived from the feature pipeline's
# feature name groupings.

# Total features per bar (for ENGINEERED_V2)
BASE_FEATURES_PER_BAR = 21

# Indices within the 21 features that belong to each group
# (these are approximate and should be tuned to match the actual feature pipeline)
FEATURE_GROUPS = {
    "trend": {
        "indices": list(range(0, 5)),      # EMAs, slopes, structure
        "description": "EMA crossovers, trend slopes, price relative to MA",
    },
    "momentum": {
        "indices": list(range(5, 9)),       # RVI, RSI-related
        "description": "RVI, RSI, momentum oscillators",
    },
    "volatility": {
        "indices": list(range(9, 13)),      # ATR, Bollinger, volatility
        "description": "ATR, Bollinger width, volatility expansion",
    },
    "volume": {
        "indices": list(range(13, 17)),     # Volume, tick volume, volume features
        "description": "Volume ratio, tick volume, volume delta",
    },
    "other": {
        "indices": list(range(17, 21)),     # Residual features
        "description": "Price levels, spread, residual",
    },
}

ALL_FEATURE_INDICES = list(range(BASE_FEATURES_PER_BAR))


# ── Synthetic Data ─────────────────────────────────────────────────────

def make_synthetic_ohlcv(
    n_bars: int = 2000,
    seed: int = 42,
    regimes: int = 3,
) -> pd.DataFrame:
    """Generate synthetic OHLCV with known regime shifts."""
    np.random.seed(seed)
    idx = pd.date_range("2026-01-01", periods=n_bars, freq="5min", tz="UTC")
    price = np.zeros(n_bars)
    base = 100.0
    chunk = n_bars // max(regimes, 1)

    for r in range(regimes):
        start = r * chunk
        end = min(start + chunk, n_bars)
        sz = end - start
        for j in range(start, end):
            i = j - start
            if r == 0:  # Trending up
                price[j] = base * (1 + 0.0003 * i + 0.0015 * np.random.randn())
            elif r == 1:  # Ranging
                price[j] = base * (1 + 0.001 * chunk) + 0.003 * base * np.random.randn()
            else:  # Volatile
                price[j] = base * (1 + 0.001 * chunk) + 0.008 * base * np.sin(i * 0.05) + 0.005 * base * np.random.randn()
        if r == 0:
            base = price[end - 1] if end < n_bars else price[-1]

    df = pd.DataFrame({
        "open": price * (1 - 0.0004 * abs(np.random.randn(n_bars))),
        "high": price * (1 + 0.003 * abs(np.random.randn(n_bars))),
        "low": price * (1 - 0.003 * abs(np.random.randn(n_bars))),
        "close": price,
        "volume": 100 + 50 * np.random.rand(n_bars),
        "tick_volume": (100 + 50 * np.random.rand(n_bars)).astype(int),
    }, index=idx)
    df.index.name = "time"
    return df


def make_synthetic_features(
    df: pd.DataFrame,
    window_size: int = 100,
    ablation_group: Optional[str] = None,
) -> tuple[np.ndarray, int]:
    """Create synthetic observation vectors from OHLCV data.

    Simulates the feature pipeline by computing bar-level features from
    OHLCV, then optionally ablating a feature group.

    Args:
        df: OHLCV DataFrame.
        window_size: Number of bars per observation window.
        ablation_group: Feature group to ablate (set to zero), or None for all features.

    Returns:
        (observations, n_features_per_bar) tuple.
        observations: (n_windows, window_size * n_features_per_bar) array.
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values
    n = len(close)

    # Compute a rich set of bar-level features (21 per bar, simulating ENGINEERED_V2)
    features_per_bar = []

    for i in range(n):
        feats = []

        # ── Trend features (indices 0-4) ──
        # EMA slopes
        ema10 = np.mean(close[max(0, i - 10): i + 1])
        ema20 = np.mean(close[max(0, i - 20): i + 1])
        ema50 = np.mean(close[max(0, i - 50): i + 1])
        feats.append(ema10 / (ema20 + 1e-8) - 1.0)  # 0: EMA10/EMA20 ratio
        feats.append(ema20 / (ema50 + 1e-8) - 1.0)  # 1: EMA20/EMA50 ratio
        feats.append((ema10 - ema20) / (ema20 + 1e-8))  # 2: EMA crossover
        # Price relative to MA
        ma50 = np.mean(close[max(0, i - 50): i + 1])
        feats.append(close[i] / (ma50 + 1e-8) - 1.0)  # 3: Price vs MA50
        # Slope
        if i >= 20:
            slope = (close[i] - close[i - 20]) / (close[i - 20] + 1e-8)
        else:
            slope = 0.0
        feats.append(slope)  # 4: 20-bar slope

        # ── Momentum features (indices 5-8) ──
        if i >= 14:
            gains = np.diff(close[max(0, i - 14): i + 1])
            avg_gain = np.mean(gains[gains > 0]) if np.any(gains > 0) else 0
            avg_loss = abs(np.mean(gains[gains < 0])) if np.any(gains < 0) else 1e-8
            rsi = 100.0 - 100.0 / (1.0 + avg_gain / (avg_loss + 1e-8))
        else:
            rsi = 50.0
        feats.append(rsi / 100.0)  # 5: RSI normalised
        feats.append(1.0 - rsi / 100.0)  # 6: 1-RSI (mean reversion signal)
        # Momentum
        if i >= 10:
            mom = close[i] / (close[i - 10] + 1e-8) - 1.0
        else:
            mom = 0.0
        feats.append(mom)  # 7: 10-bar momentum
        feats.append(abs(mom))  # 8: momentum magnitude

        # ── Volatility features (indices 9-12) ──
        atr = max(high[i] - low[i], 1e-8)
        feats.append(atr / (close[i] + 1e-8))  # 9: ATR / close
        # Bollinger width
        if i >= 20:
            ma20 = np.mean(close[max(0, i - 20): i + 1])
            std20 = np.std(close[max(0, i - 20): i + 1])
            bb_width = 4.0 * std20 / (ma20 + 1e-8)
        else:
            bb_width = 0.0
        feats.append(bb_width)  # 10: BB width
        feats.append(atr / (np.mean([max(high[j] - low[j], 1e-8) for j in range(max(0, i - 20), i + 1)]) + 1e-8))  # 11: ATR ratio
        feats.append(np.std(close[max(0, i - 10): i + 1]) / (close[i] + 1e-8))  # 12: 10-bar std dev

        # ── Volume features (indices 13-16) ──
        vol_i = volume[i]
        vol_ma10 = np.mean(volume[max(0, i - 10): i + 1])
        feats.append(vol_i / (vol_ma10 + 1e-8))  # 13: Volume ratio
        feats.append(vol_i / (np.mean(volume[max(0, i - 50): i + 1]) + 1e-8))  # 14: Volume vs long MA
        tick_vol = df.get("tick_volume", volume)[i]
        feats.append(tick_vol / (vol_i + 1e-8))  # 15: Tick volume ratio
        feats.append((vol_i - vol_ma10) / (vol_ma10 + 1e-8))  # 16: Volume delta

        # ── Residual features (indices 17-20) ──
        high_low = (high[i] - low[i]) / (low[i] + 1e-8)  # 17: Range
        feats.append(high_low)
        feats.append((close[i] - low[i]) / (high[i] - low[i] + 1e-8))  # 18: Close position in range
        feats.append(np.log(volume[i] + 1))  # 19: Log volume
        feats.append(close[i])  # 20: Raw close (normalised later)

        features_per_bar.append(feats)

    features = np.array(features_per_bar, dtype=np.float32)

    # ── Apply ablation: zero out the specified feature group ──
    if ablation_group and ablation_group in FEATURE_GROUPS:
        group = FEATURE_GROUPS[ablation_group]
        features[:, group["indices"]] = 0.0
        print(f"  Ablated group '{ablation_group}': {group['description']}")
    elif ablation_group and ablation_group != "ALL":
        print(f"  Unknown ablation group '{ablation_group}', using ALL features")

    n_features = features.shape[1]

    # Normalise each feature column
    for col in range(n_features):
        vals = features[:, col]
        mean = np.mean(vals)
        std = np.std(vals) + 1e-8
        features[:, col] = (vals - mean) / std

    # Create windowed observations (flattened window of bars)
    obs_list = []
    for i in range(window_size, n):
        window = features[i - window_size: i]  # [window_size, n_features]
        obs_list.append(window.reshape(-1))     # flatten

    observations = np.array(obs_list, dtype=np.float32)
    return observations, n_features


# ── Training Runner ────────────────────────────────────────────────────

def make_env(obs: np.ndarray, window_size: int = 100) -> DummyVecEnv:
    """
    Create a DummyVecEnv that feeds pre-computed observations.

    This replaces the full TradingEnv for ablation testing so we can
    focus on the feature utilisation question without env complexity.
    """
    index = [0]

    def _init() -> object:
        import numpy as np

        obs_dim = obs.shape[1]
        portfolio_dim = 0  # No portfolio state for synthetic data

        class _SimpleEnv(gym.Env):
            metadata = {"render_modes": []}

            def __init__(self):
                super().__init__()
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(obs_dim + portfolio_dim,), dtype=np.float32
                )
                self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
                self._step = window_size

            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                self._step = window_size
                return obs[self._step - window_size].copy(), {}

            def step(self, action):
                self._step += 1
                idx = self._step
                if idx >= len(obs):
                    return obs[-1].copy(), 0.0, True, False, {}

                state = obs[idx].copy()
                reward = float(np.mean(state[:5])) * 0.01
                return state, reward, False, False, {}

        return _SimpleEnv()

    return DummyVecEnv([_init])


def run_trial(
    ablation_group: str,
    observations: np.ndarray,
    window_size: int,
    regime_dim: int,
    total_timesteps: int,
    trial_id: int = 0,
    verbose: bool = False,
) -> dict:
    """
    Run a single training trial with a given feature ablation.

    Args:
        ablation_group: Feature group name to ablate, or "ALL" for baseline.
        observations: Pre-computed observation array.
        window_size: Number of bars per observation window.
        regime_dim: Regime feature dimension.
        total_timesteps: Number of training timesteps.
        trial_id: Trial index (for logging).
        verbose: If True, print progress.

    Returns:
        Dict with training results.
    """
    obs_dim = observations.shape[1]
    n_features = obs_dim // window_size

    if verbose:
        print(f"\n{'='*60}")
        print(f"Trial {trial_id}: Ablation '{ablation_group}'")
        print(f"  Observations: {observations.shape[0]} windows x {obs_dim} dims")
        print(f"  Features per bar: {n_features}")
        print(f"  Total timesteps: {total_timesteps}")
        print(f"{'='*60}")

    # Create environment
    env = make_env(observations, window_size=window_size)

    # Regime detector (synthetic: use a simple heuristic based on recent vol)
    # We inject regime features into the observation tail
    try:
        from drl.regime_detector import RegimeDetector, NUM_REGIMES
        rd = RegimeDetector(use_patterns=False)
    except Exception:
        rd = None

    # Build policy kwargs with regime routing
    policy_kwargs = {
        "features_extractor_class": None,  # Will use default if we don't override
        "net_arch": {"pi": [64, 64], "vf": [64, 64]},
        "num_regimes": 5,
        "regime_dim": regime_dim,
    }

    try:
        start_time = time.time()

        model = RegimeRoutedPPO(
            RegimeRoutedActorCriticPolicy,
            env,
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
            policy_kwargs=policy_kwargs,
            regime_loss_coef=0.05,
            verbose=0,
        )

        model.learn(total_timesteps=total_timesteps)

        elapsed = time.time() - start_time

        # Collect metrics
        result = {
            "ablation_group": ablation_group,
            "trial_id": trial_id,
            "total_timesteps": total_timesteps,
            "elapsed_seconds": round(elapsed, 1),
            "completed": True,
            "status": "ok",
        }

        if verbose:
            print(f"  ✓ Completed in {elapsed:.1f}s")

    except Exception as exc:
        elapsed = time.time() - start_time if 'start_time' in dir() else 0
        result = {
            "ablation_group": ablation_group,
            "trial_id": trial_id,
            "total_timesteps": total_timesteps,
            "elapsed_seconds": round(elapsed, 1),
            "completed": False,
            "status": str(exc),
        }
        if verbose:
            print(f"  ✗ Failed: {exc}")

    return result


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Feature ablation test harness for RegimeRoutedPPO"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20000,
        help="Total training timesteps per trial (default: 20000)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of trials per ablation group (default: 1)",
    )
    parser.add_argument(
        "--groups",
        type=str,
        nargs="+",
        default=["ALL", "no_volume", "no_momentum", "no_volatility", "no_trend"],
        help="Feature groups to test",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/feature_ablation_results.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )
    args = parser.parse_args()

    print(f"Feature Ablation Test Harness")
    print(f"{'='*60}")
    print(f"Steps per trial: {args.steps}")
    print(f"Trials per group: {args.trials}")
    print(f"Groups: {', '.join(args.groups)}")
    print(f"{'='*60}\n")

    # Generate synthetic data
    print("Generating synthetic data...")
    df = make_synthetic_ohlcv(n_bars=5000, regimes=4)

    # Cache observations for each ablation group to avoid recomputation
    obs_cache = {}
    for group in args.groups:
        ablation = group.replace("no_", "").upper() if group.startswith("no_") else None
        if ablation and ablation not in FEATURE_GROUPS:
            ablation = None

        print(f"\nBuilding observations for '{group}'...")
        observations, n_features = make_synthetic_features(
            df, window_size=100, ablation_group=ablation
        )
        obs_cache[group] = observations

    # Set regime_dim for the observation
    regime_dim = 6  # 5 one-hot + confidence

    # Run trials
    all_results = []
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    for group in args.groups:
        observations = obs_cache[group]

        for trial in range(args.trials):
            result = run_trial(
                ablation_group=group,
                observations=observations,
                window_size=100,
                regime_dim=regime_dim,
                total_timesteps=args.steps,
                trial_id=trial,
                verbose=args.verbose,
            )
            all_results.append(result)

            # Save incremental results
            _save_results(all_results, args.output)

    # Final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    completed = [r for r in all_results if r["completed"]]
    failed = [r for r in all_results if not r["completed"]]
    print(f"Total trials: {len(all_results)}")
    print(f"Completed: {len(completed)}")
    print(f"Failed: {len(failed)}")
    if completed:
        avg_time = np.mean([r["elapsed_seconds"] for r in completed])
        print(f"Average time per trial: {avg_time:.1f}s")
    print(f"\nResults saved to: {args.output}")


def _save_results(results: list[dict], path: str):
    """Save results to CSV."""
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
