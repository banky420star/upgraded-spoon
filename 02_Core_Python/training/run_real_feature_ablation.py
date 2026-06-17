"""
Real Feature Ablation Test Harness (ENGINEERED_V2)

Trains multiple RegimeRoutedPPO agents on real XAUUSDm data with different
feature subsets ablated from the ENGINEERED_V2 env feature matrix.

Feature groups tested (59-column env matrix):
  - ALL: all features (baseline)
  - NO_TREND: zero out trend features (htf_trend, vol_bucket)
  - NO_MOMENTUM: zero out momentum features (log_ret1, log_ret5, log_ret20)
  - NO_VOLATILITY: zero out realized volatility (rv_20)
  - NO_VOLUME: zero out volume features (rel_volume, spread_est_bps)
  - NO_CROSS_ASSET: zero out 6 live cross-asset features (USDJPYm)
  - NO_ML_SIGNAL: zero out XGBoost signal feature
  - NO_REGIME: disable regime detector (regime_dim=0)

Each group is trained for a configurable number of timesteps on real data.
Results are logged to a CSV for comparison.

Usage:
    python training/run_real_feature_ablation.py --symbol XAUUSDm --steps 30000 --trials 1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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

from drl.adaptive_feature_extractor import AdaptiveLSTMFeatureExtractor
from drl.regime_routed_policy import RegimeRoutedPPO, RegimeRoutedActorCriticPolicy

# Disable SB3 warnings
os.environ["SB3_VERBOSE"] = "0"


# ── Feature Group Definitions (name-based, not index-based) ─────────────

# ENGINEERED_V2 column order — single source of truth for column indices.
# Imported from Python.feature_registry — single source of truth
from Python.feature_registry import (
    ENGINEERED_V2_COLUMNS, FEATURE_GROUPS_BY_NAME, FEATURE_GROUPS,
    PAUSED_GROUPS, ABLATION_GROUPS,
)



# Feature set cardinality (derived from column name list)
FULL_FEATURE_COUNT = len(ENGINEERED_V2_COLUMNS)


# ── Fingerprinting ──────────────────────────────────────────────────

def matrix_fingerprint(x: np.ndarray) -> str:
    """Stable 12-char MD5 fingerprint of a numpy array.

    Converts NaN → 0, casts to float32, and hashes the raw bytes.
    """
    arr = np.nan_to_num(x).astype(np.float32)
    return hashlib.md5(arr.tobytes()).hexdigest()[:12]


def _model_weight_fingerprint(model) -> str:
    """Hash all trainable model parameters into a 12-char fingerprint.

    Two models with identical weights produce the same fingerprint.
    """
    parts = []
    for _name, param in sorted(model.policy.named_parameters()):
        parts.append(param.detach().cpu().numpy().ravel())
    if not parts:
        return "no_params"
    flat = np.concatenate(parts)
    return hashlib.md5(flat.astype(np.float32).tobytes()).hexdigest()[:12]


def _action_distribution_entropy(positions: np.ndarray, n_bins: int = 5) -> float:
    """Binned Shannon entropy of the position distribution.

    Uses adaptive binning based on the actual range of positions so that
    a policy operating in a narrow band (e.g. [-0.2, 0.2]) is NOT falsely
    reported as zero entropy. Bins are spread over [pos_min, pos_max] with
    5% padding on each side.

    Interpretation:
      - 0.000 = all positions in one bucket → policy stuck / saturated
      - 0.500 = positions spread across ~sqrt(n_bins) buckets
      - 1.000 = uniform across all buckets → maximally diverse
    """
    positions = np.asarray(positions, dtype=np.float64)
    if len(positions) == 0:
        return 0.0

    pmin, pmax = float(positions.min()), float(positions.max())
    if abs(pmax - pmin) < 1e-12:
        # All positions identical: zero entropy
        return 0.0

    # Adaptive binning: spread n_bins over [pmin, pmax] with 5% padding
    pad = 0.05 * (pmax - pmin)
    lo = pmin - pad
    hi = pmax + pad
    bins = np.linspace(lo, hi, n_bins + 1)
    bins[0] = -np.inf
    bins[-1] = np.inf

    counts, _ = np.histogram(positions, bins=bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # drop empty bins for entropy calc
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(max(n_bins, 2))
    normalised = float(entropy / max_entropy) if max_entropy > 0 else 0.0
    return round(normalised, 6) + 0.0  # coerce -0.0 to 0.0


# ── Real Data & Features ──────────────────────────────────────────────

def load_real_data(
    symbol: str = "XAUUSDm",
    n_bars: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """Load real OHLCV data for the given symbol.

    Falls back to fetching from the data_feed pipeline; if that fails,
    downloads from a Python data source or generates synthetic data as
    a last resort so the harness is always runnable.
    """
    try:
        from Python.data_feed import fetch_training_data

        df = fetch_training_data(symbol, bars=n_bars)
        if df is not None and len(df) >= 1000:
            print(f"  Loaded {len(df)} bars of {symbol} from data_feed")
            return df
    except Exception as exc:
        print(f"  data_feed unavailable ({exc}), trying MT5 download...")

    # Fallback: try downloading via MT5
    try:
        import MetaTrader5 as mt5

        if mt5.initialize():
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, n_bars)
            mt5.shutdown()
            if rates is not None and len(rates) >= 1000:
                df = pd.DataFrame(rates)
                df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
                df.set_index("time", inplace=True)
                print(f"  Loaded {len(df)} bars of {symbol} from MT5")
                return df
    except Exception as exc:
        print(f"  MT5 download failed ({exc}), using synthetic data...")

    # Last resort: synthetic data spanning a realistic price range
    print("  WARNING: Using synthetic data — results will not reflect real market structure")
    np.random.seed(seed)
    idx = pd.date_range("2026-01-01", periods=n_bars, freq="5min", tz="UTC")
    price = 100.0 * np.exp(np.cumsum(np.random.randn(n_bars) * 0.0005))
    df = pd.DataFrame({
        "open": price * (1 - 0.0003 * np.abs(np.random.randn(n_bars))),
        "high": price * (1 + 0.002 * np.abs(np.random.randn(n_bars))),
        "low": price * (1 - 0.002 * np.abs(np.random.randn(n_bars))),
        "close": price,
        "volume": 100 + 50 * np.random.rand(n_bars),
        "tick_volume": (100 + 50 * np.random.rand(n_bars)).astype(int),
    }, index=idx)
    df.index.name = "time"
    return df


def build_real_features(
    df: pd.DataFrame,
    symbol: str = "XAUUSDm",
    ablation_group: Optional[str] = None,
) -> tuple[np.ndarray, int]:
    """Build the ENGINEERED_V2 env feature matrix from real OHLCV data.

    Optionally ablates (zeros out) a feature group to measure its impact.

    Args:
        df: OHLCV DataFrame from load_real_data().
        symbol: Trading symbol (for cross-asset features).
        ablation_group: Feature group to ablate, or "ALL" / None for full set.

    Returns:
        (observations, n_features_per_bar) tuple.
        observations: (n_windows, window_size * n_features_per_bar) array
                      where each row is a flattened window of bars.
    """
    try:
        from Python.feature_pipeline import build_env_feature_matrix
        env_matrix = build_env_feature_matrix(df, symbol=symbol)
    except Exception as exc:
        print(f"  Feature pipeline failed ({exc}), building inline...")
        env_matrix = _build_features_fallback(df, symbol)

    n_features = env_matrix.shape[1]
    print(f"  Feature matrix: {n_features} columns x {len(env_matrix)} bars")

    # ── Apply ablation: zero out the specified feature group ──
    if ablation_group and ablation_group in FEATURE_GROUPS:
        group = FEATURE_GROUPS[ablation_group]
        idx = [i for i in group["indices"] if 0 <= i < n_features]
        if idx:
            env_matrix[:, idx] = 0.0
            print(f"  Ablated '{ablation_group}': zeroed {len(idx)} cols — {group['description']}")
    elif ablation_group and ablation_group != "ALL":
        print(f"  Unknown ablation group '{ablation_group}', using ALL features")

    return env_matrix, n_features


def _build_features_fallback(df: pd.DataFrame, symbol: str = "") -> np.ndarray:
    """Simple fallback feature builder when the real pipeline is unavailable."""
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values
    n = len(close)

    features = []
    for i in range(n):
        feats = []
        # Returns
        r1 = close[i] / close[max(0, i - 1)] - 1 if i >= 1 else 0.0
        r5 = close[i] / close[max(0, i - 5)] - 1 if i >= 5 else 0.0
        r20 = close[i] / close[max(0, i - 20)] - 1 if i >= 20 else 0.0
        # Volatility
        if i >= 20:
            rv = np.std(close[i - 20:i + 1]) / close[i]
        else:
            rv = 0.0
        # Volume ratio
        vol_ma10 = np.mean(volume[max(0, i - 10):i + 1])
        rel_vol = volume[i] / max(vol_ma10, 1e-8)
        # Trend
        ma50 = np.mean(close[max(0, i - 50):i + 1])
        htf_trend = (close[i] / ma50 - 1) if ma50 > 0 else 0.0

        feats.extend([close[i] / df["open"].iloc[i] - 1 if i > 0 else 0.0,  # open_rel
                      high[i] / close[i] - 1,  # high_rel
                      low[i] / close[i] - 1,  # low_rel
                      0.0,  # placeholders ...
                      0.0, r1, r5, r20,
                      0.0, 0.0, 0.0, 0.0,  # candle geometry
                      rv,
                      rel_vol,
                      0.0,  # spread
                      htf_trend,
                      0.0,  # vol_bucket
                      ])
        # Pad to 40 base cols + 18 cross + 1 ml = 59
        while len(feats) < 59:
            feats.append(0.0)
        features.append(feats)

    return np.array(features, dtype=np.float32)


def compute_reward_signal(close_prices: np.ndarray, lookahead: int = 3) -> np.ndarray:
    """Compute a realistic trading reward based on forward returns.

    Uses a clipped forward return scaled by volatility as the reward signal.
    This gives higher reward to configurations that can predict directional moves.
    """
    n = len(close_prices)
    rewards = np.zeros(n, dtype=np.float32)
    for i in range(n - lookahead):
        fwd_ret = (close_prices[i + lookahead] / close_prices[i]) - 1.0
        # Scale by recent volatility for risk-adjusted signal
        if i >= 20:
            local_vol = np.std(close_prices[i - 20:i + 1] / close_prices[i - 20] - 1) + 1e-8
        else:
            local_vol = 0.001
        rewards[i] = np.clip(fwd_ret / local_vol, -1.0, 1.0)
    return rewards


# ── Environment ────────────────────────────────────────────────────────

def make_env(
    feature_matrix: np.ndarray,
    rewards: np.ndarray,
    window_size: int = 100,
    regime_dim: int = 5,
    turnover_penalty: float = 0.0,  # 0.0 = removed; turnover penalises switching (opposite of tactical positioning)
    concentration_penalty: float = 0.001,
) -> DummyVecEnv:
    """Create a DummyVecEnv that feeds real feature observations.

    The environment exposes pre-computed feature windows as observations
    and uses the pre-computed reward signal with penalties to discourage
    all-in hold-forever behaviour:
      - turnover_penalty: cost per unit of position change (discourages
        large jumps, encourages gradual tactical adjustments)
      - concentration_penalty: cost for holding extreme positions near ±1
        (discourages hold-forever, encourages moderate sizing)

    This isolates the feature utilisation question from trading
    environment complexity.
    """
    n_bars = feature_matrix.shape[0]
    n_features = feature_matrix.shape[1]
    obs_dim = window_size * n_features + regime_dim

    def _init() -> gym.Env:
        class _FeatureEnv(gym.Env):
            metadata = {"render_modes": []}

            def __init__(self):
                super().__init__()
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
                )
                self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
                self._step = window_size
                self._max_step = n_bars - 1
                self._reward_count = 0
                self._reward_accum = {
                    "growth": 0.0,
                    "raw_reward": 0.0,
                    "turnover_cost": 0.0,
                    "concentration_cost": 0.0,
                }
                self._last_position = 0.0

            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                self._step = window_size
                self._last_position = 0.0
                # Return first observation
                obs = self._get_obs(self._step)
                return obs, {}

            def pop_reward_components(self):
                if self._reward_count == 0:
                    return {}
                result = {k: float(v / self._reward_count) for k, v in self._reward_accum.items()}
                self._reward_accum = {
                    "growth": 0.0,
                    "raw_reward": 0.0,
                    "turnover_cost": 0.0,
                    "concentration_cost": 0.0,
                }
                self._reward_count = 0
                return result

            def step(self, action):
                self._step += 1
                if self._step >= self._max_step:
                    # Terminal
                    obs = self._get_obs(self._max_step - 1)
                    return obs, 0.0, True, False, {}

                obs = self._get_obs(self._step)
                raw_reward = float(rewards[self._step])

                # ── Position-aware penalties ──
                position = float(action[0])  # first action dim = position size
                position_change = abs(position - self._last_position)
                turnover_cost = turnover_penalty * position_change
                concentration_cost = concentration_penalty * (position ** 2)
                self._last_position = position

                growth = position * raw_reward
                reward = growth - turnover_cost - concentration_cost

                # Accumulate reward components
                self._reward_count += 1
                self._reward_accum["growth"] += growth
                self._reward_accum["raw_reward"] += raw_reward
                self._reward_accum["turnover_cost"] += turnover_cost
                self._reward_accum["concentration_cost"] += concentration_cost

                return obs, reward, False, False, {}

            def _get_obs(self, idx: int) -> np.ndarray:
                """Build observation: windowed features + regime features."""
                start = idx - window_size
                window = feature_matrix[start:idx]  # (window_size, n_features)
                flat = window.reshape(-1)  # flatten

                if regime_dim > 0:
                    # Simple regime heuristic based on recent vol and price position
                    if idx >= 20:
                        recent_vol = np.std(feature_matrix[idx - 20:idx, 0])  # open_rel vol
                    else:
                        recent_vol = 0.0
                    regime_feat = np.zeros(regime_dim, dtype=np.float32)
                    regime_feat[0] = 1.0 if recent_vol > 0.5 else 0.0  # high vol regime
                    regime_feat[1] = 1.0 if feature_matrix[idx - 1, 0] > 0 else 0.0  # up bias
                    regime_feat[2] = 0.5  # confidence
                    regime_feat[3] = float(feature_matrix[idx - 1, 5])  # recent return
                    regime_feat[4] = float(feature_matrix[idx - 1, 12])  # recent vol
                    obs = np.concatenate([flat, regime_feat])
                else:
                    obs = flat

                return obs.astype(np.float32)

        return _FeatureEnv()

    return DummyVecEnv([_init])


# ── Training Runner ────────────────────────────────────────────────────

def run_trial(
    ablation_group: str,
    feature_matrix: np.ndarray,
    rewards: np.ndarray,
    window_size: int,
    regime_dim: int,
    total_timesteps: int,
    close_prices: np.ndarray,
    trial_id: int = 0,
    verbose: bool = False,
    seed: int = 42,
) -> dict:
    """Run a single training trial with a given feature ablation.

    Args:
        ablation_group: Feature group name to ablate, or "ALL" for baseline.
        feature_matrix: (n_bars, n_features) pre-computed feature matrix.
        rewards: (n_bars,) pre-computed reward signal.
        window_size: Number of bars per observation window.
        regime_dim: Regime feature dimension (0 to disable).
        total_timesteps: Number of training timesteps.
        close_prices: (n_bars,) close prices for metric computation.
        trial_id: Trial index (for logging).
        verbose: If True, print progress.

    Returns:
        Dict with training results and metrics.
    """
    n_features = feature_matrix.shape[1]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Trial {trial_id}: Ablation '{ablation_group}'")
        print(f"  Features per bar: {n_features}")
        print(f"  Regime dim: {regime_dim}")
        print(f"  Total timesteps: {total_timesteps}")
        print(f"{'='*60}")

    # Create environment
    env = make_env(
        feature_matrix, rewards,
        window_size=window_size,
        regime_dim=regime_dim,
    )
    env.seed(seed)

    # Build policy kwargs — use AdaptiveLSTMFeatureExtractor to match real training pipeline
    use_regime = regime_dim > 0
    use_bias = ablation_group in ("TREND_MOMENTUM_FIRST", "NO_BIAS_SATURATION")
    bias_fixed_temp = 0.1 if ablation_group == "NO_BIAS_SATURATION" else None
    policy_kwargs = {
        "features_extractor_class": AdaptiveLSTMFeatureExtractor,
        "features_extractor_kwargs": {
            "features_dim": 256,
            "window_size": window_size,
            "num_heads": 4,
            "regime_dim": regime_dim,
            "use_trend_momentum_bias": use_bias,
            "bias_fixed_temperature": bias_fixed_temp,
        },
        "net_arch": {"pi": [64, 64], "vf": [64, 64]},
        "num_regimes": 5 if use_regime else 1,
        "regime_dim": regime_dim,
    }

    start_time = time.time()
    result = {
        "ablation_group": ablation_group,
        "trial_id": trial_id,
        "total_timesteps": total_timesteps,
    }

    try:
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
            ent_coef=0.05,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            regime_loss_coef=0.05 if use_regime else 0.0,
            verbose=0,
        )

        model.learn(total_timesteps=total_timesteps)
        elapsed = time.time() - start_time

        # ── Bias diagnostics (for TREND_MOMENTUM_FIRST, always on) ──
        if use_bias:
            print(f"\n  [BIAS] Running bias layer diagnostics...")
            _diagnose_bias(model, feature_matrix, close_prices, window_size, regime_dim)

        # ── Evaluate on validation split ──
        val_metrics = _evaluate(model, feature_matrix, close_prices, window_size, regime_dim)

        result.update({
            "elapsed_seconds": round(elapsed, 1),
            "completed": True,
            "status": "ok",
            "weight_fingerprint": _model_weight_fingerprint(model),
            **val_metrics,
        })

        # ── Extract reward component means ──
        reward_components = env.envs[0].pop_reward_components()

        if verbose:
            sharpe = val_metrics.get("sharpe_ratio", 0)
            win_rate = val_metrics.get("win_rate", 0)
            profit_factor = val_metrics.get("profit_factor", 0)
            print(f"  [OK] Completed in {elapsed:.1f}s | Sharpe: {sharpe:.3f} | WinRate: {win_rate:.1%} | PF: {profit_factor:.2f}")
            if reward_components:
                rc = reward_components
                print(f"  [REWARD_COMPONENTS] growth={rc['growth']:.6f}  raw_reward={rc['raw_reward']:.6f}  "
                      f"turnover_cost={rc['turnover_cost']:.6f}  concentration_cost={rc['concentration_cost']:.6f}")

            # ── Save position timeseries CSV and plot ──
            positions = val_metrics.get("positions", [])
            if positions and len(positions) > 10:
                csv_path = f"runtime/positions_{ablation_group}_trial{trial_id}.csv"
                import os
                os.makedirs("runtime", exist_ok=True)
                with open(csv_path, "w") as csvf:
                    print("step,position", file=csvf)
                    for i, p in enumerate(positions):
                        print(f"{i},{p:.8f}", file=csvf)
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [3, 1]})
                    steps = list(range(len(positions)))
                    ax1.plot(steps, positions, color="#2196F3", linewidth=0.8, alpha=0.8)
                    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
                    ax1.axhline(y=0.5, color="red", linestyle=":", linewidth=0.5, alpha=0.3, label="conc. penalty threshold")
                    ax1.axhline(y=-0.5, color="red", linestyle=":", linewidth=0.5, alpha=0.3)
                    ax1.set_ylabel("Position")
                    ax1.set_title(f"Position Timeseries — {ablation_group} (trial {trial_id})")
                    ax1.legend(loc="upper right", fontsize=8)
                    ax1.grid(True, alpha=0.3)
                    # Histogram on bottom subplot
                    ax2.hist(positions, bins=50, color="#2196F3", alpha=0.7, edgecolor="none")
                    ax2.axvline(x=0, color="gray", linestyle="--", linewidth=0.5)
                    ax2.set_xlabel("Position")
                    ax2.set_ylabel("Frequency")
                    ax2.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plot_path = f"runtime/positions_{ablation_group}_trial{trial_id}.png"
                    plt.savefig(plot_path, dpi=150)
                    plt.close()
                    print(f"  [PLOT] Saved to {plot_path}")
                except Exception as plot_err:
                    print(f"  [PLOT] Skipped (matplotlib not available?): {plot_err}")

    except Exception as exc:
        elapsed = time.time() - start_time
        result.update({
            "elapsed_seconds": round(elapsed, 1),
            "completed": False,
            "status": str(exc)[:200],
        })
        if verbose:
            print(f"  [FAIL] {exc}")

    return result


def _diagnose_bias(
    model: RegimeRoutedPPO,
    feature_matrix: np.ndarray,
    close_prices: np.ndarray,
    window_size: int,
    regime_dim: int,
    val_split: float = 0.7,
) -> None:
    """Log diagnostics for the TrendMomentumBiasLayer inside the model.

    Runs a forward pass over the validation split and aggregates bias scores.
    """
    extractor = model.policy.features_extractor
    bias_layer = getattr(extractor, 'trend_momentum_bias', None)
    if bias_layer is None:
        print("  [BIAS] No TrendMomentumBiasLayer found in model")
        return

    # Reset persistent bias before diagnostic pass
    bias_layer.reset_persistent_bias()

    n = len(feature_matrix)
    val_start = int(n * val_split)
    n_features = feature_matrix.shape[1]

    if val_start + window_size + 50 >= n:
        print("  [BIAS] Not enough validation data for diagnostics")
        return

    # Collect scores over validation range
    trend_vals = []
    momentum_vals = []
    direction_vals = []
    confidence_vals = []
    agreement_vals = []
    persistent_vals = []

    with th.no_grad():
        for i in range(val_start, n - 1):
            start = i - window_size
            window = feature_matrix[start:i]
            flat = window.reshape(-1)

            if regime_dim > 0:
                regime_feat = np.zeros(regime_dim, dtype=np.float32)
                regime_feat[1] = 1.0 if feature_matrix[i - 1, 0] > 0 else 0.0
                obs = np.concatenate([flat, regime_feat]).astype(np.float32)
            else:
                obs = flat.astype(np.float32)

            # Trigger forward pass through the extractor (model.predict goes through policy)
            model.predict(obs.reshape(1, -1), deterministic=True)

            scores = bias_layer.last_scores
            if scores:
                trend_vals.append(scores["trend"][0, 0].item())
                momentum_vals.append(scores["momentum"][0, 0].item())
                direction_vals.append(scores["direction_bias"][0, 0].item())
                confidence_vals.append(scores["confidence"][0, 0].item())
                agreement_vals.append(scores["agreement"][0, 0].item())
                persistent_vals.append(scores["persistent_bias"][0, 0].item())

    if not direction_vals:
        print("  [BIAS] No bias scores collected")
        return

    direction_arr = np.array(direction_vals, dtype=np.float32)
    confidence_arr = np.array(confidence_vals, dtype=np.float32)
    trend_arr = np.array(trend_vals, dtype=np.float32)
    momentum_arr = np.array(momentum_vals, dtype=np.float32)
    persistent_arr = np.array(persistent_vals, dtype=np.float32)

    # Categorise direction into buckets
    def _bucket(v):
        if v > 0.3:
            return "bullish"
        elif v < -0.3:
            return "bearish"
        else:
            return "range"

    buckets = np.array([_bucket(v) for v in direction_arr])
    unique, counts = np.unique(buckets, return_counts=True)

    print(f"  [BIAS] direction_counts={dict(zip(unique, counts.tolist()))}")
    print(f"  [BIAS] confidence_mean={np.nanmean(confidence_arr):.6f}")
    print(f"  [BIAS] confidence_std={np.nanstd(confidence_arr):.6f}")
    print(f"  [BIAS] trend_mean={np.nanmean(trend_arr):.6f}")
    print(f"  [BIAS] momentum_mean={np.nanmean(momentum_arr):.6f}")
    print(f"  [BIAS] persistent_mean={np.nanmean(persistent_arr):.6f}")

    # Future direction agreement
    if len(direction_arr) >= 2:
        future_ret = np.sign(np.diff(close_prices[val_start:n]))
        min_len = min(len(direction_arr), len(future_ret))
        bias_dir = np.sign(direction_arr[:min_len])
        future_dir = future_ret[:min_len]
        agreement = np.mean(bias_dir == future_dir)
        print(f"  [BIAS] future_direction_agreement={agreement:.4f}")

    # Reset persistent bias after diagnostics so it doesn't leak into eval
    bias_layer.reset_persistent_bias()


def _evaluate(
    model: RegimeRoutedPPO,
    feature_matrix: np.ndarray,
    close_prices: np.ndarray,
    window_size: int,
    regime_dim: int,
    val_split: float = 0.7,
) -> dict:
    """Run a validation forward pass and compute trading metrics.

    Uses the last 30% of data as validation. Simulates a simple position
    strategy using the model's action (position size) and computes
    standard trading metrics.
    """
    n = len(feature_matrix)
    val_start = int(n * val_split)

    if val_start + window_size + 50 >= n:
        return {
            "sharpe_ratio": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "net_return_pct": 0.0,
            "trade_count": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "validation_samples": 0,
            "action_hash": "N/A",
            "position_mean": 0.0,
            "position_std": 0.0,
            "position_min": 0.0,
            "position_max": 0.0,
            "long_pct": 0.0,
            "short_pct": 0.0,
            "flat_pct": 0.0,
            "action_abs_mean": 0.0,
            "action_entropy": 0.0,
            "turnover_pct": 0.0,
            "positions": [],
        }

    n_features = feature_matrix.shape[1]

    # Run model over validation window
    positions = []
    with th.no_grad():
        for i in range(val_start, n - 1):
            # Build observation
            start = i - window_size
            window = feature_matrix[start:i]
            flat = window.reshape(-1)

            if regime_dim > 0:
                regime_feat = np.zeros(regime_dim, dtype=np.float32)
                if i >= 20:
                    recent_vol = np.std(feature_matrix[i - 20:i, 0])
                else:
                    recent_vol = 0.0
                regime_feat[0] = 1.0 if recent_vol > 0.5 else 0.0
                regime_feat[1] = 1.0 if feature_matrix[i - 1, 0] > 0 else 0.0
                regime_feat[2] = 0.5
                regime_feat[3] = float(feature_matrix[i - 1, 5])
                regime_feat[4] = float(feature_matrix[i - 1, 12])
                obs = np.concatenate([flat, regime_feat]).astype(np.float32)
            else:
                obs = flat.astype(np.float32)

            action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
            positions.append(float(action[0, 0]))  # position size from first action dim

    # Compute PnL from positions
    positions = np.array(positions, dtype=np.float32)
    val_returns = close_prices[val_start + 1:n] / close_prices[val_start:n - 1] - 1.0

    # Align lengths
    min_len = min(len(positions), len(val_returns))
    positions = positions[:min_len]
    val_returns = val_returns[:min_len]

    # Strategy returns = position * market return - transaction cost
    tc = 0.0002  # 2 bps per trade
    position_changes = np.abs(np.diff(positions, prepend=positions[0]))
    strategy_returns = positions * val_returns - tc * position_changes

    # ── Compute metrics ──
    total_return = np.sum(strategy_returns)
    avg_return = np.mean(strategy_returns)
    std_return = np.std(strategy_returns) + 1e-8

    sharpe = avg_return / std_return * np.sqrt(288 * 252)  # 5-min bars → annualised (24h, 252 days)

    # Trade analysis
    direction = np.sign(positions)
    trade_returns = direction[:-1] * val_returns[:-1]  # return when position is active
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]

    trade_count = len(trade_returns)
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = win_count / max(trade_count, 1)

    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    profit_factor = abs(np.sum(wins) / max(abs(np.sum(losses)), 1e-8))

    # Max drawdown
    cumulative = np.cumprod(1 + strategy_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_dd = float(np.min(drawdown))

    return {
        "sharpe_ratio": round(float(sharpe), 4),
        "profit_factor": round(float(profit_factor), 4),
        "max_drawdown": round(float(max_dd), 6),
        "net_return_pct": round(float(total_return * 100), 4),
        "trade_count": int(trade_count),
        "win_rate": round(float(win_rate), 6),
        "avg_win": round(float(avg_win), 6),
        "avg_loss": round(float(avg_loss), 6),
        "validation_samples": int(min_len),
        # Position-level diagnostics — reveals whether different feature sets
        # produce different trading behaviour even if aggregate metrics match.
        "action_hash": matrix_fingerprint(positions.reshape(-1, 1)),
        "position_mean": round(float(np.mean(positions)), 6),
        "position_std": round(float(np.std(positions)), 6),
        "position_min": round(float(np.min(positions)), 6),
        "position_max": round(float(np.max(positions)), 6),
        "long_pct": round(float(np.mean(positions > 0.05)), 6),
        "short_pct": round(float(np.mean(positions < -0.05)), 6),
        "flat_pct": round(float(np.mean(np.abs(positions) <= 0.05)), 6),
        "action_abs_mean": round(float(np.mean(np.abs(positions))), 6),
        # Action entropy — binned Shannon entropy of position distribution.
        # Zero means all positions fall in one bucket (policy stuck/saturated).
        # High entropy means the policy is exploring diverse position sizes.
        "action_entropy": _action_distribution_entropy(positions),
        # Turnover — fraction of steps where position changed significantly.
        # Low turnover with low entropy = hold-forever. High turnover = tactical.
        "turnover_pct": round(float(np.mean(position_changes > 0.01)), 6),
        "positions": positions.tolist(),
    }


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Real feature ablation test harness for RegimeRoutedPPO (ENGINEERED_V2)"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="XAUUSDm",
        help="Trading symbol (default: XAUUSDm)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30000,
        help="Total training timesteps per trial (default: 30000)",
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
        default=None,
        help=(
            "Feature groups to test (default: all). "
            f"Options: {', '.join(ABLATION_GROUPS)}"
        ),
    )
    parser.add_argument(
        "--n-bars",
        type=int,
        default=5000,
        help="Number of OHLCV bars to load (default: 5000)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Observation window size in bars (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/real_feature_ablation_results.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    # Set seeds for reproducibility
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
    except ImportError:
        pass

    groups = args.groups or ABLATION_GROUPS

    print(f"{'='*60}")
    print(f"REAL FEATURE ABLATION TEST HARNESS")
    print(f"{'='*60}")
    print(f"Symbol: {args.symbol}")
    print(f"Steps per trial: {args.steps}")
    print(f"Trials per group: {args.trials}")
    print(f"Groups: {', '.join(groups)}")
    print(f"Bars: {args.n_bars}")
    print(f"Window: {args.window}")
    print(f"{'='*60}")

    # ── Load data ──
    print(f"\nLoading real data for {args.symbol}...")
    df = load_real_data(symbol=args.symbol, n_bars=args.n_bars, seed=args.seed)
    close_prices = df["close"].values.astype(np.float32)
    print(f"  Got {len(df)} bars of data")

    # ── Compute reward signal ──
    print("\nComputing reward signal...")
    rewards = compute_reward_signal(close_prices)

    # ── Build feature matrices ──
    # Base features (ALL: no ablation)
    print("\nBuilding base feature matrix (ALL)...")
    all_features, n_features = build_real_features(df, symbol=args.symbol)
    print(f"  Base feature count: {n_features}")

    # ── Runtime column audit: reveal which indices actually carry signal ──
    print(f"\n[COLUMN_AUDIT] n_features={n_features}")
    col_means = np.nanmean(all_features, axis=0)
    col_stds = np.nanstd(all_features, axis=0)
    for i in range(min(n_features, len(ENGINEERED_V2_COLUMNS))):
        name = ENGINEERED_V2_COLUMNS[i]
        has_signal = col_stds[i] > 1e-8
        marker = "[SIGNAL]" if has_signal else "[ZERO] "
        print(f"  {marker} [{i:2d}] {name:<25s} mean={col_means[i]:+.4f}  std={col_stds[i]:.4f}")
    if n_features > len(ENGINEERED_V2_COLUMNS):
        for i in range(len(ENGINEERED_V2_COLUMNS), n_features):
            has_signal = col_stds[i] > 1e-8
            marker = "[SIGNAL]" if has_signal else "[ZERO] "
            print(f"  {marker} [{i:2d}] <unnamed>                  mean={col_means[i]:+.4f}  std={col_stds[i]:.4f}")

    # ── Group check: print column-name mapping before training ──
    if n_features == len(ENGINEERED_V2_COLUMNS):
        print(f"\n[GROUP_CHECK] Feature column mapping (n_features={n_features}):")
        for group in sorted(FEATURE_GROUPS.keys()):
            spec = FEATURE_GROUPS[group]
            print(f"  {group}: {spec['description']}")
            for idx, name in zip(spec["indices"], spec["columns"]):
                print(f"    [{idx}] {name}")
    else:
        print(f"\n[GROUP_CHECK] WARNING: n_features={n_features} != expected {len(ENGINEERED_V2_COLUMNS)}, skipping column-name verification")

    # ── Compute ALL baseline fingerprint for comparison ──
    all_fingerprint = matrix_fingerprint(all_features)
    print(f"\n  ALL fingerprint: {all_fingerprint}")

    # Track which groups disable regime (not a feature-column ablation)
    regime_off_groups = {"NO_REGIME"}

    # ── Train each group ──
    all_results = []
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    for group in groups:
        # Determine regime dimension
        if group in regime_off_groups:
            regime_dim = 0
        else:
            regime_dim = 5  # default regime features

        # Build ablated features (unless NO_REGIME which just disables regime)
        if group == "ALL" or group in regime_off_groups or group in ("TREND_MOMENTUM_FIRST", "NO_BIAS_SATURATION"):
            features = all_features.copy()
        elif group == "NO_TREND_MOMENTUM":
            # Special case: zero out BOTH trend AND momentum columns
            features = all_features.copy()
            all_idx = []
            for key in ["trend", "momentum"]:
                all_idx.extend(FEATURE_GROUPS[key]["indices"])
            all_idx = [i for i in all_idx if 0 <= i < features.shape[1]]
            if all_idx:
                features[:, all_idx] = 0.0
                print(f"\nAblating '{group}': zeroed {len(all_idx)} columns (trend + momentum)")
        else:
            ablation = group.replace("NO_", "").lower()
            if ablation in FEATURE_GROUPS:
                features = all_features.copy()
                idx = FEATURE_GROUPS[ablation]["indices"]
                idx = [i for i in idx if 0 <= i < features.shape[1]]
                if idx:
                    features[:, idx] = 0.0
                    print(f"\nAblating '{group}': zeroed {len(idx)} columns")
            else:
                print(f"\nUnknown group '{group}', using ALL features")
                features = all_features.copy()

        # ── Fingerprint: verify feature mask is actually applied ──
        fprint = matrix_fingerprint(features)
        print(f"  [ABLATION] group={group}")
        print(f"  [ABLATION]   shape={features.shape}")
        print(f"  [ABLATION]   fingerprint={fprint}")
        print(f"  [ABLATION]   mean={np.nanmean(features):.6f}, std={np.nanstd(features):.6f}")
        if group != "ALL" and group not in regime_off_groups and group not in ("TREND_MOMENTUM_FIRST", "NO_BIAS_SATURATION"):
            if fprint == all_fingerprint:
                # Check if the group's columns are already all-zero in ALL
                ablation = group.replace("NO_", "").lower()
                if ablation in FEATURE_GROUPS:
                    idx = FEATURE_GROUPS[ablation]["indices"]
                    idx = [i for i in idx if 0 <= i < all_features.shape[1]]
                    col_stds = np.nanstd(all_features[:, idx], axis=0)
                    dead_cols = all(col_stds < 1e-8)
                    if dead_cols:
                        print(f"  [ABLATION]   columns already all-zero in ALL (pipeline not producing them) — skipping assert")
                    else:
                        msg = (
                            f"{group} produced IDENTICAL features to ALL. "
                            f"Ablation mask was NOT applied! Check feature column indices."
                        )
                        print(f"  [ABLATION]   *** ASSERTION FAILED: {msg}")
                        raise AssertionError(msg)
                else:
                    msg = (
                        f"{group} produced IDENTICAL features to ALL. "
                        f"Ablation mask was NOT applied! Check feature column indices."
                    )
                    print(f"  [ABLATION]   *** ASSERTION FAILED: {msg}")
                    raise AssertionError(msg)
            else:
                print(f"  [ABLATION]   differs from ALL [OK]")
        else:
            print(f"  [ABLATION]   matches ALL (baseline) [OK]")

        for trial in range(args.trials):
            result = run_trial(
                ablation_group=group,
                feature_matrix=features,
                rewards=rewards,
                window_size=args.window,
                regime_dim=regime_dim,
                total_timesteps=args.steps,
                close_prices=close_prices,
                trial_id=trial,
                verbose=args.verbose,
                seed=args.seed,
            )
            all_results.append(result)

            # Save incremental results
            _save_results(all_results, args.output)

    # ── Final summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    completed = [r for r in all_results if r.get("completed")]
    failed = [r for r in all_results if not r.get("completed")]
    print(f"Total trials: {len(all_results)}")
    print(f"Completed: {len(completed)}")
    print(f"Failed: {len(failed)}")

    if completed:
        print(f"\n{'Group':<20} {'Sharpe':<10} {'WinRate':<10} {'PF':<10} {'Drawdown':<10} {'Trades':<8} {'Time':<8}")
        print(f"{'-'*70}")
        for r in completed:
            sharpe = r.get("sharpe_ratio", 0)
            wr = r.get("win_rate", 0)
            pf = r.get("profit_factor", 0)
            dd = r.get("max_drawdown", 0)
            tc = r.get("trade_count", 0)
            et = r.get("elapsed_seconds", 0)
            status = "OK" if r.get("completed") else "FAIL"
            print(f"{r['ablation_group']:<20} {sharpe:<10.3f} {wr:<10.2%} {pf:<10.2f} {dd:<10.4f} {tc:<8} {et:<8.1f}s {status}")

        # ── Determinism check: compare model weights and action trajectories ──
        weights = {r["ablation_group"]: r.get("weight_fingerprint", "N/A") for r in completed}
        actions = {r["ablation_group"]: r.get("action_hash", "N/A") for r in completed}
        unique_weights = set(weights.values())
        unique_actions = set(actions.values())
        print(f"\n[DETERMINISM] Model weight fingerprints:")
        for g, w in weights.items():
            print(f"  {g}: {w}")
        print(f"[DETERMINISM] Action trajectory hashes:")
        for g, a in actions.items():
            print(f"  {g}: {a}")
        
        if len(unique_weights) == 1 and len(unique_actions) == 1:
            print(f"[DETERMINISM] *** IDENTICAL weights AND actions — training is effectively deterministic")
        elif len(unique_weights) > 1 and len(unique_actions) == 1:
            print(f"[DETERMINISM] Different weights but IDENTICAL actions — policy internals differ but output behaviour converges")
        elif len(unique_weights) == 1 and len(unique_actions) > 1:
            print(f"[DETERMINISM] Same weights but DIFFERENT actions — action head is stochastic or validation path differs")
        else:
            print(f"[DETERMINISM] {len(unique_weights)} distinct weight sets, {len(unique_actions)} distinct action trajectories — groups produce genuinely different behaviour")

        # ── Position distribution summary ──
        print(f"\n[POSITIONS] Per-group position distribution:")
        print(f"  {'Group':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Long%':<8} {'Short%':<8} {'Flat%':<8} {'AbsMean':<10} {'Entropy':<10} {'Turn%':<8}")
        print(f"  {'-'*121}")
        for r in completed:
            pm = r.get("position_mean", 0)
            ps = r.get("position_std", 0)
            pmin = r.get("position_min", 0)
            pmax = r.get("position_max", 0)
            lp = r.get("long_pct", 0)
            sp = r.get("short_pct", 0)
            fp = r.get("flat_pct", 0)
            am = r.get("action_abs_mean", 0)
            ent = r.get("action_entropy", 0)
            to = r.get("turnover_pct", 0)
            print(f"  {r['ablation_group']:<20} {pm:<10.4f} {ps:<10.4f} {pmin:<10.4f} {pmax:<10.4f} {lp:<8.1%} {sp:<8.1%} {fp:<8.1%} {am:<10.4f} {ent:<10.4f} {to:<8.1%}")

        # Best performer by Sharpe
        best = max(completed, key=lambda r: r.get("sharpe_ratio", -999))
        print(f"\nBest Sharpe: {best['ablation_group']} ({best['sharpe_ratio']:.3f})")

    print(f"\nResults saved to: {args.output}")


def _save_results(results: list[dict], path: str):
    """Save results to CSV."""
    if not results:
        return
    # Collect all unique fieldnames across all results
    # (failed trials may have fewer fields than successful ones)
    fieldnames = list(dict.fromkeys(k for r in results for k in r.keys()))
    # Ensure status and key metrics are always first columns
    priority = ["ablation_group", "trial_id", "completed", "status", "sharpe_ratio",
                "profit_factor", "win_rate", "max_drawdown", "net_return_pct",
                "trade_count", "avg_win", "avg_loss"]
    for p in reversed(priority):
        if p in fieldnames:
            fieldnames.remove(p)
            fieldnames.insert(0, p)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
