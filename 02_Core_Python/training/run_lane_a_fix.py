"""
Lane A: Fixed Engineered Features — apply taming to the engineered feature pipeline.

Usage:
    python training/run_lane_a_fix.py
    python training/run_lane_a_fix.py --steps 100 --n-bars 1000 --seed 42

CLI args:
    --symbol      Trading symbol (default: XAUUSDm)
    --n-bars      Number of bars to load (default: 100000)
    --steps       Total training timesteps (default: 50000)
    --seed        Single seed to run (optional, default: run all 3 seeds)
    --timeframe   MT5 timeframe (default: 1m)

Output: runtime/lane_a_all_seeds.csv
"""
import sys, os, time, warnings, argparse
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "C:/supreme-chainsaw")

from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn

from training.run_real_feature_ablation import load_real_data, compute_reward_signal, build_real_features
from training.taming_shared import BaseTamedEnv, MetricsCallback, compute_weight_hash, evaluate

# ── Defaults (overridden by CLI when run directly) ──
SYMBOL = "XAUUSDm"
N_BARS = 100000
N_STEPS = 50000
SEEDS = [42, 123, 456]


# ── Fixed config ──
LOOKAHEAD = 5              # predict 5-bar forward return (better signal/noise than 1-bar)
WINDOW_SIZE = 64
HIDDEN_SIZE = 128
N_LSTM_LAYERS = 2
FEATURES_DIM = 64

# Taming (same as Lane B)
TURNOVER_COST = 0.0003      # realistic XAUUSD 1m spread + slippage (~3 bps)
CONCENTRATION_PENALTY = 0.002
SMOOTHING_ALPHA = 0.3
COOLDOWN_STEPS = 5


def clean_features(features):
    """
    Remove all-zero columns and the ml_signal column (feat_58).
    Returns cleaned feature matrix and a list of kept column indices.
    """
    n_orig = features.shape[1]
    # Find all-zero columns
    non_zero_cols = np.where(np.max(np.abs(features), axis=0) > 1e-10)[0]
    # Remove feat_58 (ml_signal) which has lookahead bias
    keep_cols = [c for c in non_zero_cols if c != 58]
    cleaned = features[:, keep_cols]

    print(f"  Feature cleanup: {n_orig} -> {cleaned.shape[1]} columns")
    print(f"    Removed {n_orig - len(non_zero_cols)} all-zero columns")
    print(f"    Removed feat_58 (ml_signal lookahead bias)")
    return cleaned, keep_cols


class FixedFeatureEnv(BaseTamedEnv):
    """
    Feature-matrix environment with taming (same as Lane B):
      - Higher turnover + concentration penalties
      - Position smoothing (EMA)
      - Cooldown after direction flip
    """
    def __init__(self, feature_matrix, signal,
                 window_size=WINDOW_SIZE,
                 turnover_cost=TURNOVER_COST,
                 concentration_penalty=CONCENTRATION_PENALTY,
                 smoothing_alpha=SMOOTHING_ALPHA,
                 cooldown_steps=COOLDOWN_STEPS):
        self._feature_matrix = feature_matrix.astype(np.float32)
        self._signal = signal.astype(np.float32)
        super().__init__(
            window_size=window_size,
            turnover_cost=turnover_cost,
            concentration_penalty=concentration_penalty,
            smoothing_alpha=smoothing_alpha,
            cooldown_steps=cooldown_steps,
            n=len(feature_matrix),
            n_features=feature_matrix.shape[1],
        )
        self._build_features()

    def _build_features(self):
        self.features = self._feature_matrix

    def _raw_reward_at(self, idx):
        return float(self._signal[idx]) if idx < len(self._signal) else 0.0


class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """Same architecture as Lane B."""
    def __init__(self, observation_space: spaces.Box, features_dim: int = FEATURES_DIM):
        super().__init__(observation_space, features_dim=features_dim)
        self.window_size = WINDOW_SIZE
        # Infer n_features from obs space
        total_dim = observation_space.shape[0]
        self.n_features = total_dim // self.window_size

        self.lstm = nn.LSTM(
            input_size=self.n_features,
            hidden_size=HIDDEN_SIZE,
            num_layers=N_LSTM_LAYERS,
            batch_first=True,
            bidirectional=False,
            dropout=0.1 if N_LSTM_LAYERS > 1 else 0,
        )
        self.projection = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, features_dim),
            nn.LayerNorm(features_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        x = observations.view(batch_size, self.window_size, self.n_features)
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]
        return self.projection(last)



def train_seed(seed, train_features, train_signal):
    """Train PPO on engineered features with taming."""
    np.random.seed(seed)

    def make_env():
        return FixedFeatureEnv(train_features, train_signal)

    train_env = DummyVecEnv([make_env])
    train_env = VecMonitor(train_env)

    # Infer n_features for the LSTM from the cleaned feature matrix
    policy_kwargs = {
        "features_extractor_class": LSTMFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": FEATURES_DIM},
        "net_arch": {"pi": [64], "vf": [64]},
    }

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=0,
    )

    cb = MetricsCallback()
    model.learn(total_timesteps=N_STEPS, callback=cb)
    return model, cb


# ── CLI parsing (only runs when executed directly, not on import) ──
def main():
    parser = argparse.ArgumentParser(description="Lane A: Fixed Engineered Features")
    parser.add_argument("--symbol", default=SYMBOL, help=f"MT5 symbol (default: {SYMBOL})")
    parser.add_argument("--n-bars", type=int, default=N_BARS, help=f"Number of bars (default: {N_BARS})")
    parser.add_argument("--steps", type=int, default=N_STEPS, help=f"Timesteps (default: {N_STEPS})")
    parser.add_argument("--seed", type=int, default=None, help="Single seed (default: run all 3)")
    parser.add_argument("--timeframe", default="1m", help="MT5 timeframe (default: 1m)")
    args = parser.parse_args()

    # Override globals with CLI values
    globals()['SYMBOL'] = args.symbol
    globals()['N_BARS'] = args.n_bars
    globals()['N_STEPS'] = args.steps
    if args.seed is not None:
        globals()['SEEDS'] = [args.seed]

    print("=" * 64)
    print("LANE A — FIXED ENGINEERED FEATURES (tamed + cleaned)")
    print("=" * 64)
    print()

    # 1. Load data and build features
    print("[1] Loading data and building features...")
    df = load_real_data(symbol=SYMBOL, n_bars=N_BARS)
    close = df["close"].values.astype(np.float64)
    features, n_feat = build_real_features(df, symbol=SYMBOL, ablation_group=None)
    signal = compute_reward_signal(close, lookahead=LOOKAHEAD)

    valid = min(len(features), len(signal))
    features = features[:valid]
    signal = signal[:valid]
    close = close[:valid]
    print(f"  Raw features: {features.shape}")

    # 2. Clean features
    print()
    print("[2] Cleaning feature matrix...")
    features, keep_cols = clean_features(features)
    print(f"  Cleaned features: {features.shape}")

    # 3. Train/val split
    split = int(len(features) * 0.7)
    trn_f, val_f = features[:split], features[split:]
    trn_s, val_s = signal[:split], signal[split:]
    print(f"  Train: {len(trn_f)} bars | Val: {len(val_f)} bars")
    print()

    # 4. Run 3 seeds
    all_val = []
    all_train = []
    all_positions = []
    all_nw = []

    for seed in SEEDS:
        print(f"  {'='*56}")
        print(f"  Seed {seed}")
        print(f"  {'='*56}")
        t0 = time.time()

        model, cb = train_seed(seed, trn_f, trn_s)
        train_time = time.time() - t0

        # Training metrics
        trn_pos = np.array(cb.positions)
        train_m = {
            "long_pct": float(np.mean(trn_pos > 0.01) * 100) if len(trn_pos) > 0 else 0,
            "short_pct": float(np.mean(trn_pos < -0.01) * 100) if len(trn_pos) > 0 else 0,
            "flat_pct": float(np.mean(np.abs(trn_pos) <= 0.01) * 100) if len(trn_pos) > 0 else 0,
            "time": train_time,
        }
        all_train.append(train_m)

        print(f"    Training: {train_time:.1f}s ({N_STEPS/train_time:.0f} steps/s)")
        print(f"    Train: {train_m['long_pct']:.1f}%L / {train_m['short_pct']:.1f}%S / {train_m['flat_pct']:.1f}%F")

        # Validation
        val_m = evaluate(model, lambda: FixedFeatureEnv(val_f, val_s, window_size=WINDOW_SIZE),
                         turnover_cost=TURNOVER_COST)
        val_m["weight_hash"] = compute_weight_hash(model)
        all_val.append(val_m)
        all_positions.append(val_m.pop("positions"))
        all_nw.append(val_m.pop("net_worth"))

        print(f"    Val:   {val_m['long_pct']:.1f}%L / {val_m['short_pct']:.1f}%S / {val_m['flat_pct']:.1f}%F")
        print(f"    Sharpe={val_m['sharpe']:.2f} Ret={val_m['total_return']:.2f}% "
              f"DD={val_m['max_drawdown']:.2f}% To={val_m['turnover']:.1f}%")
        print(f"    action_hash={val_m['action_hash']}  weight_hash={val_m['weight_hash']}")
        print()

    # 5. Aggregate
    print()
    print("=" * 64)
    print("AGGREGATED — LANE A (Fixed Engineered Features)")
    print("=" * 64)

    metrics_names = ["long_pct", "short_pct", "flat_pct", "pos_mean", "pos_std",
                     "sharpe", "total_return", "max_drawdown", "turnover", "n_steps"]

    print(f"\n  {'Metric':<20} {'Mean':>10} {'Std':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10}")
    for name in metrics_names:
        vals = [m[name] for m in all_val]
        print(f"  {name:<20} {np.mean(vals):>10.2f} {np.std(vals):>10.2f}")

    # 6. Head-to-head with Lane B
    print()
    print("=" * 64)
    print("HEAD-TO-HEAD: Lane A (Fixed Engineered) vs Lane B (Tamed LSTM)")
    print("=" * 64)

    # Lane B results from the previous run
    lane_b = {
        "long_pct": 17.59, "short_pct": 66.13, "flat_pct": 16.28,
        "sharpe": 4.01, "total_return": 0.53, "turnover": 75.36,
    }

    print(f"\n  {'Metric':<20} {'Lane A':>10} {'Lane B':>10} {'Winner':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    for name in metrics_names:
        if name in ("pos_mean", "pos_std", "n_steps", "max_drawdown"):
            continue
        a_vals = [m[name] for m in all_val]
        a_mean = np.mean(a_vals)
        b_val = lane_b.get(name, 0.0)
        if name == "flat_pct":
            winner = "LaneA" if a_mean < b_val else "LaneB"
        elif name in ("sharpe", "total_return"):
            winner = "LaneA" if a_mean > b_val else "LaneB"
        elif name == "turnover":
            winner = "LaneA" if a_mean < b_val else "LaneB"
        else:
            winner = "--"
        print(f"  {name:<20} {a_mean:>10.2f} {b_val:>10.2f} {winner:>10}")

    print()
    print("CONCLUSION:")
    a_flat = np.mean([m["flat_pct"] for m in all_val])
    a_sharpe = np.mean([m["sharpe"] for m in all_val])

    if a_flat > 50:
        print("  Lane A still goes mostly flat — cleaning features alone didn't fix the signal problem.")
    elif a_sharpe > 0:
        print("  Lane A shows positive Sharpe — the cleaned features have SOME signal with taming.")
    else:
        print("  Lane A still negative — the engineered features (minus ml_signal) have no predictive power.")

    # Save
    os.makedirs("runtime", exist_ok=True)
    all_rows = []
    for i, seed_num in enumerate(SEEDS):
        pos = all_positions[i]
        nw = all_nw[i]
        for j in range(len(pos)):
            all_rows.append({
                "seed": seed_num, "step": j, "position": pos[j],
                "net_worth": nw[min(j+1, len(nw)-1)]
            })
    pd.DataFrame(all_rows).to_csv("runtime/lane_a_all_seeds.csv", index=False)
    print(f"\n  CSV: runtime/lane_a_all_seeds.csv")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
