"""
Lane B: Tamed Raw OHLCV + LSTM PPO — 3-seed walk-forward validation.

Tier 3 fixes (Jun 2026):
  - Discrete actions: Long(+1) / Flat(0) / Short(-1) — prevents Gaussian collapse
  - No EMA smoothing — removes action inertia
  - Market symmetry: inverted price series concatenated (2x training bars)
  - No session features (they were spurious row-number regime detectors)
  - Raw 5-bar forward returns (not demeaned — market symmetry handles trend bias)
  - Inactivity penalty: small cost for staying flat, forces model to take positions
  - Higher ent_coef (0.05) for discrete exploration
  - RSI(14) + MACD histogram features

Usage:
    python training/run_lane_b_raw_lstm.py
    python training/run_lane_b_raw_lstm.py --steps 100 --n-bars 1000 --seed 42

CLI args:
    --symbol      Trading symbol (default: XAUUSDm)
    --n-bars      Number of bars to load (default: 100000)
    --steps       Total training timesteps (default: 50000)
    --seed        Single seed to run (optional, default: run all 3 seeds)
    --timeframe   MT5 timeframe (default: M5)

Output: runtime/lane_b_raw_all_seeds.csv
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

from training.run_real_feature_ablation import load_real_data
from training.taming_shared import BaseTamedEnv, MetricsCallback, compute_weight_hash, evaluate


# ── Defaults (overridden by CLI when run directly) ──
SYMBOL = "XAUUSDm"
N_BARS = 100000
N_STEPS = 50000
SEEDS = [42, 123, 456, 789, 234, 567, 890, 345]
TIMEFRAME = "M5"


# ── Fixed config (not CLI-overridable) ──
N_FEATURES = 7          # 5 OHLCV log-returns + 1 RSI + 1 MACD histogram
WINDOW_SIZE = 64        # 64 x 5min = 320min ~ 5.3 hours of context
HIDDEN_SIZE = 128
N_LSTM_LAYERS = 2
FEATURES_DIM = 64

# Taming parameters
TURNOVER_COST = 0.0           # zero — removed direction trap
CONCENTRATION_PENALTY = 0.0   # zero for discrete — no "extreme position" problem with Long/Flat/Short
COOLDOWN_STEPS = 5
REWARD_HORIZON = 5            # predict 5-bar forward return
REWARD_SCALE = 1000.0         # scale raw returns (~0.001) to PPO-friendly range (~1.0)
INACTIVITY_PENALTY = 0.0003   # cost for staying flat (0.0003*1000 = 0.3 in reward space)
HOLDING_PENALTY = 0.00005     # small penalty when holding > 0.3 position unchanged
ENT_COEF = 0.05               # higher entropy for discrete actions


# ── Tamed environment ──
class TamedOHLCVEnv(BaseTamedEnv):
    """Discrete-action env: OHLCV + RSI + MACD with raw forward returns + inactivity penalty."""
    def __init__(self, df, window_size=WINDOW_SIZE,
                 turnover_cost=TURNOVER_COST,
                 concentration_penalty=CONCENTRATION_PENALTY,
                 smoothing_alpha=0.3,
                 cooldown_steps=COOLDOWN_STEPS):
        self.df = df.reset_index(drop=True)
        n = len(self.df)
        super().__init__(
            window_size=window_size,
            turnover_cost=turnover_cost,
            concentration_penalty=concentration_penalty,
            smoothing_alpha=smoothing_alpha,
            cooldown_steps=cooldown_steps,
            n=n,
            n_features=N_FEATURES,
            reward_scale=REWARD_SCALE,
            holding_penalty=HOLDING_PENALTY,
            inactivity_penalty=INACTIVITY_PENALTY,
            discrete=True,  # Tier 3: Long/Flat/Short, no EMA
        )
        self._build_features()

    def _build_features(self):
        """Build features: OHLCV log-returns, RSI(14), MACD histogram, H-bar forward signal."""
        o = self.df["open"].values.astype(np.float64)
        h = self.df["high"].values.astype(np.float64)
        l = self.df["low"].values.astype(np.float64)
        c = self.df["close"].values.astype(np.float64)
        v = self.df["volume"].values.astype(np.float64)

        n = len(self.df)
        H = REWARD_HORIZON

        # ── OHLCV log-return features (columns 0-4) ──
        ohlcv = np.zeros((n, 5), dtype=np.float32)
        for i in range(n):
            if i == 0:
                continue
            ohlcv[i, 0] = np.log(o[i] / o[i-1]) if o[i-1] > 0 else 0.0
            ohlcv[i, 1] = np.log(h[i] / h[i-1]) if h[i-1] > 0 else 0.0
            ohlcv[i, 2] = np.log(l[i] / l[i-1]) if l[i-1] > 0 else 0.0
            ohlcv[i, 3] = np.log(c[i] / c[i-1]) if c[i-1] > 0 else 0.0
            ohlcv[i, 4] = np.log(v[i] / v[i-1]) if v[i-1] > 0 else 0.0

        # ── RSI(14) — column 5 ──
        rsi = np.zeros(n, dtype=np.float32)
        delta = np.diff(c, prepend=c[0])
        gains = np.maximum(delta, 0.0)
        losses = np.maximum(-delta, 0.0)
        rsi_period = 14
        for i in range(rsi_period, n):
            avg_gain = float(np.mean(gains[i - rsi_period + 1 : i + 1]))
            avg_loss = float(np.mean(losses[i - rsi_period + 1 : i + 1]))
            if avg_loss < 1e-10:
                rsi[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100.0 - 100.0 / (1.0 + rs)

        # ── MACD histogram — column 6 ──
        alpha12 = 2.0 / 13.0
        alpha26 = 2.0 / 27.0
        ema12 = np.zeros(n, dtype=np.float64)
        ema26 = np.zeros(n, dtype=np.float64)
        ema12[0] = c[0]
        ema26[0] = c[0]
        for i in range(1, n):
            ema12[i] = alpha12 * c[i] + (1.0 - alpha12) * ema12[i-1]
            ema26[i] = alpha26 * c[i] + (1.0 - alpha26) * ema26[i-1]
        macd_line = ema12 - ema26
        signal_line = np.zeros(n, dtype=np.float64)
        signal_line[0] = macd_line[0]
        alpha_sig = 2.0 / 10.0
        for i in range(1, n):
            signal_line[i] = alpha_sig * macd_line[i] + (1.0 - alpha_sig) * signal_line[i-1]
        macd_hist = macd_line - signal_line

        # Normalize MACD histogram by price level
        macd_hist_norm = np.zeros(n, dtype=np.float32)
        for i in range(n):
            denom = max(c[i], 1.0)
            macd_hist_norm[i] = float(macd_hist[i] / denom)

        # ── Combine all features ──
        raw_features = np.zeros((n, N_FEATURES), dtype=np.float32)
        raw_features[:, 0:5] = ohlcv          # OHLCV log returns
        raw_features[:, 5] = rsi              # RSI(14)
        raw_features[:, 6] = macd_hist_norm   # MACD histogram / price

        # Z-score normalize all features
        self.features = raw_features.copy()
        for col in range(N_FEATURES):
            col_data = raw_features[:, col]
            mean = np.mean(col_data)
            std = np.std(col_data)
            self.features[:, col] = (col_data - mean) / max(std, 1e-10)

        # ── H-bar forward return (raw, no demeaning) ──
        # Market symmetry (real + inverted data) handles trend bias —
        # the model must use RSI/MACD features to decide direction
        self._forward_ret = np.zeros(n, dtype=np.float32)
        for i in range(n - H):
            self._forward_ret[i] = float((c[i+H] / c[i]) - 1.0)

    def _raw_reward_at(self, idx):
        """Raw forward return — market symmetry removes directional bias."""
        return self._forward_ret[idx]


# ── LSTM feature extractor ──
class LSTMFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = FEATURES_DIM):
        super().__init__(observation_space, features_dim=features_dim)
        self.window_size = WINDOW_SIZE
        self.n_features = N_FEATURES

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


# ── Market symmetry augmentation ──
def make_inverted_df(df):
    """Create a mirror-image dataset where uptrends become downtrends and vice versa.

    This forces the model to use features (RSI, MACD) rather than simply
    riding the dataset's global trend. A 100% short strategy will have zero
    expected return when trained on both real and inverted data.
    """
    inverted = df.copy()
    mean_price = float(df["close"].mean())
    for col in ["open", "high", "low", "close"]:
        inverted[col] = 2.0 * mean_price - df[col].values
    # Swap high/low since mirror flips them
    inverted_high = inverted["high"].values.copy()
    inverted_low = inverted["low"].values.copy()
    inverted["high"] = inverted_low
    inverted["low"] = inverted_high
    return inverted


# ── Train one seed ──
def train_seed(seed, train_df):
    """Train a PPO model with market symmetry (real + inverted data)."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Market symmetry: concatenate real and inverted datasets
    # Without session features, model must use RSI/MACD/OHLCV for direction
    inverted_df = make_inverted_df(train_df)
    combined_df = pd.concat([train_df, inverted_df], ignore_index=True)

    def make_env():
        return TamedOHLCVEnv(combined_df)

    train_env = DummyVecEnv([make_env])
    train_env = VecMonitor(train_env)

    policy_kwargs = {
        "features_extractor_class": LSTMFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": FEATURES_DIM},
        "net_arch": {"pi": [64], "vf": [64]},
    }

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ENT_COEF,         # 0.05 for discrete — forces exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=0,
    )

    cb = MetricsCallback()
    model.learn(total_timesteps=N_STEPS, callback=cb)
    return model, cb


# ── CLI parsing + main (only runs when executed directly) ──
def main():
    parser = argparse.ArgumentParser(description="Lane B: Tamed Raw OHLCV + LSTM (Tier 3)")
    parser.add_argument("--symbol", default=SYMBOL, help=f"MT5 symbol (default: {SYMBOL})")
    parser.add_argument("--n-bars", type=int, default=N_BARS, help=f"Bars to load (default: {N_BARS})")
    parser.add_argument("--steps", type=int, default=N_STEPS, help=f"Timesteps (default: {N_STEPS})")
    parser.add_argument("--seed", type=int, default=None, help="Single seed (default: run all 3)")
    parser.add_argument("--timeframe", default=TIMEFRAME, help=f"MT5 timeframe (default: {TIMEFRAME})")
    args = parser.parse_args()

    # Override globals with CLI values
    globals()['SYMBOL'] = args.symbol
    globals()['N_BARS'] = args.n_bars
    globals()['N_STEPS'] = args.steps
    globals()['TIMEFRAME'] = args.timeframe
    if args.seed is not None:
        globals()['SEEDS'] = [args.seed]

    # ── Begin experiment ──
    print("=" * 64)
    print("LANE B — TAMED RAW OHLCV + LSTM (Tier 3: Discrete + Symmetry, raw returns)")
    print("=" * 64)
    print()
    print(f"  Config:")
    print(f"    action_space:        Discrete(3) [Long/Flat/Short] — no EMA inertia")
    print(f"    market_symmetry:     inverted dataset concatenated (2x bars)")
    print(f"    reward:              raw 5-bar forward return (no demeaning)")
    print(f"    session_features:    REMOVED (were spurious regime detector)")
    print(f"    timeframe:           {TIMEFRAME}")
    print(f"    n_features:          {N_FEATURES} (5 OHLCV + RSI + MACD hist)")
    print(f"    window_size:         {WINDOW_SIZE} bars ({WINDOW_SIZE*5} min)")
    print(f"    turnover_cost:       {TURNOVER_COST}")
    print(f"    inactivity_penalty:  {INACTIVITY_PENALTY} (cost for staying flat)")
    print(f"    concentration_penalty: {CONCENTRATION_PENALTY}")
    print(f"    holding_penalty:     {HOLDING_PENALTY}")
    print(f"    cooldown_steps:      {COOLDOWN_STEPS}")
    print(f"    reward_horizon:      {REWARD_HORIZON} bars")
    print(f"    reward_scale:        {REWARD_SCALE}")
    print(f"    total_timesteps:     {N_STEPS}")
    print(f"    ppo n_steps:         2048")
    print(f"    ppo ent_coef:        {ENT_COEF}")
    print(f"    ppo learning_rate:   1e-4")
    print(f"    seeds:               {SEEDS}")
    print(f"    symbol:              {SYMBOL}")
    print(f"    n_bars:              {N_BARS}")
    print()

    # Load data
    print(f"  Loading {SYMBOL} data ({TIMEFRAME}, {N_BARS} bars)...")
    df = load_real_data(symbol=SYMBOL, n_bars=N_BARS)
    print(f"  Loaded {len(df)} bars of {SYMBOL}")
    split = int(len(df) * 0.7)
    train_df = df.iloc[:split].reset_index(drop=True)
    val_df = df.iloc[split:].reset_index(drop=True)
    print(f"  Train: {len(train_df)} bars (+ inverted = {len(train_df)*2} combined)")
    print(f"  Val:   {len(val_df)} bars (real only, honest eval)")
    print()

    # Run seeds
    all_val_metrics = []
    all_train_metrics = []
    all_positions = []
    all_nw = []

    for seed in SEEDS:
        print(f"  {'='*56}")
        print(f"  Seed {seed}")
        print(f"  {'='*56}")
        t0 = time.time()

        model, cb = train_seed(seed, train_df)
        train_time = time.time() - t0

        # Training metrics
        trn_pos = np.array(cb.positions)
        train_metrics = {
            "pos_mean": float(np.mean(trn_pos)) if len(trn_pos) > 0 else 0,
            "pos_std": float(np.std(trn_pos)) if len(trn_pos) > 0 else 0,
            "long_pct": float(np.mean(trn_pos > 0.01) * 100) if len(trn_pos) > 0 else 0,
            "short_pct": float(np.mean(trn_pos < -0.01) * 100) if len(trn_pos) > 0 else 0,
            "flat_pct": float(np.mean(np.abs(trn_pos) <= 0.01) * 100) if len(trn_pos) > 0 else 0,
            "ep_reward_mean": float(np.mean(cb.episode_rewards)) if cb.episode_rewards else 0,
            "time": train_time,
        }
        all_train_metrics.append(train_metrics)

        print(f"    Training: {train_time:.1f}s ({N_STEPS/train_time:.0f} steps/s)")
        print(f"    Train long%={train_metrics['long_pct']:.1f} short%={train_metrics['short_pct']:.1f} "
              f"flat%={train_metrics['flat_pct']:.1f}")

        # Validation — on REAL data only (honest eval)
        val_metrics = evaluate(model, lambda: TamedOHLCVEnv(val_df, window_size=WINDOW_SIZE),
                               turnover_cost=TURNOVER_COST)
        val_metrics["weight_hash"] = compute_weight_hash(model)
        all_val_metrics.append(val_metrics)
        all_positions.append(val_metrics["positions"])
        all_nw.append(val_metrics["net_worth"])

        print(f"    Val long%={val_metrics['long_pct']:.1f} short%={val_metrics['short_pct']:.1f} "
              f"flat%={val_metrics['flat_pct']:.1f}")
        print(f"    Val Sharpe={val_metrics['sharpe']:.2f} Return={val_metrics['total_return']:.2f}% "
              f"DD={val_metrics['max_drawdown']:.2f}%")
        print(f"    Turnover={val_metrics['turnover']:.1f}% PosStd={val_metrics['pos_std']:.4f}")
        print(f"    action_hash={val_metrics['action_hash']}  weight_hash={val_metrics['weight_hash']}")
        print()

    # ── Aggregate ──
    print()
    print("=" * 64)
    print("AGGREGATED RESULTS (mean +/- std across 3 seeds)")
    print("=" * 64)

    metrics_names = ["long_pct", "short_pct", "flat_pct", "pos_mean", "pos_std",
                     "sharpe", "total_return", "max_drawdown", "turnover", "n_steps"]

    print(f"\n  {'Metric':<20} {'Mean':>10} {'Std':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10}")

    for name in metrics_names:
        vals = [m[name] for m in all_val_metrics]
        mean_v = np.mean(vals)
        std_v = np.std(vals)
        if name == "sharpe":
            print(f"  {name:<20} {mean_v:>10.2f} {std_v:>10.2f}")
        elif name in ("n_steps",):
            print(f"  {name:<20} {mean_v:>10.0f} {std_v:>10.0f}")
        else:
            print(f"  {name:<20} {mean_v:>10.2f} {std_v:>10.2f}")

    # Training metrics
    print()
    print("  TRAINING METRICS (mean across seeds):")
    mean_ep_ret = np.mean([m["ep_reward_mean"] for m in all_train_metrics])
    mean_long = np.mean([m["long_pct"] for m in all_train_metrics])
    mean_short = np.mean([m["short_pct"] for m in all_train_metrics])
    mean_flat = np.mean([m["flat_pct"] for m in all_train_metrics])
    print(f"    Ep reward mean: {mean_ep_ret:.2f}")
    print(f"    Long/Short/Flat: {mean_long:.1f}% / {mean_short:.1f}% / {mean_flat:.1f}%")

    # Save CSV
    os.makedirs("runtime", exist_ok=True)
    all_data = []
    for i, seed in enumerate(SEEDS):
        pos = all_positions[i]
        nw = all_nw[i]
        for j in range(len(pos)):
            all_data.append({"seed": seed, "step": j, "position": pos[j],
                             "net_worth": nw[min(j+1, len(nw)-1)]})
    pd.DataFrame(all_data).to_csv("runtime/lane_b_raw_8seeds.csv", index=False)
    print(f"\n  CSV: runtime/lane_b_raw_8seeds.csv")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
        colors = ["steelblue", "darkorange", "seagreen"]
        for i, seed in enumerate(SEEDS):
            pos = all_positions[i]
            nw = all_nw[i]
            ax = axes[i, 0]
            ax.plot(pos, color=colors[i], linewidth=0.6, alpha=0.8)
            ax.axhline(0, color="gray", ls="--", lw=0.4)
            ax.set_ylabel(f"Pos (seed {seed})")
            ax.grid(alpha=0.3)
            ax2 = axes[i, 1]
            ax2.plot(nw, color=colors[i], linewidth=0.6, alpha=0.8)
            ax2.set_ylabel(f"NW (seed {seed})")
            ax2.grid(alpha=0.3)
        axes[-1, 0].set_xlabel("Validation step")
        axes[-1, 1].set_xlabel("Validation step")
        plt.suptitle("Tamed Raw OHLCV + LSTM — Tier 3: Discrete + Symmetry, Raw Returns, 8 Seeds (M5)", fontsize=13)
        plt.tight_layout()
        plt.savefig("runtime/lane_b_raw_8seeds.png", dpi=150)
        plt.close()
        print(f"  Plot: runtime/lane_b_raw_8seeds.png")
    except Exception as e:
        print(f"  Plot failed: {e}")

    # Conclusion
    print()
    print("=" * 64)
    print("CONCLUSION")
    print("=" * 64)
    avg_sharpe = np.mean([m["sharpe"] for m in all_val_metrics])
    avg_return = np.mean([m["total_return"] for m in all_val_metrics])
    avg_long = np.mean([m["long_pct"] for m in all_val_metrics])
    avg_short = np.mean([m["short_pct"] for m in all_val_metrics])
    avg_turnover = np.mean([m["turnover"] for m in all_val_metrics])

    print(f"  Avg Sharpe:  {avg_sharpe:.2f}")
    print(f"  Avg Return:  {avg_return:.2f}%")
    print(f"  Avg Long:    {avg_long:.1f}%")
    print(f"  Avg Short:   {avg_short:.1f}%")
    print(f"  Avg Turnover: {avg_turnover:.1f}%")

    if avg_sharpe > 0:
        print("\n  PROFITABLE!")
    else:
        print("\n  Not profitable yet.")

    if avg_long > 10 and avg_short > 10 and avg_turnover > 5:
        print("  TRAP BROKEN — model trades both directions dynamically.")
    elif avg_long > 5 and avg_short > 5:
        print("  Model trades both sides but one dominates.")
    else:
        print("  Direction trap persists.")
    print()
    print("  Done.")


if __name__ == "__main__":
    main()
