#!/usr/bin/env python3
"""Plot policy loss convergence: ALL (RegimeRoutedPPO) vs no_regime (plain PPO)."""
import os, sys, time, warnings, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

from drl.regime_routed_policy import RegimeRoutedPPO, RegimeRoutedActorCriticPolicy

# Import from ablation harness
import importlib.util
spec = importlib.util.spec_from_file_location("ablation", os.path.join(PROJECT_ROOT, "training", "run_feature_ablation.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
make_synthetic_ohlcv = mod.make_synthetic_ohlcv
make_synthetic_features = mod.make_synthetic_features
make_env = mod.make_env


class LoggingRegimeRoutedPPO(RegimeRoutedPPO):
    """Subclass that logs per-update loss values to a list."""
    def __init__(self, *args, loss_log=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_log = loss_log if loss_log is not None else []

    def train(self):
        super().train()
        # Capture losses BEFORE logger.dump() clears them
        keys = ["policy_gradient_loss", "value_loss", "entropy_loss",
                "approx_kl", "clip_fraction", "regime_supervised_loss"]
        vals = {}
        for k in keys:
            v = self.logger.name_to_value.get("train/" + k)
            if v is not None:
                vals[k] = float(v)
        if vals:
            self.loss_log.append(vals)


class LoggingPPO(PPO):
    """Subclass that logs per-update loss values to a list."""
    def __init__(self, *args, loss_log=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_log = loss_log if loss_log is not None else []

    def train(self):
        super().train()
        keys = ["policy_gradient_loss", "value_loss", "entropy_loss",
                "approx_kl", "clip_fraction"]
        vals = {}
        for k in keys:
            v = self.logger.name_to_value.get("train/" + k)
            if v is not None:
                vals[k] = float(v)
        if vals:
            self.loss_log.append(vals)


def main():
    parser = argparse.ArgumentParser(description="Plot loss convergence: ALL vs no_regime")
    parser.add_argument("--timesteps", type=int, default=50000, help="Training timesteps per model")
    parser.add_argument("--output", default="logs/loss_convergence.png", help="Output plot path")
    parser.add_argument("--window", type=int, default=5, help="Smoothing window")
    args = parser.parse_args()

    print("=" * 60)
    print("  Loss Convergence: ALL (RegimeRoutedPPO) vs no_regime (PPO)")
    print("=" * 60)
    print(f"  Timesteps per model: {args.timesteps}")

    # 1. Generate data
    print("\n[1/3] Generating data...")
    df = make_synthetic_ohlcv(5000, 4)
    obs, nf, _ = make_synthetic_features(df, window_size=100, ablation_group=None, regime_dim=6)
    split = int(obs.shape[0] * 0.8)
    train_obs = obs[:split]
    print(f"  Train obs: {train_obs.shape}, Feature dims: {nf}")

    # 2. Train models
    # ALL: RegimeRoutedPPO with regime routing
    print("\n[2/3] Training models...")
    train_env = make_env(train_obs, 100, df=df)

    print("  Training ALL (RegimeRoutedPPO)...")
    all_loss = []
    t0 = time.time()
    all_model = LoggingRegimeRoutedPPO(
        RegimeRoutedActorCriticPolicy, train_env,
        learning_rate=3e-4, n_steps=256, batch_size=64, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs={
            "features_extractor_class": FlattenExtractor,
            "net_arch": {"pi": [64, 64], "vf": [64, 64]},
            "num_regimes": 5, "regime_dim": 6,
        },
        regime_loss_coef=0.05, verbose=0,
    )
    all_model.loss_log = all_loss
    all_model.learn(total_timesteps=args.timesteps)
    print(f"    {len(all_loss)} updates in {time.time()-t0:.1f}s")

    # no_regime: plain PPO (same env, MlpPolicy)
    print("  Training no_regime (plain PPO)...")
    train_env2 = make_env(train_obs, 100, df=df)
    nr_loss = []
    t0 = time.time()
    nr_model = LoggingPPO(
        "MlpPolicy", train_env2,
        learning_rate=3e-4, n_steps=256, batch_size=64, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        vf_coef=0.5, max_grad_norm=0.5,
        verbose=0,
    )
    nr_model.loss_log = nr_loss
    nr_model.learn(total_timesteps=args.timesteps)
    print(f"    {len(nr_loss)} updates in {time.time()-t0:.1f}s")

    # 3. Plot
    print("\n[3/3] Plotting convergence curves...")

    def smooth(vals, w):
        if len(vals) < w:
            return vals
        return list(np.convolve(vals, np.ones(w) / w, mode="valid"))

    w = args.window
    colors = {"ALL": "#1f77b4", "no_regime": "#ff7f0e"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plots = [
        ("Policy Loss", "policy_gradient_loss", axes[0, 0]),
        ("Value Loss", "value_loss", axes[0, 1]),
        ("Entropy", "entropy_loss", axes[1, 0]),
        ("Approx KL", "approx_kl", axes[1, 1]),
    ]

    for title, key, ax in plots:
        for label, loss_log in [("ALL", all_loss), ("no_regime", nr_loss)]:
            vals = [x.get(key) for x in loss_log if x.get(key) is not None]
            if not vals:
                continue
            smoothed = smooth(vals, w)
            ax.plot(smoothed, label=label, color=colors[label], alpha=0.8, linewidth=1.2)
            # Also show raw with low alpha
            ax.plot(vals, color=colors[label], alpha=0.15, linewidth=0.5)
        ax.set_title(title)
        ax.set_xlabel("Update step")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Extra: regime supervised loss if available
    if all_loss and all_loss[0].get("regime_supervised_loss") is not None:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        vals = [x["regime_supervised_loss"] for x in all_loss if "regime_supervised_loss" in x]
        smoothed = smooth(vals, w)
        ax2.plot(smoothed, color=colors["ALL"], linewidth=1.5, label="Regime supervised loss")
        ax2.plot(vals, color=colors["ALL"], alpha=0.15, linewidth=0.5)
        ax2.set_title("Regime Supervised Loss (ALL only)")
        ax2.set_xlabel("Update step")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        regime_path = os.path.join(PROJECT_ROOT, "logs/regime_supervised_loss.png")
        os.makedirs(os.path.dirname(regime_path), exist_ok=True)
        fig2.savefig(regime_path, dpi=150, bbox_inches="tight")
        print(f"  Regime loss plot saved to {regime_path}")
        plt.close(fig2)

    plt.suptitle(f"Policy Convergence: ALL (RegimeRoutedPPO) vs no_regime (PPO)\n{args.timesteps} timesteps | smoothed w={w}", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(PROJECT_ROOT, args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Main plot saved to {out_path}")
    plt.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
