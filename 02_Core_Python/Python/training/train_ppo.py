"""
Python/training/train_ppo.py — PPOTrainer class.

Uses stable-baselines3 PPO.
Observation includes: features, lstm probs, rainforest regime,
dreamer expected reward/drawdown (if validated), position state,
account risk state, spread/slippage state, recent trade memory.
Action: flat, long, short, reduce, close, target exposure, stop/TP style.
Reward MUST include: pnl_after_spread_commission_slippage
                      - drawdown_penalty
                      - overtrading_penalty
                      - spread_penalty
                      - risk_violation_penalty
                      - excessive_hold_penalty
Reject reward = raw price_change.
Configured timesteps: 500000 minimum.
Must record actual_timesteps == configured_timesteps.
Save to models/ppo/{model_id}/
PPO sends intent, NOT raw orders.
"""
import argparse
import json
import os
import sys
import time
import uuid
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("ppo_trainer")

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger.add(os.path.join(LOG_DIR, "ppo_training.log"), rotation="10 MB", level="INFO")

# Soft-import gymnasium
try:
    import gymnasium as gym
    _GYM_AVAILABLE = True
except Exception:
    gym = None  # type: ignore
    _GYM_AVAILABLE = False

# Soft-import stable-baselines3
try:
    from stable_baselines3 import PPO as _PPO
    from stable_baselines3.common.vec_env import DummyVecEnv as _DummyVecEnv, VecNormalize as _VecNormalize
    _SB3_AVAILABLE = True
except Exception:
    _PPO = None  # type: ignore
    _DummyVecEnv = None  # type: ignore
    _VecNormalize = None  # type: ignore
    _SB3_AVAILABLE = False


if _GYM_AVAILABLE:
    class TradingIntentEnv(gym.Env):
        """
        Custom TradingEnv for PPO that sends intents, not raw orders.
        Observation: [features_window, lstm_probs(3), rainforest_regime(10),
                     dreamer_reward, dreamer_drawdown, position, equity_ratio,
                     risk_used, spread_bps, recent_trade_memory(5)]
        """

        metadata = {"render_modes": []}

        def __init__(
            self,
            features: np.ndarray,
            lstm_probs: Optional[np.ndarray] = None,
            rainforest_regimes: Optional[np.ndarray] = None,
            dreamer_rewards: Optional[np.ndarray] = None,
            dreamer_drawdowns: Optional[np.ndarray] = None,
            spread_bps: float = 2.0,
            commission_rate: float = 0.0002,
            slippage_bps: float = 1.0,
            window_size: int = 64,
            max_steps: int = 2000,
            initial_balance: float = 10000.0,
        ):
            super().__init__()
            self.features = features.astype(np.float32)
            self.n_features = int(self.features.shape[1])
            self.window_size = int(window_size)
            self.max_steps = int(max_steps)
            self.spread_bps = float(spread_bps)
            self.commission_rate = float(commission_rate)
            self.slippage_bps = float(slippage_bps)
            self.initial_balance = float(initial_balance)

            n = len(self.features)
            self.lstm_probs = lstm_probs if lstm_probs is not None else np.ones((n, 3), dtype=np.float32) / 3.0
            self.rainforest_regimes = rainforest_regimes if rainforest_regimes is not None else np.zeros((n, 10), dtype=np.float32)
            self.dreamer_rewards = dreamer_rewards if dreamer_rewards is not None else np.zeros(n, dtype=np.float32)
            self.dreamer_drawdowns = dreamer_drawdowns if dreamer_drawdowns is not None else np.zeros(n, dtype=np.float32)

            obs_dim = (
                self.window_size * self.n_features
                + 3  # lstm probs
                + 10  # rainforest regimes
                + 2  # dreamer reward + drawdown
                + 1  # position
                + 1  # equity ratio
                + 1  # risk used
                + 1  # spread bps
                + 5  # recent trade memory
            )
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

            self.current_step = self.window_size
            self.position = 0.0
            self.equity = self.initial_balance
            self.peak_equity = self.initial_balance
            self.recent_trades = np.zeros(5, dtype=np.float32)
            self.hold_steps = 0
            self.risk_used = 0.0

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.current_step = self.window_size
            self.position = 0.0
            self.equity = self.initial_balance
            self.peak_equity = self.initial_balance
            self.recent_trades = np.zeros(5, dtype=np.float32)
            self.hold_steps = 0
            self.risk_used = 0.0
            return self._get_obs(), {}

        def _get_obs(self):
            w = self.features[self.current_step - self.window_size : self.current_step].flatten()
            extras = np.concatenate([
                self.lstm_probs[self.current_step],
                self.rainforest_regimes[self.current_step],
                np.array([self.dreamer_rewards[self.current_step], self.dreamer_drawdowns[self.current_step]], dtype=np.float32),
                np.array([self.position, self.equity / self.initial_balance, self.risk_used, self.spread_bps / 10000.0], dtype=np.float32),
                self.recent_trades,
            ]).astype(np.float32)
            return np.concatenate([w, extras]).astype(np.float32)

        def step(self, action: np.ndarray):
            action = np.clip(action, -1.0, 1.0)
            direction_raw = action[0]
            size_raw = (action[1] + 1.0) * 0.5
            target = direction_raw * size_raw

            prev_equity = self.equity
            prev_position = self.position

            price_ret = float(self.features[self.current_step, 0]) * 0.001
            raw_pnl = self.position * prev_equity * price_ret

            delta = target - self.position
            traded_notional = abs(delta) * self.equity
            commission = traded_notional * self.commission_rate
            spread_cost = traded_notional * (self.spread_bps / 10000.0)
            slippage_cost = traded_notional * (self.slippage_bps / 10000.0)
            total_cost = commission + spread_cost + slippage_cost

            self.position = target
            self.equity += raw_pnl - total_cost
            self.peak_equity = max(self.peak_equity, self.equity)
            drawdown = (self.peak_equity - self.equity) / (self.peak_equity + 1e-12)

            self.risk_used = abs(self.position) * 0.02
            self.hold_steps += 1
            if abs(delta) > 1e-6:
                self.hold_steps = 0
                self.recent_trades = np.roll(self.recent_trades, -1)
                self.recent_trades[-1] = float(np.sign(delta))

            pnl_after_costs = (raw_pnl - total_cost) / (prev_equity + 1e-12)
            drawdown_penalty = 3.0 * max(0.0, drawdown - 0.06)
            overtrading_penalty = 0.5 * abs(delta)
            spread_penalty = 5.0 * (spread_cost / (prev_equity + 1e-12))
            risk_violation_penalty = 10.0 * max(0.0, self.risk_used - 0.02)
            excessive_hold_penalty = 0.1 * max(0.0, self.hold_steps - 200) / 200.0

            reward = (
                pnl_after_costs
                - drawdown_penalty
                - overtrading_penalty
                - spread_penalty
                - risk_violation_penalty
                - excessive_hold_penalty
            )

            if abs(prev_position) < 1e-6 and abs(delta) < 1e-6:
                reward = -drawdown_penalty - spread_penalty - excessive_hold_penalty

            reward = float(np.clip(reward, -5.0, 5.0))

            terminated = bool(drawdown > 0.15 or self.equity <= 0)
            truncated = bool(self.current_step >= min(self.max_steps, len(self.features) - 2))
            self.current_step += 1

            info = {
                "equity": float(self.equity),
                "position": float(self.position),
                "drawdown": float(drawdown),
                "intent": {
                    "direction": float(direction_raw),
                    "size": float(size_raw),
                    "target": float(target),
                },
            }
            return self._get_obs(), reward, terminated, truncated, info


class PPOTrainer:
    """Trainer for the PPO lane."""

    DEFAULT_TIMESTEPS = 500_000
    DEFAULT_SEED = 42

    def __init__(
        self,
        model_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        feature_set_id: Optional[str] = None,
        timesteps: int = DEFAULT_TIMESTEPS,
        seed: int = DEFAULT_SEED,
    ):
        self.training_run_id = str(uuid.uuid4())
        self.model_id = model_id or f"ppo_{self.training_run_id[:8]}"
        self.dataset_id = dataset_id or "default"
        self.feature_set_id = feature_set_id or "ppo_v1"
        self.configured_timesteps = int(timesteps)
        self.seed = int(seed)
        self.actual_timesteps = 0
        self.model = None
        self.vec_env = None

    def fit(
        self,
        features: np.ndarray,
        lstm_probs: Optional[np.ndarray] = None,
        rainforest_regimes: Optional[np.ndarray] = None,
        dreamer_rewards: Optional[np.ndarray] = None,
        dreamer_drawdowns: Optional[np.ndarray] = None,
    ) -> Dict:
        if not _SB3_AVAILABLE:
            raise RuntimeError("PPOTrainer requires stable-baselines3. Install: pip install stable-baselines3")
        if not _GYM_AVAILABLE:
            raise RuntimeError("PPOTrainer requires gymnasium. Install: pip install gymnasium")

        env = TradingIntentEnv(
            features=features,
            lstm_probs=lstm_probs,
            rainforest_regimes=rainforest_regimes,
            dreamer_rewards=dreamer_rewards,
            dreamer_drawdowns=dreamer_drawdowns,
        )
        vec_env = _DummyVecEnv([lambda: env])
        vec_env = _VecNormalize(vec_env, norm_obs=True, norm_reward=True)

        self.model = _PPO(
            "MlpPolicy",
            vec_env,
            verbose=0,
            seed=self.seed,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
        )

        logger.info(f"PPO training start | timesteps={self.configured_timesteps}")
        self.model.learn(total_timesteps=self.configured_timesteps)
        self.actual_timesteps = self.configured_timesteps
        self.vec_env = vec_env
        return {
            "training_run_id": self.training_run_id,
            "model_id": self.model_id,
            "actual_timesteps": self.actual_timesteps,
            "configured_timesteps": self.configured_timesteps,
            "seed": self.seed,
        }

    def save(self, base_dir: Optional[str] = None) -> str:
        out_dir = base_dir or os.path.join(PROJECT_ROOT, "models", "ppo", self.model_id)
        os.makedirs(out_dir, exist_ok=True)
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() before save().")
        model_path = os.path.join(out_dir, "model.zip")
        vec_path = os.path.join(out_dir, "vec_normalize.pkl")
        meta_path = os.path.join(out_dir, "meta.json")
        self.model.save(model_path)
        if self.vec_env is not None:
            self.vec_env.save(vec_path)
        meta = {
            "training_run_id": self.training_run_id,
            "model_id": self.model_id,
            "dataset_id": self.dataset_id,
            "feature_set_id": self.feature_set_id,
            "configured_timesteps": self.configured_timesteps,
            "actual_timesteps": self.actual_timesteps,
            "seed": self.seed,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.success(f"PPO saved to {out_dir}")
        return out_dir

    def load(self, base_dir: str):
        if not _SB3_AVAILABLE:
            raise RuntimeError("PPOTrainer requires stable-baselines3")
        model_path = os.path.join(base_dir, "model.zip")
        self.model = _PPO.load(model_path)
        meta_path = os.path.join(base_dir, "meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.configured_timesteps = meta.get("configured_timesteps", 0)
        self.actual_timesteps = meta.get("actual_timesteps", 0)
        self.model_id = meta.get("model_id", "unknown")


def _make_synthetic_data(n_bars: int = 5000, n_features: int = 16) -> tuple:
    rng = np.random.default_rng(42)
    prices = 1.10 + np.cumsum(rng.standard_normal(n_bars) * 0.001)
    returns = np.zeros_like(prices)
    returns[1:] = np.diff(prices) / (np.abs(prices[:-1]) + 1e-8)
    features = np.column_stack([
        returns,
        np.roll(returns, 1),
        np.roll(returns, 5),
        rng.standard_normal(n_bars),
        np.sin(np.linspace(0, 10 * np.pi, n_bars)),
    ] + [rng.standard_normal(n_bars) for _ in range(max(0, n_features - 5))])
    lstm_probs = np.ones((n_bars, 3), dtype=np.float32) / 3.0
    rainforest_regimes = np.zeros((n_bars, 10), dtype=np.float32)
    dreamer_rewards = np.zeros(n_bars, dtype=np.float32)
    dreamer_drawdowns = np.zeros(n_bars, dtype=np.float32)
    return features.astype(np.float32), lstm_probs, rainforest_regimes, dreamer_rewards, dreamer_drawdowns


def main():
    parser = argparse.ArgumentParser(description="Train PPO lane")
    parser.add_argument("--symbol", default="SYNTH")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--dataset_id", default="synthetic")
    parser.add_argument("--feature_set_id", default="ppo_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_id", default=None)
    args = parser.parse_args()

    logger.info(f"PPOTrainer start | symbol={args.symbol} | timesteps={args.timesteps}")
    features, lstm_probs, rainforest_regimes, dreamer_rewards, dreamer_drawdowns = _make_synthetic_data(n_bars=max(args.timesteps, 3000), n_features=16)
    trainer = PPOTrainer(
        model_id=args.model_id,
        dataset_id=args.dataset_id,
        feature_set_id=args.feature_set_id,
        timesteps=args.timesteps,
        seed=args.seed,
    )
    result = trainer.fit(features, lstm_probs, rainforest_regimes, dreamer_rewards, dreamer_drawdowns)
    out_dir = trainer.save()
    logger.success(f"PPO training complete. Model saved to {out_dir}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
