"""
Python/training/train_dreamer.py — DreamerTrainer class.

RSSM world model: latent_state, transition_model, reward_model, drawdown_model, done_model
Action space: flat, long_small, long_normal, short_small, short_normal, reduce_position, close_position
Uses PyTorch.
If Dreamer cannot be properly trained (missing dependencies, data too small),
mark status as stub_disabled and return gracefully.
Save to models/dreamer/{model_id}/
Validation metrics: latent_prediction_loss, reward_prediction_error, drawdown_prediction_error, rollout_stability
"""
import argparse
import json
import os
import sys
import time
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("dreamer_trainer")

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger.add(os.path.join(LOG_DIR, "dreamer_training.log"), rotation="10 MB", level="INFO")

# Soft-import torch so the module is importable even when torch is missing.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except Exception as _torch_err:
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _TORCH_AVAILABLE = False

ACTION_SPACE = [
    "flat",
    "long_small",
    "long_normal",
    "short_small",
    "short_normal",
    "reduce_position",
    "close_position",
]
ACTION_DIM = len(ACTION_SPACE)

# Define PyTorch classes only when torch is available.
if _TORCH_AVAILABLE:
    class _SimpleRSSM(nn.Module):
        """Lightweight RSSM for trading."""

        def __init__(self, embed_dim: int = 128, hidden_dim: int = 256, stoch_dim: int = 32, action_dim: int = ACTION_DIM):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.stoch_dim = stoch_dim
            self.action_dim = action_dim
            self.gru = nn.GRUCell(stoch_dim + action_dim, hidden_dim)
            self.prior = nn.Sequential(nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, stoch_dim))
            self.posterior = nn.Sequential(nn.Linear(hidden_dim + embed_dim, 128), nn.ReLU(), nn.Linear(128, stoch_dim))

        def initial_state(self, batch_size: int, device: str):
            h = torch.zeros(batch_size, self.hidden_dim, device=device)
            z = torch.zeros(batch_size, self.stoch_dim, device=device)
            return h, z

        def observe(self, embed: torch.Tensor, action_onehot: torch.Tensor, h_prev: torch.Tensor, z_prev: torch.Tensor):
            x = torch.cat([z_prev, action_onehot], dim=-1)
            h = self.gru(x, h_prev)
            z = torch.tanh(self.posterior(torch.cat([h, embed], dim=-1)))
            return h, z

        def imagine(self, action_onehot: torch.Tensor, h_prev: torch.Tensor, z_prev: torch.Tensor):
            x = torch.cat([z_prev, action_onehot], dim=-1)
            h = self.gru(x, h_prev)
            z = torch.tanh(self.prior(h))
            return h, z

    class _DreamerNet(nn.Module):
        """Dreamer world model with reward and drawdown heads."""

        def __init__(self, obs_dim: int, hidden_dim: int = 256, stoch_dim: int = 32):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(obs_dim, 256), nn.ReLU(), nn.Linear(256, 128))
            self.rssm = _SimpleRSSM(embed_dim=128, hidden_dim=hidden_dim, stoch_dim=stoch_dim, action_dim=ACTION_DIM)
            state_dim = hidden_dim + stoch_dim
            self.decoder = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU(), nn.Linear(256, obs_dim))
            self.reward_model = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 1))
            self.drawdown_model = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid())
            self.done_model = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid())

        def forward(self, obs: torch.Tensor, action: torch.Tensor, h: torch.Tensor, z: torch.Tensor):
            embed = self.encoder(obs)
            h_new, z_new = self.rssm.observe(embed, action, h, z)
            state = torch.cat([h_new, z_new], dim=-1)
            obs_pred = self.decoder(state)
            reward_pred = self.reward_model(state).squeeze(-1)
            drawdown_pred = self.drawdown_model(state).squeeze(-1)
            done_pred = self.done_model(state).squeeze(-1)
            return obs_pred, reward_pred, drawdown_pred, done_pred, h_new, z_new


class DreamerTrainer:
    """Trainer for the Dreamer world-model lane."""

    DEFAULT_WINDOW = 64
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_SEED = 42

    def __init__(
        self,
        model_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        feature_set_id: Optional[str] = None,
        window_size: int = DEFAULT_WINDOW,
        batch_size: int = DEFAULT_BATCH_SIZE,
        seed: int = DEFAULT_SEED,
        device: Optional[str] = None,
    ):
        self.status = "initializing"
        self.stub_disabled = False
        if not _TORCH_AVAILABLE:
            logger.warning("DreamerTrainer disabled: PyTorch not available")
            self.status = "stub_disabled"
            self.stub_disabled = True
            self._disabled_reason = "PyTorch not available"
            self.training_run_id = str(uuid.uuid4())
            self.model_id = model_id or f"dreamer_{self.training_run_id[:8]}"
            self.dataset_id = dataset_id or "default"
            self.feature_set_id = feature_set_id or "dreamer_v1"
            self.window_size = int(window_size)
            self.batch_size = int(batch_size)
            self.seed = int(seed)
            self.device = "cpu"
            self.model = None
            return

        self.training_run_id = str(uuid.uuid4())
        self.model_id = model_id or f"dreamer_{self.training_run_id[:8]}"
        self.dataset_id = dataset_id or "default"
        self.feature_set_id = feature_set_id or "dreamer_v1"
        self.window_size = int(window_size)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))
        self.model = None
        self.history: List[Dict] = []
        self.obs_dim = 0

    def _set_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _build_sequences(self, features: np.ndarray, returns: np.ndarray, drawdowns: np.ndarray, dones: np.ndarray) -> List[Dict]:
        seqs = []
        w = self.window_size
        for i in range(w, len(features)):
            seqs.append({
                "obs": features[i - w : i].flatten(),
                "return": float(returns[i]),
                "drawdown": float(drawdowns[i]),
                "done": bool(dones[i]),
            })
        return seqs

    def fit(self, features: np.ndarray, returns: np.ndarray, drawdowns: np.ndarray, dones: np.ndarray, steps: int = 2000) -> Dict:
        if self.stub_disabled:
            return {
                "status": "stub_disabled",
                "reason": getattr(self, "_disabled_reason", "unknown"),
                "training_run_id": self.training_run_id,
                "model_id": self.model_id,
            }

        self._set_seed()
        self.obs_dim = int(features.shape[1] * self.window_size)
        seqs = self._build_sequences(features, returns, drawdowns, dones)
        if len(seqs) < 100:
            logger.warning(f"DreamerTrainer: too few sequences ({len(seqs)}), marking stub_disabled")
            self.status = "stub_disabled"
            self.stub_disabled = True
            return {
                "status": "stub_disabled",
                "reason": f"too_few_sequences: {len(seqs)}",
                "training_run_id": self.training_run_id,
                "model_id": self.model_id,
            }

        self.model = _DreamerNet(self.obs_dim).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        mse = nn.MSELoss()
        bce = nn.BCELoss()

        # Replay buffer from sequences
        buffer = seqs
        best_loss = float("inf")
        patience = 10
        patience_counter = 0

        for step in range(steps):
            self.model.train()
            batch = np.random.choice(len(buffer), min(self.batch_size, len(buffer)), replace=False)
            obs_batch = []
            act_batch = []
            ret_batch = []
            dd_batch = []
            done_batch = []
            for idx in batch:
                s = buffer[idx]
                obs_batch.append(s["obs"])
                a = np.zeros(ACTION_DIM, dtype=np.float32)
                a[np.random.randint(0, ACTION_DIM)] = 1.0
                act_batch.append(a)
                ret_batch.append(s["return"])
                dd_batch.append(s["drawdown"])
                done_batch.append(1.0 if s["done"] else 0.0)

            obs_t = torch.tensor(np.stack(obs_batch), dtype=torch.float32).to(self.device)
            act_t = torch.tensor(np.stack(act_batch), dtype=torch.float32).to(self.device)
            ret_t = torch.tensor(np.array(ret_batch, dtype=np.float32), dtype=torch.float32).to(self.device)
            dd_t = torch.tensor(np.array(dd_batch, dtype=np.float32), dtype=torch.float32).to(self.device)
            done_t = torch.tensor(np.array(done_batch, dtype=np.float32), dtype=torch.float32).to(self.device)

            h, z = self.model.rssm.initial_state(len(obs_t), self.device)
            obs_pred, reward_pred, drawdown_pred, done_pred, _, _ = self.model(obs_t, act_t, h, z)
            loss_obs = mse(obs_pred, obs_t)
            loss_reward = mse(reward_pred, ret_t)
            loss_drawdown = mse(drawdown_pred, dd_t)
            loss_done = bce(done_pred, done_t)
            loss = loss_obs + loss_reward + loss_drawdown + loss_done

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            if step % 50 == 0:
                val_metrics = self._validate(buffer)
                val_metrics["step"] = step
                val_metrics["train_loss"] = float(loss.item())
                self.history.append(val_metrics)
                logger.info(f"Dreamer step {step}/{steps} | loss={loss.item():.4f} | latent_loss={val_metrics.get('latent_prediction_loss', 0):.4f}")
                if val_metrics.get("val_loss", float("inf")) < best_loss:
                    best_loss = val_metrics.get("val_loss", float("inf"))
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at step {step}")
                        break

        self.status = "trained"
        return {
            "training_run_id": self.training_run_id,
            "model_id": self.model_id,
            "status": self.status,
            "steps_trained": len(self.history) * 50,
            "best_val_loss": best_loss,
            "final_metrics": self.history[-1] if self.history else {},
        }

    def _validate(self, buffer: List[Dict]) -> Dict:
        if self.model is None:
            return {}
        self.model.eval()
        # Sample a small validation batch
        idxs = np.random.choice(len(buffer), min(32, len(buffer)), replace=False)
        obs_batch = []
        act_batch = []
        ret_batch = []
        dd_batch = []
        done_batch = []
        for idx in idxs:
            s = buffer[idx]
            obs_batch.append(s["obs"])
            a = np.zeros(ACTION_DIM, dtype=np.float32)
            a[np.random.randint(0, ACTION_DIM)] = 1.0
            act_batch.append(a)
            ret_batch.append(s["return"])
            dd_batch.append(s["drawdown"])
            done_batch.append(1.0 if s["done"] else 0.0)

        with torch.no_grad():
            obs_t = torch.tensor(np.stack(obs_batch), dtype=torch.float32).to(self.device)
            act_t = torch.tensor(np.stack(act_batch), dtype=torch.float32).to(self.device)
            h, z = self.model.rssm.initial_state(len(obs_t), self.device)
            obs_pred, reward_pred, drawdown_pred, done_pred, _, _ = self.model(obs_t, act_t, h, z)
            latent_loss = float(torch.mean((obs_pred - obs_t) ** 2).item())
            reward_err = float(torch.mean((reward_pred - torch.tensor(np.array(ret_batch, dtype=np.float32), device=self.device)) ** 2).item())
            drawdown_err = float(torch.mean((drawdown_pred - torch.tensor(np.array(dd_batch, dtype=np.float32), device=self.device)) ** 2).item())

        # Rollout stability: measure variance across imagined steps
        rollout_stability = 0.0
        with torch.no_grad():
            h0, z0 = self.model.rssm.initial_state(1, self.device)
            obs0 = obs_t[:1]
            act0 = act_t[:1]
            _, _, _, _, h1, z1 = self.model(obs0, act0, h0, z0)
            states = [torch.cat([h1, z1], dim=-1).cpu().numpy().flatten()]
            for _ in range(4):
                a = torch.zeros(1, ACTION_DIM, device=self.device)
                a[0, np.random.randint(0, ACTION_DIM)] = 1.0
                h1, z1 = self.model.rssm.imagine(a, h1, z1)
                states.append(torch.cat([h1, z1], dim=-1).cpu().numpy().flatten())
            rollout_stability = float(np.std(np.array(states)))

        return {
            "val_loss": latent_loss + reward_err + drawdown_err,
            "latent_prediction_loss": round(latent_loss, 6),
            "reward_prediction_error": round(reward_err, 6),
            "drawdown_prediction_error": round(drawdown_err, 6),
            "rollout_stability": round(rollout_stability, 6),
        }

    def save(self, base_dir: Optional[str] = None) -> str:
        out_dir = base_dir or os.path.join(PROJECT_ROOT, "models", "dreamer", self.model_id)
        os.makedirs(out_dir, exist_ok=True)
        meta = {
            "training_run_id": self.training_run_id,
            "model_id": self.model_id,
            "dataset_id": self.dataset_id,
            "feature_set_id": self.feature_set_id,
            "window_size": self.window_size,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "status": self.status,
            "stub_disabled": self.stub_disabled,
            "obs_dim": self.obs_dim,
            "history": self.history,
        }
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        if self.model is not None and not self.stub_disabled:
            torch.save(self.model.state_dict(), os.path.join(out_dir, "model.pt"))
            logger.success(f"Dreamer saved to {out_dir}")
        else:
            logger.info(f"Dreamer stub saved meta to {out_dir} (model not trained)")
        return out_dir

    def load(self, base_dir: str) -> None:
        meta_path = os.path.join(base_dir, "meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.status = meta.get("status", "unknown")
        self.stub_disabled = meta.get("stub_disabled", False)
        self.obs_dim = meta.get("obs_dim", 0)
        if self.obs_dim > 0 and not self.stub_disabled and _TORCH_AVAILABLE:
            self.model = _DreamerNet(self.obs_dim).to(self.device)
            self.model.load_state_dict(torch.load(os.path.join(base_dir, "model.pt"), map_location=self.device))
            self.model.eval()


def _make_synthetic_data(n_bars: int = 5000, n_features: int = 16) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    prices = 1.10 + np.cumsum(rng.standard_normal(n_bars) * 0.001)
    returns = np.zeros_like(prices)
    returns[1:] = np.diff(prices) / (np.abs(prices[:-1]) + 1e-8)
    drawdowns = np.abs(np.minimum(0.0, np.cumsum(returns))) + 0.001
    dones = np.zeros(n_bars, dtype=bool)
    dones[-1] = True
    features = np.column_stack([
        returns,
        np.roll(returns, 1),
        np.roll(returns, 5),
        drawdowns,
        rng.standard_normal(n_bars),
        np.sin(np.linspace(0, 10 * np.pi, n_bars)),
    ] + [rng.standard_normal(n_bars) for _ in range(max(0, n_features - 6))])
    return features.astype(np.float32), returns.astype(np.float32), drawdowns.astype(np.float32), dones


def main():
    parser = argparse.ArgumentParser(description="Train Dreamer world-model lane")
    parser.add_argument("--symbol", default="SYNTH")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--timesteps", type=int, default=5000)
    parser.add_argument("--dataset_id", default="synthetic")
    parser.add_argument("--feature_set_id", default="dreamer_v1")
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_id", default=None)
    args = parser.parse_args()

    logger.info(f"DreamerTrainer start | symbol={args.symbol} | dataset={args.dataset_id}")
    features, returns, drawdowns, dones = _make_synthetic_data(n_bars=args.timesteps, n_features=16)
    trainer = DreamerTrainer(
        model_id=args.model_id,
        dataset_id=args.dataset_id,
        feature_set_id=args.feature_set_id,
        window_size=args.window,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    result = trainer.fit(features, returns, drawdowns, dones, steps=500)
    out_dir = trainer.save()
    logger.success(f"Dreamer training complete. Artifacts saved to {out_dir}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
