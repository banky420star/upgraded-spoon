"""
Python/training/train_lstm.py — LSTMTrainer class.

Input: [batch, lookback_window=128, feature_count]
Outputs: p_up, p_down, p_flat, expected_return_bps, volatility_forecast, confidence, embedding_vector
Uses PyTorch.
"""
import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Soft-import torch
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

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import precision_score, recall_score, accuracy_score
    _SKLEARN_AVAILABLE = True
except Exception:
    StandardScaler = None  # type: ignore
    precision_score = None  # type: ignore
    recall_score = None  # type: ignore
    accuracy_score = None  # type: ignore
    _SKLEARN_AVAILABLE = False

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("lstm_trainer")

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger.add(os.path.join(LOG_DIR, "lstm_training.log"), rotation="10 MB", level="INFO")


class LSTMModel(nn.Module):
    """Multi-output LSTM for trading signals."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout, bidirectional=False,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        # Direction head: 3 classes (flat, up, down)
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )
        # Expected return in bps
        self.return_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        # Volatility forecast
        self.vol_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )
        # Confidence scalar [0,1]
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        # Embedding vector
        self.embedding_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out, _ = self.lstm(x)
        last = self.norm(out[:, -1, :])
        logits = self.direction_head(last)
        probs = F.softmax(logits, dim=-1)
        return {
            "logits": logits,
            "p_flat": probs[:, 0],
            "p_up": probs[:, 1],
            "p_down": probs[:, 2],
            "expected_return_bps": self.return_head(last).squeeze(-1) * 10000.0,
            "volatility_forecast": self.vol_head(last).squeeze(-1),
            "confidence": self.confidence_head(last).squeeze(-1),
            "embedding_vector": self.embedding_head(last),
        }


class LSTMTrainer:
    """
    Trainer for the LSTM lane.
    """

    DEFAULT_LOOKBACK = 128
    DEFAULT_EPOCHS = 50
    DEFAULT_BATCH_SIZE = 64
    DEFAULT_SEED = 42

    def __init__(
        self,
        model_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        feature_set_id: Optional[str] = None,
        label_set_id: Optional[str] = None,
        lookback_window: int = DEFAULT_LOOKBACK,
        horizons: Optional[List[int]] = None,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        seed: int = DEFAULT_SEED,
        device: Optional[str] = None,
    ):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("LSTMTrainer requires PyTorch. Install: pip install torch")
        if not _SKLEARN_AVAILABLE:
            raise RuntimeError("LSTMTrainer requires scikit-learn. Install: pip install scikit-learn")

        self.training_run_id = str(uuid.uuid4())
        self.model_id = model_id or f"lstm_{self.training_run_id[:8]}"
        self.dataset_id = dataset_id or "default"
        self.feature_set_id = feature_set_id or "engineered_v2"
        self.label_set_id = label_set_id or "directional"
        self.lookback_window = int(lookback_window)
        self.horizons = horizons or [1, 5, 15]
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.seed = int(seed)

        self.device = device or ("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))
        self.model: Optional[LSTMModel] = None
        self.scaler = StandardScaler()
        self.feature_count = 0
        self.history: List[Dict] = []

    def _set_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def build_sequences(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        volatility: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build lookback sequences and multi-horizon labels.
        Labels: 0=flat, 1=up, 2=down.
        """
        seq_len = self.lookback_window
        X, y_dir, y_ret, y_vol, y_conf = [], [], [], [], []

        max_horizon = max(self.horizons)
        for i in range(seq_len, len(features) - max_horizon):
            X.append(features[i - seq_len : i])
            # Directional label using largest horizon
            h = max_horizon
            future_ret = returns[i : i + h].sum()
            vol_window = volatility[i - seq_len : i] if volatility is not None else np.std(returns[i - seq_len : i])
            vol_est = float(np.std(vol_window)) if isinstance(vol_window, np.ndarray) else float(vol_window)
            thr = max(0.0005, vol_est * 0.35)
            if future_ret > thr:
                y_dir.append(1)
            elif future_ret < -thr:
                y_dir.append(2)
            else:
                y_dir.append(0)
            y_ret.append(future_ret * 10000.0)  # bps
            y_vol.append(vol_est)
            y_conf.append(1.0 if abs(future_ret) > thr else 0.0)

        return (
            np.array(X, dtype=np.float32),
            np.array(y_dir, dtype=np.int64),
            np.array(y_ret, dtype=np.float32),
            np.array(y_vol, dtype=np.float32),
            np.array(y_conf, dtype=np.float32),
        )

    def fit(self, features: np.ndarray, returns: np.ndarray, volatility: Optional[np.ndarray] = None) -> Dict:
        self._set_seed()
        self.feature_count = int(features.shape[1])

        features_scaled = self.scaler.fit_transform(features)
        X, y_dir, y_ret, y_vol, y_conf = self.build_sequences(features_scaled, returns, volatility)
        if len(X) < 100:
            raise ValueError(f"Too few sequences ({len(X)}). Need >=100.")

        n = len(X)
        split = int(n * 0.85)
        X_train, X_val = X[:split], X[split:]
        yd_train, yd_val = y_dir[:split], y_dir[split:]
        yr_train, yr_val = y_ret[:split], y_ret[split:]
        yv_train, yv_val = y_vol[:split], y_vol[split:]
        yc_train, yc_val = y_conf[:split], y_conf[split:]

        self.model = LSTMModel(input_dim=self.feature_count).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        ce_criterion = nn.CrossEntropyLoss()
        mse_criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience = 7
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            perm = np.random.permutation(len(X_train))
            epoch_losses = []
            for i in range(0, len(X_train), self.batch_size):
                idx = perm[i : i + self.batch_size]
                xb = torch.tensor(X_train[idx], dtype=torch.float32).to(self.device)
                yd = torch.tensor(yd_train[idx], dtype=torch.long).to(self.device)
                yr = torch.tensor(yr_train[idx], dtype=torch.float32).to(self.device)
                yv = torch.tensor(yv_train[idx], dtype=torch.float32).to(self.device)
                yc = torch.tensor(yc_train[idx], dtype=torch.float32).to(self.device)

                optimizer.zero_grad()
                out = self.model(xb)
                loss_dir = ce_criterion(out["logits"], yd)
                loss_ret = mse_criterion(out["expected_return_bps"], yr)
                loss_vol = mse_criterion(out["volatility_forecast"], yv)
                loss_conf = mse_criterion(out["confidence"], yc)
                loss = loss_dir + 0.001 * loss_ret + 0.001 * loss_vol + 0.001 * loss_conf
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(float(loss.item()))

            val_metrics = self._validate(X_val, yd_val, yr_val, yv_val, yc_val)
            avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
            val_metrics["epoch"] = epoch + 1
            val_metrics["train_loss"] = round(avg_loss, 6)
            self.history.append(val_metrics)
            logger.info(f"LSTM epoch {epoch + 1}/{self.epochs} | train_loss={avg_loss:.4f} | val_dir_acc={val_metrics.get('directional_accuracy', 0):.3f}")

            val_loss = val_metrics.get("val_loss", float("inf"))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        final_metrics = self.history[-1] if self.history else {}
        return {
            "training_run_id": self.training_run_id,
            "model_id": self.model_id,
            "epochs_trained": len(self.history),
            "best_val_loss": best_val_loss,
            "final_metrics": final_metrics,
        }

    def _validate(
        self,
        X_val: np.ndarray,
        yd_val: np.ndarray,
        yr_val: np.ndarray,
        yv_val: np.ndarray,
        yc_val: np.ndarray,
    ) -> Dict:
        if self.model is None:
            return {}
        self.model.eval()
        with torch.no_grad():
            xb = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            out = self.model(xb)
            preds = out["logits"].argmax(dim=-1).cpu().numpy()
            probs = F.softmax(out["logits"], dim=-1).cpu().numpy()

        y_true = yd_val
        directional_accuracy = float(accuracy_score(y_true, preds)) if accuracy_score else float(np.mean(preds == y_true))

        # Per-class precision/recall
        def _pr(label):
            tp = int(np.sum((preds == label) & (y_true == label)))
            fp = int(np.sum((preds == label) & (y_true != label)))
            fn = int(np.sum((preds != label) & (y_true == label)))
            prec = tp / (tp + fp + 1e-12)
            rec = tp / (tp + fn + 1e-12)
            return prec, rec

        precision_long, recall_long = _pr(1)
        precision_short, recall_short = _pr(2)

        expected_return_error = float(np.mean(np.abs(out["expected_return_bps"].cpu().numpy() - yr_val)))
        vol_error = float(np.mean(np.abs(out["volatility_forecast"].cpu().numpy() - yv_val)))

        # Calibration error: average |confidence - accuracy| per bin
        conf = out["confidence"].cpu().numpy()
        correct = (preds == y_true).astype(float)
        bins = 5
        cal_error = 0.0
        for b in range(bins):
            lo, hi = b / bins, (b + 1) / bins
            mask = (conf >= lo) & (conf < hi)
            if mask.any():
                cal_error += abs(np.mean(conf[mask]) - np.mean(correct[mask]))
        calibration_error = cal_error / bins

        confidence_vs_accuracy = {
            "mean_confidence": float(np.mean(conf)),
            "accuracy": directional_accuracy,
            "gap": abs(float(np.mean(conf)) - directional_accuracy),
        }

        # Performance by regime (simple split by volatility quartile)
        regimes = np.digitize(yv_val, np.quantile(yv_val, [0.25, 0.5, 0.75]))
        performance_by_regime = {}
        for r in [0, 1, 2, 3]:
            mask = regimes == r
            if mask.any():
                performance_by_regime[f"vol_quartile_{r}"] = {
                    "accuracy": float(np.mean(preds[mask] == y_true[mask])),
                    "count": int(mask.sum()),
                }

        val_loss = float(np.mean(np.abs(probs[np.arange(len(y_true)), y_true] - 1.0)))

        return {
            "val_loss": val_loss,
            "directional_accuracy": round(directional_accuracy, 6),
            "precision_long": round(precision_long, 6),
            "precision_short": round(precision_short, 6),
            "recall_long": round(recall_long, 6),
            "recall_short": round(recall_short, 6),
            "expected_return_error": round(expected_return_error, 6),
            "volatility_error": round(vol_error, 6),
            "calibration_error": round(calibration_error, 6),
            "confidence_vs_accuracy": confidence_vs_accuracy,
            "performance_by_regime": performance_by_regime,
        }

    def save(self, base_dir: Optional[str] = None) -> str:
        out_dir = base_dir or os.path.join(PROJECT_ROOT, "models", "lstm", self.model_id)
        os.makedirs(out_dir, exist_ok=True)
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() before save().")

        model_path = os.path.join(out_dir, "model.pt")
        scaler_path = os.path.join(out_dir, "scaler.pkl")
        meta_path = os.path.join(out_dir, "meta.json")

        torch.save(self.model.state_dict(), model_path)
        import joblib
        joblib.dump(self.scaler, scaler_path)

        meta = {
            "training_run_id": self.training_run_id,
            "model_id": self.model_id,
            "dataset_id": self.dataset_id,
            "feature_set_id": self.feature_set_id,
            "label_set_id": self.label_set_id,
            "lookback_window": self.lookback_window,
            "horizons": self.horizons,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "feature_count": self.feature_count,
            "device": self.device,
            "history": self.history,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.success(f"LSTM saved to {out_dir}")
        return out_dir

    def load(self, base_dir: str) -> None:
        meta_path = os.path.join(base_dir, "meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.feature_count = meta.get("feature_count", 0)
        self.model = LSTMModel(input_dim=self.feature_count).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(base_dir, "model.pt"), map_location=self.device))
        import joblib
        self.scaler = joblib.load(os.path.join(base_dir, "scaler.pkl"))
        self.model.eval()


def _make_synthetic_data(n_bars: int = 5000, n_features: int = 32) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    prices = 1.10 + np.cumsum(rng.standard_normal(n_bars) * 0.001)
    features = np.column_stack([
        prices,
        rng.standard_normal(n_bars),
        rng.standard_normal(n_bars) * 0.5,
        np.sin(np.linspace(0, 20 * np.pi, n_bars)),
    ] + [rng.standard_normal(n_bars) for _ in range(max(0, n_features - 4))])
    returns = np.zeros_like(prices)
    returns[1:] = np.diff(prices) / (np.abs(prices[:-1]) + 1e-8)
    volatility = np.abs(returns)
    return features.astype(np.float32), returns.astype(np.float32), volatility.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Train LSTM lane")
    parser.add_argument("--symbol", default="SYNTH")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--timesteps", type=int, default=5000)
    parser.add_argument("--dataset_id", default="synthetic")
    parser.add_argument("--feature_set_id", default="engineered_v2")
    parser.add_argument("--lookback", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_id", default=None)
    args = parser.parse_args()

    logger.info(f"LSTMTrainer start | symbol={args.symbol} | timeframe={args.timeframe} | dataset={args.dataset_id}")

    features, returns, volatility = _make_synthetic_data(n_bars=args.timesteps, n_features=32)
    trainer = LSTMTrainer(
        model_id=args.model_id,
        dataset_id=args.dataset_id,
        feature_set_id=args.feature_set_id,
        lookback_window=args.lookback,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    result = trainer.fit(features, returns, volatility)
    out_dir = trainer.save()
    logger.success(f"LSTM training complete. Model saved to {out_dir}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
