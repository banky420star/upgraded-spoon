"""
Python/training/train_rainforest.py — RainforestTrainer class.

Uses sklearn RandomForestClassifier.
Regime labels: trend_up, trend_down, range_low_vol, range_high_vol,
              breakout_expansion, mean_reversion_zone, chaos_spike,
              spread_danger, liquidity_thin, unknown
Outputs: regime, regime_confidence, allowed_policy_modes, blocked_policy_modes,
         risk_bias, top_features
Save to: models/rainforest/{model_id}/
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
    logger = logging.getLogger("rainforest_trainer")

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger.add(os.path.join(LOG_DIR, "rainforest_training.log"), rotation="10 MB", level="INFO")

try:
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    _SKLEARN_AVAILABLE = True
except Exception:
    RandomForestClassifier = None  # type: ignore
    joblib = None  # type: ignore
    _SKLEARN_AVAILABLE = False

REGIME_LABELS = [
    "trend_up",
    "trend_down",
    "range_low_vol",
    "range_high_vol",
    "breakout_expansion",
    "mean_reversion_zone",
    "chaos_spike",
    "spread_danger",
    "liquidity_thin",
    "unknown",
]

REGIME_POLICY_MAP = {
    "trend_up": {
        "allowed": ["long_normal", "long_small", "reduce_position"],
        "blocked": ["short_small", "short_normal"],
        "risk_bias": "long",
    },
    "trend_down": {
        "allowed": ["short_normal", "short_small", "reduce_position"],
        "blocked": ["long_small", "long_normal"],
        "risk_bias": "short",
    },
    "range_low_vol": {
        "allowed": ["flat", "long_small", "short_small", "reduce_position"],
        "blocked": ["long_normal", "short_normal"],
        "risk_bias": "neutral",
    },
    "range_high_vol": {
        "allowed": ["flat", "reduce_position", "close_position"],
        "blocked": ["long_small", "long_normal", "short_small", "short_normal"],
        "risk_bias": "neutral",
    },
    "breakout_expansion": {
        "allowed": ["long_small", "short_small", "reduce_position"],
        "blocked": ["long_normal", "short_normal"],
        "risk_bias": "mixed",
    },
    "mean_reversion_zone": {
        "allowed": ["long_small", "short_small", "flat"],
        "blocked": ["long_normal", "short_normal"],
        "risk_bias": "mixed",
    },
    "chaos_spike": {
        "allowed": ["flat", "close_position", "reduce_position"],
        "blocked": ["long_small", "long_normal", "short_small", "short_normal"],
        "risk_bias": "neutral",
    },
    "spread_danger": {
        "allowed": ["flat", "close_position"],
        "blocked": ["long_small", "long_normal", "short_small", "short_normal", "reduce_position"],
        "risk_bias": "neutral",
    },
    "liquidity_thin": {
        "allowed": ["flat", "reduce_position", "close_position"],
        "blocked": ["long_small", "long_normal", "short_small", "short_normal"],
        "risk_bias": "neutral",
    },
    "unknown": {
        "allowed": ["flat"],
        "blocked": ["long_small", "long_normal", "short_small", "short_normal", "reduce_position", "close_position"],
        "risk_bias": "neutral",
    },
}


class RainforestTrainer:
    """
    Trainer for the Rainforest (RandomForest) regime-classifier lane.
    """

    DEFAULT_N_ESTIMATORS = 500
    DEFAULT_MAX_DEPTH = 12
    DEFAULT_SEED = 42

    def __init__(
        self,
        model_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        feature_set_id: Optional[str] = None,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        max_depth: int = DEFAULT_MAX_DEPTH,
        seed: int = DEFAULT_SEED,
    ):
        if not _SKLEARN_AVAILABLE:
            raise RuntimeError("RainforestTrainer requires scikit-learn. Install: pip install scikit-learn")

        self.training_run_id = str(uuid.uuid4())
        self.model_id = model_id or f"rainforest_{self.training_run_id[:8]}"
        self.dataset_id = dataset_id or "default"
        self.feature_set_id = feature_set_id or "rainforest_v1"
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.seed = int(seed)

        self.clf: Optional[RandomForestClassifier] = None
        self.feature_names: List[str] = []
        self.top_features: List[Dict] = []
        self.trained_at: Optional[float] = None

    def _label_regimes(self, returns: np.ndarray, volatility: np.ndarray, spread: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """
        Auto-label regimes from price dynamics.
        """
        n = len(returns)
        labels = np.full(n, "unknown", dtype=object)

        med_vol = float(np.median(volatility))
        med_spread = float(np.median(spread))
        med_vol_pct = float(np.median(volume))

        # Trend
        ma5 = np.convolve(returns, np.ones(5) / 5, mode="same")
        trend_up = (ma5 > med_vol * 1.5) & (volatility > med_vol * 0.5)
        trend_down = (ma5 < -med_vol * 1.5) & (volatility > med_vol * 0.5)

        # Range / low vol
        range_low = (np.abs(returns) < med_vol * 0.5) & (volatility < med_vol * 0.8)
        range_high = (np.abs(returns) < med_vol * 0.8) & (volatility >= med_vol * 1.2)

        # Breakout / expansion
        breakout = (np.abs(returns) > med_vol * 2.5) & (volatility > med_vol * 1.5)

        # Mean reversion
        mean_rev = (returns * np.roll(returns, 1) < 0) & (volatility < med_vol * 1.5)

        # Chaos spike
        chaos = (np.abs(returns) > med_vol * 4.0) | (volatility > med_vol * 3.0)

        # Spread danger
        spread_danger = spread > med_spread * 2.0

        # Liquidity thin
        liquidity_thin = volume < med_vol_pct * 0.3

        labels[range_low] = "range_low_vol"
        labels[range_high] = "range_high_vol"
        labels[trend_up] = "trend_up"
        labels[trend_down] = "trend_down"
        labels[breakout] = "breakout_expansion"
        labels[mean_rev] = "mean_reversion_zone"
        labels[chaos] = "chaos_spike"
        labels[spread_danger] = "spread_danger"
        labels[liquidity_thin] = "liquidity_thin"

        return labels

    def fit(self, features: np.ndarray, returns: np.ndarray, volatility: np.ndarray, spread: np.ndarray, volume: np.ndarray, feature_names: Optional[List[str]] = None) -> Dict:
        self.feature_names = feature_names or [f"f_{i}" for i in range(features.shape[1])]
        labels = self._label_regimes(returns, volatility, spread, volume)

        # Drop rows with NaNs
        valid = ~np.isnan(features).any(axis=1)
        X, y = features[valid], labels[valid]
        if len(X) < 100:
            raise ValueError(f"Too few valid rows ({len(X)}). Need >=100.")

        self.clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.seed,
            n_jobs=-1,
            class_weight="balanced",
        )
        self.clf.fit(X, y)
        self.trained_at = time.time()

        importances = dict(zip(self.feature_names, self.clf.feature_importances_.tolist()))
        sorted_imp = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
        self.top_features = [
            {"feature": k, "importance": round(float(v), 6)} for k, v in sorted_imp[:20]
        ]

        logger.success(f"Rainforest trained on {len(X)} rows | classes={list(self.clf.classes_)} | top={self.top_features[0]['feature'] if self.top_features else 'none'}")
        return {
            "training_run_id": self.training_run_id,
            "model_id": self.model_id,
            "samples": len(X),
            "classes": list(self.clf.classes_),
            "top_features": self.top_features,
        }

    def predict(self, X: np.ndarray) -> Dict:
        if self.clf is None:
            return {
                "regime": "unknown",
                "regime_confidence": 0.0,
                "allowed_policy_modes": REGIME_POLICY_MAP["unknown"]["allowed"],
                "blocked_policy_modes": REGIME_POLICY_MAP["unknown"]["blocked"],
                "risk_bias": "neutral",
                "top_features": [],
            }
        x_last = X[-1:].reshape(1, -1)
        proba = self.clf.predict_proba(x_last)[0]
        class_proba = dict(zip(self.clf.classes_, proba.tolist()))
        best = max(class_proba, key=class_proba.get)
        conf = float(class_proba[best])
        policy = REGIME_POLICY_MAP.get(best, REGIME_POLICY_MAP["unknown"])
        return {
            "regime": best,
            "regime_confidence": round(conf, 6),
            "allowed_policy_modes": policy["allowed"],
            "blocked_policy_modes": policy["blocked"],
            "risk_bias": policy["risk_bias"],
            "top_features": self.top_features[:10],
        }

    def save(self, base_dir: Optional[str] = None) -> str:
        out_dir = base_dir or os.path.join(PROJECT_ROOT, "models", "rainforest", self.model_id)
        os.makedirs(out_dir, exist_ok=True)
        if self.clf is None:
            raise RuntimeError("Model not trained. Call fit() before save().")
        model_path = os.path.join(out_dir, "model.pkl")
        meta_path = os.path.join(out_dir, "meta.json")
        payload = {
            "model": self.clf,
            "feature_names": self.feature_names,
            "top_features": self.top_features,
            "trained_at": self.trained_at,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "seed": self.seed,
            "training_run_id": self.training_run_id,
            "model_id": self.model_id,
            "dataset_id": self.dataset_id,
            "feature_set_id": self.feature_set_id,
            "classes": list(self.clf.classes_),
        }
        joblib.dump(payload, model_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in payload.items() if k != "model"}, f, indent=2)
        logger.success(f"Rainforest saved to {out_dir}")
        return out_dir

    def load(self, base_dir: str) -> None:
        model_path = os.path.join(base_dir, "model.pkl")
        payload = joblib.load(model_path)
        self.clf = payload["model"]
        self.feature_names = payload.get("feature_names", [])
        self.top_features = payload.get("top_features", [])
        self.trained_at = payload.get("trained_at")
        self.n_estimators = payload.get("n_estimators", self.n_estimators)
        self.max_depth = payload.get("max_depth", self.max_depth)
        self.seed = payload.get("seed", self.seed)


def _make_synthetic_data(n_bars: int = 5000, n_features: int = 14) -> tuple:
    rng = np.random.default_rng(42)
    prices = 1.10 + np.cumsum(rng.standard_normal(n_bars) * 0.001)
    returns = np.zeros_like(prices)
    returns[1:] = np.diff(prices) / (np.abs(prices[:-1]) + 1e-8)
    volatility = np.abs(returns) + 0.0001
    spread = rng.uniform(1, 5, n_bars)
    volume = rng.uniform(100, 10000, n_bars)
    features = np.column_stack([
        returns,
        np.roll(returns, 1),
        np.roll(returns, 5),
        volatility,
        np.roll(volatility, 1),
        spread / 10000.0,
        volume / 10000.0,
        np.sin(np.linspace(0, 10 * np.pi, n_bars)),
    ] + [rng.standard_normal(n_bars) for _ in range(max(0, n_features - 8))])
    return features.astype(np.float32), returns.astype(np.float32), volatility.astype(np.float32), spread.astype(np.float32), volume.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Train Rainforest regime lane")
    parser.add_argument("--symbol", default="SYNTH")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--timesteps", type=int, default=5000)
    parser.add_argument("--dataset_id", default="synthetic")
    parser.add_argument("--feature_set_id", default="rainforest_v1")
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--max_depth", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_id", default=None)
    args = parser.parse_args()

    logger.info(f"RainforestTrainer start | symbol={args.symbol} | dataset={args.dataset_id}")
    features, returns, volatility, spread, volume = _make_synthetic_data(n_bars=args.timesteps, n_features=14)
    trainer = RainforestTrainer(
        model_id=args.model_id,
        dataset_id=args.dataset_id,
        feature_set_id=args.feature_set_id,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        seed=args.seed,
    )
    result = trainer.fit(features, returns, volatility, spread, volume, feature_names=[f"f_{i}" for i in range(features.shape[1])])
    out_dir = trainer.save()
    logger.success(f"Rainforest training complete. Model saved to {out_dir}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
