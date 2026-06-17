"""
ML-based directional signal for trading.
Trains an XGBoost classifier on engineered features to predict next-bar direction,
then pipes the probability into the PPO observation as an additional feature.
"""
from __future__ import annotations

import numpy as np
from loguru import logger

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    xgb = None

try:
    from sklearn.ensemble import RandomForestClassifier
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

ML_SIGNAL_FEATURES = 1
"""Number of ML signal features appended to the observation matrix (1 = probability)."""

# Module-level cached model to avoid re-training across parallel vectorized envs
_ml_model_cache = {"model": None, "n_samples": 0, "n_features": 0}


def compute_ml_signal(
    feature_matrix: np.ndarray,
    close_prices: np.ndarray,
) -> np.ndarray:
    """
    Train a tree-based classifier on features to predict next-bar direction.

    Args:
        feature_matrix: (n_timesteps, n_features) feature matrix.
        close_prices: (n_timesteps,) close prices for computing the target.

    Returns:
        (n_timesteps, 1) probability of next bar being UP, in [0, 1].
        Returns zeros if no model is available.
    """
    n = len(feature_matrix)
    if n < 50:
        return np.zeros((n, 1), dtype=np.float32)

    # Check cache: if same data dimensions, reuse model (avoids re-training per parallel env)
    global _ml_model_cache
    if _ml_model_cache["model"] is not None:
        if _ml_model_cache["n_samples"] == n and _ml_model_cache["n_features"] == feature_matrix.shape[1]:
            try:
                proba = _ml_model_cache["model"].predict_proba(feature_matrix)[:, 1]
                return proba.astype(np.float32).reshape(-1, 1)
            except Exception:
                _ml_model_cache["model"] = None

    # Target: next-bar binary direction
    target = np.zeros(n, dtype=np.int32)
    target[:-1] = (close_prices[1:] > close_prices[:-1]).astype(np.int32)

    # Try XGBoost first (fast, handles non-linearity well)
    model = None
    if _HAS_XGB:
        try:
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=42,
                verbosity=0,
                n_jobs=1,
            )
            model.fit(feature_matrix, target)
            logger.info(f"ML Signal: XGBoost trained on {n} x {feature_matrix.shape[1]}")
        except Exception as e:
            logger.warning(f"ML Signal: XGBoost failed ({e})")

    # Fallback to sklearn RandomForest
    if model is None and _HAS_SKLEARN:
        try:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(feature_matrix, target)
            logger.info(f"ML Signal: RandomForest trained on {n} x {feature_matrix.shape[1]}")
        except Exception as e:
            logger.warning(f"ML Signal: RandomForest failed ({e})")

    if model is None:
        logger.warning("ML Signal: No model available, returning zeros")
        return np.zeros((n, 1), dtype=np.float32)

    proba = model.predict_proba(feature_matrix)[:, 1]
    # Cache the trained model for reuse across parallel envs
    _ml_model_cache["model"] = model
    _ml_model_cache["n_samples"] = n
    _ml_model_cache["n_features"] = feature_matrix.shape[1]
    return proba.astype(np.float32).reshape(-1, 1)
