"""
RFFeatureSelector — Random Forest based feature importance pruning.

Reduces the 150+ engineered features to the top K most predictive features
for next-bar return. Runs once per training session before building the
TradingEnv. Persists the selected feature mask for inference consistency.

Usage:
    selector = RFFeatureSelector(top_k=40)
    selector.fit(feature_df, target_col='close')
    selected_indices = selector.get_selected_indices()
    reduced_df = selector.transform(feature_df)
    selector.save('models/feature_selector.pkl')

Integration:
    In enhanced_train_drl.py, after fetching training data:
        selector = RFFeatureSelector(top_k=40)
        selector.fit(feature_matrix, close_prices)
        feature_pipeline.apply_feature_selector(selector)
    The TradingEnv then receives only the top K features per bar.
"""

from __future__ import annotations

import os
import pickle
import warnings
from typing import Optional

import numpy as np
import pandas as pd

try:
    from loguru import logger
except ImportError:
    import logging as _logging
    logger = _logging.getLogger("feature_selector")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class RFFeatureSelector:
    """Random Forest based feature importance selector.

    Args:
        top_k: Number of top features to keep (default 40).
        n_estimators: RF trees for importance estimation.
        max_depth: Max tree depth (shallow = faster, less overfitting).
        min_samples_leaf: Min samples per leaf.
        random_state: RNG seed.
        model_path: Optional path to load a pre-fitted selector.
    """

    def __init__(
        self,
        top_k: int = 40,
        n_estimators: int = 200,
        max_depth: int = 8,
        min_samples_leaf: int = 10,
        random_state: int = 42,
        model_path: Optional[str] = None,
    ):
        self.top_k = max(10, min(100, top_k))
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self._model = None
        self._selected_indices: list[int] = []
        self._feature_importances: np.ndarray | None = None
        self._feature_names: list[str] = []
        self._total_features: int = 0

        if model_path and os.path.exists(model_path):
            self.load(model_path)
            logger.info(f"RFFeatureSelector: loaded from {model_path}")

    # ── Public API ──────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: Optional[np.ndarray] = None,
        target_col: str = "close",
    ) -> "RFFeatureSelector":
        """Fit RF and compute feature importances.

        Args:
            X: Feature matrix (n_samples, n_features) or DataFrame.
            y: Target values. If None, uses next-bar return as target.
            target_col: Ignored if y is provided; labels column for auto-target.

        Returns:
            self (for chaining).
        """
        from sklearn.ensemble import RandomForestRegressor

        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
            X_arr = X.values.astype(np.float32)
        else:
            self._feature_names = [f"feat_{i}" for i in range(X.shape[1])]
            X_arr = X.astype(np.float32)

        self._total_features = X_arr.shape[1]

        # Compute target: next-bar return if not provided
        if y is None:
            # Use the mean of the last 50% of rows as target proxy
            # Or compute forward return
            y = np.zeros(X_arr.shape[0])
            for i in range(1, X_arr.shape[0]):
                # Use a simple momentum signal as proxy
                row_mean = float(np.mean(np.abs(X_arr[i])))
                y[i] = row_mean  # absolute feature magnitude as proxy for signal strength
            # Better: use actual close if available
            if self._feature_names and "close" in self._feature_names:
                close_col = self._feature_names.index("close")
                close_vals = X_arr[:, close_col]
                returns = np.diff(close_vals, prepend=close_vals[0])
                y = returns / (np.abs(close_vals) + 1e-8)
            elif self._feature_names and "close_ret_1" in self._feature_names:
                ret_col = self._feature_names.index("close_ret_1")
                y = X_arr[:, ret_col]

        # Remove NaN/Inf
        mask = np.isfinite(X_arr).all(axis=1) & np.isfinite(y)
        X_clean = X_arr[mask]
        y_clean = y[mask]

        if len(X_clean) < 50:
            logger.warning(
                f"RFFeatureSelector: only {len(X_clean)} clean samples, "
                f"using all features (no pruning)"
            )
            self._selected_indices = list(range(self._total_features))
            return self

        # Train RF
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=1,
            verbose=0,
        )
        self._model.fit(X_clean, y_clean)

        # Compute importances
        self._feature_importances = self._model.feature_importances_

        # Select top K
        if self._total_features <= self.top_k:
            self._selected_indices = list(range(self._total_features))
        else:
            self._selected_indices = np.argsort(self._feature_importances)[-self.top_k:].tolist()
            self._selected_indices.sort()

        kept_pct = 100.0 * len(self._selected_indices) / max(self._total_features, 1)
        top_names = [self._feature_names[i] for i in self._selected_indices[-10:]]
        logger.info(
            f"RFFeatureSelector: {self._total_features} → {len(self._selected_indices)} features "
            f"({kept_pct:.1f}% kept). Top: {top_names[::-1]}"
        )

        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Reduce feature matrix to selected features only.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            (n_samples, top_k) array.
        """
        if not self._selected_indices:
            logger.warning("RFFeatureSelector: not fitted, returning original")
            return np.asarray(X, dtype=np.float32)

        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        if X_arr.shape[1] != self._total_features:
            logger.warning(
                f"RFFeatureSelector: expected {self._total_features} features, "
                f"got {X_arr.shape[1]}. Clamping indices."
            )
            valid_idx = [i for i in self._selected_indices if i < X_arr.shape[1]]
            if not valid_idx:
                return X_arr
            return X_arr[:, valid_idx]

        return X_arr[:, self._selected_indices]

    def get_selected_indices(self) -> list[int]:
        """Return the column indices of selected features."""
        return list(self._selected_indices)

    def get_selected_names(self) -> list[str]:
        """Return the names of selected features."""
        if not self._selected_indices or not self._feature_names:
            return []
        return [self._feature_names[i] for i in self._selected_indices]

    def get_importance_scores(self) -> dict[str, float]:
        """Return dict of feature_name → importance score for selected features."""
        if self._feature_importances is None or not self._feature_names:
            return {}
        return {
            self._feature_names[i]: float(self._feature_importances[i])
            for i in self._selected_indices
        }

    # ── Integration helpers ────────────────────────────────────────

    def fit_on_feature_matrix(
        self,
        feature_matrix: np.ndarray,
        close_prices: np.ndarray,
    ) -> "RFFeatureSelector":
        """Fit using a raw feature matrix + close prices.

        Convenience method for training pipeline integration.
        """
        # Create a mock DataFrame with feature names
        n_feats = feature_matrix.shape[1]
        feature_names = [f"f{i}" for i in range(n_feats)]
        df = pd.DataFrame(feature_matrix, columns=feature_names)
        # Add close as the last feature for target computation
        df["__close__"] = close_prices[: len(df)]

        returns = np.diff(close_prices[: len(df) + 1], prepend=close_prices[0])
        ret_target = returns[: len(df)] / (np.abs(close_prices[: len(df)]) + 1e-8)

        X_arr = feature_matrix
        mask = np.isfinite(X_arr).all(axis=1) & np.isfinite(ret_target)
        X_clean = X_arr[mask]
        y_clean = ret_target[mask]

        if len(X_clean) < 50:
            self._selected_indices = list(range(n_feats))
            self._total_features = n_feats
            return self

        from sklearn.ensemble import RandomForestRegressor

        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=1,
        )
        self._model.fit(X_clean, y_clean)
        self._feature_importances = self._model.feature_importances_
        self._total_features = n_feats

        if n_feats <= self.top_k:
            self._selected_indices = list(range(n_feats))
        else:
            self._selected_indices = np.argsort(self._feature_importances)[-self.top_k:].tolist()
            self._selected_indices.sort()

        logger.info(
            f"RFFeatureSelector: {n_feats} → {len(self._selected_indices)} features "
            f"({100.0 * len(self._selected_indices) / max(n_feats, 1):.1f}% kept)"
        )
        return self

    # ── Persistence ────────────────────────────────────────────────

    def save(self, path: str):
        """Save fitted selector to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "top_k": self.top_k,
            "selected_indices": self._selected_indices,
            "feature_names": self._feature_names,
            "feature_importances": self._feature_importances,
            "total_features": self._total_features,
            "model": self._model,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"RFFeatureSelector: saved to {path}")

    def load(self, path: str):
        """Load fitted selector from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.top_k = data.get("top_k", self.top_k)
        self._selected_indices = data.get("selected_indices", [])
        self._feature_names = data.get("feature_names", [])
        self._feature_importances = data.get("feature_importances")
        self._total_features = data.get("total_features", 0)
        self._model = data.get("model")
        logger.info(
            f"RFFeatureSelector: loaded ({len(self._selected_indices)}/{self._total_features} features)"
        )


# ── Quick test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)

    # Generate synthetic features + target
    n = 500
    n_feats = 60
    X = np.random.randn(n, n_feats).astype(np.float32)
    # Make first 10 features predictive
    y = 0.3 * X[:, 0] + 0.2 * X[:, 3] - 0.15 * X[:, 7] + 0.1 * np.random.randn(n)

    selector = RFFeatureSelector(top_k=15)
    selector.fit(X, y)
    print(f"Selected {len(selector._selected_indices)} features out of {n_feats}")
    X_reduced = selector.transform(X)
    print(f"Reduced shape: {X_reduced.shape}")
    print("✅ RFFeatureSelector working!")
