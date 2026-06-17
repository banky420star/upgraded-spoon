"""
RegimeDetector — Random Forest Market Regime + Candle Pattern Augmented Observation.

Integrates with TradingEnv to append a regime one-hot (5 classes) to the
observation vector, letting the PPO policy learn conditional strategies
(e.g., go long in trending_up, stay flat in ranging, reduce size in volatile).

Features fed to the RF:
  - ADX (trend strength)
  - ATR ratio (volatility relative to history)
  - BB width (squeeze detection)
  - Volume relative to MA
  - MA slope (trend direction)
  - Candle pattern features from PatternDetector (11+ dims)

Two modes:
  - Online: fits a lightweight RF on-the-fly using heuristic labels from recent data
  - Pretrained: loads a saved RF model from disk
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
    logger = _logging.getLogger("regime_detector")

try:
    from Python.patterns.pattern_detector import PatternDetector, PATTERN_FEATURE_NAMES
    _HAS_PATTERNS = True
except Exception:
    _HAS_PATTERNS = True
    PATTERN_FEATURE_NAMES = [
        "doji", "hammer", "shooting_star", "engulfing_bullish", "engulfing_bearish",
        "harami_bullish", "harami_bearish", "morning_star", "evening_star",
        "piercing_line", "dark_cloud_cover",
    ]

    class PatternDetector:  # noqa: F811
        """Stub detector when Python.patterns.pattern_detector is unavailable."""
        def get_pattern_feature_vector(self, df):
            return [0.0] * len(PATTERN_FEATURE_NAMES)

warnings.filterwarnings("ignore", message=".*F score.*", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ── Regime labels ──────────────────────────────────────────────────────
REGIME_LABELS = ["trending_up", "trending_down", "ranging", "volatile", "breakout"]
NUM_REGIMES = len(REGIME_LABELS)
REGIME_LABEL_MAP = {name: i for i, name in enumerate(REGIME_LABELS)}


# ── Heuristic labelling for online training ────────────────────────────
def _label_regime(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    atr: np.ndarray,
    adx: np.ndarray,
    bb_width: np.ndarray,
    atr_ma: np.ndarray,
    vol_ma: np.ndarray,
    ma50: np.ndarray,
    i: int,
) -> int:
    """Heuristic regime label for bar i (used as training target)."""
    # Need enough lookback
    if i < 20:
        return REGIME_LABEL_MAP["ranging"]

    # Check breakout first (price breaking recent range with volume)
    lookback = min(12, i)
    recent_high = high[i - lookback : i].max()
    recent_low = low[i - lookback : i].min()
    atr_i = max(atr[i], 1e-8)
    vol_i = max(volume[i], 1e-8)

    is_breakout_up = close[i] > recent_high + 0.5 * atr_i and vol_i > 1.5 * vol_ma[i]
    is_breakout_dn = close[i] < recent_low - 0.5 * atr_i and vol_i > 1.5 * vol_ma[i]

    if is_breakout_up or is_breakout_dn:
        return REGIME_LABEL_MAP["breakout"]

    # Volatile: ATR much higher than recent average
    if atr[i] > 1.8 * atr_ma[i] and bb_width[i] > np.percentile(bb_width[max(0, i - 100) : i + 1], 80):
        return REGIME_LABEL_MAP["volatile"]

    # Trending: ADX > 25, clear direction
    if adx[i] > 25:
        # Check slope of 20-bar MA
        ma_slope = (ma50[i] - ma50[max(0, i - 20)]) / (ma50[max(0, i - 20)] + 1e-8)
        if ma_slope > 0.003:
            return REGIME_LABEL_MAP["trending_up"]
        elif ma_slope < -0.003:
            return REGIME_LABEL_MAP["trending_down"]
        else:
            return REGIME_LABEL_MAP["ranging"]

    # Default: ranging
    return REGIME_LABEL_MAP["ranging"]


def _compute_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Vectorized ADX computation."""
    n = len(close)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    tr[0] = high[0] - low[0]

    # Up move / Down move
    up = np.diff(high, prepend=high[0])
    down = np.diff(low, prepend=low[0])
    pos_dm = np.where((up > down) & (up > 0), up, 0.0)
    neg_dm = np.where((down > up) & (down > 0), down, 0.0)

    # Smoothed ATR and DMs
    atr_smooth = np.zeros(n)
    pos_smooth = np.zeros(n)
    neg_smooth = np.zeros(n)
    atr_smooth[0] = tr[0]
    pos_smooth[0] = pos_dm[0]
    neg_smooth[0] = neg_dm[0]

    for i in range(1, n):
        atr_smooth[i] = (atr_smooth[i - 1] * (period - 1) + tr[i]) / period
        pos_smooth[i] = (pos_smooth[i - 1] * (period - 1) + pos_dm[i]) / period
        neg_smooth[i] = (neg_smooth[i - 1] * (period - 1) + neg_dm[i]) / period

    # Directional indicators
    pdi = 100.0 * pos_smooth / (atr_smooth + 1e-8)
    ndi = 100.0 * neg_smooth / (atr_smooth + 1e-8)
    dx = 100.0 * np.abs(pdi - ndi) / (pdi + ndi + 1e-8)

    # ADX is smoothed DX
    adx = np.zeros(n)
    adx[0] = dx[0]
    for i in range(1, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx


# ── Main RegimeDetector ────────────────────────────────────────────────
class RegimeDetector:
    """Random Forest based market regime classifier.

    Args:
        n_estimators: Number of RF trees (lower = faster for online training).
        max_depth: Max tree depth.
        use_patterns: Whether to include PatternDetector features.
        model_path: Path to pre-trained RF model pickle. If None, trains online.
        confidence_threshold: Min confidence to emit a non-ranging regime.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        use_patterns: bool = True,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.45,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.use_patterns = use_patterns and _HAS_PATTERNS
        self.confidence_threshold = confidence_threshold
        self._model = None
        self._pattern_detector = PatternDetector() if self.use_patterns else None
        self._feature_dim = 7  # base regime features
        if self.use_patterns:
            self._feature_dim += len(PATTERN_FEATURE_NAMES)  # + 11 patterns

        # Load pre-trained or lazy-init
        if model_path and os.path.exists(model_path):
            self._load(model_path)
            logger.info(f"RegimeDetector: loaded pre-trained model from {model_path}")

        # Training buffer for online fitting
        self._train_features: list[np.ndarray] = []
        self._train_labels: list[int] = []

        # Memoization: avoid recomputing features when the same DataFrame is
        # passed multiple times (e.g., get_regime_observation + get_regime_label in one step)
        self._cached_features_df_id: int = 0
        self._cached_features_df_len: int = 0
        self._cached_features: np.ndarray | None = None

    # ── Public API ──────────────────────────────────────────────────

    def compute_features_batch(self, df: pd.DataFrame) -> np.ndarray:
        """Compute regime feature vectors for ALL bars.

        Returns:
            (n, self._feature_dim) float32 array.
        """
        _require_ohlcv(df)

        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        volume = df["volume"].values.astype(float) if "volume" in df.columns else np.ones_like(close)
        n = len(close)
        eps = 1e-8

        # ── Compute regime indicators ──
        atr = self._compute_atr(high, low, close, 14)
        adx = _compute_adx(high, low, close, 14)

        # BB width
        ma20 = pd.Series(close).rolling(20, min_periods=1).mean().values
        std20 = pd.Series(close).rolling(20, min_periods=1).std().fillna(0).values
        bb_width = (4.0 * std20) / (ma20 + eps)

        # ATR ratio
        atr_ma = pd.Series(atr).rolling(50, min_periods=1).mean().fillna(atr.mean()).values
        atr_ratio = atr / (atr_ma + eps)

        # Volume relative
        vol_ma20 = pd.Series(volume).rolling(20, min_periods=1).mean().fillna(volume.mean()).values
        vol_rel = volume / (vol_ma20 + eps)

        # MA slope (50-bar)
        ma50 = pd.Series(close).rolling(50, min_periods=1).mean().values
        ma_slope = np.zeros(n)
        for i in range(20, n):
            ma_slope[i] = (ma50[i] - ma50[i - 20]) / (ma50[i - 20] + eps)

        # Price relative to MA50
        price_rel_ma = close / (ma50 + eps) - 1.0

        # ── Build per-bar features ──
        base_feats = np.column_stack([
            np.clip(adx / 100.0, 0, 1),
            np.clip(atr_ratio, 0, 5),
            np.clip(bb_width, 0, 5),
            np.clip(vol_rel, 0, 10),
            np.clip(ma_slope * 100, -1, 1),
            np.clip(price_rel_ma, -1, 1),
            (adx > 25) & (atr < atr_ma * 1.3),
        ]).astype(np.float32)

        # ── Pattern features per bar ──
        if self.use_patterns and self._pattern_detector is not None:
            pat_dim = len(PATTERN_FEATURE_NAMES)
            pat_features = np.zeros((n, pat_dim), dtype=np.float32)
            for i in range(n):
                try:
                    row_df = df.iloc[max(0, i - 5): i + 1].copy()
                    pat_vec = self._pattern_detector.get_pattern_feature_vector(row_df)
                    for j, v in enumerate(pat_vec):
                        if j < pat_dim:
                            pat_features[i, j] = float(v)
                except Exception:
                    pass
            return np.column_stack([base_feats, pat_features]).astype(np.float32)

        return base_feats

    def compute_features(self, df: pd.DataFrame) -> np.ndarray:
        """Compute regime feature vector for the latest bar.

        Returns:
            (self._feature_dim,) float32 array.
        """
        _require_ohlcv(df)

        # ── Memoization ──────────────────────────────────────────
        # Avoid recomputing when the same DataFrame is passed multiple
        # times per step (e.g. get_regime_observation + get_regime_label).
        df_id = id(df)
        df_len = len(df)
        if (
            df_id == self._cached_features_df_id
            and df_len == self._cached_features_df_len
            and self._cached_features is not None
        ):
            return self._cached_features

        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        volume = df["volume"].values.astype(float) if "volume" in df.columns else np.ones_like(close)
        n = len(close)

        eps = 1e-8

        # ── Compute regime indicators ──
        atr = self._compute_atr(high, low, close, 14)
        adx = _compute_adx(high, low, close, 14)

        # BB width
        ma20 = pd.Series(close).rolling(20, min_periods=1).mean().values
        std20 = pd.Series(close).rolling(20, min_periods=1).std().fillna(0).values
        bb_width = (4.0 * std20) / (ma20 + eps)

        # ATR ratio (current ATR vs 50-bar average)
        atr_ma = pd.Series(atr).rolling(50, min_periods=1).mean().fillna(atr.mean()).values
        atr_ratio = atr / (atr_ma + eps)

        # Volume relative to MA(20)
        vol_ma20 = pd.Series(volume).rolling(20, min_periods=1).mean().fillna(volume.mean()).values
        vol_rel = volume / (vol_ma20 + eps)

        # MA slope (50-bar)
        ma50 = pd.Series(close).rolling(50, min_periods=1).mean().values
        ma_slope_20 = np.zeros(n)
        for i in range(20, n):
            ma_slope_20[i] = (ma50[i] - ma50[i - 20]) / (ma50[i - 20] + eps)
        ma_slope = ma_slope_20

        # Price relative to MA50
        price_rel_ma = close / (ma50 + eps) - 1.0

        # ── Latest values (last bar) ──
        feats = [
            float(np.clip(adx[-1] / 100.0, 0, 1)),           # 0: ADX normalised
            float(np.clip(atr_ratio[-1], 0, 5)),              # 1: ATR ratio
            float(np.clip(bb_width[-1], 0, 5)),               # 2: BB width
            float(np.clip(vol_rel[-1], 0, 10)),               # 3: Volume relative
            float(np.clip(ma_slope[-1] * 100, -1, 1)),        # 4: MA slope * 100
            float(np.clip(price_rel_ma[-1], -1, 1)),          # 5: Price vs MA50
            float(adx[-1] > 25 and atr[-1] < atr_ma[-1] * 1.3),  # 6: trending + not volatile flag
        ]

        # ── Pattern features ──
        if self.use_patterns and self._pattern_detector is not None:
            try:
                pat_vec = self._pattern_detector.get_pattern_feature_vector(df)
                for v in pat_vec:
                    feats.append(float(v))
            except Exception:
                feats.extend([0.0] * len(PATTERN_FEATURE_NAMES))

        result = np.array(feats, dtype=np.float32)
        self._cached_features_df_id = df_id
        self._cached_features_df_len = df_len
        self._cached_features = result
        return result

    def classify(self, features: np.ndarray) -> tuple[int, float]:
        """Classify regime from feature vector.

        Returns:
            (regime_index, confidence)
            If no model is fitted yet, returns ranging with low confidence.
        """
        if self._model is None:
            return REGIME_LABEL_MAP["ranging"], 0.3

        probs = self._model.predict_proba(features.reshape(1, -1))[0]
        pred = int(np.argmax(probs))
        confidence = float(probs[pred])
        return pred, confidence

    def classify_batch(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Classify a batch of feature vectors.

        Args:
            features: 2D array of shape (n_samples, n_features).

        Returns:
            (preds, confs) — 1D arrays of regime indices and confidences.
        """
        if self._model is None:
            n = len(features)
            return np.full(n, REGIME_LABEL_MAP["ranging"]), np.full(n, 0.3)

        probs = self._model.predict_proba(features)
        preds = np.argmax(probs, axis=1)
        confs = np.max(probs, axis=1)
        return preds, confs

    def get_regime_observation(self, df: pd.DataFrame) -> np.ndarray:
        """Get the full regime observation: 5-dim one-hot + confidence.

        Returns:
            (6,) float32: [regime_onehot(5), confidence]
        """
        features = self.compute_features(df)
        regime_idx, confidence = self.classify(features)

        # Online training: store this sample for future fitting
        self._train_features.append(features)
        self._train_labels.append(regime_idx)
        # Cap buffer size
        if len(self._train_features) > 5000:
            self._train_features = self._train_features[-2500:]
            self._train_labels = self._train_labels[-2500:]

        onehot = np.zeros(NUM_REGIMES, dtype=np.float32)
        onehot[regime_idx] = 1.0
        return np.concatenate([onehot, np.array([confidence], dtype=np.float32)])

    def get_regime_label(self, df: pd.DataFrame) -> str:
        """Human-readable regime name."""
        features = self.compute_features(df)
        idx, _ = self.classify(features)
        return REGIME_LABELS[idx]

    # ── Training ───────────────────────────────────────────────────

    def fit_online(self, force: bool = False):
        """Fit/refit RF on accumulated heuristic training data.

        Should be called periodically (e.g., every 1000 env steps).
        """
        if len(self._train_features) < 50 and not force:
            return

        from sklearn.ensemble import RandomForestClassifier

        X = np.array(self._train_features, dtype=np.float32)
        y = np.array(self._train_labels, dtype=np.int32)

        # Balance classes (if we have at least 5 samples of each)
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2:
            return

        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        )
        self._model.fit(X, y)
        logger.info(
            f"RegimeDetector: fitted RF on {len(X)} samples, "
            f"classes={dict(zip(unique, counts))}"
        )

    def fit_heuristic(self, df: pd.DataFrame):
        """Fit RF using heuristic labels computed from the full DataFrame."""
        _require_ohlcv(df)
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        volume = df["volume"].values.astype(float) if "volume" in df.columns else np.ones_like(close)
        n = len(close)

        if n < 50:
            logger.warning("RegimeDetector: too few rows for heuristic fit")
            return

        # Compute all indicators
        atr = self._compute_atr(high, low, close, 14)
        adx = _compute_adx(high, low, close, 14)
        ma20 = pd.Series(close).rolling(20, min_periods=1).mean().values
        std20 = pd.Series(close).rolling(20, min_periods=1).std().fillna(0).values
        bb_width = (4.0 * std20) / (ma20 + 1e-8)
        atr_ma = pd.Series(atr).rolling(50, min_periods=1).mean().fillna(atr.mean()).values
        vol_ma = pd.Series(volume).rolling(20, min_periods=1).mean().fillna(volume.mean()).values
        ma50 = pd.Series(close).rolling(50, min_periods=1).mean().values

        # Build features and labels for each bar
        features_list = []
        labels = []

        for i in range(50, n):
            label = _label_regime(close, high, low, volume, atr, adx, bb_width, atr_ma, vol_ma, ma50, i)
            # Feature vector for this bar (use a fake df slice)
            row_df = df.iloc[max(0, i - 5) : i + 1].copy()
            try:
                fv = self.compute_features(row_df)
            except Exception:
                continue
            features_list.append(fv)
            labels.append(label)

        if len(features_list) < 50:
            logger.warning(f"RegimeDetector: only {len(features_list)} samples, skipping fit")
            return

        from sklearn.ensemble import RandomForestClassifier

        X = np.array(features_list, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)

        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        )
        self._model.fit(X, y)

        unique, counts = np.unique(y, return_counts=True)
        logger.info(
            f"RegimeDetector: fitted RF on {len(X)} heuristic samples, "
            f"classes={dict(zip(unique, counts))}"
        )

    # ── Persistence ────────────────────────────────────────────────

    def save(self, path: str):
        """Save fitted model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self._model, "feature_dim": self._feature_dim}, f)
        logger.info(f"RegimeDetector: saved to {path}")

    def _load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._model = data.get("model")
        self._feature_dim = data.get("feature_dim", self._feature_dim)

    # ── Internal helpers ───────────────────────────────────────────

    @staticmethod
    def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr[0] = tr1[0]
        atr = np.zeros_like(tr)
        atr[0] = tr[0]
        for i in range(1, len(tr)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        return np.maximum(atr, 1e-8)

    def get_feature_names(self) -> list[str]:
        """Human-readable feature names for debugging."""
        names = [
            "adx_norm",
            "atr_ratio",
            "bb_width",
            "vol_rel",
            "ma_slope",
            "price_rel_ma",
            "trending_flag",
        ]
        if self.use_patterns:
            names.extend([f"pat_{n}" for n in PATTERN_FEATURE_NAMES])
        return names


def _require_ohlcv(df: pd.DataFrame):
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain column '{col}'")


# ── Quick test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    # Generate synthetic data
    n = 500
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "open": np.cumsum(np.random.randn(n) * 0.001) + 100,
        "high": np.zeros(n),
        "low": np.zeros(n),
        "close": np.zeros(n),
        "volume": np.random.rand(n) * 1000 + 500,
    }, index=idx)
    df["close"] = df["open"] + np.random.randn(n) * 0.002
    df["high"] = np.maximum(df["open"], df["close"]) + np.random.rand(n) * 0.003
    df["low"] = np.minimum(df["open"], df["close"]) - np.random.rand(n) * 0.003

    rd = RegimeDetector()
    obs = rd.get_regime_observation(df)
    print(f"Regime observation shape: {obs.shape}")
    print(f"Regime onehot: {obs[:5]}")
    print(f"Confidence: {obs[5]:.3f}")
    print(f"Regime label: {rd.get_regime_label(df)}")

    # Test online
    rd.fit_heuristic(df)
    obs = rd.get_regime_observation(df)
    print(f"\nAfter fit:")
    print(f"Regime onehot: {obs[:5]}")
    print(f"Confidence: {obs[5]:.3f}")
    print(f"Regime label: {rd.get_regime_label(df)}")
    print("✅ RegimeDetector working!")
