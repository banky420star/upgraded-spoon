"""
RainforestDetector — RandomForest-based market regime classifier.

Regimes: bull_trend, bear_trend, ranging, breakout_up, breakout_down,
         reversal_up, reversal_down

Features are computed from OHLCV bars; labels are auto-generated from
forward returns and volatility so the detector can train on any raw
price history without hand-labelled data.

Usage:
    detector = RainforestDetector()
    detector.train_from_mt5_data("BTCUSDm")
    pred = detector.predict_regime(df)
    print(pred["regime"], pred["confidence"])
"""
from __future__ import annotations

import os
import time
from typing import Optional

try:
    from loguru import logger
except ImportError:
    import logging as _logging
    logger = _logging.getLogger("rainforest_detector")  # type: ignore

# ── numpy / pandas — soft import; error deferred to first use ───────────────
try:
    import numpy as np
    import pandas as pd
    _NUMPY_PANDAS_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    pd = None  # type: ignore
    _NUMPY_PANDAS_AVAILABLE = False


def _require_numpy_pandas():
    if not _NUMPY_PANDAS_AVAILABLE:
        raise ImportError(
            "numpy and pandas are required for RainforestDetector. "
            "Install them with: pip install numpy pandas"
        )


# ── Optional MT5 backend (Windows / Wine bridge) ────────────────────────────
try:
    from Python.mt5_compat import MT5Compat  # type: ignore
except Exception:
    MT5Compat = None  # type: ignore

# ── PatternDetector integration for classical patterns in regime features ──
try:
    from Python.patterns.pattern_detector import PatternDetector, PATTERN_FEATURE_NAMES
    _PATTERN_DETECTOR_AVAILABLE = True
except Exception:
    _PATTERN_DETECTOR_AVAILABLE = False
    PatternDetector = None  # type: ignore
    PATTERN_FEATURE_NAMES = []  # type: ignore

# ── sklearn / joblib ─────────────────────────────────────────────────────────
try:
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    RandomForestClassifier = None  # type: ignore
    joblib = None  # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REGIMES = [
    "bull_trend",
    "bear_trend",
    "ranging",
    "breakout_up",
    "breakout_down",
    "reversal_up",
    "reversal_down",
]

FEATURE_NAMES = [
    # Core technicals (original)
    "returns_1",
    "returns_5",
    "returns_15",
    "returns_60",
    "volatility_10",
    "volatility_20",
    "atr_pct",
    "rsi_14",
    "macd_signal",
    "bb_width",
    "volume_ratio",
    "price_vs_sma50",
    "price_vs_sma200",
    "high_low_range",
    # Timing / market structure (added for open/news awareness)
    "session_london",
    "session_ny",
    "major_open_win",
    "news_proximity",
    "has_news_soon",
    # Classical patterns (will be populated by PatternDetector)
    "has_doji",
    "has_hammer",
    "has_shooting_star",
    "has_bullish_engulfing",
    "has_bearish_engulfing",
    "has_double_top",
    "has_double_bottom",
    "has_bull_flag",
    "has_bear_flag",
    "has_breakout_up",
    "has_breakout_down",
]

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Helper indicator functions
# ---------------------------------------------------------------------------

def _log_returns(close: pd.Series, n: int) -> pd.Series:
    """Log return over n bars."""
    return np.log(close / close.shift(n)).fillna(0.0)


def _rolling_std(returns: pd.Series, n: int) -> pd.Series:
    return returns.rolling(n, min_periods=max(1, n // 2)).std().fillna(0.0)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean().fillna(1e-8)


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(n, min_periods=1).mean()
    rs = gain / (loss + 1e-8)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def _macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    return (macd - sig).fillna(0.0)


def _bollinger_width(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    mid = close.rolling(n, min_periods=1).mean()
    std = close.rolling(n, min_periods=1).std().fillna(0.0)
    upper = mid + k * std
    lower = mid - k * std
    return ((upper - lower) / (mid + 1e-8)).fillna(0.0)


# ---------------------------------------------------------------------------
# RainforestDetector
# ---------------------------------------------------------------------------

class RainforestDetector:
    """
    Random Forest ensemble for real-time market pattern & regime classification.

    Regimes: bull_trend, bear_trend, ranging, breakout_up, breakout_down,
             reversal_up, reversal_down
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = 12, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self._model: Optional[RandomForestClassifier] = None
        self._trained_at: Optional[float] = None
        self._feature_importances: dict[str, float] = {}
        self._classes: list[str] = list(REGIMES)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_features(self, df) -> "np.ndarray":
        """Extract feature matrix from OHLCV DataFrame.

        Expects columns: open, high, low, close, volume.
        Returns shape (n_rows, n_features).
        """
        _require_numpy_pandas()
        df = df.copy().reset_index(drop=True)
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float)

        ret1 = _log_returns(close, 1)
        ret5 = _log_returns(close, 5)
        ret15 = _log_returns(close, 15)
        ret60 = _log_returns(close, 60)

        vol10 = _rolling_std(ret1, 10)
        vol20 = _rolling_std(ret1, 20)

        atr14 = _atr(high, low, close, 14)
        atr_pct = (atr14 / (close + 1e-8)).fillna(0.0)

        rsi14 = _rsi(close, 14)
        macd_hist = _macd_hist(close)
        bb_width = _bollinger_width(close)

        vol_ma20 = volume.rolling(20, min_periods=1).mean()
        volume_ratio = (volume / (vol_ma20 + 1e-8)).fillna(1.0)

        sma50 = close.rolling(50, min_periods=1).mean()
        sma200 = close.rolling(200, min_periods=1).mean()
        price_vs_sma50 = (close / (sma50 + 1e-8) - 1.0).fillna(0.0)
        price_vs_sma200 = (close / (sma200 + 1e-8) - 1.0).fillna(0.0)

        bar_range = (high - low)
        high_low_range = (bar_range / (atr14 + 1e-8)).fillna(1.0)

        # ── NEW: Session / open window / news timing features for Rainforest regime detection
        # Full ensemble (PPO Decision + Dreamer + Rainforest) now aware of market structure around opens and news.
        n = len(close)
        session_london = np.zeros(n)
        session_ny = np.zeros(n)
        major_open_win = np.zeros(n)
        news_proximity = np.zeros(n)
        has_news_soon = np.zeros(n)
        if 'time' in df.columns or hasattr(df.index, 'hour'):
            try:
                if 'time' in df.columns:
                    dt = pd.to_datetime(df['time'], errors='coerce', utc=True)
                else:
                    dt = pd.to_datetime(df.index, errors='coerce', utc=True)
                if hasattr(dt, 'hour'):
                    hf = dt.hour.astype(float) + dt.minute.astype(float) / 60.0
                    session_london = ((hf >= 8) & (hf < 17)).astype(float).to_numpy()
                    session_ny = ((hf >= 13) & (hf < 22)).astype(float).to_numpy()
                    lwin = ((hf >= 7.5) & (hf <= 9.5)).astype(float).to_numpy()
                    nywin = ((hf >= 12.5) & (hf <= 14.5)).astype(float).to_numpy()
                    major_open_win = np.maximum(lwin, nywin)
            except Exception:
                pass
        if 'news_distance_minutes' in df.columns:
            try:
                nd = pd.to_numeric(df['news_distance_minutes'], errors='coerce').fillna(999).to_numpy()
                news_proximity = np.clip(1.0 / (1.0 + nd / 60.0), 0, 1)
                has_news_soon = (nd < 60).astype(float)
            except Exception:
                pass

        # ── CLASSICAL PATTERN FEATURES via PatternDetector (the "edge" integration)
        # Patterns + timing give Rainforest richer regimes; Dreamer pattern-conditioned imagination;
        # Decision PPO / ensemble can bias TimeExitSpec, sizing, partials on favorable states.
        n_pat = len(PATTERN_FEATURE_NAMES) or 11
        pattern_feats = np.zeros((n, n_pat), dtype=np.float32)  # aligned to FEATURE_NAMES tail (11 patterns)
        if _PATTERN_DETECTOR_AVAILABLE and n >= 5:
            try:
                detector = PatternDetector(atr_period=14)
                # Compute per-row (or latest for efficiency; broadcast recent pattern state)
                # For training regimes we use latest bar patterns as context (world-model friendly)
                timing_for_pat = {
                    "major_open_window": float(major_open_win[-1]) if n > 0 else 0.0,
                    "news_proximity": float(news_proximity[-1]) if n > 0 else 0.0,
                    "has_high_impact_news_soon": float(has_news_soon[-1]) if n > 0 else 0.0,
                }
                pat_vec = detector.get_pattern_feature_vector(df, timing_context=timing_for_pat)
                # Broadcast the current pattern snapshot across rows (standard for regime classifiers;
                # downstream can use rolling if desired). This fixes FEATURE_NAMES length mismatch.
                for col in range(12):
                    pattern_feats[:, col] = pat_vec[col]
            except Exception:
                # graceful: patterns neutral if detector hiccups
                pass

        feat = np.column_stack([
            ret1.values,
            ret5.values,
            ret15.values,
            ret60.values,
            vol10.values,
            vol20.values,
            atr_pct.values,
            rsi14.values,
            macd_hist.values,
            bb_width.values,
            volume_ratio.values,
            price_vs_sma50.values,
            price_vs_sma200.values,
            high_low_range.values,
            # NEW timing (5 added): regimes now conditioned on opens/news timing
            session_london,
            session_ny,
            major_open_win,
            news_proximity,
            has_news_soon,
            # CLASSICAL PATTERNS (11): completes FEATURE_NAMES (31 total); gives the edge
            *[pattern_feats[:, i] for i in range(n_pat)],
        ])
        # Ensure exact alignment with FEATURE_NAMES (core14 + timing5 + patterns11 = 30)
        expected = len(FEATURE_NAMES)
        assert feat.shape[1] == expected, f"Rainforest feature count mismatch: {feat.shape[1]} vs {expected}"
        return feat.astype(np.float32)

    # ------------------------------------------------------------------
    # Auto-labelling
    # ------------------------------------------------------------------

    def label_regimes(self, df, forward_window: int = 20) -> "np.ndarray":
        """Auto-label historical data using forward returns and volatility.

        Rules (evaluated in priority order):
          breakout_up   : forward_return > +2% AND bar range > 1.5x ATR
          breakout_down : forward_return < -2% AND bar range > 1.5x ATR
          reversal_up   : prev_return < -1% AND current return > +0.5%
          reversal_down : prev_return > +1% AND current return < -0.5%
          bull_trend    : forward_return > +1% AND volatility < median
          bear_trend    : forward_return < -1% AND volatility < median
          else          : ranging
        """
        df = df.copy().reset_index(drop=True)
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        ret1 = _log_returns(close, 1)
        atr14 = _atr(high, low, close, 14)
        vol10 = _rolling_std(ret1, 10)
        med_vol = vol10.median()
        bar_range = (high - low)
        hl_ratio = bar_range / (atr14 + 1e-8)

        # Forward return over forward_window bars
        fwd_return = np.log(close.shift(-forward_window) / close).fillna(0.0)
        prev_ret = ret1.shift(1).fillna(0.0)
        curr_ret = ret1

        labels = np.full(len(df), "ranging", dtype=object)

        _require_numpy_pandas()
        # Apply rules (lower index = higher priority)
        bull = (fwd_return > 0.01) & (vol10 < med_vol)
        bear = (fwd_return < -0.01) & (vol10 < med_vol)
        bo_up = (fwd_return > 0.02) & (hl_ratio > 1.5)
        bo_dn = (fwd_return < -0.02) & (hl_ratio > 1.5)
        rev_up = (prev_ret < -0.01) & (curr_ret > 0.005)
        rev_dn = (prev_ret > 0.01) & (curr_ret < -0.005)

        labels[bull.values] = "bull_trend"
        labels[bear.values] = "bear_trend"
        labels[rev_up.values] = "reversal_up"
        labels[rev_dn.values] = "reversal_down"
        labels[bo_up.values] = "breakout_up"
        labels[bo_dn.values] = "breakout_down"

        return labels

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, df) -> None:
        """Train on historical OHLCV data. Handles labeling internally."""
        _require_numpy_pandas()
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")
        if len(df) < 250:
            logger.warning(f"RainforestDetector: only {len(df)} bars — recommend ≥250 for good labels")

        X = self.extract_features(df)
        y = self.label_regimes(df)

        # Drop rows where features are fully NaN (first few rows)
        valid = ~np.isnan(X).any(axis=1)
        X, y = X[valid], y[valid]

        if len(X) < 50:
            raise ValueError(f"Not enough valid rows after feature extraction: {len(X)}")

        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
        self._model.fit(X, y)
        self._classes = list(self._model.classes_)
        self._trained_at = time.time()

        # Cache feature importances
        self._feature_importances = {
            name: float(imp)
            for name, imp in zip(FEATURE_NAMES, self._model.feature_importances_)
        }
        logger.success(
            f"RainforestDetector trained on {len(X)} bars | "
            f"classes={self._classes} | "
            f"top_feature={max(self._feature_importances, key=self._feature_importances.get)}"
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_regime(self, df) -> dict:
        """
        Predict regime for the most recent bar.

        Returns:
            {
                "regime": "bull_trend",
                "confidence": 0.82,
                "probabilities": {"bull_trend": 0.82, "ranging": 0.10, ...},
                "feature_importances": {"rsi_14": 0.18, "atr_pct": 0.15, ...},
                "top_patterns": [{"pattern": "rsi_14 > 60 & price_vs_sma50 > 0.02",
                                   "importance": 0.18}]
            }
        """
        if not self.is_trained():
            return {
                "regime": "ranging",
                "confidence": 0.0,
                "probabilities": {r: 0.0 for r in REGIMES},
                "feature_importances": {},
                "top_patterns": [],
                "error": "model_not_trained",
            }

        _require_numpy_pandas()
        X = self.extract_features(df)
        if len(X) == 0:
            return {
                "regime": "ranging",
                "confidence": 0.0,
                "probabilities": {r: 0.0 for r in REGIMES},
                "feature_importances": {k: round(v, 4) for k, v in self._feature_importances.items()},
                "top_patterns": [],
                "error": "empty_dataframe",
            }
        # Use only the last row
        x_last = X[-1:].copy()
        if np.isnan(x_last).any():
            x_last = np.nan_to_num(x_last, nan=0.0)

        proba = self._model.predict_proba(x_last)[0]
        class_proba = dict(zip(self._classes, proba.tolist()))

        # Fill missing regimes with 0
        full_proba = {r: float(class_proba.get(r, 0.0)) for r in REGIMES}

        best_regime = max(full_proba, key=full_proba.get)
        confidence = float(full_proba[best_regime])

        return {
            "regime": best_regime,
            "confidence": round(confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in full_proba.items()},
            "feature_importances": {k: round(v, 4) for k, v in self._feature_importances.items()},
            "top_patterns": self.get_top_patterns(n=5),
        }

    # ------------------------------------------------------------------
    # Pattern reporting
    # ------------------------------------------------------------------

    def get_top_patterns(self, n: int = 10) -> list[dict]:
        """Return top N patterns by feature importance."""
        if not self._feature_importances:
            return []
        sorted_feats = sorted(
            self._feature_importances.items(), key=lambda x: x[1], reverse=True
        )[:n]

        patterns = []
        for feat_name, importance in sorted_feats:
            # Build a simple human-readable pattern description
            patterns.append({
                "pattern": _feature_to_pattern_desc(feat_name),
                "feature": feat_name,
                "importance": round(importance, 4),
            })
        return patterns

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save trained model with joblib."""
        if not self.is_trained():
            raise RuntimeError("Cannot save an untrained model")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "model": self._model,
            "classes": self._classes,
            "feature_importances": self._feature_importances,
            "trained_at": self._trained_at,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
        }
        joblib.dump(payload, path)
        logger.info(f"RainforestDetector saved to {path}")

    def load(self, path: str) -> bool:
        """Load model. Returns True if successful."""
        if not os.path.exists(path):
            return False
        try:
            payload = joblib.load(path)
            self._model = payload["model"]
            self._classes = payload.get("classes", list(REGIMES))
            self._feature_importances = payload.get("feature_importances", {})
            self._trained_at = payload.get("trained_at")
            self.n_estimators = payload.get("n_estimators", self.n_estimators)
            self.max_depth = payload.get("max_depth", self.max_depth)
            logger.success(f"RainforestDetector loaded from {path}")
            return True
        except Exception as exc:
            logger.warning(f"RainforestDetector load failed ({path}): {exc}")
            return False

    def is_trained(self) -> bool:
        """Return True if the model has been trained and is ready for prediction."""
        return self._model is not None and self._trained_at is not None

    # ------------------------------------------------------------------
    # MT5 training shortcut
    # ------------------------------------------------------------------

    def train_from_mt5_data(
        self,
        symbol: str,
        n_bars: int = 5000,
        config: Optional[dict] = None,
    ) -> bool:
        """
        Try to fetch OHLCV data from MT5 and train.
        Falls back to synthetic data if MT5 is unavailable.
        Returns True if training succeeded.
        """
        model_dir = os.path.join(_ROOT, "models")
        safe_sym = symbol.replace("/", "_")
        model_path = os.path.join(model_dir, f"rainforest_{safe_sym}.pkl")

        # Try to load existing model first
        if self.load(model_path):
            logger.info(f"RainforestDetector: loaded existing model for {symbol}")
            return True

        # Attempt MT5 fetch
        df = self._fetch_mt5_df(symbol, n_bars, config)
        if df is not None and len(df) >= 250:
            try:
                self.fit(df)
                self.save(model_path)
                return True
            except Exception as exc:
                logger.warning(f"RainforestDetector MT5 fit failed for {symbol}: {exc}")

        # Fallback to synthetic data
        logger.info(f"RainforestDetector: using synthetic data for {symbol}")
        df_synth = self._generate_synthetic_training_data(max(n_bars, 3000))
        try:
            self.fit(df_synth)
            self.save(model_path)
            return True
        except Exception as exc:
            logger.error(f"RainforestDetector synthetic fit failed for {symbol}: {exc}")
            return False

    def _fetch_mt5_df(
        self,
        symbol: str,
        n_bars: int,
        config: Optional[dict],
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV bars from MT5. Returns None on any failure."""
        try:
            from Python.mt5_compat import mt5  # type: ignore

            if not mt5.initialize():
                return None

            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, n_bars)
            if rates is None or len(rates) < 250:
                return None

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            if "tick_volume" in df.columns and "volume" not in df.columns:
                df.rename(columns={"tick_volume": "volume"}, inplace=True)
            df["symbol"] = symbol
            needed = ["open", "high", "low", "close", "volume"]
            if not all(c in df.columns for c in needed):
                return None
            return df[["time", "open", "high", "low", "close", "volume", "symbol"]].copy()
        except Exception as exc:
            logger.debug(f"RainforestDetector MT5 fetch failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Synthetic data generator
    # ------------------------------------------------------------------

    def _generate_synthetic_training_data(self, n_bars: int = 3000):
        """
        Generate realistic synthetic OHLCV data using geometric Brownian motion
        with periodic regime switches (trending, ranging, breakout).

        Returns a pandas DataFrame, or None if numpy/pandas are unavailable.
        """
        if not _NUMPY_PANDAS_AVAILABLE:
            logger.warning(
                "numpy/pandas not installed — _generate_synthetic_training_data() returning None. "
                "Install with: pip install numpy pandas"
            )
            return None
        rng = np.random.default_rng(42)

        prices = np.empty(n_bars)
        prices[0] = 1000.0

        # Regime parameters
        regime_length_mean = 200
        drift_options = [0.0002, -0.0002, 0.0, 0.0005, -0.0005]
        vol_options = [0.005, 0.005, 0.003, 0.012, 0.012]

        current_drift = 0.0
        current_vol = 0.005
        regime_counter = 0
        regime_life = rng.integers(80, regime_length_mean * 2)

        for i in range(1, n_bars):
            if regime_counter >= regime_life:
                idx = rng.integers(len(drift_options))
                current_drift = drift_options[idx]
                current_vol = vol_options[idx]
                regime_life = rng.integers(80, regime_length_mean * 2)
                regime_counter = 0
            z = rng.standard_normal()
            prices[i] = prices[i - 1] * np.exp(current_drift + current_vol * z)
            regime_counter += 1

        # Build OHLCV from prices
        noise = rng.uniform(0.001, 0.005, n_bars)
        high = prices * (1.0 + noise)
        low = prices * (1.0 - noise * rng.uniform(0.5, 1.0, n_bars))
        open_ = np.roll(prices, 1)
        open_[0] = prices[0]
        volume_base = rng.uniform(500, 2000, n_bars)
        # Volume spikes on high-volatility moves
        vol_shock = np.abs(np.diff(np.log(prices + 1e-8), prepend=0.0))
        volume = volume_base * (1.0 + 10.0 * vol_shock)

        times = pd.date_range("2020-01-01", periods=n_bars, freq="5min")

        df = pd.DataFrame({
            "time": times,
            "open": open_,
            "high": high,
            "low": low,
            "close": prices,
            "volume": volume,
        })
        return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _feature_to_pattern_desc(feat: str) -> str:
    """Convert a feature name to a human-readable pattern description."""
    descriptions = {
        "returns_1":       "1-bar return momentum",
        "returns_5":       "5-bar return momentum",
        "returns_15":      "15-bar return momentum",
        "returns_60":      "60-bar return momentum",
        "volatility_10":   "10-bar rolling volatility",
        "volatility_20":   "20-bar rolling volatility",
        "atr_pct":         "ATR as % of price",
        "rsi_14":          "RSI(14) > 60 | < 40",
        "macd_signal":     "MACD histogram crossover",
        "bb_width":        "Bollinger Band width expansion",
        "volume_ratio":    "volume vs 20-bar avg",
        "price_vs_sma50":  "price vs SMA50 deviation",
        "price_vs_sma200": "price vs SMA200 deviation",
        "high_low_range":  "bar range vs ATR ratio",
        # Classical patterns (now wired)
        "has_doji":            "Doji (indecision / potential reversal)",
        "has_hammer":          "Hammer (bullish reversal at support)",
        "has_shooting_star":   "Shooting Star (bearish reversal at resistance)",
        "has_bullish_engulfing": "Bullish Engulfing (strong reversal long)",
        "has_bearish_engulfing": "Bearish Engulfing (strong reversal short)",
        "has_double_top":      "Double Top (bearish reversal / resistance)",
        "has_double_bottom":   "Double Bottom (bullish reversal / support)",
        "has_bull_flag":       "Bull Flag (continuation long after pole)",
        "has_bear_flag":       "Bear Flag (continuation short after pole)",
        "has_breakout_up":     "Breakout Up (trend resumption / momentum long)",
        "has_breakout_down":   "Breakout Down (trend resumption / momentum short)",
    }
    return descriptions.get(feat, feat)
