"""
PatternDetector — Classical candlestick + chart pattern detection with timing context.

Detects patterns the user requested:
- Candlesticks: Doji, Hammer, Shooting Star, Engulfing (bullish/bearish)
- Chart patterns: Double Top, Double Bottom, Bear Flag, Bull Flag, Breakout (up/down)

Combines detections with session/news timing context so the full ensemble
(Decision PPO + Dreamer + Rainforest) can make informed rich TradeDecisions,
especially around TimeExitSpec for news and market opens.

This gives Dreamer better-conditioned imagination rollouts and Rainforest richer regimes.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
import os
import pandas as pd

try:
    from loguru import logger
except ImportError:
    import logging as _logging
    logger = _logging.getLogger("pattern_detector")


@dataclass
class DetectedPattern:
    name: str
    direction: str          # "bullish", "bearish", "neutral"
    strength: float         # 0.0 - 1.0
    bar_index: int
    metadata: dict = field(default_factory=dict)


@dataclass
class PatternState:
    """Rich pattern + timing context for the ensemble."""
    active_patterns: List[DetectedPattern]
    dominant_pattern: Optional[DetectedPattern]
    timing_context: dict          # from timing features (open window, news proximity, etc.)
    regime_hint: str = "ranging"  # can be enriched by Rainforest


class PatternDetector:
    """
    Lightweight classical pattern detector.
    Designed to be fast enough for real-time + rich enough to condition Dreamer imagination
    and influence Decision PPO rich TradeDecisions (especially TimeExitSpec).
    """

    def __init__(self, atr_period: int = 14):
        self.atr_period = atr_period
        self._pattern_feature_names = PATTERN_FEATURE_NAMES[:]  # for consumers


    def detect(self, df: pd.DataFrame, timing_context: Optional[dict] = None) -> PatternState:
        """
        Main entry point.
        df must have at least: open, high, low, close, (optionally volume, time)
        timing_context can come from the enriched feature pipeline (major_open_window, news_proximity, etc.)
        """
        _require_ohlcv(df)
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        open_ = df["open"].values.astype(float)

        n = len(close)
        if n < 5:
            return PatternState(active_patterns=[], dominant_pattern=None,
                                timing_context=timing_context or {}, regime_hint="ranging")

        # Robust ATR for strength normalization (key expansion for reliable edge)
        atr = _compute_atr(high, low, close, self.atr_period)

        patterns: List[DetectedPattern] = []

        # --- Candlestick patterns (last 1-3 bars) ---
        patterns.extend(self._detect_doji(close, high, low, open_, atr))
        patterns.extend(self._detect_hammer_shooting_star(close, high, low, open_, atr))
        patterns.extend(self._detect_engulfing(close, high, low, open_, atr))

        # --- Robust chart patterns (lookback window, extrema + vol normalized) ---
        patterns.extend(self._detect_double_top_bottom(close, high, low, atr))
        patterns.extend(self._detect_flags(close, high, low, atr, open_))
        patterns.extend(self._detect_breakouts(close, high, low, atr))

        # Pick strongest
        dominant = max(patterns, key=lambda p: p.strength) if patterns else None

        timing = timing_context or self._default_timing_context(df)

        # Optional: boost pattern strength near favorable timing (opens / low news)
        if timing.get("major_open_window", 0.0) > 0.5 or timing.get("news_proximity", 0.0) < 0.2:
            for p in patterns:
                if p.direction in ("bullish", "bearish") and "engulfing" in p.name or "flag" in p.name or "breakout" in p.name:
                    p.strength = float(np.clip(p.strength * 1.15, 0.0, 1.0))
                    p.metadata["timing_boost"] = True

        # === ARTIFACT-DRIVEN TOP_BOOST_PATTERNS SUPPORT (from next_training_overrides.json) ===
        # Boosts specific patterns (double_bottom, breakout_up, etc.) by 1.29x+ factor when listed.
        # This directly implements the MetaOptimizer suggestion: emphasize high-edge patterns from validation evidence.
        try:
            boost_env = os.environ.get("AGI_TOP_BOOST_PATTERNS", "")
            if boost_env:
                boost_set = set([p.strip() for p in boost_env.split(",") if p.strip()])
                boost_factor = 1.29  # default multiplier aligned with feature_importance "patterns"=1.29
                # allow per-run tuning via companion env if needed
                try:
                    custom_f = float(os.environ.get("AGI_TOP_BOOST_FACTOR", "1.29"))
                    if custom_f > 0.5:
                        boost_factor = custom_f
                except Exception:
                    pass
                for p in patterns:
                    name_match = p.name
                    # normalize common aliases (double_bottom vs has_double_bottom)
                    if name_match in boost_set or f"has_{name_match}" in boost_set or name_match.replace("_", "") in [b.replace("_","") for b in boost_set]:
                        p.strength = float(np.clip(p.strength * boost_factor, 0.0, 1.0))
                        p.metadata["artifact_boost"] = True
                        p.metadata["boost_factor"] = boost_factor
        except Exception:
            pass

        return PatternState(
            active_patterns=patterns,
            dominant_pattern=dominant,
            timing_context=timing,
            regime_hint=self._infer_regime_hint(patterns, timing)
        )

    # ------------------------------------------------------------------
    # Individual pattern detectors (simple but effective versions)
    # ------------------------------------------------------------------

    def _detect_doji(self, close, high, low, open_, atr=None) -> List[DetectedPattern]:
        patterns = []
        body = np.abs(close - open_)
        range_ = high - low + 1e-8
        doji_ratio = body / range_

        # Last 1-3 bars, ATR-normalized strength for robustness
        for i in range(min(3, len(close))):
            idx = len(close) - 1 - i
            if doji_ratio[idx] < 0.12:
                strength = 1.0 - (doji_ratio[idx] / 0.12)
                atr_norm = float(atr[idx] / (close[idx] + 1e-8)) if atr is not None else 1.0
                strength = min(1.0, strength * (1.0 + 0.2 * atr_norm))  # slight boost in volatile
                patterns.append(DetectedPattern(
                    name="doji",
                    direction="neutral",
                    strength=float(np.clip(strength, 0.35, 1.0)),
                    bar_index=idx,
                    metadata={"body_range_ratio": float(doji_ratio[idx]), "atr_norm": atr_norm}
                ))
        return patterns

    def _detect_hammer_shooting_star(self, close, high, low, open_, atr=None) -> List[DetectedPattern]:
        patterns = []
        body = np.abs(close - open_)
        lower_wick = np.minimum(close, open_) - low
        upper_wick = high - np.maximum(close, open_)
        total = high - low + 1e-8

        for i in range(min(3, len(close))):
            idx = len(close) - 1 - i
            if body[idx] / total[idx] < 0.30:
                atr_factor = 1.0
                if atr is not None and close[idx] > 0:
                    atr_factor = 1.0 + min(0.4, float(atr[idx] / (close[idx] + 1e-8)))  # robust vol scaling
                # Hammer (bullish reversal at support) - require decent range
                if lower_wick[idx] > 2.0 * body[idx] and upper_wick[idx] < 0.7 * body[idx] and total[idx] > 0.5 * (atr[idx] if atr is not None else total[idx]):
                    strength = min(1.0, lower_wick[idx] / (2.3 * body[idx])) * atr_factor
                    patterns.append(DetectedPattern("hammer", "bullish", float(np.clip(strength, 0.4, 1.0)), idx,
                                                    metadata={"wick_ratio": float(lower_wick[idx]/max(body[idx],1e-8))}))
                # Shooting Star / Inverted Hammer (bearish reversal at resistance)
                if upper_wick[idx] > 2.0 * body[idx] and lower_wick[idx] < 0.7 * body[idx] and total[idx] > 0.5 * (atr[idx] if atr is not None else total[idx]):
                    strength = min(1.0, upper_wick[idx] / (2.3 * body[idx])) * atr_factor
                    patterns.append(DetectedPattern("shooting_star", "bearish", float(np.clip(strength, 0.4, 1.0)), idx,
                                                    metadata={"wick_ratio": float(upper_wick[idx]/max(body[idx],1e-8))}))
        return patterns

    def _detect_engulfing(self, close, high, low, open_, atr=None) -> List[DetectedPattern]:
        if len(close) < 2:
            return []
        patterns = []

        for i in range(min(4, len(close) - 1)):
            idx = len(close) - 1 - i
            prev_body = abs(close[idx-1] - open_[idx-1])
            curr_body = abs(close[idx] - open_[idx])
            prev_bull = close[idx-1] > open_[idx-1]
            rng = max(high[idx] - low[idx], 1e-8)
            vol_factor = 1.0
            if atr is not None:
                vol_factor = 1.0 + min(0.35, float(atr[idx] / (close[idx] + 1e-8)))

            # Bullish engulfing - strict body dominance + close beyond
            if (not prev_bull and close[idx] > open_[idx] and
                open_[idx] < close[idx-1] and close[idx] > open_[idx-1] and
                curr_body > prev_body * 1.03 and curr_body > 0.6 * rng):
                strength = min(1.0, curr_body / (prev_body + 1e-8)) * vol_factor
                patterns.append(DetectedPattern("bullish_engulfing", "bullish", float(np.clip(strength, 0.5, 1.0)), idx,
                                                metadata={"body_ratio": float(curr_body / max(prev_body, 1e-8))}))
            # Bearish engulfing
            if (prev_bull and close[idx] < open_[idx] and
                open_[idx] > close[idx-1] and close[idx] < open_[idx-1] and
                curr_body > prev_body * 1.03 and curr_body > 0.6 * rng):
                strength = min(1.0, curr_body / (prev_body + 1e-8)) * vol_factor
                patterns.append(DetectedPattern("bearish_engulfing", "bearish", float(np.clip(strength, 0.5, 1.0)), idx,
                                                metadata={"body_ratio": float(curr_body / max(prev_body, 1e-8))}))
        return patterns

    def _detect_double_top_bottom(self, close, high, low, atr=None) -> List[DetectedPattern]:
        patterns = []
        if len(close) < 20:
            return patterns

        lookback = min(40, len(close) - 3)
        maxima, minima = _find_local_extrema(high[-lookback:], order=2)
        maxima = [m + (len(high) - lookback) for m in maxima]
        minima = [m + (len(low) - lookback) for m in minima]

        # Double top: two peaks of similar height, separated, recent
        for k in range(len(maxima) - 1):
            i, j = maxima[k], maxima[k + 1]
            if j - i < 4 or j - i > 18:
                continue
            h1, h2 = high[i], high[j]
            if abs(h1 - h2) / max(h1, 1e-8) < 0.032:
                strength = 0.78 - (abs(h1 - h2) / max(h1, 1e-8)) * 8
                if atr is not None:
                    strength *= (1.0 + min(0.2, float(atr[j] / (close[j] + 1e-8))))
                patterns.append(DetectedPattern("double_top", "bearish", float(np.clip(strength, 0.55, 0.92)), j,
                                                metadata={"peak_sep": int(j - i)}))
                break

        # Double bottom
        for k in range(len(minima) - 1):
            i, j = minima[k], minima[k + 1]
            if j - i < 4 or j - i > 18:
                continue
            l1, l2 = low[i], low[j]
            if abs(l1 - l2) / max(abs(l1), 1e-8) < 0.032:
                strength = 0.78 - (abs(l1 - l2) / max(abs(l1), 1e-8)) * 8
                if atr is not None:
                    strength *= (1.0 + min(0.2, float(atr[j] / (close[j] + 1e-8))))
                patterns.append(DetectedPattern("double_bottom", "bullish", float(np.clip(strength, 0.55, 0.92)), j,
                                                metadata={"trough_sep": int(j - i)}))
                break

        return patterns

    def _detect_flags(self, close, high, low, atr=None, open_=None) -> List[DetectedPattern]:
        patterns = []
        if len(close) < 15:
            return patterns

        n = len(close)
        # Look for strong impulse pole (recent 4-8 bars) + tight consolidation flag (ATR normalized)
        for pole_end in range(n - 13, n - 4):
            pole_start = max(0, pole_end - 6)
            pole_move = (close[pole_end] - close[pole_start]) / max(abs(close[pole_start]), 1e-8)
            flag_high = high[pole_end + 1 : pole_end + 7].max() if pole_end + 7 <= n else high[-1]
            flag_low = low[pole_end + 1 : pole_end + 7].min() if pole_end + 7 <= n else low[-1]
            flag_range = (flag_high - flag_low) / max(close[pole_end + 3 : pole_end + 7].mean() if pole_end + 7 <= n else close[-1], 1e-8)
            atr_ref = atr[pole_end] if atr is not None else (high[pole_end] - low[pole_end])

            # Pole + flag contraction (flag range << prior ATR)
            if abs(pole_move) > 0.038 and flag_range < 0.55 * (atr_ref / max(close[pole_end], 1e-8)):
                direction = "bull" if pole_move > 0 else "bear"
                strength = min(1.0, abs(pole_move) / 0.085 + (0.25 if flag_range < 0.018 else 0))
                bar_idx = min(n-1, pole_end + 6)
                patterns.append(DetectedPattern(f"{direction}_flag", "bullish" if pole_move > 0 else "bearish",
                                                float(np.clip(strength, 0.5, 0.95)), bar_idx,
                                                metadata={"pole_move": float(pole_move), "flag_range": float(flag_range)}))
        return patterns

    def _detect_breakouts(self, close, high, low, atr=None) -> List[DetectedPattern]:
        patterns = []
        if len(close) < 12:
            return patterns

        n = len(close)
        # Focus on most recent possible breakout (robust last-window check)
        for i in range(max(0, n-25), n-1):
            win = min(12, n - i - 1)
            if win < 3: continue
            recent_high = high[i : i + win].max()
            recent_low = low[i : i + win].min()
            atr_ref = (atr[i + win] if atr is not None else np.mean(high[i:i+win] - low[i:i+win]))
            atr_ref = max(atr_ref, 1e-8)
            curr_close = close[i + win]

            vol_mult = 1.15
            if atr is not None and close[i+win] > 0:
                vol_mult = 1.0 + min(0.4, float(atr[i+win] / close[i+win]))

            if curr_close > recent_high + 0.28 * atr_ref:
                strength = min(0.92, 0.72 * vol_mult)
                patterns.append(DetectedPattern("breakout_up", "bullish", float(strength), i + win,
                                                metadata={"dist_atr": float((curr_close - recent_high) / atr_ref)}))
            elif curr_close < recent_low - 0.28 * atr_ref:
                strength = min(0.92, 0.72 * vol_mult)
                patterns.append(DetectedPattern("breakout_down", "bearish", float(strength), i + win,
                                                metadata={"dist_atr": float((recent_low - curr_close) / atr_ref)}))
        return patterns

    def _infer_regime_hint(self, patterns: List[DetectedPattern], timing: dict) -> str:
        bullish = sum(1 for p in patterns if p.direction == "bullish")
        bearish = sum(1 for p in patterns if p.direction == "bearish")
        has_breakout = any("breakout" in p.name for p in patterns)
        has_flag = any("flag" in p.name for p in patterns)
        has_reversal_candle = any(p.name in ("hammer", "shooting_star", "bullish_engulfing", "bearish_engulfing") for p in patterns)

        if bullish > bearish + 1:
            if has_breakout:
                return "bullish_breakout"
            if has_flag:
                return "bull_trend"
            return "bull_trend" if not has_reversal_candle else "reversal_up"
        if bearish > bullish + 1:
            if has_breakout:
                return "bearish_breakout"
            if has_flag:
                return "bear_trend"
            return "bear_trend" if not has_reversal_candle else "reversal_down"
        # timing aware ranging bias
        if timing.get("major_open_window", 0) > 0.6:
            return "ranging"  # caution at open unless strong signal
        return "ranging"

    def _default_timing_context(self, df: pd.DataFrame) -> dict:
        return {
            "major_open_window": 0.0,
            "news_proximity": 0.0,
            "has_high_impact_news_soon": 0.0
        }

    # ------------------------------------------------------------------
    # Integration APIs: pattern feature vectors for Rainforest + Dreamer obs
    # ------------------------------------------------------------------

    def extract_pattern_features(self, df: pd.DataFrame, timing_context: Optional[dict] = None) -> dict[str, float]:
        """Return dict of the canonical has_* pattern indicators (strength in [0,1]).
        Used by RainforestDetector.extract_features and feature_pipeline to enrich regimes/obs.
        """
        _require_ohlcv(df)
        state = self.detect(df, timing_context=timing_context)
        feats: dict[str, float] = {name: 0.0 for name in self._pattern_feature_names}

        name_map = {
            "doji": "has_doji",
            "hammer": "has_hammer",
            "shooting_star": "has_shooting_star",
            "bullish_engulfing": "has_bullish_engulfing",
            "bearish_engulfing": "has_bearish_engulfing",
            "double_top": "has_double_top",
            "double_bottom": "has_double_bottom",
            "bull_flag": "has_bull_flag",
            "bear_flag": "has_bear_flag",
            "breakout_up": "has_breakout_up",
            "breakout_down": "has_breakout_down",
        }

        for p in state.active_patterns:
            key = name_map.get(p.name, f"has_{p.name}")
            if key in feats:
                # presence weighted by strength (allows soft signals for world model)
                feats[key] = max(feats[key], float(np.clip(p.strength, 0.0, 1.0)))

        # Add a few summary stats for richer state (Dreamer can learn from these too)
        feats["_dominant_strength"] = float(state.dominant_pattern.strength) if state.dominant_pattern else 0.0
        feats["_num_active_patterns"] = float(len(state.active_patterns))
        feats["_regime_hint_bull"] = 1.0 if "bull" in state.regime_hint else 0.0
        feats["_regime_hint_bear"] = 1.0 if "bear" in state.regime_hint else 0.0
        return feats

    def get_pattern_feature_vector(self, df: pd.DataFrame, timing_context: Optional[dict] = None) -> np.ndarray:
        """Return 1D float32 vector aligned to PATTERN_FEATURE_NAMES (for stacking into obs/features)."""
        feats = self.extract_pattern_features(df, timing_context)
        vec = np.array([feats.get(name, 0.0) for name in PATTERN_FEATURE_NAMES], dtype=np.float32)
        return vec

    def enrich_dataframe_with_patterns(self, df: pd.DataFrame, timing_context: Optional[dict] = None) -> pd.DataFrame:
        """Return df with pattern columns appended (non-destructive). Useful for training sets."""
        out = df.copy().reset_index(drop=True)
        vec = self.get_pattern_feature_vector(out, timing_context)
        for i, name in enumerate(PATTERN_FEATURE_NAMES):
            out[name] = vec[i]  # broadcast scalar (latest bar patterns) to all rows; caller can ffill/rolling if needed
        # Also attach latest state summary
        state = self.detect(out, timing_context)
        out["_pattern_regime"] = state.regime_hint
        out["_dominant_pattern"] = state.dominant_pattern.name if state.dominant_pattern else ""
        return out


def _require_ohlcv(df: pd.DataFrame):
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain column '{col}'")


# ── Public constants for integration with Rainforest + feature pipelines ──
PATTERN_FEATURE_NAMES: list[str] = [
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


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Vectorized ATR for normalized pattern strength (robustness)."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3 = np.abs(low - np.roll(close, 1))
    tr3[0] = tr1[0]
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return np.maximum(atr, 1e-8)


def _find_local_extrema(arr: np.ndarray, order: int = 2) -> tuple[list[int], list[int]]:
    """Simple local maxima / minima indices (robust, no scipy dep)."""
    n = len(arr)
    maxima, minima = [], []
    for i in range(order, n - order):
        if all(arr[i] > arr[i - k] for k in range(1, order + 1)) and all(arr[i] > arr[i + k] for k in range(1, order + 1)):
            maxima.append(i)
        if all(arr[i] < arr[i - k] for k in range(1, order + 1)) and all(arr[i] < arr[i + k] for k in range(1, order + 1)):
            minima.append(i)
    return maxima, minima
