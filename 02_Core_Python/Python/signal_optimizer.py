"""
Signal Optimizer — Advanced signal filtering for improved edge detection.

This module provides sophisticated entry filtering to improve risk-adjusted returns:
1. Multi-timeframe confluence analysis
2. Market structure detection (support/resistance avoidance)
3. Trend strength confirmation
4. Volatility regime filtering
5. Consecutive loss protection
"""
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class SignalQuality:
    """Signal quality assessment."""
    score: float  # 0.0 to 1.0
    passed: bool
    filters: Dict[str, bool]
    notes: List[str]


class SignalOptimizer:
    """
    Advanced signal filtering for improved entry quality.

    Philosophy: Better to miss a good trade than take a bad one.
    Focus on high-conviction setups with multiple confluences.
    """

    def __init__(self):
        # Consecutive loss tracking per symbol
        self._loss_streaks: Dict[str, deque] = {}
        self._max_streak_window = 10

        # Market structure cache
        self._structure_cache: Dict[str, dict] = {}

        # Minimum signal quality threshold
        self._min_quality_score = float(os.environ.get("AGI_MIN_QUALITY_SCORE", "0.65"))

        logger.success(f"SignalOptimizer initialized (min_quality={self._min_quality_score})")

    def evaluate_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        action: str,
        ppo_score: float,
        regime: str,
        confidence: float,
    ) -> SignalQuality:
        """
        Evaluate signal quality with multiple filters.

        Returns SignalQuality with score and filter results.
        Score >= 0.65 recommended for entry.
        """
        filters = {}
        notes = []
        score = 0.0

        # Filter 1: Consecutive Loss Protection (weight: 0.20)
        loss_score = self._check_loss_streak(symbol)
        filters["loss_streak"] = loss_score >= 0.5
        score += loss_score * 0.20
        if loss_score < 0.5:
            notes.append(f"Recent loss streak ({loss_score:.2f})")

        # Filter 2: Trend Strength (weight: 0.25)
        trend_score = self._assess_trend_strength(df)
        filters["trend_strength"] = trend_score >= 0.6
        score += trend_score * 0.25
        if trend_score < 0.6:
            notes.append(f"Weak trend ({trend_score:.2f})")

        # Filter 3: Market Structure (weight: 0.25)
        structure_score = self._check_market_structure(symbol, df, action)
        filters["market_structure"] = structure_score >= 0.5
        score += structure_score * 0.25
        if structure_score < 0.5:
            notes.append("Near S/R zone - avoid")

        # Filter 4: Volatility Quality (weight: 0.15)
        vol_score = self._assess_volatility_quality(df, regime)
        filters["volatility_quality"] = vol_score >= 0.5
        score += vol_score * 0.15
        if vol_score < 0.5:
            notes.append(f"Poor volatility conditions ({vol_score:.2f})")

        # Filter 5: Signal Momentum (weight: 0.15)
        momentum_score = self._assess_signal_momentum(df, action)
        filters["signal_momentum"] = momentum_score >= 0.5
        score += momentum_score * 0.15

        # Final quality assessment
        passed = score >= self._min_quality_score and all([
            filters["trend_strength"],
            filters["market_structure"],
        ])

        return SignalQuality(
            score=round(score, 3),
            passed=passed,
            filters=filters,
            notes=notes,
        )

    def _check_loss_streak(self, symbol: str) -> float:
        """
        Check recent loss streak and return score (0.0-1.0).

        Score reduces as losses accumulate. After 3 consecutive losses,
        score drops to 0.3 (70% reduction in position size).
        """
        if symbol not in self._loss_streaks:
            self._loss_streaks[symbol] = deque(maxlen=self._max_streak_window)

        streak = self._loss_streaks[symbol]
        if len(streak) < 3:
            return 1.0  # No history = no penalty

        recent = list(streak)[-3:]
        loss_count = sum(1 for r in recent if r < 0)

        if loss_count == 0:
            return 1.0
        elif loss_count == 1:
            return 0.9
        elif loss_count == 2:
            return 0.6
        else:
            return 0.3  # 3 consecutive losses

    def _assess_trend_strength(self, df: pd.DataFrame) -> float:
        """
        Assess trend strength using ADX-like calculation.

        Returns score 0.0-1.0 where:
        - >0.7 = strong trend
        - 0.5-0.7 = moderate trend
        - <0.5 = weak trend/chop
        """
        if len(df) < 20:
            return 0.5

        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values

        # Calculate directional movement
        plus_dm = []
        minus_dm = []
        tr_list = []

        for i in range(1, min(20, len(df))):
            plus_dm.append(max(highs[i] - highs[i-1], 0))
            minus_dm.append(max(lows[i-1] - lows[i], 0))
            tr_list.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            ))

        if not tr_list or sum(tr_list) == 0:
            return 0.5

        avg_tr = np.mean(tr_list)
        avg_plus_dm = np.mean(plus_dm) if plus_dm else 0
        avg_minus_dm = np.mean(minus_dm) if minus_dm else 0

        # Normalize
        plus_di = avg_plus_dm / avg_tr if avg_tr > 0 else 0
        minus_di = avg_minus_dm / avg_tr if avg_tr > 0 else 0

        # ADX-like calculation
        dx = abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100

        # Convert to 0-1 score
        if dx > 40:
            return 1.0
        elif dx > 25:
            return 0.7 + (dx - 25) / 50
        elif dx > 15:
            return 0.4 + (dx - 15) / 25
        else:
            return 0.2 + dx / 75

    def _check_market_structure(self, symbol: str, df: pd.DataFrame, action: str) -> float:
        """
        Check if price is near support/resistance zones.

        Returns lower score if near S/R to avoid false breakouts.
        """
        if len(df) < 50:
            return 0.7  # Not enough data

        current_price = df["close"].iloc[-1]

        # Calculate recent highs/lows as S/R levels
        recent_highs = df["high"].rolling(20).max().iloc[-20:].values
        recent_lows = df["low"].rolling(20).min().iloc[-20:].values

        # Find key levels (clustered highs/lows)
        resistance = np.percentile(recent_highs, 90)
        support = np.percentile(recent_lows, 10)

        # Calculate distance to nearest level
        price_range = resistance - support
        if price_range == 0:
            return 0.7

        if action == "BUY":
            # For buys, avoid prices near resistance
            dist_to_resistance = (resistance - current_price) / price_range
            if dist_to_resistance < 0.05:  # Within 5% of resistance
                return 0.3  # Avoid
            elif dist_to_resistance < 0.10:
                return 0.6  # Caution
            else:
                return 0.9  # Good distance
        else:  # SELL
            # For sells, avoid prices near support
            dist_to_support = (current_price - support) / price_range
            if dist_to_support < 0.05:  # Within 5% of support
                return 0.3  # Avoid
            elif dist_to_support < 0.10:
                return 0.6  # Caution
            else:
                return 0.9  # Good distance

    def _assess_volatility_quality(self, df: pd.DataFrame, regime: str) -> float:
        """
        Assess if current volatility conditions are favorable.

        Avoid:
        - Extreme volatility (unstable)
        - Very low volatility (chop)
        """
        if len(df) < 20:
            return 0.6

        # Calculate ATR-based volatility
        atr = self._calculate_atr(df, 14)
        current_price = df["close"].iloc[-1]

        if current_price == 0 or atr == 0:
            return 0.5

        vol_pct = atr / current_price

        # Optimal range: 0.1% to 0.5% daily volatility
        if regime == "LOW_VOLATILITY":
            if vol_pct < 0.0005:  # Very low vol
                return 0.4
            elif vol_pct < 0.001:
                return 0.7
            else:
                return 0.9
        elif regime == "HIGH_VOLATILITY":
            if vol_pct > 0.005:  # Extreme vol
                return 0.4
            elif vol_pct > 0.003:
                return 0.6
            else:
                return 0.8
        else:  # MED
            if 0.001 < vol_pct < 0.003:
                return 1.0
            elif vol_pct < 0.0005 or vol_pct > 0.005:
                return 0.5
            else:
                return 0.8

    def _assess_signal_momentum(self, df: pd.DataFrame, action: str) -> float:
        """
        Check if price momentum confirms the signal direction.

        Score based on:
        - Short-term momentum alignment
        - Volume confirmation
        """
        if len(df) < 10:
            return 0.6

        closes = df["close"].values
        volumes = df.get("volume", pd.Series(np.ones(len(df)))).values

        # Calculate momentum
        momentum_5 = (closes[-1] - closes[-5]) / closes[-5] if closes[-5] != 0 else 0
        momentum_10 = (closes[-1] - closes[-10]) / closes[-10] if closes[-10] != 0 else 0

        # Volume trend
        vol_avg = np.mean(volumes[-10:-5])
        vol_recent = np.mean(volumes[-5:])
        volume_confirming = vol_recent > vol_avg * 1.1  # 10% above average

        # Score based on alignment
        if action == "BUY":
            if momentum_5 > 0 and momentum_10 > 0:
                score = 0.9 if momentum_5 > momentum_10 else 0.7
            elif momentum_5 > 0:
                score = 0.6
            else:
                score = 0.3
        else:  # SELL
            if momentum_5 < 0 and momentum_10 < 0:
                score = 0.9 if momentum_5 < momentum_10 else 0.7
            elif momentum_5 < 0:
                score = 0.6
            else:
                score = 0.3

        # Volume boost
        if volume_confirming:
            score = min(1.0, score + 0.1)

        return score

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(df) < period + 1:
            return df["high"].iloc[-1] - df["low"].iloc[-1]

        highs = df["high"].values[-period-1:]
        lows = df["low"].values[-period-1:]
        closes = df["close"].values[-period-1:]

        tr_list = []
        for i in range(1, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_list.append(tr)

        return np.mean(tr_list) if tr_list else 0

    def record_trade_result(self, symbol: str, pnl: float):
        """Record trade result for loss streak tracking."""
        if symbol not in self._loss_streaks:
            self._loss_streaks[symbol] = deque(maxlen=self._max_streak_window)
        self._loss_streaks[symbol].append(pnl)


class KellyPositionSizer:
    """
    Kelly Criterion Position Sizing with fractional application.

    The Kelly Criterion calculates the optimal fraction of capital to bet
    given the probability of winning and the win/loss ratio.

    Formula: f* = (p*b - q) / b
    where:
        p = probability of win
        q = probability of loss (1-p)
        b = average win / average loss (win/loss ratio)

    We use HALF-KELLY for safety: f = 0.5 * f*
    """

    def __init__(self, fraction: float = 0.5):
        """
        Args:
            fraction: Kelly fraction to use (0.5 = Half-Kelly, safer)
        """
        self.fraction = fraction
        self._trade_history: Dict[str, deque] = {}
        self._min_trades_for_kelly = 20

        logger.success(f"KellyPositionSizer initialized (fraction={fraction})")

    def calculate_size(
        self,
        symbol: str,
        base_risk_pct: float,
        equity: float,
    ) -> float:
        """
        Calculate position size using Kelly Criterion.

        Args:
            symbol: Trading symbol
            base_risk_pct: Base risk per trade (e.g., 0.01 for 1%)
            equity: Current equity

        Returns:
            Kelly-adjusted risk percentage
        """
        # Get symbol-specific trade history
        history = self._trade_history.get(symbol, deque())

        if len(history) < self._min_trades_for_kelly:
            # Not enough history - use base risk with conservative multiplier
            return base_risk_pct * 0.7

        # Calculate Kelly parameters
        pnls = list(history)
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        if not wins or not losses:
            return base_risk_pct * 0.5  # No edge detected

        p = len(wins) / len(pnls)  # Win probability
        q = 1 - p

        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))

        if avg_loss == 0:
            return base_risk_pct  # Avoid division by zero

        b = avg_win / avg_loss  # Win/loss ratio

        # Kelly formula
        kelly_f = (p * b - q) / b if b > 0 else 0

        # Apply fractional Kelly for safety
        adjusted_f = kelly_f * self.fraction

        # Clamp to reasonable limits (0.5% to 3% of equity)
        min_risk = 0.005
        max_risk = 0.03

        final_risk = base_risk_pct * max(min_risk, min(adjusted_f, max_risk))

        logger.debug(
            f"Kelly sizing for {symbol}: p={p:.2f}, b={b:.2f}, "
            f"kelly={kelly_f:.3f}, adjusted={adjusted_f:.3f}, risk={final_risk:.4f}"
        )

        return final_risk

    def record_trade(self, symbol: str, pnl: float):
        """Record trade PnL for Kelly calculations."""
        if symbol not in self._trade_history:
            self._trade_history[symbol] = deque(maxlen=100)
        self._trade_history[symbol].append(pnl)


class SpreadOptimizer:
    """
    Optimize entry timing based on spread conditions.

    Reduces trading costs by:
    1. Avoiding entries during wide spread periods
    2. Timing entries during liquid sessions
    3. Spread-aware limit order placement
    """

    def __init__(self):
        self._spread_history: Dict[str, deque] = {}
        self._optimal_spread_threshold = 1.5  # 1.5x average spread

        logger.success("SpreadOptimizer initialized")

    def is_spread_favorable(self, symbol: str, current_spread: float) -> Tuple[bool, float]:
        """
        Check if current spread is favorable for entry.

        Returns:
            (is_favorable, quality_score)
        """
        if symbol not in self._spread_history:
            self._spread_history[symbol] = deque(maxlen=50)

        history = self._spread_history[symbol]

        if len(history) < 10:
            # Not enough history - accept if spread < 3 pips (0.0003)
            is_good = current_spread < 0.0003
            return is_good, 0.7 if is_good else 0.3

        avg_spread = np.mean(history)
        if avg_spread == 0:
            avg_spread = 0.0001

        spread_ratio = current_spread / avg_spread

        # Update history
        history.append(current_spread)

        if spread_ratio < 1.0:
            return True, 1.0  # Better than average
        elif spread_ratio < self._optimal_spread_threshold:
            return True, 0.8  # Slightly wider but acceptable
        elif spread_ratio < 2.0:
            return False, 0.5  # Too wide
        else:
            return False, 0.2  # Very wide spread

    def get_session_quality(self, hour_utc: int) -> float:
        """
        Get trading session quality based on hour (UTC).

        Returns:
            Quality score 0.0-1.0
        """
        # London session (8-16 UTC) - highest quality
        if 8 <= hour_utc < 16:
            return 1.0
        # NY session (13-21 UTC) - high quality
        elif 13 <= hour_utc < 21:
            return 0.9
        # London/NY overlap (13-16 UTC) - best
        elif 13 <= hour_utc < 16:
            return 1.0
        # Asian session (0-8 UTC) - moderate
        elif 0 <= hour_utc < 8:
            return 0.7
        # Weekend/illiquid - poor
        else:
            return 0.5


# Global optimizer instance
_signal_optimizer = None
_kelly_sizer = None
_spread_optimizer = None


def get_signal_optimizer() -> SignalOptimizer:
    """Get or create global signal optimizer."""
    global _signal_optimizer
    if _signal_optimizer is None:
        _signal_optimizer = SignalOptimizer()
    return _signal_optimizer


def get_kelly_sizer() -> KellyPositionSizer:
    """Get or create global Kelly sizer."""
    global _kelly_sizer
    if _kelly_sizer is None:
        fraction = float(os.environ.get("AGI_KELLY_FRACTION", "0.5"))
        _kelly_sizer = KellyPositionSizer(fraction=fraction)
    return _kelly_sizer


def get_spread_optimizer() -> SpreadOptimizer:
    """Get or create global spread optimizer."""
    global _spread_optimizer
    if _spread_optimizer is None:
        _spread_optimizer = SpreadOptimizer()
    return _spread_optimizer
