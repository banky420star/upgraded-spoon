"""
MarketGuardian — Regime classification and market quality scoring.

Determines:
  1. Market regime (LOW_VOL_RANGE, MED_VOL_TREND, HIGH_VOL_BREAKOUT, etc.)
  2. MarketQuality Score (0-100) — composite score before every trade

Usage:
    from Python.market_guardian import MarketGuardian, MarketQualityScorer
    guardian = MarketGuardian(config)
    regime = guardian.classify(df)  # Returns MarketRegime enum

    quality = MarketQualityScorer(config)
    score = quality.calculate(symbol, setup, regime, event_guard_result)
    if score < 70:
        logger.info(f"Trade blocked: quality score {score} < 70")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
from loguru import logger


class MarketRegime(Enum):
    """Market regime classification for strategy selection."""
    LOW_VOL_RANGE = "low_vol_range"           # Mean reversion only, small size
    MED_VOL_TREND = "med_vol_trend"           # Trend following allowed
    HIGH_VOL_BREAKOUT = "high_vol_breakout"   # Only strong momentum setups
    NEWS_SHOCK = "news_shock"                 # No new trades
    SPREAD_DANGER = "spread_danger"           # No trading
    CHOP = "chop"                             # No trading
    NO_EDGE = "no_edge"                       # No trading


@dataclass
class RegimeResult:
    """Result of regime classification."""
    regime: MarketRegime
    confidence: float  # 0-1
    atr_14: float
    adx_14: float
    rsi_14: float
    bb_width: float  # Bollinger Band width as % of price
    trend_strength: float  # -1 to 1
    volatility_percentile: float  # 0-1, relative to last 100 bars
    is_trending: bool
    is_ranging: bool
    description: str


@dataclass
class QualityResult:
    """Result of market quality calculation."""
    score: int  # 0-100
    allowed: bool  # score >= 70
    regime: MarketRegime
    breakdown: Dict[str, int]  # Per-component scores
    reason: str
    recommendations: List[str] = field(default_factory=list)


class MarketGuardian:
    """
    Classifies market regime using ATR, ADX, RSI, and Bollinger Band analysis.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.regime_config = self.config.get("market_guardian", {})

        # Thresholds (overrideable via config)
        self.atr_low_vol_pct = self.regime_config.get("atr_low_vol_pct", 0.3)  # ATR as % of price
        self.atr_high_vol_pct = self.regime_config.get("atr_high_vol_pct", 1.0)
        self.adx_trending_threshold = self.regime_config.get("adx_trending_threshold", 25)
        self.adx_strong_trend_threshold = self.regime_config.get("adx_strong_trend_threshold", 40)
        self.bb_width_low = self.regime_config.get("bb_width_low", 0.5)  # % of price
        self.bb_width_high = self.regime_config.get("bb_width_high", 2.0)
        self.chop_rsi_range = self.regime_config.get("chop_rsi_range", 10)  # RSI stuck in middle

        self._history: Dict[str, List[RegimeResult]] = {}
        self._max_history = 100

    def classify(self, df: pd.DataFrame, symbol: str = "") -> RegimeResult:
        """
        Classify market regime from OHLCV dataframe.

        Required columns: high, low, close
        Recommended: 20+ bars of data
        """
        if len(df) < 20:
            return RegimeResult(
                regime=MarketRegime.NO_EDGE,
                confidence=0.0,
                atr_14=0.0,
                adx_14=0.0,
                rsi_14=50.0,
                bb_width=0.0,
                trend_strength=0.0,
                volatility_percentile=0.5,
                is_trending=False,
                is_ranging=False,
                description="Insufficient data for regime classification"
            )

        # Calculate indicators
        atr_14 = self._calculate_atr(df, 14)
        adx_14, plus_di, minus_di = self._calculate_adx(df, 14)
        rsi_14 = self._calculate_rsi(df['close'], 14)
        bb_upper, bb_lower, bb_middle = self._calculate_bollinger_bands(df['close'], 20)

        current_price = df['close'].iloc[-1]
        atr_pct = (atr_14 / current_price) * 100 if current_price > 0 else 0
        bb_width = ((bb_upper - bb_lower) / bb_middle) * 100 if bb_middle > 0 else 0

        # Trend strength from ADX and DI
        trend_strength = (plus_di - minus_di) / 100 if (plus_di + minus_di) > 0 else 0
        is_trending = adx_14 > self.adx_trending_threshold
        is_strong_trend = adx_14 > self.adx_strong_trend_threshold

        # RSI position (for range detection)
        rsi_middle = abs(rsi_14 - 50)
        is_ranging = (rsi_14 > 40 and rsi_14 < 60 and adx_14 < 20)

        # Volatility percentile (relative to recent history)
        volatility_percentile = self._calculate_volatility_percentile(df, atr_pct)

        # Classify regime
        regime, confidence, description = self._determine_regime(
            atr_pct=atr_pct,
            adx_14=adx_14,
            is_trending=is_trending,
            is_strong_trend=is_strong_trend,
            is_ranging=is_ranging,
            bb_width=bb_width,
            volatility_percentile=volatility_percentile
        )

        result = RegimeResult(
            regime=regime,
            confidence=confidence,
            atr_14=atr_14,
            adx_14=adx_14,
            rsi_14=rsi_14,
            bb_width=bb_width,
            trend_strength=trend_strength,
            volatility_percentile=volatility_percentile,
            is_trending=is_trending,
            is_ranging=is_ranging,
            description=description
        )

        # Store history
        if symbol:
            if symbol not in self._history:
                self._history[symbol] = []
            self._history[symbol].append(result)
            if len(self._history[symbol]) > self._max_history:
                self._history[symbol].pop(0)

        return result

    def _determine_regime(
        self,
        atr_pct: float,
        adx_14: float,
        is_trending: bool,
        is_strong_trend: bool,
        is_ranging: bool,
        bb_width: float,
        volatility_percentile: float
    ) -> tuple[MarketRegime, float, str]:
        """Determine market regime from indicators."""

        # Priority: No trade conditions first
        if atr_pct > self.atr_high_vol_pct * 2:
            return MarketRegime.SPREAD_DANGER, 0.9, f"Extreme volatility: ATR {atr_pct:.2f}%"

        if is_ranging and bb_width < self.bb_width_low:
            return MarketRegime.CHOP, 0.85, "Price chopping in tight range, ADX low"

        # Classify by volatility and trend
        if atr_pct < self.atr_low_vol_pct:
            # Low volatility
            if is_trending:
                return MarketRegime.MED_VOL_TREND, 0.7, f"Low vol but trending, ADX {adx_14:.1f}"
            else:
                return MarketRegime.LOW_VOL_RANGE, 0.8, f"Low vol range, BB width {bb_width:.2f}%"

        elif atr_pct > self.atr_high_vol_pct:
            # High volatility
            if is_strong_trend:
                return MarketRegime.HIGH_VOL_BREAKOUT, 0.75, f"High vol breakout, ADX {adx_14:.1f}"
            else:
                return MarketRegime.SPREAD_DANGER, 0.7, f"High vol no trend, risky"

        else:
            # Medium volatility
            if is_strong_trend:
                return MarketRegime.MED_VOL_TREND, 0.9, f"Strong trend, ADX {adx_14:.1f}"
            elif is_trending:
                return MarketRegime.MED_VOL_TREND, 0.75, f"Moderate trend, ADX {adx_14:.1f}"
            elif is_ranging:
                return MarketRegime.LOW_VOL_RANGE, 0.7, f"Ranging market, RSI {bb_width:.1f}"
            else:
                return MarketRegime.NO_EDGE, 0.6, "Unclear regime"

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        return atr if not np.isnan(atr) else 0.0

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> tuple[float, float, float]:
        """Calculate ADX, +DI, -DI."""
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        tr = self._calculate_true_range(df)
        atr = tr.rolling(window=period).mean()

        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(window=period).mean()

        return (
            adx.iloc[-1] if not np.isnan(adx.iloc[-1]) else 0.0,
            plus_di.iloc[-1] if not np.isnan(plus_di.iloc[-1]) else 0.0,
            minus_di.iloc[-1] if not np.isnan(minus_di.iloc[-1]) else 0.0
        )

    def _calculate_true_range(self, df: pd.DataFrame) -> pd.Series:
        """Calculate True Range."""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50.0

    def _calculate_bollinger_bands(
        self, prices: pd.Series, period: int = 20, std_dev: float = 2.0
    ) -> tuple[float, float, float]:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return (
            upper.iloc[-1] if not np.isnan(upper.iloc[-1]) else prices.iloc[-1],
            lower.iloc[-1] if not np.isnan(lower.iloc[-1]) else prices.iloc[-1],
            middle.iloc[-1] if not np.isnan(middle.iloc[-1]) else prices.iloc[-1]
        )

    def _calculate_volatility_percentile(self, df: pd.DataFrame, current_atr_pct: float) -> float:
        """Calculate where current volatility stands relative to recent history."""
        if len(df) < 50:
            return 0.5

        # Calculate ATR % for historical windows
        atr_pcts = []
        for i in range(20, min(100, len(df))):
            window = df.iloc[i-20:i]
            atr = self._calculate_atr(window, 14)
            price = window['close'].iloc[-1]
            if price > 0:
                atr_pcts.append((atr / price) * 100)

        if not atr_pcts:
            return 0.5

        return sum(1 for x in atr_pcts if x < current_atr_pct) / len(atr_pcts)

    def is_tradable(self, regime: MarketRegime) -> bool:
        """Check if regime allows trading."""
        return regime not in {
            MarketRegime.NEWS_SHOCK,
            MarketRegime.SPREAD_DANGER,
            MarketRegime.CHOP,
            MarketRegime.NO_EDGE
        }

    def get_regime_history(self, symbol: str, n: int = 10) -> List[MarketRegime]:
        """Get recent regime history for a symbol."""
        if symbol not in self._history:
            return []
        return [r.regime for r in self._history[symbol][-n:]]


class MarketQualityScorer:
    """
    Calculates MarketQuality Score (0-100) before every trade.
    Higher score = better conditions for trading.
    """

    # Component weights (must sum to 100)
    DEFAULT_WEIGHTS = {
        "no_major_news": 20,
        "spread_normal": 15,
        "volatility_tradable": 15,
        "session_liquid": 10,
        "trend_or_range_clear": 15,
        "reward_risk_available": 10,
        "no_recent_sl_chop": 10,
        "model_agrees": 5,
    }

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.quality_config = self.config.get("market_quality", {})
        self.weights = self.quality_config.get("weights", self.DEFAULT_WEIGHTS)
        self.min_score_trade = self.quality_config.get("min_score_trade", 70)
        self.min_score_full_size = self.quality_config.get("min_score_full_size", 85)

        self._recent_stops: Dict[str, List[datetime]] = {}

    def calculate(
        self,
        symbol: str,
        setup: Optional[Dict[str, Any]],
        regime: RegimeResult,
        event_guard_result: Optional[Any] = None,
        spread_bps: Optional[float] = None,
        session_liquid: bool = True,
        recent_sl_count: int = 0,
        model_confidence: float = 0.5
    ) -> QualityResult:
        """
        Calculate market quality score.

        Args:
            symbol: Trading symbol
            setup: Trade setup dict with entry, stop, target prices
            regime: RegimeResult from MarketGuardian
            event_guard_result: Result from EventGuard.check()
            spread_bps: Current spread in basis points
            session_liquid: Is current session liquid for this symbol
            recent_sl_count: Number of recent stop losses
            model_confidence: Model confidence 0-1
        """
        breakdown = {}
        recommendations = []

        # 1. No major news (20 points)
        news_ok = True
        if event_guard_result:
            news_ok = getattr(event_guard_result, 'allowed', True)
            if not news_ok:
                recommendations.append("Wait for news window to pass")
        breakdown["no_major_news"] = 20 if news_ok else 0

        # 2. Spread normal (15 points)
        spread_ok = spread_bps is None or spread_bps < 25  # 2.5 pips
        if not spread_ok:
            recommendations.append(f"Spread too wide: {spread_bps}bps")
        breakdown["spread_normal"] = 15 if spread_ok else 0

        # 3. Volatility tradable (15 points)
        vol_ok = regime.regime in {
            MarketRegime.LOW_VOL_RANGE,
            MarketRegime.MED_VOL_TREND,
            MarketRegime.HIGH_VOL_BREAKOUT
        }
        vol_score = 15 if vol_ok else 0
        if regime.regime == MarketRegime.HIGH_VOL_BREAKOUT:
            vol_score = 10  # Reduced score for high vol
            recommendations.append("High volatility - reduce size")
        breakdown["volatility_tradable"] = vol_score

        # 4. Session liquid (10 points)
        if not session_liquid:
            recommendations.append("Low liquidity session")
        breakdown["session_liquid"] = 10 if session_liquid else 0

        # 5. Trend or range clear (15 points)
        regime_ok = regime.confidence > 0.7
        if not regime_ok:
            recommendations.append("Unclear market regime")
        breakdown["trend_or_range_clear"] = 15 if regime_ok else 0

        # 6. Reward:risk available (10 points)
        r_r_ok = False
        if setup:
            entry = setup.get('entry_price', 0)
            stop = setup.get('stop_price', 0)
            target = setup.get('target_price', 0)
            if entry and stop and target:
                risk = abs(entry - stop)
                reward = abs(target - entry)
                if risk > 0:
                    r_r = reward / risk
                    r_r_ok = r_r >= 1.5  # Minimum 1.5:1 R/R
                    if not r_r_ok:
                        recommendations.append(f"Poor R/R ratio: {r_r:.2f}:1")
        breakdown["reward_risk_available"] = 10 if r_r_ok else 0

        # 7. No recent SL chop (10 points)
        sl_ok = recent_sl_count < 2
        if not sl_ok:
            recommendations.append(f"Recent SL chop ({recent_sl_count} stops)")
        breakdown["no_recent_sl_chop"] = 10 if sl_ok else 5

        # 8. Model agrees (5 points)
        model_ok = model_confidence > 0.6
        if not model_ok:
            recommendations.append("Low model confidence")
        breakdown["model_agrees"] = 5 if model_ok else 0

        # Calculate total score
        score = sum(breakdown.values())

        # Determine if allowed
        allowed = score >= self.min_score_trade

        # Build reason
        if allowed:
            if score >= self.min_score_full_size:
                reason = f"Quality score {score}/100 - Full size allowed"
            else:
                reason = f"Quality score {score}/100 - Reduced size only"
        else:
            reason = f"Quality score {score}/100 - Below minimum {self.min_score_trade}"

        return QualityResult(
            score=score,
            allowed=allowed,
            regime=regime.regime,
            breakdown=breakdown,
            reason=reason,
            recommendations=recommendations
        )

    def get_position_size_multiplier(self, quality: QualityResult) -> float:
        """Get position size multiplier based on quality score."""
        if quality.score >= self.min_score_full_size:
            return 1.0
        elif quality.score >= self.min_score_trade:
            return 0.5  # Half size for marginal setups
        else:
            return 0.0  # No trade

    def record_stop_loss(self, symbol: str):
        """Record a stop loss for recent SL tracking."""
        if symbol not in self._recent_stops:
            self._recent_stops[symbol] = []
        self._recent_stops[symbol].append(datetime.now(timezone.utc))

        # Clean old entries (> 1 hour)
        cutoff = datetime.now(timezone.utc) - pd.Timedelta(hours=1)
        self._recent_stops[symbol] = [
            t for t in self._recent_stops[symbol] if t > cutoff
        ]

    def get_recent_sl_count(self, symbol: str) -> int:
        """Get count of recent stop losses for a symbol."""
        if symbol not in self._recent_stops:
            return 0
        cutoff = datetime.now(timezone.utc) - pd.Timedelta(hours=1)
        return sum(1 for t in self._recent_stops[symbol] if t > cutoff)


# Convenience function for quick usage
def check_market_conditions(
    df: pd.DataFrame,
    symbol: str,
    config: Optional[Dict] = None,
    **kwargs
) -> tuple[RegimeResult, QualityResult]:
    """
    Quick check of market conditions.

    Returns:
        (regime_result, quality_result)
    """
    guardian = MarketGuardian(config)
    regime = guardian.classify(df, symbol)

    scorer = MarketQualityScorer(config)
    quality = scorer.calculate(
        symbol=symbol,
        setup=kwargs.get('setup'),
        regime=regime,
        event_guard_result=kwargs.get('event_guard_result'),
        spread_bps=kwargs.get('spread_bps'),
        session_liquid=kwargs.get('session_liquid', True),
        recent_sl_count=kwargs.get('recent_sl_count', 0),
        model_confidence=kwargs.get('model_confidence', 0.5)
    )

    return regime, quality