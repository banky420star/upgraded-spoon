"""
Reversal Detection Module for Chain Gambler

Detects potential trend reversals using multiple confirmation methods:
1. Divergence detection (price vs momentum)
2. Trend exhaustion patterns
3. Support/Resistance breaks with volume
4. Candlestick reversal patterns
5. Multi-timeframe alignment

When reversal is detected, can flip trade direction for counter-trend entries.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class ReversalSignal:
    """Reversal detection result."""
    detected: bool
    direction: str  # "bullish_reversal" or "bearish_reversal"
    confidence: float  # 0.0 to 1.0
    methods: List[str]  # Which methods detected the reversal
    entry_price: float
    stop_loss: float
    take_profit: float
    notes: List[str]


class ReversalDetector:
    """
    Advanced reversal detection for trend exhaustion and momentum shifts.

    Philosophy: "The trend is your friend until it bends"
    Detect when trends are likely ending for early reversal entries.
    """

    def __init__(self):
        # Configuration
        self.divergence_lookback = int(os.environ.get("AGI_DIVERGENCE_LOOKBACK", "20"))
        self.min_bars_for_trend = int(os.environ.get("AGI_MIN_TREND_BARS", "10"))
        self.reversal_confidence_threshold = float(os.environ.get("AGI_REVERSAL_THRESHOLD", "0.65"))

        logger.success(f"ReversalDetector initialized (threshold={self.reversal_confidence_threshold})")

    def detect_reversal(
        self,
        symbol: str,
        df: pd.DataFrame,
        current_action: str,
    ) -> ReversalSignal:
        """
        Detect potential trend reversal.

        Args:
            symbol: Trading symbol
            df: OHLCV DataFrame
            current_action: Current intended action (BUY/SELL/HOLD)

        Returns:
            ReversalSignal with detection result
        """
        if len(df) < self.divergence_lookback + 10:
            return ReversalSignal(
                detected=False,
                direction="none",
                confidence=0.0,
                methods=[],
                entry_price=0.0,
                stop_loss=0.0,
                take_profit=0.0,
                notes=["Insufficient data"],
            )

        methods_triggered = []
        confidence_scores = []
        notes = []

        # Method 1: Divergence Detection
        div_score, div_direction, div_notes = self._check_divergence(df)
        if div_score > 0.5:
            methods_triggered.append("divergence")
            confidence_scores.append(div_score)
            notes.extend(div_notes)

        # Method 2: Trend Exhaustion
        exhaust_score, exhaust_notes = self._check_trend_exhaustion(df)
        if exhaust_score > 0.5:
            methods_triggered.append("trend_exhaustion")
            confidence_scores.append(exhaust_score)
            notes.extend(exhaust_notes)

        # Method 3: Support/Resistance Break
        sr_score, sr_direction, sr_notes = self._check_sr_break(df)
        if sr_score > 0.5:
            methods_triggered.append("sr_break")
            confidence_scores.append(sr_score)
            notes.extend(sr_notes)

        # Method 4: Candlestick Patterns
        candle_score, candle_direction, candle_notes = self._check_candlestick_patterns(df)
        if candle_score > 0.5:
            methods_triggered.append("candlestick")
            confidence_scores.append(candle_score)
            notes.extend(candle_notes)

        # Method 5: Volume Confirmation
        vol_score, vol_notes = self._check_volume_confirmation(df)
        if vol_score > 0.5:
            methods_triggered.append("volume")
            confidence_scores.append(vol_score)
            notes.extend(vol_notes)

        # Calculate overall confidence
        if not methods_triggered:
            return ReversalSignal(
                detected=False,
                direction="none",
                confidence=0.0,
                methods=[],
                entry_price=0.0,
                stop_loss=0.0,
                take_profit=0.0,
                notes=["No reversal signals detected"],
            )

        # Weight by number of confirming methods
        avg_confidence = np.mean(confidence_scores)
        method_bonus = min(0.2, len(methods_triggered) * 0.05)  # Up to +0.2 for multiple methods
        final_confidence = min(1.0, avg_confidence + method_bonus)

        # Determine reversal direction
        # Take majority vote from methods that have direction
        directions = []
        if div_direction:
            directions.append(div_direction)
        if sr_direction:
            directions.append(sr_direction)
        if candle_direction:
            directions.append(candle_direction)

        if not directions:
            return ReversalSignal(
                detected=False,
                direction="none",
                confidence=0.0,
                methods=[],
                entry_price=0.0,
                stop_loss=0.0,
                take_profit=0.0,
                notes=["Direction unclear"] + notes,
            )

        reversal_direction = max(set(directions), key=directions.count)

        # Only confirm if confidence is high enough
        if final_confidence < self.reversal_confidence_threshold:
            return ReversalSignal(
                detected=False,
                direction=reversal_direction,
                confidence=final_confidence,
                methods=methods_triggered,
                entry_price=0.0,
                stop_loss=0.0,
                take_profit=0.0,
                notes=[f"Confidence too low ({final_confidence:.2f})"] + notes,
            )

        # Calculate entry/stop/profit levels
        current_price = df['close'].iloc[-1]
        atr = self._calculate_atr(df)

        if reversal_direction == "bullish_reversal":
            entry = current_price
            stop = current_price - atr * 1.5
            profit = current_price + atr * 3.0
        else:  # bearish_reversal
            entry = current_price
            stop = current_price + atr * 1.5
            profit = current_price - atr * 3.0

        logger.info(
            f"[ReversalDetector] {symbol}: {reversal_direction} detected "
            f"(confidence={final_confidence:.2f}, methods={methods_triggered})"
        )

        return ReversalSignal(
            detected=True,
            direction=reversal_direction,
            confidence=final_confidence,
            methods=methods_triggered,
            entry_price=entry,
            stop_loss=stop,
            take_profit=profit,
            notes=notes,
        )

    def _check_divergence(
        self,
        df: pd.DataFrame,
    ) -> Tuple[float, Optional[str], List[str]]:
        """
        Check for price-momentum divergence.

        Bullish divergence: Price making lower lows, RSI making higher lows
        Bearish divergence: Price making higher highs, RSI making lower highs
        """
        notes = []

        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Get recent swing points
        lookback = min(self.divergence_lookback, len(df) - 5)
        prices = df['close'].iloc[-lookback:].values
        rsi_vals = rsi.iloc[-lookback:].values

        if len(prices) < 10:
            return 0.0, None, ["Insufficient data for divergence"]

        # Find local extrema
        price_highs = []
        price_lows = []
        rsi_highs = []
        rsi_lows = []

        for i in range(2, len(prices) - 2):
            # Price extrema
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                price_highs.append((i, prices[i]))
            if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                price_lows.append((i, prices[i]))

            # RSI extrema
            if rsi_vals[i] > rsi_vals[i-1] and rsi_vals[i] > rsi_vals[i+1]:
                rsi_highs.append((i, rsi_vals[i]))
            if rsi_vals[i] < rsi_vals[i-1] and rsi_vals[i] < rsi_vals[i+1]:
                rsi_lows.append((i, rsi_vals[i]))

        # Check for divergence
        confidence = 0.0
        direction = None

        # Bullish divergence: lower price lows, higher RSI lows
        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            last_price_low = price_lows[-1][1]
            prev_price_low = price_lows[-2][1]
            last_rsi_low = rsi_lows[-1][1]
            prev_rsi_low = rsi_lows[-2][1]

            if last_price_low < prev_price_low and last_rsi_low > prev_rsi_low:
                confidence = 0.7
                direction = "bullish_reversal"
                notes.append(f"Bullish divergence: price ↓, RSI ↑")

        # Bearish divergence: higher price highs, lower RSI highs
        if len(price_highs) >= 2 and len(rsi_highs) >= 2:
            last_price_high = price_highs[-1][1]
            prev_price_high = price_highs[-2][1]
            last_rsi_high = rsi_highs[-1][1]
            prev_rsi_high = rsi_highs[-2][1]

            if last_price_high > prev_price_high and last_rsi_high < prev_rsi_high:
                conf = 0.7
                if conf > confidence:  # Take stronger signal
                    confidence = conf
                    direction = "bearish_reversal"
                    notes.append(f"Bearish divergence: price ↑, RSI ↓")

        return confidence, direction, notes

    def _check_trend_exhaustion(
        self,
        df: pd.DataFrame,
    ) -> Tuple[float, List[str]]:
        """
        Check for trend exhaustion signals.

        Signs:
        - ADX declining from high levels
        - Price moving in smaller increments
        - Reduced momentum in trend direction
        """
        notes = []

        # Calculate ADX (simplified)
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        if len(close) < 20:
            return 0.0, ["Insufficient data"]

        # True Range calculation
        tr1 = high[-20:] - low[-20:]
        tr2 = np.abs(high[-20:] - close[-21:-1])
        tr3 = np.abs(low[-20:] - close[-21:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        atr = np.mean(tr)

        # Check trend strength decay
        recent_range = np.max(close[-10:]) - np.min(close[-10:])
        previous_range = np.max(close[-20:-10]) - np.min(close[-20:-10])

        if previous_range == 0:
            return 0.0, ["No trend established"]

        range_decay = recent_range / previous_range

        confidence = 0.0

        if range_decay < 0.5:  # Recent range is half of previous
            confidence = 0.6
            notes.append(f"Trend exhaustion: range decay {range_decay:.2f}")

        # Check for momentum divergence
        returns = np.diff(close[-20:]) / close[-20:-1]
        recent_momentum = np.mean(returns[-5:])
        previous_momentum = np.mean(returns[-10:-5])

        if abs(recent_momentum) < abs(previous_momentum) * 0.5:
            confidence = max(confidence, 0.55)
            notes.append("Momentum decay detected")

        return confidence, notes

    def _check_sr_break(
        self,
        df: pd.DataFrame,
    ) -> Tuple[float, Optional[str], List[str]]:
        """
        Check for support/resistance breaks with confirmation.

        Bullish: Break above recent resistance
        Bearish: Break below recent support
        """
        notes = []

        if len(df) < 30:
            return 0.0, None, ["Insufficient data"]

        # Calculate recent support/resistance
        recent_highs = df['high'].iloc[-30:-5].values
        recent_lows = df['low'].iloc[-30:-5].values

        resistance = np.percentile(recent_highs, 95)
        support = np.percentile(recent_lows, 5)

        current_price = df['close'].iloc[-1]
        current_high = df['high'].iloc[-1]
        current_low = df['low'].iloc[-1]

        confidence = 0.0
        direction = None

        # Break above resistance
        if current_price > resistance * 1.001:  # 0.1% break
            # Check volume confirmation
            avg_volume = df['volume'].iloc[-20:-5].mean()
            current_volume = df['volume'].iloc[-1]

            if current_volume > avg_volume * 1.2:  # 20% above average
                confidence = 0.75
                direction = "bullish_reversal"
                notes.append(f"Resistance break: {current_price:.5f} > {resistance:.5f}")

        # Break below support
        elif current_price < support * 0.999:  # 0.1% break
            avg_volume = df['volume'].iloc[-20:-5].mean()
            current_volume = df['volume'].iloc[-1]

            if current_volume > avg_volume * 1.2:
                confidence = 0.75
                direction = "bearish_reversal"
                notes.append(f"Support break: {current_price:.5f} < {support:.5f}")

        return confidence, direction, notes

    def _check_candlestick_patterns(
        self,
        df: pd.DataFrame,
    ) -> Tuple[float, Optional[str], List[str]]:
        """
        Detect candlestick reversal patterns.

        Bullish: Hammer, Morning Star, Engulfing
        Bearish: Shooting Star, Evening Star, Engulfing
        """
        notes = []

        if len(df) < 5:
            return 0.0, None, ["Insufficient data"]

        # Get last 3 candles
        o = df['open'].iloc[-3:].values
        h = df['high'].iloc[-3:].values
        l = df['low'].iloc[-3:].values
        c = df['close'].iloc[-3:].values

        confidence = 0.0
        direction = None

        # Current candle
        body = abs(c[-1] - o[-1])
        total_range = h[-1] - l[-1]
        upper_wick = h[-1] - max(c[-1], o[-1])
        lower_wick = min(c[-1], o[-1]) - l[-1]

        # Hammer (bullish)
        if total_range > 0:
            if lower_wick > body * 2 and upper_wick < body * 0.5 and c[-1] > o[-1]:
                confidence = 0.65
                direction = "bullish_reversal"
                notes.append("Hammer pattern detected")

        # Shooting Star (bearish)
        if total_range > 0:
            if upper_wick > body * 2 and lower_wick < body * 0.5 and c[-1] < o[-1]:
                conf = 0.65
                if conf > confidence:
                    confidence = conf
                    direction = "bearish_reversal"
                    notes.append("Shooting star pattern detected")

        # Engulfing patterns (need 2 candles)
        if len(df) >= 2:
            prev_body = abs(c[-2] - o[-2])
            curr_body = abs(c[-1] - o[-1])

            # Bullish engulfing
            if c[-2] < o[-2] and c[-1] > o[-1]:  # Prev bearish, curr bullish
                if c[-1] > o[-2] and o[-1] < c[-2]:  # Current engulfs previous
                    if curr_body > prev_body * 1.2:  # 20% larger body
                        conf = 0.70
                        if conf > confidence:
                            confidence = conf
                            direction = "bullish_reversal"
                            notes.append("Bullish engulfing pattern")

            # Bearish engulfing
            if c[-2] > o[-2] and c[-1] < o[-1]:  # Prev bullish, curr bearish
                if o[-1] > c[-2] and c[-1] < o[-2]:  # Current engulfs previous
                    if curr_body > prev_body * 1.2:
                        conf = 0.70
                        if conf > confidence:
                            confidence = conf
                            direction = "bearish_reversal"
                            notes.append("Bearish engulfing pattern")

        return confidence, direction, notes

    def _check_volume_confirmation(
        self,
        df: pd.DataFrame,
    ) -> Tuple[float, List[str]]:
        """
        Check for volume confirmation of reversal.

        High volume on reversal candles = stronger signal
        """
        notes = []

        if 'volume' not in df.columns or len(df) < 20:
            return 0.0, ["No volume data"]

        recent_volume = df['volume'].iloc[-5:].mean()
        avg_volume = df['volume'].iloc[-20:-5].mean()

        if avg_volume == 0:
            return 0.0, ["Zero average volume"]

        volume_ratio = recent_volume / avg_volume

        confidence = 0.0
        if volume_ratio > 1.5:
            confidence = 0.6
            notes.append(f"High volume confirmation: {volume_ratio:.1f}x average")
        elif volume_ratio > 1.2:
            confidence = 0.4
            notes.append(f"Above average volume: {volume_ratio:.1f}x")

        return confidence, notes

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(df) < period + 1:
            return df['close'].iloc[-1] * 0.001

        highs = df['high'].values[-period-1:]
        lows = df['low'].values[-period-1:]
        closes = df['close'].values[-period-1:]

        tr_list = []
        for i in range(1, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_list.append(tr)

        return np.mean(tr_list) if tr_list else closes[-1] * 0.001

    def should_flip_direction(
        self,
        current_action: str,
        reversal: ReversalSignal,
    ) -> Tuple[bool, str]:
        """
        Determine if trade direction should be flipped based on reversal.

        Returns:
            (should_flip, reason)
        """
        if not reversal.detected:
            return False, "No reversal detected"

        if reversal.confidence < self.reversal_confidence_threshold:
            return False, f"Reversal confidence too low ({reversal.confidence:.2f})"

        # Check if reversal opposes current action
        if current_action == "BUY" and reversal.direction == "bearish_reversal":
            return True, f"Bearish reversal detected (conf={reversal.confidence:.2f})"

        if current_action == "SELL" and reversal.direction == "bullish_reversal":
            return True, f"Bullish reversal detected (conf={reversal.confidence:.2f})"

        # Reversal aligns with current action - enhance it
        if current_action == "BUY" and reversal.direction == "bullish_reversal":
            return False, "Reversal confirms BUY direction"

        if current_action == "SELL" and reversal.direction == "bearish_reversal":
            return False, "Reversal confirms SELL direction"

        return False, "No flip needed"


# Global instance
_reversal_detector = None


def get_reversal_detector() -> ReversalDetector:
    """Get or create global reversal detector."""
    global _reversal_detector
    if _reversal_detector is None:
        _reversal_detector = ReversalDetector()
    return _reversal_detector
