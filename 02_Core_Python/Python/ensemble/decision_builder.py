"""DecisionBuilder — converts raw model votes into trade intents."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

try:
    from loguru import logger
except ImportError:
    import logging as _logging
    logger = _logging.getLogger("decision_builder")  # type: ignore


@dataclass
class TradeIntent:
    """Structured trade intent emitted by DecisionBuilder.
    Now enriched with pattern + timing context for the full ensemble to produce
    rich TradeDecisions (bias TimeExitSpec, risk sizing, partial ladders for favorable states e.g. engulfing at open/low-news).
    """

    intent_id: str
    decision_id: str
    symbol: str
    side: str  # LONG / SHORT / FLAT
    target_exposure_pct: float
    stop_atr: float
    take_profit_atr: float
    max_hold_bars: int
    confidence: float
    source_bundle_id: str
    metadata: dict = field(default_factory=dict)
    # NEW: pattern+timing edge for rich decisions
    pattern_context: dict = field(default_factory=dict)  # e.g. {"dominant": "bullish_engulfing", "strength":0.9, "timing_favorable":True}
    time_exit_hints: dict = field(default_factory=dict)  # hints for TimeExitSpec e.g. {"close_before_news": True, "max_hold_minutes": 90}


class DecisionBuilder:
    """Converts raw ensemble votes into a concrete trade intent."""

    # Default intent sizing parameters
    DEFAULT_STOP_ATR = 2.0
    DEFAULT_TAKE_PROFIT_ATR = 3.0
    DEFAULT_MAX_HOLD_BARS = 48
    MAX_EXPOSURE_PCT = 0.35

    def __init__(
        self,
        stop_atr: float = DEFAULT_STOP_ATR,
        take_profit_atr: float = DEFAULT_TAKE_PROFIT_ATR,
        max_hold_bars: int = DEFAULT_MAX_HOLD_BARS,
    ):
        self.stop_atr = stop_atr
        self.take_profit_atr = take_profit_atr
        self.max_hold_bars = max_hold_bars

    def build_intent(
        self,
        raw_votes: dict,
        decision_id: str,
        symbol: str,
        source_bundle_id: str,
        regime: str = "ranging",
        pattern_context: Optional[dict] = None,
        timing_context: Optional[dict] = None,
        dreamer_sim: Optional[dict] = None,  # e.g. simulated pattern outcome rewards from Dreamer imagination
    ) -> Optional[TradeIntent]:
        """Build a TradeIntent from raw model votes.
        Now consumes Rainforest pattern+regime + Dreamer simulated pattern outcomes
        to bias rich parameters (for downstream conversion to full TradeDecision with TimeExitSpec etc).

        Raw vote object + new keys:
          ...
          rainforest may include "patterns": {...}
        """
        ppo_vote = self._extract_vote(raw_votes.get("ppo", {}))
        lstm_vote = self._extract_vote(raw_votes.get("lstm", {}))
        dreamer_vote = self._extract_vote(raw_votes.get("dreamer", {}))
        rainforest_vote = self._extract_vote(raw_votes.get("rainforest", {}))

        # Determine side via weighted consensus
        side, confidence = self._resolve_side(
            lstm=lstm_vote,
            ppo=ppo_vote,
            dreamer=dreamer_vote,
            rainforest=rainforest_vote,
            raw=raw_votes,
        )

        if side == "NO_TRADE":
            return None

        target_exposure = self._resolve_target_exposure(raw_votes, side)

        # Adjust sizing by regime risk
        if regime in ("ranging", "reversal_up", "reversal_down"):
            target_exposure *= 0.5
        elif regime in ("breakout_up", "breakout_down"):
            target_exposure *= 0.8

        # NEW: Pattern + timing + dreamer sim bias for rich edge
        pat_ctx = pattern_context or raw_votes.get("rainforest", {}).get("patterns", {}) or {}
        t_ctx = timing_context or {}
        d_sim = dreamer_sim or raw_votes.get("dreamer", {}).get("pattern_sim", {}) or {}
        favorable = self._is_favorable_pattern_timing(pat_ctx, t_ctx, regime, d_sim)
        max_hold = self.max_hold_bars
        stop_atr = self.stop_atr
        tp_atr = self.take_profit_atr
        time_hints = {}

        if favorable:
            # Favorable pattern (e.g. engulfing/hammer/flag/breakout) + good timing (open or low news) + positive dreamer sim
            # -> bias toward runner (longer hold, wider TP, tighter risk? or scaled)
            target_exposure = min(self.MAX_EXPOSURE_PCT, target_exposure * 1.25)
            max_hold = int(self.max_hold_bars * 1.4)  # allow more time for follow-through
            tp_atr = self.take_profit_atr * 1.2
            time_hints = {
                "close_before_high_impact_news": False,  # let it run if pattern strong
                "max_hold_minutes": 180,
                "partials_aggressive": True,  # hint for ladder in rich TradeDecision
            }
            # dreamer sim can further modulate
            if d_sim.get("simulated_reward", 0) > 0.8:
                tp_atr *= 1.1
        else:
            # Caution: tight time exit, smaller size already handled
            time_hints = {
                "close_before_high_impact_news": bool(t_ctx.get("news_proximity", 0) > 0.3),
                "max_hold_minutes": 75,
            }
            if "ranging" in regime or pat_ctx.get("has_doji", 0) > 0.6:
                max_hold = max(8, int(self.max_hold_bars * 0.6))

        # Clamp exposure
        target_exposure = max(0.0, min(target_exposure, self.MAX_EXPOSURE_PCT))

        if target_exposure <= 0.0:
            return None

        intent = TradeIntent(
            intent_id=f"intent_{uuid.uuid4().hex[:12]}",
            decision_id=decision_id,
            symbol=symbol,
            side=side,
            target_exposure_pct=round(target_exposure, 4),
            stop_atr=round(stop_atr, 3),
            take_profit_atr=round(tp_atr, 3),
            max_hold_bars=max_hold,
            confidence=round(confidence, 4),
            source_bundle_id=source_bundle_id,
            metadata={
                "votes": {
                    "lstm": lstm_vote,
                    "ppo": ppo_vote,
                    "dreamer": dreamer_vote,
                    "rainforest": rainforest_vote,
                },
                "regime": regime,
                "pattern_context": pat_ctx,
                "dreamer_sim": d_sim,
            },
            pattern_context={
                "dominant": pat_ctx.get("dominant_pattern", regime),
                "strength": float(pat_ctx.get("strength", 0.0)),
                "timing_favorable": bool(favorable),
                "has_engulfing_or_reversal": bool(pat_ctx.get("has_bullish_engulfing", 0) > 0.4 or pat_ctx.get("has_hammer", 0) > 0.4),
            },
            time_exit_hints=time_hints,
        )
        logger.debug(f"DecisionBuilder emitted intent {intent.intent_id} side={intent.side} exp={intent.target_exposure_pct} pattern_fav={favorable}")
        return intent

    @staticmethod
    def _extract_vote(vote_payload: dict) -> dict:
        return {
            "vote": str(vote_payload.get("vote", "FLAT")).upper(),
            "confidence": float(vote_payload.get("confidence", 0.0)),
            "expected_return": float(vote_payload.get("expected_return", 0.0)),
            "expected_reward": float(vote_payload.get("expected_reward", 0.0)),
            "ruin_probability": float(vote_payload.get("ruin_probability", 0.0)),
            "target_exposure": float(vote_payload.get("target_exposure", 0.0)),
        }

    def _resolve_side(self, lstm: dict, ppo: dict, dreamer: dict, rainforest: dict, raw: dict) -> tuple[str, float]:
        """Resolve final side and confidence from votes."""
        votes = [lstm["vote"], ppo["vote"], dreamer["vote"], rainforest["vote"]]
        confidences = [lstm["confidence"], ppo["confidence"], dreamer["confidence"], rainforest["confidence"]]

        # Count weighted votes
        long_score = 0.0
        short_score = 0.0
        flat_score = 0.0
        for v, c in zip(votes, confidences):
            weight = max(0.0, min(1.0, c))
            if v == "LONG":
                long_score += weight
            elif v == "SHORT":
                short_score += weight
            else:
                flat_score += weight

        # Dreamer ruin penalty: if ruin prob high, suppress trade
        if dreamer["ruin_probability"] > 0.30:
            long_score *= 0.5
            short_score *= 0.5

        max_score = max(long_score, short_score, flat_score)
        total_score = long_score + short_score + flat_score + 1e-8
        agreement = max_score / total_score

        # Need at least 2 models agreeing with non-trivial confidence
        if agreement < 0.40 or max_score < 0.30:
            return "NO_TRADE", round(agreement, 4)

        if long_score == max_score:
            return "LONG", round(agreement, 4)
        if short_score == max_score:
            return "SHORT", round(agreement, 4)
        return "FLAT", round(agreement, 4)

    def _resolve_target_exposure(self, raw_votes: dict, side: str) -> float:
        """Determine target exposure percentage from votes."""
        ppo = raw_votes.get("ppo", {})
        lstm = raw_votes.get("lstm", {})

        # Start from PPO target exposure if available and aligned
        ppo_target = float(ppo.get("target_exposure", 0.0))
        ppo_vote = str(ppo.get("vote", "FLAT")).upper()

        if ppo_vote == side and ppo_target > 0:
            return ppo_target

        # Default sizing based on LSTM confidence
        lstm_conf = float(lstm.get("confidence", 0.0))
        if lstm_conf >= 0.75:
            return 0.20
        if lstm_conf >= 0.60:
            return 0.15
        if lstm_conf >= 0.50:
            return 0.10
        return 0.05

    def _is_favorable_pattern_timing(self, pat_ctx: dict, t_ctx: dict, regime: str, dreamer_sim: dict) -> bool:
        """Core logic: favorable when classical pattern + supportive timing + dreamer sim positive."""
        score = 0.0
        # Strong reversal/continuation patterns
        if pat_ctx.get("has_bullish_engulfing", 0) > 0.5 or pat_ctx.get("has_hammer", 0) > 0.55:
            score += 1.0
        if pat_ctx.get("has_bull_flag", 0) > 0.5 or pat_ctx.get("has_breakout_up", 0) > 0.6:
            score += 0.9
        if pat_ctx.get("has_bearish_engulfing", 0) > 0.5 or pat_ctx.get("has_shooting_star", 0) > 0.55:
            score += 1.0
        if pat_ctx.get("has_bear_flag", 0) > 0.5 or pat_ctx.get("has_breakout_down", 0) > 0.6:
            score += 0.9

        # Timing edge (opens or away from news)
        if t_ctx.get("major_open_window", 0) > 0.4:
            score += 0.6
        if t_ctx.get("news_proximity", 1.0) < 0.25:
            score += 0.7
        if t_ctx.get("has_high_impact_news_soon", 0) > 0.5:
            score -= 0.8

        # Dreamer imagination simulation (pattern-conditioned rollout reward)
        if dreamer_sim.get("simulated_reward", 0.0) > 0.15 or dreamer_sim.get("expected_reward", 0.0) > 0.1:
            score += 0.8

        # Regime alignment
        if ("bull" in regime and score > 0) or ("bear" in regime and score > 0):
            score += 0.3

        return score >= 1.4
