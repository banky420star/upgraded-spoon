"""MetaController — ensemble meta-controller for Chain Gambler."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

try:
    from loguru import logger
except ImportError:
    import logging as _logging
    logger = _logging.getLogger("meta_controller")  # type: ignore

from Python.ensemble.decision_builder import DecisionBuilder, TradeIntent
try:
    from Python.patterns.pattern_detector import PatternDetector
except Exception:
    PatternDetector = None


@dataclass
class EnsembleDecision:
    """Final decision emitted by MetaController."""

    decision_id: str
    symbol: str
    timeframe: str
    bundle_id: str
    final_action: str  # LONG / SHORT / FLAT / NO_TRADE
    target_exposure_pct: float
    confidence: float
    agreement_score: float
    regime: str
    risk_adjustments: dict = field(default_factory=dict)
    status: str = "pending"  # pending / executed / rejected / expired
    intent: Optional[TradeIntent] = None
    metadata: dict = field(default_factory=dict)


class MetaController:
    """Combines LSTM, Rainforest, Dreamer, PPO, Risk state, and Execution state
    into a final trading decision with strict safety gates."""

    # Decision thresholds
    LSTM_LONG_THRESHOLD = 0.60
    LSTM_SHORT_THRESHOLD = 0.40
    DREAMER_RUIN_THRESHOLD = 0.25
    MAX_SPREAD_BPS = 25.0
    MIN_ACCOUNT_HEALTH = 0.50  # equity / balance ratio

    def __init__(
        self,
        bundle_id: str,
        symbol: str,
        timeframe: str,
        decision_builder: Optional[DecisionBuilder] = None,
        enable_dreamer: bool = True,
        max_exposure_pct: float = 0.35,
    ):
        self.bundle_id = bundle_id
        self.symbol = symbol
        self.timeframe = timeframe
        self.decision_builder = decision_builder or DecisionBuilder()
        self.enable_dreamer = enable_dreamer
        self.max_exposure_pct = max_exposure_pct

    def decide(
        self,
        lstm_output: dict,
        rainforest_output: dict,
        dreamer_output: dict,
        ppo_output: dict,
        risk_state: dict,
        execution_state: dict,
        pattern_state: Optional[dict] = None,  # NEW: from PatternDetector or rainforest enriched
    ) -> EnsembleDecision:
        """Evaluate all model outputs and return a final EnsembleDecision.
        Now fully consumes Rainforest pattern+regime + Dreamer simulated pattern outcomes
        (via pattern_state + dreamer_output) to bias rich TradeDecisions (TimeExitSpec etc).
        """
        decision_id = f"dec_{uuid.uuid4().hex[:12]}"

        # --- unpack inputs ---
        lstm_p_up = float(lstm_output.get("p_up", 0.5))
        lstm_vote = self._lstm_vote(lstm_p_up)
        lstm_conf = float(lstm_output.get("confidence", 0.0))
        lstm_exp_ret = float(lstm_output.get("expected_return", 0.0))

        regime = str(rainforest_output.get("regime", "ranging"))
        rainforest_conf = float(rainforest_output.get("confidence", 0.0))
        rainforest_vote = self._rainforest_vote(regime)
        allowed_behaviors = self._regime_allowed_behaviors(regime)

        # Extract pattern context (from Rainforest or explicit)
        pat_ctx = pattern_state or rainforest_output.get("patterns", {}) or rainforest_output.get("top_patterns", {})
        if isinstance(pat_ctx, list):
            pat_ctx = {"top": pat_ctx[:3]}  # normalize

        dreamer_exp_reward = float(dreamer_output.get("expected_reward", 0.0))
        dreamer_ruin = float(dreamer_output.get("ruin_probability", 0.0))
        dreamer_conf = float(dreamer_output.get("confidence", 0.0))
        dreamer_vote = self._dreamer_vote(dreamer_exp_reward, dreamer_ruin)
        # Dreamer pattern sim (if imagination provided conditioned rollouts)
        dreamer_sim = dreamer_output.get("pattern_simulated_outcomes", {}) or {"simulated_reward": dreamer_exp_reward}

        ppo_vote_raw = str(ppo_output.get("vote", "FLAT")).upper()
        ppo_target_exposure = float(ppo_output.get("target_exposure", 0.0))
        ppo_conf = float(ppo_output.get("confidence", 0.0))

        spread_bps = float(execution_state.get("spread_bps", 0.0))
        account_health = float(risk_state.get("account_health", 1.0))
        max_daily_loss_hit = bool(risk_state.get("max_daily_loss_hit", False))
        max_drawdown_hit = bool(risk_state.get("max_drawdown_hit", False))
        telemetry_valid = bool(risk_state.get("telemetry_valid", True))

        # --- build raw vote object for DecisionBuilder ---
        raw_votes = {
            "lstm": {
                "vote": lstm_vote,
                "confidence": lstm_conf,
                "expected_return": lstm_exp_ret,
            },
            "rainforest": {
                "regime": regime,
                "vote": rainforest_vote,
                "confidence": rainforest_conf,
                "patterns": pat_ctx,  # NEW: classical patterns passed through
            },
            "dreamer": {
                "vote": dreamer_vote,
                "expected_reward": dreamer_exp_reward,
                "ruin_probability": dreamer_ruin,
                "confidence": dreamer_conf,
                "pattern_simulated_outcomes": dreamer_sim,  # NEW: Dreamer imagination on pattern+timing states
            },
            "ppo": {
                "vote": ppo_vote_raw,
                "target_exposure": ppo_target_exposure,
                "confidence": ppo_conf,
            },
        }

        # --- safety gates (NO_TRADE conditions) ---
        risk_adjustments: dict = {}
        no_trade_reasons: list[str] = []

        if not telemetry_valid:
            no_trade_reasons.append("telemetry_invalid")

        if spread_bps > self.MAX_SPREAD_BPS:
            no_trade_reasons.append(f"spread_danger:{spread_bps:.1f}>{self.MAX_SPREAD_BPS}")
            risk_adjustments["spread_penalty"] = True

        if account_health < self.MIN_ACCOUNT_HEALTH:
            no_trade_reasons.append(f"account_unhealthy:{account_health:.2f}")
            risk_adjustments["account_health_penalty"] = True

        if max_daily_loss_hit or max_drawdown_hit:
            no_trade_reasons.append("risk_limits_breached")
            risk_adjustments["risk_limit_lock"] = True

        # Model disagreement gate
        model_votes = [lstm_vote, ppo_vote_raw, rainforest_vote]
        if self.enable_dreamer:
            model_votes.append(dreamer_vote)
        unique_votes = set(v for v in model_votes if v in ("LONG", "SHORT", "FLAT"))
        if len(unique_votes) > 2:
            no_trade_reasons.append("models_disagree")
            risk_adjustments["disagreement_penalty"] = True

        if regime == "ranging" and rainforest_conf < 0.40:
            no_trade_reasons.append("regime_unknown")
            risk_adjustments["regime_uncertainty"] = True

        if self.enable_dreamer and dreamer_ruin > self.DREAMER_RUIN_THRESHOLD:
            no_trade_reasons.append(f"dreamer_ruin_high:{dreamer_ruin:.2f}")
            risk_adjustments["dreamer_ruin_penalty"] = True

        # --- strong long/short rules ---
        strong_long = (
            lstm_p_up > self.LSTM_LONG_THRESHOLD
            and ppo_vote_raw == "LONG"
            and (not self.enable_dreamer or dreamer_exp_reward > 0)
            and "trend_following" in allowed_behaviors
            and spread_bps <= self.MAX_SPREAD_BPS
            and account_health >= self.MIN_ACCOUNT_HEALTH
            and not (max_daily_loss_hit or max_drawdown_hit)
        )

        strong_short = (
            lstm_p_up < self.LSTM_SHORT_THRESHOLD
            and ppo_vote_raw == "SHORT"
            and (not self.enable_dreamer or dreamer_exp_reward > 0)
            and "trend_following" in allowed_behaviors
            and spread_bps <= self.MAX_SPREAD_BPS
            and account_health >= self.MIN_ACCOUNT_HEALTH
            and not (max_daily_loss_hit or max_drawdown_hit)
        )

        if no_trade_reasons:
            final_action = "NO_TRADE"
            confidence = 0.0
            agreement = 0.0
            target_exposure = 0.0
        elif strong_long:
            final_action = "LONG"
            confidence = round(min(1.0, (lstm_p_up + ppo_conf + lstm_conf) / 3), 4)
            agreement = round(confidence, 4)
            target_exposure = min(ppo_target_exposure, self.max_exposure_pct) if ppo_target_exposure > 0 else 0.20
        elif strong_short:
            final_action = "SHORT"
            confidence = round(min(1.0, ((1.0 - lstm_p_up) + ppo_conf + lstm_conf) / 3), 4)
            agreement = round(confidence, 4)
            target_exposure = min(ppo_target_exposure, self.max_exposure_pct) if ppo_target_exposure > 0 else 0.20
        else:
            # Not strong enough — let DecisionBuilder resolve
            intent = self.decision_builder.build_intent(
                raw_votes=raw_votes,
                decision_id=decision_id,
                symbol=self.symbol,
                source_bundle_id=self.bundle_id,
                regime=regime,
                pattern_context=pat_ctx,
                timing_context=execution_state.get("timing", {}),
                dreamer_sim=dreamer_sim,
            )
            if intent is None:
                final_action = "NO_TRADE"
                confidence = 0.0
                agreement = 0.0
                target_exposure = 0.0
            else:
                final_action = intent.side
                confidence = intent.confidence
                agreement = confidence
                target_exposure = intent.target_exposure_pct
                risk_adjustments["builder_sizing"] = True

        # Re-derive intent if we have a trade action
        intent = None
        if final_action in ("LONG", "SHORT"):
            intent = self.decision_builder.build_intent(
                raw_votes=raw_votes,
                decision_id=decision_id,
                symbol=self.symbol,
                source_bundle_id=self.bundle_id,
                regime=regime,
                pattern_context=pat_ctx,
                timing_context=execution_state.get("timing", {}),
                dreamer_sim=dreamer_sim,
            )
            if intent:
                # Override with meta-controller sizing
                intent.side = final_action
                intent.target_exposure_pct = round(target_exposure, 4)
                intent.confidence = round(confidence, 4)

        decision = EnsembleDecision(
            decision_id=decision_id,
            symbol=self.symbol,
            timeframe=self.timeframe,
            bundle_id=self.bundle_id,
            final_action=final_action,
            target_exposure_pct=round(target_exposure, 4),
            confidence=round(confidence, 4),
            agreement_score=round(agreement, 4),
            regime=regime,
            risk_adjustments=risk_adjustments,
            status="pending" if final_action in ("LONG", "SHORT") else "rejected",
            intent=intent,
            metadata={
                "raw_votes": raw_votes,
                "no_trade_reasons": no_trade_reasons,
                "strong_long": strong_long,
                "strong_short": strong_short,
                "account_health": account_health,
                "spread_bps": spread_bps,
            },
        )

        if final_action == "NO_TRADE":
            logger.info(
                f"MetaController NO_TRADE for {self.symbol} reasons={no_trade_reasons}"
            )
        else:
            logger.info(
                f"MetaController {final_action} for {self.symbol} "
                f"conf={confidence} exp={target_exposure} regime={regime}"
            )
        return decision

    @staticmethod
    def _lstm_vote(p_up: float) -> str:
        if p_up > 0.60:
            return "LONG"
        if p_up < 0.40:
            return "SHORT"
        return "FLAT"

    @staticmethod
    def _rainforest_vote(regime: str) -> str:
        mapping = {
            "bull_trend": "LONG",
            "bear_trend": "SHORT",
            "breakout_up": "LONG",
            "breakout_down": "SHORT",
            "reversal_up": "LONG",
            "reversal_down": "SHORT",
            "ranging": "FLAT",
        }
        return mapping.get(regime, "FLAT")

    @staticmethod
    def _dreamer_vote(expected_reward: float, ruin_probability: float) -> str:
        if ruin_probability > 0.30:
            return "FLAT"
        if expected_reward > 0.0:
            return "LONG"
        if expected_reward < 0.0:
            return "SHORT"
        return "FLAT"

    @staticmethod
    def _regime_allowed_behaviors(regime: str) -> list[str]:
        allowed = {
            "bull_trend": ["trend_following", "pyramiding"],
            "bear_trend": ["trend_following", "shorting"],
            "ranging": ["mean_reversion", "scalping"],
            "breakout_up": ["momentum", "trend_following"],
            "breakout_down": ["momentum", "shorting"],
            "reversal_up": ["counter_trend", "mean_reversion"],
            "reversal_down": ["counter_trend", "mean_reversion"],
        }
        return allowed.get(regime, ["trend_following"])
