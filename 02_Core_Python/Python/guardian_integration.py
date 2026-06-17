"""
Guardian Integration — Integration of Guardian Trader components into Server_AGI.

This module provides the main trading loop integration for:
  1. MarketGuardian — Regime classification
  2. MarketQualityScorer — Quality score 0-100
  3. StrategySelector — Choose trend/mean-reversion/breakout
  4. ExitEngineR — R-based exits
  5. TradeJournal — Post-trade learning

Usage (in Server_AGI.py):
    from Python.guardian_integration import GuardianTraderIntegration
    guardian = GuardianTraderIntegration(config)

    # In main loop:
    if guardian.should_trade(symbol, df, setup):
        signal = guardian.generate_signal(symbol, df)
        if signal.signal_type != SignalType.HOLD:
            trade_id = executor.execute(signal)
            guardian.register_position(trade_id, signal)

    # Update exits:
    for pos_id, current_price in positions:
        action = guardian.update_exit(pos_id, current_price)
        if action.volume_to_close > 0:
            executor.close_partial(pos_id, action)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List
import pandas as pd
from loguru import logger

from Python.market_guardian import (
    MarketGuardian, MarketQualityScorer, MarketRegime,
    RegimeResult, QualityResult
)
from Python.strategy_selector import (
    StrategySelector, StrategyContext, StrategySignal, SignalType
)
from Python.exit_engine_r import ExitEngineR, ExitRecommendation, ExitAction
from Python.trade_journal import TradeJournal, TradeRecord


@dataclass
class GuardianDecision:
    """Decision result from Guardian Trader."""
    allowed: bool
    reason: str
    regime: Optional[MarketRegime] = None
    quality_score: int = 0
    signal: Optional[StrategySignal] = None
    position_size_mult: float = 0.0


class GuardianTraderIntegration:
    """
    Integrated Guardian Trader system.
    Combines all components into a unified trading interface.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.gt_config = self.config.get("guardian_trader", {})
        self.enabled = self.gt_config.get("enabled", True)

        if not self.enabled:
            logger.info("Guardian Trader integration disabled")
            return

        # Initialize components
        self.market_guardian = MarketGuardian(config)
        self.quality_scorer = MarketQualityScorer(config)
        self.strategy_selector = StrategySelector(config)
        self.exit_engine = ExitEngineR(config)
        self.trade_journal = TradeJournal(config)

        # Session tracking
        self._recent_quality_scores: Dict[str, int] = {}
        self._session_trade_count = 0
        self._session_start = datetime.utcnow()

        logger.success("Guardian Trader integration initialized")

    def should_trade(
        self,
        symbol: str,
        df: pd.DataFrame,
        setup: Optional[Dict[str, Any]] = None,
        spread_bps: Optional[float] = None,
        event_guard_result: Optional[Any] = None,
        session_liquid: bool = True,
        model_confidence: float = 0.5
    ) -> GuardianDecision:
        """
        Full Guardian check before trading.

        Returns:
            GuardianDecision with allowed status and full context
        """
        if not self.enabled:
            return GuardianDecision(
                allowed=True,
                reason="Guardian disabled - allowing trade"
            )

        # 1. Classify market regime
        regime_result = self.market_guardian.classify(df, symbol)

        # Check if regime is tradable
        if not self.market_guardian.is_tradable(regime_result.regime):
            return GuardianDecision(
                allowed=False,
                reason=f"Regime {regime_result.regime.value} not tradable: {regime_result.description}",
                regime=regime_result.regime,
                quality_score=0
            )

        # 2. Calculate quality score
        recent_sl_count = self.quality_scorer.get_recent_sl_count(symbol)

        quality_result = self.quality_scorer.calculate(
            symbol=symbol,
            setup=setup,
            regime=regime_result,
            event_guard_result=event_guard_result,
            spread_bps=spread_bps,
            session_liquid=session_liquid,
            recent_sl_count=recent_sl_count,
            model_confidence=model_confidence
        )

        self._recent_quality_scores[symbol] = quality_result.score

        # 3. Check quality threshold
        if not quality_result.allowed:
            return GuardianDecision(
                allowed=False,
                reason=f"Quality score {quality_result.score}/100 below minimum",
                regime=regime_result.regime,
                quality_score=quality_result.score
            )

        # Calculate position size multiplier
        size_mult = self.quality_scorer.get_position_size_multiplier(quality_result)

        return GuardianDecision(
            allowed=True,
            reason=quality_result.reason,
            regime=regime_result.regime,
            quality_score=quality_result.score,
            position_size_mult=size_mult
        )

    def generate_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        spread_bps: float = 0.0,
        session_quality: str = "high"
    ) -> StrategySignal:
        """
        Generate trading signal using appropriate strategy.
        """
        if not self.enabled:
            return StrategySignal(
                signal_type=SignalType.HOLD,
                confidence=0.0,
                reason="Guardian disabled"
            )

        # Get regime
        regime_result = self.market_guardian.classify(df, symbol)

        # Build strategy context
        context = StrategyContext(
            regime=regime_result.regime,
            atr_14=regime_result.atr_14,
            adx_14=regime_result.adx_14,
            rsi_14=regime_result.rsi_14,
            trend_direction=1 if regime_result.trend_strength > 0 else -1 if regime_result.trend_strength < 0 else 0,
            volatility_percentile=regime_result.volatility_percentile,
            spread_bps=spread_bps,
            session_quality=session_quality
        )

        # Generate signal
        signal = self.strategy_selector.generate_signal(symbol, df, context)

        logger.debug(f"Guardian signal for {symbol}: {signal.signal_type.value} "
                    f"via {signal.strategy_name} (conf: {signal.confidence:.2f})")

        return signal

    def register_position(
        self,
        position_id: str,
        symbol: str,
        side: str,
        entry_price: float,
        stop_price: float,
        target_price: float,
        volume: float,
        regime: str,
        strategy: str,
        quality_score: int,
        **context
    ):
        """
        Register a new position for R-based exit management.
        """
        if not self.enabled:
            return

        # Register with exit engine
        self.exit_engine.register_position(
            position_id=position_id,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            stop_price=stop_price,
            initial_volume=volume
        )

        # Record in journal
        self.trade_journal.record_entry(
            trade_id=position_id,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            volume=volume,
            context={
                "regime": regime,
                "strategy": strategy,
                "quality_score": quality_score,
                **context
            }
        )

        self._session_trade_count += 1

        logger.info(f"Registered Guardian position {position_id}: {side} {volume} {symbol}")

    def update_exit(
        self,
        position_id: str,
        current_price: float,
        timestamp: Optional[datetime] = None
    ) -> ExitRecommendation:
        """
        Update position and get exit recommendation.
        """
        if not self.enabled:
            return ExitRecommendation(
                action=ExitAction.HOLD,
                current_r=0.0,
                profit_pct=0.0,
                volume_to_close=0.0,
                reason="Guardian disabled"
            )

        return self.exit_engine.update_position(
            position_id=position_id,
            current_price=current_price,
            timestamp=timestamp
        )

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_reason: str,
        partial_exits: Optional[List[Dict]] = None
    ):
        """
        Close position and record in journal.
        """
        if not self.enabled:
            return

        # Get state for PnL calculation
        state = self.exit_engine.get_position_state(position_id)
        if state:
            if state.side == "BUY":
                pnl = (exit_price - state.entry_price) * state.initial_volume
            else:
                pnl = (state.entry_price - exit_price) * state.initial_volume

            # Record exit
            self.trade_journal.record_exit(
                trade_id=position_id,
                exit_price=exit_price,
                pnl=pnl,
                exit_reason=exit_reason,
                max_r_reached=state.highest_r_reached,
                partial_exits=partial_exits
            )

            # Record stop loss for quality scoring
            if pnl < 0:
                self.quality_scorer.record_stop_loss(state.symbol)

        # Remove from exit engine
        self.exit_engine.remove_position(position_id)

    def get_insights(self, lookback: int = 50) -> Dict[str, Any]:
        """
        Get learning insights from trade journal.
        """
        if not self.enabled:
            return {"error": "Guardian disabled"}

        insights = self.trade_journal.analyze(lookback)

        return {
            "total_trades": insights.total_trades,
            "win_rate": insights.win_rate,
            "total_pnl": insights.total_pnl,
            "profit_factor": insights.profit_factor,
            "regime_performance": insights.regime_performance,
            "symbol_performance": insights.symbol_performance,
            "strategy_performance": insights.strategy_performance,
            "recommendations": insights.recommendations
        }

    def get_status(self) -> Dict[str, Any]:
        """
        Get current Guardian Trader status.
        """
        if not self.enabled:
            return {"enabled": False}

        exit_summary = self.exit_engine.get_summary()
        journal_stats = self.trade_journal.get_stats()

        return {
            "enabled": True,
            "active_positions": exit_summary["active_positions"],
            "runner_positions": exit_summary["runner_positions"],
            "completed_trades": journal_stats["completed_trades"],
            "total_pnl": journal_stats["total_pnl"],
            "session_trades": self._session_trade_count,
            "exit_config": exit_summary["config"]
        }


def create_guardian_trader(config_path: Optional[str] = None) -> GuardianTraderIntegration:
    """
    Factory function to create Guardian Trader with micro account config.

    Args:
        config_path: Path to config YAML (default: guardian_trader_micro.yaml)

    Returns:
        Configured GuardianTraderIntegration
    """
    import yaml

    if config_path is None:
        config_path = "configs/guardian_trader_micro.yaml"

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Failed to load Guardian config from {config_path}: {e}")
        config = {}

    return GuardianTraderIntegration(config)
