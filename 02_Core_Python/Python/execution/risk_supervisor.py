"""Execution-layer RiskSupervisor — thin wrapper around the canonical RiskEngine.

This reduces duplication with the top-level RiskEngine (Python/risk_engine.py)
while adding a few execution-specific fields and the can_trade() helper used
by the executor layer.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

# Delegate the core daily risk counters / halt logic to the single canonical implementation
from Python.risk_engine import RiskEngine as _CanonicalRiskEngine


class RiskSupervisor:
    """Execution companion that delegates most behavior to canonical RiskEngine.

    Only the extra fields needed by the execution path (max open positions,
    drawdown guard, can_trade with position limits) are added here.
    """

    def __init__(self, config: dict | None = None):
        self._engine = _CanonicalRiskEngine()

        # Execution-specific configuration only
        risk_cfg = (config or {}).get("risk", {}) if isinstance(config, dict) else {}
        self.max_open_positions = int(risk_cfg.get("max_open_positions", 8))
        self.max_positions_per_symbol = int(risk_cfg.get("max_positions_per_symbol", 2))
        self.max_drawdown_pct = float(risk_cfg.get("max_drawdown_pct", 8.0))

        # Mirror commonly accessed attributes for backward compatibility
        self.max_daily_loss = self._engine.max_daily_loss
        self.max_daily_trades = self._engine.max_daily_trades
        self.max_daily_trades_per_symbol = self._engine.max_daily_trades_per_symbol
        self.max_daily_losing_trades_per_symbol = self._engine.max_daily_losing_trades_per_symbol
        self.max_lots = self._engine.max_lots
        self.default_symbol_profile = self._engine.default_symbol_profile
        self.symbol_profiles = self._engine.symbol_profiles

        # Lightweight mirrors of mutable state (updated in delegation methods)
        self.realized_pnl_today = self._engine.realized_pnl_today
        self.daily_trades = self._engine.daily_trades
        self.daily_trades_by_symbol = self._engine.daily_trades_by_symbol
        self.daily_losing_trades_by_symbol = self._engine.daily_losing_trades_by_symbol
        self.halt = self._engine.halt
        self.error_halt = self._engine.error_halt
        self.error_count = self._engine.error_count
        self.current_dd = self._engine.current_dd
        self.peak_equity = self._engine.peak_equity
        self._current_equity = 0.0
        self._halt_reason = getattr(self._engine, "_halt_reason", "")
        self.last_reset_day = self._engine.last_reset_day

    # --- Delegation to canonical engine (eliminates duplicated logic) ---

    def reset_daily(self) -> None:
        self._engine.reset_daily()
        self._sync_from_engine()

    def maybe_roll_day(self) -> None:
        self._engine.maybe_roll_day()
        self._sync_from_engine()

    def record_trade(self, symbol: str | None = None) -> None:
        self._engine.record_trade(symbol)
        self._sync_from_engine()

    def record_pnl(self, pnl: float) -> None:
        self._engine.record_pnl(pnl)
        self._sync_from_engine()

    def record_pnl_with_equity(self, pnl: float, equity: float | None = None) -> bool:
        triggered = self._engine.record_pnl_with_equity(pnl, equity)
        self._sync_from_engine()
        return triggered

    def trigger_rollback(self, reason: str = "harness") -> None:
        # Mirror to engine halt
        self.halt = True
        self._halt_reason = f"rollback:{reason}"
        if hasattr(self._engine, "_halt_reason"):
            self._engine._halt_reason = self._halt_reason

    def record_trade_result(self, symbol: str | None, pnl: float) -> None:
        self._engine.record_trade_result(symbol, pnl)
        self._sync_from_engine()

    def update_equity(self, equity: float) -> None:
        self._engine.update_equity(equity)
        self._sync_from_engine()
        self._current_equity = float(equity)

    def record_error(self) -> None:
        self._engine.record_error()
        self._sync_from_engine()

    def can_trade(self, symbol: str | None = None) -> bool:
        self.maybe_roll_day()
        if self.halt:
            logger.debug(f"RiskSupervisor: trade blocked — halt ({self._halt_reason})")
            return False
        if self.daily_trades >= self.max_daily_trades:
            logger.debug(
                f"RiskSupervisor: trade blocked — daily trade limit "
                f"({self.daily_trades}/{self.max_daily_trades})"
            )
            return False
        if symbol:
            key = str(symbol)
            if int(self.daily_trades_by_symbol.get(key, 0)) >= self.max_daily_trades_per_symbol:
                return False
            if int(self.daily_losing_trades_by_symbol.get(key, 0)) >= self.max_daily_losing_trades_per_symbol:
                return False
        return True

    # Delegate timing-aware daily loss for rich Decision + TimeExitSpec (production hardening)
    def is_high_impact_news_window(self, symbol: str = None) -> bool:
        return self._engine.is_high_impact_news_window(symbol) if hasattr(self._engine, "is_high_impact_news_window") else False

    def should_respect_time_exit_for_loss_limit(self, active_time_exits: list = None) -> bool:
        return self._engine.should_respect_time_exit_for_loss_limit(active_time_exits) if hasattr(self._engine, "should_respect_time_exit_for_loss_limit") else False

    def record_pnl_with_equity_timing_aware(self, pnl: float, equity: float | None = None, active_time_exits: list = None) -> bool:
        return self._engine.record_pnl_with_equity_timing_aware(pnl, equity, active_time_exits) if hasattr(self._engine, "record_pnl_with_equity_timing_aware") else self.record_pnl_with_equity(pnl, equity)

    def get_symbol_profile(self, symbol: str) -> dict:
        return self._engine.get_symbol_profile(symbol) if hasattr(self._engine, "get_symbol_profile") else self.default_symbol_profile.copy()

    # --- Internal ---

    def _sync_from_engine(self) -> None:
        """Keep our lightweight mirrors in sync after delegating to the engine."""
        self.realized_pnl_today = self._engine.realized_pnl_today
        self.daily_trades = self._engine.daily_trades
        self.daily_trades_by_symbol = self._engine.daily_trades_by_symbol
        self.daily_losing_trades_by_symbol = self._engine.daily_losing_trades_by_symbol
        self.halt = self._engine.halt
        self.error_halt = self._engine.error_halt
        self.error_count = self._engine.error_count
        self.current_dd = self._engine.current_dd
        self.peak_equity = self._engine.peak_equity
        self.last_reset_day = self._engine.last_reset_day
        if hasattr(self._engine, "_halt_reason"):
            self._halt_reason = self._engine._halt_reason
        # expose for harness
        self._halt_reason = getattr(self._engine, "_halt_reason", self._halt_reason)
