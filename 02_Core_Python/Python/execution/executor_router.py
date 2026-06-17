"""ExecutorRouter — routes trade intents to the correct executor.

  paper_sim   → PaperExecutor
  demo_live   → MT5DemoExecutor
  real_live*  → MT5DemoExecutor (locked) or raw MT5Executor (unlocked)

No raw orders are emitted from the PPO policy; all intents pass through
GateEngine before reaching an executor.
"""

from __future__ import annotations

from typing import Any, Union

from Python.execution.mode_resolver import resolve_mode
from Python.execution.paper_executor import PaperExecutor
from Python.execution.mt5_demo_executor import MT5DemoExecutor

# Rich support
try:
    from Python.execution.trade_decision import TradeDecision
except Exception:
    TradeDecision = None  # type: ignore
from Python.execution.trade_decision import TradeDecision


class ExecutorRouter:
    """Routes trade intents based on the current execution mode."""

    def __init__(
        self,
        config: dict | None = None,
        risk_supervisor=None,
        mt5_executor=None,
    ):
        self.config = config or {}
        self.mode = resolve_mode(self.config)
        self.risk = risk_supervisor

        if self.mode == "paper_sim":
            self._executor = PaperExecutor(
                config=self.config,
                risk_supervisor=risk_supervisor,
            )
        elif self.mode in ("demo_live", "real_live_locked", "real_live"):
            # Demo and locked-real both route through the guarded demo executor.
            # Only an explicit, fully-gated real_live path could reach the raw
            # MT5Executor, and that is blocked here by design.
            self._executor = MT5DemoExecutor(
                config=self.config,
                risk_supervisor=risk_supervisor,
                mt5_executor=mt5_executor,
            )
        else:
            # Safe fallback
            self._executor = PaperExecutor(
                config=self.config,
                risk_supervisor=risk_supervisor,
            )

    def submit(self, intent: dict[str, Any] | TradeDecision) -> dict[str, Any]:
        """Send a gated trade intent (or rich TradeDecision) to the active executor.

        Accepts legacy dicts or new TradeDecision (auto-normalized for compatibility).
        Returns an execution metadata dict with at least:
          { "executed": bool, "mode": str, "reason": str }
        """
        if isinstance(intent, TradeDecision):
            # Convert rich decision to minimal intent the old executors understand.
            # Full rich logic lives in ExecutionAgent; router remains the low-level primitive.
            side = "BUY" if intent.side.value == "LONG" else ("SELL" if intent.side.value == "SHORT" else "FLAT")
            size = intent.size.value if intent.size.mode.value == "fixed_lots" else 0.01
            intent = {
                "symbol": intent.symbol,
                "side": side,
                "size": size,
                "comment": f"from_TradeDecision:{intent.decision_id}",
                "magic": intent.magic,
            }
        return self._executor.execute(intent)

    def get_positions(self, symbol: str | None = None) -> list[Any]:
        """Return open positions from the active executor."""
        return self._executor.get_positions(symbol)

    def force_flatten_all(self, reason: str = "router_force_flatten") -> dict:
        """Delegate emergency flatten to active executor (supports DecisionPPO rollback path + legacy)."""
        try:
            if hasattr(self._executor, "force_flatten_all"):
                return self._executor.force_flatten_all(reason)  # type: ignore
            # Fallback: close all known positions via executor API
            for p in self._executor.get_positions() or []:
                try:
                    self._executor.close_position(p.get("ticket"), p.get("open_price", 0.0))
                except Exception:
                    pass
            return {"executed": True, "fallback": True, "reason": reason}
        except Exception as e:
            return {"executed": False, "error": str(e), "reason": reason}

    @property
    def active_executor_name(self) -> str:
        return type(self._executor).__name__

    def force_flatten_all(self, reason: str = "supervisor_trigger") -> dict[str, Any]:
        """Delegate force flatten (rollback safety) to active executor. Works for both paper and demo MT5 paths."""
        if hasattr(self._executor, "force_flatten_all"):
            return self._executor.force_flatten_all(reason)
        # Fallback for executors without it (e.g. legacy)
        logger = __import__("loguru", fromlist=["logger"]).logger
        logger.warning("Executor has no force_flatten_all; using get+close best effort")
        positions = self.get_positions()
        closed = 0
        for p in positions:
            if hasattr(self._executor, "close_position"):
                self._executor.close_position(p.get("ticket", 0), p.get("open_price", 0.0))
                closed += 1
        return {"executed": True, "mode": self.mode, "closed": closed, "reason": reason, "fallback": True}
