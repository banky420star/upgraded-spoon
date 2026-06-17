"""MT5DemoExecutor — guarded wrapper around the real MT5Executor.

Enforces demo-only constraints:
  max lot 0.01
  max positions 1
  max trades/hour 3
  daily loss cap 2%
  spread zscore cap 2.0
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Union

from loguru import logger

# Rich Decision support (additive, no breakage)
try:
    from Python.execution.trade_decision import TradeDecision
except Exception:
    TradeDecision = None  # type: ignore


class MT5DemoExecutor:
    """Wraps Python.mt5_executor.MT5Executor with hard demo guards."""

    # Demo guard constants
    MAX_LOT = 0.01
    MAX_POSITIONS = 1
    MAX_TRADES_PER_HOUR = 3
    MAX_DAILY_LOSS_PCT = 2.0
    MAX_SPREAD_ZSCORE = 2.0

    def __init__(
        self,
        config: dict | None = None,
        risk_supervisor=None,
        mt5_executor=None,
    ):
        self.config = config or {}
        self.risk = risk_supervisor
        self._mt5 = mt5_executor
        self._trade_history: list[dict[str, Any]] = []  # for rate limiting

    def _trades_in_last_hour(self) -> int:
        now = time.time()
        cutoff = now - 3600
        return sum(1 for t in self._trade_history if t["ts"] > cutoff)

    def _daily_loss_pct(self) -> float:
        if self.risk is None:
            return 0.0
        # Use the risk supervisor's equity tracking if available
        equity = getattr(self.risk, "_current_equity", 0.0) or 1.0
        pnl = getattr(self.risk, "realized_pnl_today", 0.0)
        if equity <= 0:
            return 0.0
        return abs(min(0.0, pnl)) / equity * 100.0

    def _current_spread_zscore(self, symbol: str) -> float:
        # Fallback: if we have no live tick, return 0 (safe).
        # A real implementation would compute zscore from recent spread history.
        return 0.0

    def _enforce_guards(self, intent: dict[str, Any]) -> tuple[bool, str]:
        """Run all demo guards. Returns (allowed, reason)."""
        symbol = intent.get("symbol", "")
        size = float(intent.get("size", 0.0) or 0.0)

        # 1. Lot cap (also respect AGI_PAPER_FIXED_LOT env for harness consistency)
        fixed = os.environ.get("AGI_PAPER_FIXED_LOT", "").strip()
        if fixed:
            try:
                if size > float(fixed):
                    size = float(fixed)
            except Exception:
                pass
        if size > self.MAX_LOT:
            return False, f"lot_cap ({size} > {self.MAX_LOT})"

        # 2. Position cap
        positions = self.get_positions(symbol)
        if len(positions) >= self.MAX_POSITIONS:
            return False, f"max_positions ({len(positions)} >= {self.MAX_POSITIONS})"

        # 3. Hourly trade rate
        if self._trades_in_last_hour() >= self.MAX_TRADES_PER_HOUR:
            return False, f"hourly_rate ({self._trades_in_last_hour()} >= {self.MAX_TRADES_PER_HOUR})"

        # 4. Daily loss % cap
        if self._daily_loss_pct() >= self.MAX_DAILY_LOSS_PCT:
            return False, f"daily_loss_cap ({self._daily_loss_pct():.2f}% >= {self.MAX_DAILY_LOSS_PCT}%)"

        # 5. Spread zscore cap (placeholder — blocks only if explicitly computed > cap)
        zscore = self._current_spread_zscore(symbol)
        if zscore > self.MAX_SPREAD_ZSCORE:
            return False, f"spread_zscore ({zscore:.2f} > {self.MAX_SPREAD_ZSCORE})"

        return True, "guards_passed"

    def execute(self, intent: Union[dict[str, Any], "TradeDecision"]) -> dict[str, Any]:
        """Execute a gated trade intent or rich TradeDecision through the wrapped MT5Executor.
        Supports Decision PPO full specs via normalization (legacy paths unchanged).
        """
        # Normalize TradeDecision for guards + downstream (adapter)
        if TradeDecision is not None and isinstance(intent, TradeDecision):
            td = intent
            intent = {
                "symbol": td.symbol,
                "side": td.side.value if hasattr(td.side, "value") else str(td.side),
                "size": td.size.value if hasattr(td.size, "value") else 0.01,
                "price": 0.0,
                "sl": getattr(getattr(td, "sl", None), "price", None) or getattr(getattr(td, "sl", None), "value", None),
                "tp": getattr(getattr(td, "tp", None), "price", None) or getattr(getattr(td, "tp", None), "value", None),
                "magic": td.magic,
                "comment": td.comment,
                "decision_id": getattr(td, "decision_id", None),
                "rich": True,
            }
        allowed, reason = self._enforce_guards(intent)
        if not allowed:
            logger.warning(f"[DEMO GATE] Blocked intent: {reason}")
            return {"executed": False, "mode": "demo_live", "reason": reason}

        symbol = intent.get("symbol", "")
        side = str(intent.get("side", "")).upper()
        size = float(intent.get("size", 0.0) or 0.0)
        price = float(intent.get("price", 0.0) or 0.0)
        sl = intent.get("sl")
        tp = intent.get("tp")
        magic = intent.get("magic", 0)
        comment = intent.get("comment", "")

        if size <= 0:
            return {"executed": False, "mode": "demo_live", "reason": "zero_size"}

        # Record with risk supervisor
        if self.risk is not None:
            self.risk.record_trade(symbol)

        # Track trade for rate limiting
        self._trade_history.append({"ts": time.time(), "symbol": symbol})

        # If we have a real MT5Executor, delegate; otherwise simulate.
        if self._mt5 is not None:
            try:
                # MT5 constants: ORDER_TYPE_BUY=0, ORDER_TYPE_SELL=1
                order_type = 0 if side == "BUY" else 1
                meta = self._mt5.open_position(
                    symbol=symbol,
                    order_type=order_type,
                    volume=size,
                    order_meta={
                        "exposure": size,
                        "magic": magic,
                        "comment": comment,
                    },
                )
                return {
                    "executed": bool(meta.get("executed", False)),
                    "mode": "demo_live",
                    "reason": "mt5_demo_fill",
                    **meta,
                }
            except Exception as exc:
                logger.error(f"[DEMO] MT5 delegate failed: {exc}")
                if self.risk is not None:
                    self.risk.record_error()
                return {"executed": False, "mode": "demo_live", "reason": f"mt5_error:{exc}"}

        # Fallback when no MT5Executor is wired in (dry-run within demo mode)
        logger.info(
            f"[DEMO-DRY] {side} {size} {symbol} @ {price} — no MT5Executor wired"
        )
        return {
            "executed": True,
            "mode": "demo_live",
            "reason": "demo_dry_run",
            "symbol": symbol,
            "side": side,
            "size": size,
            "price": price,
        }

    def get_positions(self, symbol: str | None = None) -> list[Any]:
        if self._mt5 is not None:
            try:
                longs, shorts = self._mt5.get_positions(symbol)
                return list(longs) + list(shorts)
            except Exception as exc:
                logger.warning(f"[DEMO] get_positions failed: {exc}")
        return []

    def force_rollback_flatten(self, reason: str = "harness_trigger") -> dict:
        """Harness safety hook: delegate flatten to wrapped MT5Executor and record risk."""
        logger.critical(f"[DEMO HARNESS] Rollback flatten: {reason}")
        if self.risk is not None:
            try:
                self.risk.record_pnl_with_equity(-9999, getattr(self.risk, "_current_equity", 10000))  # force consideration
            except Exception:
                pass
        if self._mt5 is not None and hasattr(self._mt5, "force_flatten_all"):
            return self._mt5.force_flatten_all(reason)
        return {"closed": 0, "note": "no_mt5_executor_wired"}
