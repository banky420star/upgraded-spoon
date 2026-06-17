"""PaperExecutor — simulated fills with position tracking.

Records every intent and fill to `logs/trade_journal.jsonl`.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Union

from loguru import logger

# Support rich Decision path (additive)
try:
    from Python.execution.trade_decision import TradeDecision
except Exception:
    TradeDecision = None  # type: ignore


_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_JOURNAL_PATH = os.path.join(_PROJECT_ROOT, "logs", "trade_journal.jsonl")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PaperExecutor:
    """Simulated broker execution for paper mode.

    Maintains a simple in-memory position book and writes every action
    to the trade journal.
    """

    def __init__(
        self,
        config: dict | None = None,
        risk_supervisor=None,
    ):
        self.config = config or {}
        self.risk = risk_supervisor
        self._positions: list[dict[str, Any]] = []
        self._ticket_seq = 900_000
        os.makedirs(os.path.dirname(_JOURNAL_PATH), exist_ok=True)

    def _next_ticket(self) -> int:
        self._ticket_seq += 1
        return self._ticket_seq

    def _journal(self, record: dict[str, Any]) -> None:
        record["ts"] = _now_iso()
        record["executor"] = "paper"
        try:
            with open(_JOURNAL_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as exc:
            logger.warning(f"PaperExecutor journal write failed: {exc}")

    def execute(self, intent: Union[dict[str, Any], "TradeDecision"]) -> dict[str, Any]:
        """Simulate execution of a gated trade intent or rich TradeDecision.

        Supports both legacy simple dict and new Decision PPO TradeDecision (via to_dict or direct).
        Zero breakage for simple paths.
        """
        # Normalize rich Decision to dict for legacy handling inside (adapter path)
        if TradeDecision is not None and isinstance(intent, TradeDecision):
            td = intent
            intent = {
                "symbol": td.symbol,
                "side": td.side.value if hasattr(td.side, "value") else str(td.side),
                "size": td.size.value if hasattr(td.size, "value") else float(getattr(td.size, "value", 0.01)),
                "price": 0.0,
                "sl": getattr(td.sl, "price", None) or getattr(td.sl, "value", None),
                "tp": getattr(td.tp, "price", None) or getattr(td.tp, "value", None),
                "magic": td.magic,
                "comment": td.comment,
                "decision_id": td.decision_id,
                "source": td.source,
                "rich": True,
            }
        symbol = intent.get("symbol", "")
        side = str(intent.get("side", "")).upper()
        size = float(intent.get("size", 0.0) or 0.0)
        price = float(intent.get("price", 0.0) or 0.0)
        sl = intent.get("sl")
        tp = intent.get("tp")
        magic = intent.get("magic", 0)
        comment = intent.get("comment", "")

        if size <= 0:
            return {"executed": False, "mode": "paper_sim", "reason": "zero_size"}

        # Record trade with risk supervisor if available
        if self.risk is not None:
            self.risk.record_trade(symbol)

        ticket = self._next_ticket()
        pos = {
            "ticket": ticket,
            "symbol": symbol,
            "side": side,
            "size": size,
            "open_price": price,
            "sl": sl,
            "tp": tp,
            "magic": magic,
            "comment": comment,
            "open_time": time.time(),
        }
        self._positions.append(pos)

        fill_record = {
            "action": "open",
            "symbol": symbol,
            "side": side,
            "size": size,
            "price": price,
            "ticket": ticket,
            "magic": magic,
            "comment": comment,
        }
        self._journal(fill_record)
        logger.info(f"[PAPER] Filled {side} {size} {symbol} @ {price} (ticket={ticket})")

        return {
            "executed": True,
            "mode": "paper_sim",
            "reason": "paper_fill",
            "ticket": ticket,
            "symbol": symbol,
            "side": side,
            "size": size,
            "price": price,
        }

    def get_positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Return open positions, optionally filtered by symbol."""
        positions = self._positions
        if symbol:
            positions = [p for p in positions if p["symbol"] == symbol]
        return positions

    def force_flatten_all(self, reason: str = "paper_executor_flatten") -> dict[str, Any]:
        """Emergency flatten for rollback / supervisor (DecisionPPO + legacy paths)."""
        count = len(self._positions)
        for pos in list(self._positions):
            try:
                self._journal({"action": "force_flatten", "ticket": pos["ticket"], "symbol": pos["symbol"], "reason": reason})
            except Exception:
                pass
        self._positions.clear()
        logger.warning(f"[PAPER-EXEC] force_flatten_all: {count} positions closed ({reason})")
        return {"executed": True, "closed": count, "reason": reason}

    def close_position(self, ticket: int, price: float) -> dict[str, Any]:
        """Close a paper position by ticket."""
        for i, pos in enumerate(self._positions):
            if pos["ticket"] == ticket:
                pnl = 0.0
                side = pos["side"]
                size = pos["size"]
                open_price = pos["open_price"]
                if side == "BUY":
                    pnl = (price - open_price) * size * 100_000
                else:
                    pnl = (open_price - price) * size * 100_000

                self._positions.pop(i)
                self._journal(
                    {
                        "action": "close",
                        "symbol": pos["symbol"],
                        "ticket": ticket,
                        "price": price,
                        "pnl": pnl,
                    }
                )
                if self.risk is not None:
                    self.risk.record_trade_result(pos["symbol"], pnl)

                logger.info(
                    f"[PAPER] Closed #{ticket} {pos['symbol']} @ {price} PnL={pnl:.2f}"
                )
                return {
                    "executed": True,
                    "mode": "paper_sim",
                    "reason": "paper_close",
                    "ticket": ticket,
                    "pnl": pnl,
                }

        return {"executed": False, "mode": "paper_sim", "reason": "ticket_not_found"}

    def force_flatten_all(self, reason: str = "supervisor_trigger") -> dict[str, Any]:
        """Force close all open paper positions (for rollback / safety). Compatible with new execution layer."""
        closed = 0
        total_pnl = 0.0
        for pos in list(self._positions):
            # Use mid price 0 for sim close (journal will note)
            close_res = self.close_position(pos["ticket"], pos.get("open_price", 0.0))
            if close_res.get("executed"):
                closed += 1
                total_pnl += close_res.get("pnl", 0.0)
        self._journal({
            "action": "force_flatten_all",
            "reason": reason,
            "closed_count": closed,
            "total_pnl": total_pnl,
        })
        logger.warning(f"[PAPER] force_flatten_all: closed {closed} positions (reason={reason})")
        return {"executed": True, "mode": "paper_sim", "closed": closed, "pnl": total_pnl, "reason": reason}
