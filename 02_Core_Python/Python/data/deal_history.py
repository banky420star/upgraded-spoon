"""
DealHistory — Record closed deals from MT5.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger

from Python.mt5_compat import mt5, MT5_AVAILABLE

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DealHistory:
    """Record closed deals from MT5 history_deals_get."""

    def __init__(self, project_root: str | None = None):
        self.project_root = project_root or _PROJECT_ROOT
        self.raw_dir = os.path.join(self.project_root, "data", "raw", "mt5", "deals")
        os.makedirs(self.raw_dir, exist_ok=True)

    def fetch(self, days_back: int = 7) -> list[dict[str, Any]]:
        """Fetch closed deals from MT5."""
        if not MT5_AVAILABLE:
            logger.debug("MT5 unavailable — returning empty deals")
            return []
        try:
            if not mt5.initialize():
                logger.debug("MT5 initialize failed — returning empty deals")
                return []
        except Exception as exc:
            logger.debug(f"MT5 initialize error: {exc}")
            return []
        try:
            since = datetime.now(timezone.utc) - timedelta(days=days_back)
            until = datetime.now(timezone.utc)
            deals = mt5.history_deals_get(since, until)
        except Exception as exc:
            logger.warning(f"MT5 history_deals_get error: {exc}")
            return []
        if deals is None or len(deals) == 0:
            return []
        now = datetime.now(timezone.utc).isoformat()
        records = []
        for d in deals:
            ts = datetime.fromtimestamp(int(getattr(d, "time", 0)), tz=timezone.utc).isoformat()
            entry = int(getattr(d, "entry", 0))
            records.append({
                "ticket": int(getattr(d, "ticket", 0)),
                "order": int(getattr(d, "order", 0)),
                "position_id": int(getattr(d, "position_id", 0)),
                "symbol": getattr(d, "symbol", ""),
                "type": "BUY" if getattr(d, "type", 0) == 0 else "SELL",
                "entry": entry,
                "reason": int(getattr(d, "reason", 0)),
                "time": ts,
                "volume": float(getattr(d, "volume", 0) or 0),
                "price": float(getattr(d, "price", 0) or 0),
                "profit": float(getattr(d, "profit", 0) or 0),
                "commission": float(getattr(d, "commission", 0) or 0),
                "swap": float(getattr(d, "swap", 0) or 0),
                "fee": float(getattr(d, "fee", 0) or 0),
                "comment": getattr(d, "comment", ""),
                "raw_ingested_at": now,
            })
        # Pair entry/exit prices for closing deals
        entry_prices: dict[int, float] = {}
        for r in records:
            if r["entry"] == 0 and r["position_id"]:
                entry_prices[r["position_id"]] = r["price"]
        for r in records:
            if r["entry"] == 1 and r["position_id"] in entry_prices:
                r["entry_price"] = entry_prices[r["position_id"]]
                r["exit_price"] = r["price"]
        return records

    def store(self, deals: list[dict[str, Any]]) -> str:
        path = self._today_path()
        with open(path, "a", encoding="utf-8") as f:
            for d in deals:
                f.write(json.dumps(d, default=str) + "\n")
        return path

    def ingest(self, days_back: int = 7) -> list[dict[str, Any]]:
        deals = self.fetch(days_back)
        if deals:
            self.store(deals)
        return deals

    def _today_path(self) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        return os.path.join(self.raw_dir, f"{stamp}.jsonl")
