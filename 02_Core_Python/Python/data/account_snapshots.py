"""
AccountSnapshot — Record MT5 account telemetry at regular intervals.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from Python.mt5_compat import mt5, MT5_AVAILABLE

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AccountSnapshot:
    """Record balance, equity, margin, margin_free, leverage, currency, server, login."""

    def __init__(self, project_root: str | None = None):
        self.project_root = project_root or _PROJECT_ROOT
        self.raw_dir = os.path.join(self.project_root, "data", "raw", "mt5", "account")
        os.makedirs(self.raw_dir, exist_ok=True)

    def fetch(self) -> dict[str, Any] | None:
        """Fetch current account snapshot from MT5."""
        if not MT5_AVAILABLE:
            logger.debug("MT5 unavailable — skipping account snapshot")
            return None
        try:
            if not mt5.initialize():
                logger.debug("MT5 initialize failed — skipping account snapshot")
                return None
        except Exception as exc:
            logger.debug(f"MT5 initialize error: {exc}")
            return None
        try:
            ai = mt5.account_info()
        except Exception as exc:
            logger.debug(f"MT5 account_info error: {exc}")
            return None
        if ai is None:
            return None
        now = datetime.now(timezone.utc).isoformat()
        return {
            "balance": float(getattr(ai, "balance", 0) or 0),
            "equity": float(getattr(ai, "equity", 0) or 0),
            "margin": float(getattr(ai, "margin", 0) or 0),
            "margin_free": float(getattr(ai, "margin_free", 0) or 0),
            "leverage": int(getattr(ai, "leverage", 0) or 0),
            "currency": getattr(ai, "currency", ""),
            "server": getattr(ai, "server", ""),
            "login": int(getattr(ai, "login", 0) or 0),
            "snapshot_at": now,
        }

    def record(self, snapshot: dict[str, Any] | None = None) -> str | None:
        """Persist snapshot to append-only JSONL."""
        snap = snapshot or self.fetch()
        if snap is None:
            return None
        path = self._today_path()
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(snap, default=str) + "\n")
        return path

    def _today_path(self) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        return os.path.join(self.raw_dir, f"{stamp}.jsonl")
