"""
Provenance — Dataset lineage and approval tracking.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Provenance:
    dataset_id: str = ""
    symbol: str = ""
    timeframe: str = ""
    source: str = ""
    broker: str = ""
    start: str = ""
    end: str = ""
    rows: int = 0
    missing_candles: int = 0
    duplicate_timestamps: int = 0
    spread_included: bool = False
    commission_model: str = ""
    slippage_model: str = ""
    timezone_checked: bool = False
    leakage_checked: bool = False
    approved_for_training: bool = False
    approved_for_champion_training: bool = False
    dataset_hash: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    meta: dict[str, Any] = field(default_factory=dict)

    def compute_hash(
        self,
        df=None,
        records: list[dict] | None = None,
        file_path: str | None = None,
    ) -> str:
        """Compute SHA-256 hash of dataset content."""
        hasher = hashlib.sha256()
        if df is not None and not df.empty:
            try:
                import pandas as pd
                hasher.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
            except Exception:
                hasher.update(str(df.to_dict()).encode())
        elif records:
            for r in sorted(records, key=lambda x: json.dumps(x, sort_keys=True)):
                hasher.update(json.dumps(r, sort_keys=True, default=str).encode())
        elif file_path and os.path.exists(file_path):
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
        self.dataset_hash = hasher.hexdigest()
        return self.dataset_hash

    def approve_for_training(
        self,
        min_rows: int = 1000,
        require_spread: bool = True,
        require_timezone: bool = True,
        require_leakage: bool = True,
        max_missing_ratio: float = 0.05,
        max_dupe_ratio: float = 0.01,
    ) -> bool:
        """Approve dataset for standard training if criteria met."""
        if self.rows < min_rows:
            return False
        if require_spread and not self.spread_included:
            return False
        if require_timezone and not self.timezone_checked:
            return False
        if require_leakage and not self.leakage_checked:
            return False
        if self.rows and (self.missing_candles / self.rows) > max_missing_ratio:
            return False
        if self.rows and (self.duplicate_timestamps / self.rows) > max_dupe_ratio:
            return False
        self.approved_for_training = True
        return True

    def approve_for_champion_training(
        self,
        min_rows: int = 5000,
        max_missing_ratio: float = 0.01,
        max_dupe_ratio: float = 0.005,
    ) -> bool:
        """Approve dataset for champion (production-grade) training."""
        if self.rows < min_rows:
            return False
        if not self.approved_for_training:
            return False
        if self.rows and (self.missing_candles / self.rows) > max_missing_ratio:
            return False
        if self.rows and (self.duplicate_timestamps / self.rows) > max_dupe_ratio:
            return False
        self.approved_for_champion_training = True
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "source": self.source,
            "broker": self.broker,
            "start": self.start,
            "end": self.end,
            "rows": self.rows,
            "missing_candles": self.missing_candles,
            "duplicate_timestamps": self.duplicate_timestamps,
            "spread_included": self.spread_included,
            "commission_model": self.commission_model,
            "slippage_model": self.slippage_model,
            "timezone_checked": self.timezone_checked,
            "leakage_checked": self.leakage_checked,
            "approved_for_training": self.approved_for_training,
            "approved_for_champion_training": self.approved_for_champion_training,
            "dataset_hash": self.dataset_hash,
            "created_at": self.created_at,
            "meta": self.meta,
        }
