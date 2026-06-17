"""
TradeJournal — append-only JSONL trade ledger for the feedback loop.

Records every trade with full lineage (decision_id, intent_id, bundle_id)
and post-trade metrics (mfe, mae, outcome_label, mistake_label).
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
from loguru import logger


# Valid outcome labels used by the feedback / coroner pipeline
VALID_OUTCOME_LABELS: set[str] = {
    "winner_clean",
    "winner_lucky",
    "loser_expected",
    "loser_bad_entry",
    "loser_bad_exit",
    "loser_spread",
    "loser_regime_shift",
    "loser_news_spike",
    "loser_overfit_signal",
    "loser_execution_slippage",
    "flat_noise",
}

VALID_MISTAKE_LABELS: set[str] = {
    "none",
    "bad_entry_timing",
    "bad_exit_timing",
    "ignored_spread",
    "ignored_slippage",
    "overfit_signal",
    "regime_miss",
    "news_miss",
    "risk_oversize",
    "execution_error",
    "model_miscalibration",
}


class TradeJournal:
    """
    Lightweight trade journal that appends every trade to a JSONL file.
    """

    def __init__(
        self,
        log_path: str = "logs/trade_journal.jsonl",
    ):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def record_trade(self, trade: Dict[str, Any]) -> str:
        """
        Validate and append a trade record.

        Required fields:
          trade_id, decision_id, intent_id, bundle_id, symbol, timeframe,
          execution_mode, account_type, side, entry_time, exit_time,
          entry_price, exit_price, volume, spread_paid, slippage, fees,
          pnl, pnl_pct, mfe, mae, exit_reason, outcome_label, mistake_label
        """
        trade_id = str(trade.get("trade_id", f"trade_{uuid.uuid4().hex[:8]}"))
        trade["trade_id"] = trade_id

        # Normalise timestamps to ISO strings
        for key in ("entry_time", "exit_time"):
            if key in trade and isinstance(trade[key], datetime):
                trade[key] = trade[key].isoformat()

        # Validate labels
        outcome = trade.get("outcome_label", "")
        if outcome and outcome not in VALID_OUTCOME_LABELS:
            logger.warning(f"Trade {trade_id} has unexpected outcome_label '{outcome}'")

        mistake = trade.get("mistake_label", "")
        if mistake and mistake not in VALID_MISTAKE_LABELS:
            logger.warning(f"Trade {trade_id} has unexpected mistake_label '{mistake}'")

        # Enforce numeric fields
        for num_key in (
            "entry_price",
            "exit_price",
            "volume",
            "spread_paid",
            "slippage",
            "fees",
            "pnl",
            "pnl_pct",
            "mfe",
            "mae",
        ):
            if num_key in trade:
                trade[num_key] = float(trade[num_key])

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(trade, default=str) + "\n")

        logger.debug(f"TradeJournal recorded {trade_id}")
        return trade_id

    def load_trades(self, n: Optional[int] = None) -> pd.DataFrame:
        """Load trades from JSONL into a DataFrame."""
        if not self.log_path.exists():
            return pd.DataFrame()

        records: List[Dict[str, Any]] = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if n is not None:
            records = records[-n:]

        return pd.DataFrame(records)

    def get_trade(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """Scan JSONL for a single trade_id (slow — use for debugging)."""
        if not self.log_path.exists():
            return None
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("trade_id") == trade_id:
                        return rec
                except json.JSONDecodeError:
                    continue
        return None

    def count_by_outcome(self) -> Dict[str, int]:
        """Quick tally of outcome labels."""
        df = self.load_trades()
        if df.empty or "outcome_label" not in df.columns:
            return {}
        return df["outcome_label"].value_counts().to_dict()
