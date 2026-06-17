"""
ReplayBuilder — constructs replay datasets for retraining from
closed trades, blocked trades, and their original context.

Output format: Parquet (preferred) or CSV under data/replay/
Schema includes original features, model votes, regime, Dreamer rollouts,
outcome labels and mistake labels.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


class ReplayBuilder:
    """
    Accumulates rows from the live / demo loop and flushes them to disk
    as replay datasets ready for model retraining.
    """

    def __init__(
        self,
        data_dir: str = "data/replay",
        format: str = "parquet",
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.format = format.lower()
        self._buffer: List[Dict[str, Any]] = []

    def add_closed_trade(
        self,
        trade: Dict[str, Any],
        original_features: Optional[Dict[str, Any]] = None,
        model_votes: Optional[Dict[str, float]] = None,
        regime: Optional[str] = None,
        dreamer_rollout_summary: Optional[Dict[str, Any]] = None,
        outcome_label: Optional[str] = None,
        mistake_label: Optional[str] = None,
        blocked: bool = False,
    ) -> None:
        """
        Append a row to the in-memory replay buffer.

        Args:
            trade: trade dict (closed or blocked).
            original_features: feature vector at decision time.
            model_votes: {ppo: 0.6, dreamer: 0.3, ...}.
            regime: rainforest regime label.
            dreamer_rollout_summary: {expected_return, imagined_risk, value_estimate}.
            outcome_label: from OutcomeLabeler.
            mistake_label: from OutcomeLabeler.
            blocked: True if the trade was blocked by risk/guardian.
        """
        row: Dict[str, Any] = {
            "replay_id": f"replay_{uuid.uuid4().hex[:8]}",
            "trade_id": trade.get("trade_id"),
            "symbol": trade.get("symbol"),
            "timeframe": trade.get("timeframe"),
            "side": trade.get("side"),
            "entry_time": trade.get("entry_time"),
            "exit_time": trade.get("exit_time"),
            "entry_price": float(trade.get("entry_price", 0.0)),
            "exit_price": float(trade.get("exit_price", 0.0)),
            "volume": float(trade.get("volume", 0.0)),
            "pnl": float(trade.get("pnl", 0.0)),
            "pnl_pct": float(trade.get("pnl_pct", 0.0)),
            "mfe": float(trade.get("mfe", 0.0)),
            "mae": float(trade.get("mae", 0.0)),
            "spread_paid": float(trade.get("spread_paid", 0.0)),
            "slippage": float(trade.get("slippage", 0.0)),
            "fees": float(trade.get("fees", 0.0)),
            "exit_reason": trade.get("exit_reason"),
            "outcome_label": outcome_label or "flat_noise",
            "mistake_label": mistake_label or "none",
            "blocked": blocked,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if original_features:
            for k, v in original_features.items():
                row[f"feat_{k}"] = v

        if model_votes:
            row["vote_ppo"] = model_votes.get("ppo", np.nan)
            row["vote_dreamer"] = model_votes.get("dreamer", np.nan)
            row["vote_ensemble"] = model_votes.get("ensemble", np.nan)

        if regime:
            row["regime"] = regime

        if dreamer_rollout_summary:
            row["dreamer_expected_return"] = dreamer_rollout_summary.get("expected_return", np.nan)
            row["dreamer_imagined_risk"] = dreamer_rollout_summary.get("imagined_risk", np.nan)
            row["dreamer_value_estimate"] = dreamer_rollout_summary.get("value_estimate", np.nan)

        self._buffer.append(row)

    def flush(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> Path:
        """
        Write buffered rows to disk and clear the buffer.

        Returns path to written file.
        """
        if not self._buffer:
            logger.debug("ReplayBuilder flush called with empty buffer")
            return Path()

        df = pd.DataFrame(self._buffer)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        sym = symbol or str(df["symbol"].iloc[0]) if "symbol" in df.columns else "UNKNOWN"
        tf = timeframe or str(df["timeframe"].iloc[0]) if "timeframe" in df.columns else "M15"

        if self.format == "parquet":
            path = self.data_dir / f"{sym}_{tf}_{ts}.parquet"
            df.to_parquet(path, index=False)
        else:
            path = self.data_dir / f"{sym}_{tf}_{ts}.csv"
            df.to_csv(path, index=False)

        logger.info(f"ReplayBuilder flushed {len(df)} rows to {path}")
        self._buffer.clear()
        return path

    def load_recent(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        n_files: int = 5,
    ) -> pd.DataFrame:
        """
        Load the N most recent replay files and concatenate.
        """
        files = sorted(self.data_dir.glob(f"{symbol or '*'}_{timeframe or '*'}_*.*"), reverse=True)
        files = files[:n_files]
        if not files:
            return pd.DataFrame()

        frames: List[pd.DataFrame] = []
        for f in files:
            try:
                if f.suffix == ".parquet":
                    frames.append(pd.read_parquet(f))
                elif f.suffix == ".csv":
                    frames.append(pd.read_csv(f))
            except Exception as e:
                logger.warning(f"Failed to load replay file {f}: {e}")

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
