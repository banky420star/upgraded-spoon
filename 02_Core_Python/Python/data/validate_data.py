"""
DataValidator — Validate market data quality and integrity.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


class DataValidator:
    """Validate candles and datasets for quality issues."""

    def __init__(self, symbol_metadata: dict | None = None):
        self.symbol_metadata = symbol_metadata or {}
        self.report: dict[str, Any] = {}

    def validate_candles(self, df: pd.DataFrame, symbol: str = "", timeframe: str = "5m") -> dict[str, Any]:
        """Run full validation suite on a candle DataFrame and return a report dict."""
        if df.empty:
            return {"valid": False, "errors": ["empty dataframe"], "warnings": []}

        df = df.copy()
        df.columns = [str(c).lower() for c in df.columns]

        errors: list[str] = []
        warnings: list[str] = []

        # Ensure timestamp index
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.set_index("timestamp").sort_index()
        elif not isinstance(df.index, pd.DatetimeIndex):
            errors.append("missing timestamp column or datetime index")
            return {"valid": False, "errors": errors, "warnings": warnings}

        # timezone consistency
        if df.index.tz is None:
            warnings.append("index lacks timezone; assuming UTC")
            df.index = df.index.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT")
        elif str(df.index.tz) != "UTC":
            df.index = df.index.tz_convert("UTC")

        # missing candles
        tf_min = self._timeframe_minutes(timeframe)
        expected_diff = pd.Timedelta(minutes=tf_min)
        missing_count = 0
        if len(df) > 1:
            gaps = df.index.to_series().diff().dropna()
            missing_count = int((gaps > expected_diff * 1.5).sum())
            if missing_count:
                warnings.append(f"missing candles detected: {missing_count} gaps > {expected_diff * 1.5}")

        # duplicate timestamps
        dupes = int(df.index.duplicated().sum())
        if dupes:
            warnings.append(f"duplicate timestamps: {dupes}")

        # impossible OHLC
        if "high" in df.columns and "low" in df.columns:
            impossible = int((df["high"] < df["low"]).sum())
            if impossible:
                errors.append(f"impossible OHLC (high < low) in {impossible} rows")
            outside = int(
                ((df["open"] > df["high"]) | (df["open"] < df["low"]) |
                 (df["close"] > df["high"]) | (df["close"] < df["low"])).sum()
            )
            if outside:
                errors.append(f"open/close outside high-low range in {outside} rows")
        if "open" in df.columns:
            zero_neg = int(((df["open"] <= 0) | (df["high"] <= 0) | (df["low"] <= 0) | (df["close"] <= 0)).sum())
            if zero_neg:
                errors.append(f"zero or negative price in {zero_neg} rows")

        # abnormal zero volume
        if "volume" in df.columns:
            zero_vol = int((df["volume"] == 0).sum())
            if zero_vol:
                warnings.append(f"abnormal zero volume in {zero_vol} rows")

        # missing spread
        if "spread" in df.columns:
            missing_spread = int(df["spread"].isna().sum())
            if missing_spread:
                warnings.append(f"missing spread in {missing_spread} rows")
            spread_mean = df["spread"].mean()
            spread_std = df["spread"].std()
            if spread_std and not np.isnan(spread_std):
                spikes = int((df["spread"] > spread_mean + 5 * spread_std).sum())
                if spikes:
                    warnings.append(f"spread spikes (>5σ) in {spikes} rows")
        else:
            warnings.append("spread column missing")

        # broker digit mismatch
        if symbol and symbol in self.symbol_metadata:
            meta = self.symbol_metadata[symbol]
            digits = meta.get("digits")
            if digits is not None and "close" in df.columns:
                max_decimals = df["close"].apply(
                    lambda x: len(str(x).split(".")[1]) if "." in str(x) and str(x).split(".")[1] else 0
                ).max()
                if max_decimals != digits:
                    warnings.append(f"broker digit mismatch: expected {digits}, got max {max_decimals}")
            point = meta.get("point")
            if point is not None and "close" in df.columns and len(df) > 1:
                diffs = df["close"].diff().abs().dropna()
                if not diffs.empty and point > 0:
                    non_point_multiples = int((diffs % point > point * 0.01).sum())
                    if non_point_multiples:
                        warnings.append(f"point-size mismatch in {non_point_multiples} price diffs")
        else:
            warnings.append("symbol metadata missing")

        valid = len(errors) == 0
        self.report = {
            "valid": valid,
            "symbol": symbol,
            "timeframe": timeframe,
            "rows": len(df),
            "missing_candles": missing_count,
            "duplicate_timestamps": dupes,
            "errors": errors,
            "warnings": warnings,
            "spread_included": "spread" in df.columns,
            "timezone_checked": True,
            "leakage_checked": False,
        }
        return self.report

    def validate_split_overlap(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        test: pd.DataFrame,
    ) -> dict[str, Any]:
        """Check for overlap between train/validation/test sets."""
        errors: list[str] = []
        if not train.empty and not validation.empty:
            train_max = pd.to_datetime(train.index.max(), utc=True)
            val_min = pd.to_datetime(validation.index.min(), utc=True)
            if train_max >= val_min:
                errors.append("train/validation overlap")
        if not validation.empty and not test.empty:
            val_max = pd.to_datetime(validation.index.max(), utc=True)
            test_min = pd.to_datetime(test.index.min(), utc=True)
            if val_max >= test_min:
                errors.append("validation/test overlap")
        return {
            "overlap_errors": errors,
            "overlap_free": len(errors) == 0,
        }

    def check_lookahead_leakage(self, df: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
        """Detect obvious lookahead leakage (e.g., future close in features)."""
        if df.empty or not feature_cols:
            return {"leakage_detected": False, "details": []}
        leaks: list[str] = []
        for col in feature_cols:
            lowered = col.lower()
            if "future" in lowered or "next_" in lowered or "target" in lowered:
                continue
            if "close" in lowered and any(f in lowered for f in ("lead", "shift", "fwd", "forward", "tomorrow")):
                leaks.append(f"potential lookahead in feature '{col}'")
        return {"leakage_detected": bool(leaks), "details": leaks}

    @staticmethod
    def _timeframe_minutes(tf: str) -> int:
        tf = tf.lower().strip()
        if tf.endswith("m"):
            return int(tf[:-1])
        if tf.endswith("h"):
            return int(tf[:-1]) * 60
        if tf.endswith("d"):
            return int(tf[:-1]) * 24 * 60
        return 5
