"""
Cross-asset feature computation for trading.
Fetches related symbols (USDJPY, US10Y, DXY) alongside the primary trading symbol
and computes lagged correlations, relative strength, and divergence features.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from Python.data_feed import fetch_training_data

# Cross-asset symbol configuration
CROSS_ASSET_SYMBOLS = ["USDJPYm", "US10Y", "DXY"]
FEATURES_PER_SYMBOL = 6  # ret_1, ret_5, ret_20, corr_5, corr_20, zscore_5
TOTAL_CROSS_FEATURES = len(CROSS_ASSET_SYMBOLS) * FEATURES_PER_SYMBOL

_ALT_SYMBOLS = {
    "USDJPYm": ["USDJPY", "USDJPY.cfd"],
    "US10Y": ["US10YR", "US10Y.BOND", "US10Y.cfd", "US10YT"],
    "DXY": ["USDX", "DOLLAR_INDEX", "USDOLLAR", "DXY.cfd", "DX"],
}

_cross_cache: dict[tuple[str, str], Optional[pd.DataFrame]] = {}


def compute_cross_asset_features(
    primary_symbol: str,
    primary_df: pd.DataFrame,
    interval: str | None = None,
) -> np.ndarray:
    """Compute cross-asset features aligned to the primary DataFrame's index."""
    if interval is None:
        interval = _infer_interval(primary_df)

    # Infer period from DataFrame date range
    n = len(primary_df)
    if n >= 2 and hasattr(primary_df, 'index') and isinstance(primary_df.index, pd.DatetimeIndex):
        total_days = (primary_df.index[-1] - primary_df.index[0]).days
        period = f"{max(total_days + 5, 30)}d"
    else:
        period = "90d"
    result = np.zeros((n, TOTAL_CROSS_FEATURES), dtype=np.float32)

    primary_close = pd.Series(
        primary_df["close"].values.astype(np.float64), index=primary_df.index
    )
    primary_ret = primary_close.pct_change().fillna(0.0)

    for si, sym in enumerate(CROSS_ASSET_SYMBOLS):
        base_idx = si * FEATURES_PER_SYMBOL

        cross_df = _fetch_cross_data(sym, interval, period=period)
        if cross_df is None or cross_df.empty:
            continue

        try:
            aligned = cross_df.reindex(
                primary_df.index, method="ffill", tolerance=pd.Timedelta(minutes=30)
            )
        except Exception:
            try:
                aligned = cross_df.reindex(primary_df.index, method="ffill")
            except Exception:
                continue

        if aligned is None or aligned.empty:
            continue

        cross_close = aligned["close"].ffill().fillna(0.0).astype(np.float64)
        cross_ret = cross_close.pct_change().fillna(0.0)

        # [0] ret_1
        result[:, base_idx] = cross_ret.values.astype(np.float32)
        # [1] ret_5
        result[:, base_idx + 1] = cross_close.pct_change(5).fillna(0.0).values.astype(np.float32)
        # [2] ret_20
        result[:, base_idx + 2] = cross_close.pct_change(20).fillna(0.0).values.astype(np.float32)
        # [3] corr_5
        corr_5 = primary_ret.rolling(5).corr(cross_ret).fillna(0.0)
        result[:, base_idx + 3] = corr_5.values.astype(np.float32)
        # [4] corr_20
        corr_20 = primary_ret.rolling(20).corr(cross_ret).fillna(0.0)
        result[:, base_idx + 4] = corr_20.values.astype(np.float32)
        # [5] zscore_5
        zmean = cross_ret.rolling(5).mean()
        zstd = cross_ret.rolling(5).std().replace(0.0, np.nan)
        zscore = ((cross_ret - zmean) / zstd).fillna(0.0).replace([np.inf, -np.inf], 0.0)
        result[:, base_idx + 5] = zscore.values.astype(np.float32)

    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def clear_cache() -> None:
    """Clear the cross-asset data cache."""
    _cross_cache.clear()


def _infer_interval(df: pd.DataFrame) -> str:
    if df is None or df.empty or len(df) < 2:
        return "5m"
    diff_s = (df.index[-1] - df.index[-2]).total_seconds()
    if diff_s <= 120:
        return "1m"
    if diff_s <= 360:
        return "5m"
    if diff_s <= 900:
        return "15m"
    if diff_s <= 3600:
        return "1h"
    if diff_s <= 14400:
        return "4h"
    return "1d"


def _fetch_cross_data(symbol: str, interval: str, period: str = "90d") -> Optional[pd.DataFrame]:
    cache_key = (symbol, interval)
    if cache_key in _cross_cache:
        return _cross_cache[cache_key]

        pass  # period is now a parameter of this function
    for try_sym in [symbol] + _ALT_SYMBOLS.get(symbol, []):
        try:
            df = fetch_training_data(
                try_sym, period=period, interval=interval, strict=False
            )
            if df is not None and not df.empty and len(df) > 50:
                logger.info(f"Cross-asset: fetched {len(df)} bars for {try_sym} ({interval})")
                _cross_cache[cache_key] = df
                return df
        except Exception as exc:
            logger.debug(f"Cross-asset: {try_sym} @ {interval} failed: {exc}")
            continue

    logger.debug(f"Cross-asset: {symbol} not available after all fallbacks")
    _cross_cache[cache_key] = None
    return None
