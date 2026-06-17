#!/usr/bin/env python3
"""Standalone column audit tool for ENGINEERED_V2 feature matrix."

Usage:
    python training/run_column_audit.py --symbol XAUUSDm --bars 5000
    python training/run_column_audit.py --symbol XAUUSDm --bars 3000 --csv audit.csv
"""
from __future__ import annotations

import argparse
import sys
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Python.feature_registry import (
    ENGINEERED_V2_COLUMNS,
    FEATURE_GROUPS,
    PAUSED_GROUPS,
    N_FEATURES,
    print_column_audit,
    audit_column,
    detect_dead_columns,
)


def load_data(symbol: str, n_bars: int) -> pd.DataFrame:
    """Load OHLCV data - tries MT5, then data_feed, then synthetic."""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            raise RuntimeError("mt5.initialize() failed")
        try:
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, n_bars + 200)
        finally:
            mt5.shutdown()
        if rates is not None and len(rates) >= 100:
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)
            df.rename(columns={"tick_volume": "volume"}, inplace=True)
            print(f"  [MT5]  {len(df)} bars for {symbol}")
            return df
    except Exception as e:
        print(f"  [MT5]  unavailable ({e})")

    try:
        from Python.data_feed import fetch_training_data
        df = fetch_training_data(symbol, n_bars + 200)
        if df is not None and len(df) >= 100:
            print(f"  [DATA_FEED]  {len(df)} bars for {symbol}")
            return df
    except Exception as e:
        print(f"  [DATA_FEED]  unavailable ({e})")

    print(f"  [SYNTHETIC]  generating {n_bars} bars")
    rng = np.random.default_rng(42)
    close = 2000.0 + np.cumsum(rng.normal(0, 2, n_bars + 200))
    high = close + np.abs(rng.normal(0, 3, n_bars + 200))
    low = close - np.abs(rng.normal(0, 3, n_bars + 200))
    open_ = close - rng.normal(0, 1, n_bars + 200)
    volume = np.abs(rng.normal(1000, 200, n_bars + 200))
    idx = pd.date_range("2024-01-01", periods=n_bars + 200, freq="5min")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def build_features(df: pd.DataFrame, symbol: str) -> np.ndarray:
    """Build the ENGINEERED_V2 feature matrix."""
    try:
        from Python.feature_pipeline import build_env_feature_matrix
        matrix = build_env_feature_matrix(df, symbol=symbol)
        print(f"  [PIPELINE]  feature_pipeline.build_env_feature_matrix -> {matrix.shape}")
        return matrix
    except Exception as e:
        print(f"  [PIPELINE FAILED]  {e}, using fallback")

    from training.run_real_feature_ablation import _build_features_fallback
    matrix = _build_features_fallback(df, symbol)
    print(f"  [FALLBACK]  _build_features_fallback -> {matrix.shape}")
    return matrix


def compute_forward_returns(close: np.ndarray, horizon: int = 20) -> np.ndarray:
    """Compute forward returns for correlation analysis."""
    fwd = np.full(len(close), np.nan)
    for i in range(len(close) - horizon):
        fwd[i] = (close[i + horizon] / (close[i] + 1e-12)) - 1.0
    return fwd


def run_audit(symbol: str, n_bars: int, csv_path=None):
    """Run the full column audit."""
    print()
    print(f"=== COLUMN AUDIT: {symbol}  {n_bars} bars ===")

    df = load_data(symbol, n_bars)
    close = df["close"].to_numpy(dtype=np.float64)

    features = build_features(df, symbol)
    n_cols = features.shape[1]

    fwd_ret = compute_forward_returns(close)

    print_column_audit(features, fwd_ret, title=f"{symbol}  {n_bars} bars")

    dead, reasons = detect_dead_columns(features)
    live_names = [ENGINEERED_V2_COLUMNS[i] for i in range(min(n_cols, len(ENGINEERED_V2_COLUMNS))) if i not in dead]
    print(f"Summary: {len(live_names)}/{n_cols} columns live, {len(dead)} dead")
    if dead:
        print(f"Dead: {[ENGINEERED_V2_COLUMNS[i] for i in dead if i < len(ENGINEERED_V2_COLUMNS)]}")

    if csv_path:
        rows = []
        for i in range(min(n_cols, len(ENGINEERED_V2_COLUMNS))):
            s = audit_column(features, i, fwd_ret)
            s["dead"] = i in dead
            s["dead_reason"] = reasons.get(i, "")
            rows.append(s)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"  Audit CSV saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="ENGINEERED_V2 column audit")
    parser.add_argument("--symbol", default="XAUUSDm", help="Trading symbol")
    parser.add_argument("--bars", type=int, default=5000, help="Number of bars")
    parser.add_argument("--csv", default=None, help="Optional CSV output path")
    args = parser.parse_args()
    run_audit(args.symbol, args.bars, args.csv)


if __name__ == "__main__":
    main()
