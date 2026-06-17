"""
Multi-Timeframe Feature Builder with Per-Symbol Best Parameters

This is the new standard way to prepare data for training when using
the 1m + 5m + 15m + 1h pipeline.

It automatically loads the best known feature parameters for each symbol
from configs/best_features_per_symbol.yaml
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import numpy as np

from Python.feature_pipeline import (
    _normalize_ohlcv,
    build_env_feature_matrix,
    build_lstm_feature_frame,
    normalize_feature_version,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEST_FEATURES_CONFIG = PROJECT_ROOT / "configs" / "best_features_per_symbol.yaml"

# Cache for loaded config
_best_features_cache: Dict[str, Any] = {}


def load_best_feature_params(symbol: str) -> Dict[str, Any]:
    """Load the best known feature parameters for a symbol."""
    global _best_features_cache
    
    if not _best_features_cache:
        if BEST_FEATURES_CONFIG.exists():
            with open(BEST_FEATURES_CONFIG, "r", encoding="utf-8") as f:
                _best_features_cache = yaml.safe_load(f) or {}
        else:
            _best_features_cache = {"symbols": {}}
    
    symbols = _best_features_cache.get("symbols", {})
    if symbol in symbols:
        return symbols[symbol]
    
    # Fallback to default baseline parameters
    default = {
        "sma_fast": 20,
        "sma_slow": 50,
        "ema_fast": 12,
        "ema_slow": 26,
        "rsi_period": 14,
        "atr_period": 14,
        "higher_tf_vol_lookback": 50,
    }
    return default


def build_multitimeframe_features(
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    symbol: str,
    feature_version: str = "engineered_v2",
) -> pd.DataFrame:
    """
    Build a rich multi-timeframe feature matrix using the best known
    parameters for the given symbol.
    
    This is the new standard for the training pipeline.
    """
    params = load_best_feature_params(symbol)
    
    # Use the best parameters for this symbol
    sma_fast = params.get("sma_fast", 20)
    sma_slow = params.get("sma_slow", 50)
    ema_fast = params.get("ema_fast", 12)
    ema_slow = params.get("ema_slow", 26)
    rsi_period = params.get("rsi_period", 14)
    atr_period = params.get("atr_period", 14)
    higher_tf_vol_lookback = params.get("higher_tf_vol_lookback", 50)
    
    def add_feats(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
        df = _normalize_ohlcv(df)
        df[f"{prefix}ret1"] = df["close"].pct_change()
        df[f"{prefix}sma_fast"] = df["close"].rolling(sma_fast).mean()
        df[f"{prefix}sma_slow"] = df["close"].rolling(sma_slow).mean()
        df[f"{prefix}ema_fast"] = df["close"].ewm(ema_fast).mean()
        df[f"{prefix}ema_slow"] = df["close"].ewm(ema_slow).mean()
        df[f"{prefix}atr"] = (df["high"] - df["low"]).rolling(atr_period).mean()
        df[f"{prefix}rsi"] = 100 - (100 / (1 + df[f"{prefix}ret1"].rolling(rsi_period).apply(
            lambda x: x[x > 0].sum() / -x[x < 0].sum() if x[x < 0].sum() != 0 else 1, raw=False)))
        df[f"{prefix}vol"] = df[f"{prefix}ret1"].rolling(20).std()
        return df
    
    d1 = add_feats(df_1m, "1m_")
    d5 = add_feats(df_5m if (df_5m is not None and not df_5m.empty) else df_1m, "5m_")
    d15 = add_feats(df_15m if (df_15m is not None and not df_15m.empty) else df_1m, "15m_")
    d60 = add_feats(df_1h if (df_1h is not None and not df_1h.empty) else df_1m, "1h_")
    
    # Higher TF context features aligned to 1m (graceful: if higher==base then ctx features become neutral-ish but never crash)
    def add_ctx(base: pd.DataFrame, higher: pd.DataFrame, p: str) -> pd.DataFrame:
        if higher is None or higher.empty or len(higher) < 2:
            # graceful degradation: no higher TF signal for this level
            base[f"{p}trend"] = 0
            base[f"{p}volreg"] = 0
            return base
        aligned = higher.reindex(base.index, method="ffill").ffill().bfill()
        base[f"{p}trend"] = (aligned["close"] > aligned[f"{p}sma_fast"]).astype(int)
        base[f"{p}volreg"] = (aligned[f"{p}vol"] > aligned[f"{p}vol"].rolling(max(2, higher_tf_vol_lookback)).mean()).astype(int)
        return base
    
    d1 = add_ctx(d1, d5, "5m_")
    d1 = add_ctx(d1, d15, "15m_")
    d1 = add_ctx(d1, d60, "1h_")
    
    # Combine into final feature matrix (base 1m + context)
    feature_cols = [c for c in d1.columns if c not in ["open", "high", "low", "close", "volume"]]
    final = d1[feature_cols].copy()
    
    return final


def get_multitimeframe_feature_count(symbol: str, feature_version: str = "engineered_v2") -> int:
    """Return how many features the best config for this symbol produces."""
    # Rough count based on current builder (can be made dynamic)
    return 14  # Adjust if you change the feature set significantly
