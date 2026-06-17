import argparse
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from drl.trading_env import TradingEnv


def load_real_data(symbol="XAUUSDm", n_bars=3000, seed=42):
    """Load real MT5 data, fallback to synthetic."""
    np.random.seed(seed)
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize(): raise ConnectionError("MT5 unavailable")
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, n_bars)
        mt5.shutdown()
        if rates is None or len(rates) == 0: raise ValueError(f"No data for {symbol}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        print(f"  [DATA] Loaded {len(df)} bars of {symbol} from MT5")
        return df
    except Exception as e:
        print(f"  [DATA] MT5 unavailable ({e}), using synthetic")
        n = n_bars
        df = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=n, freq="5min"),
            "open": 2000 + np.cumsum(np.random.randn(n) * 0.5),
            "high": np.zeros(n), "low": np.zeros(n), "close": np.zeros(n),
            "volume": np.random.randint(100, 1000, n),
        })
        df["close"] = df["open"] + np.random.randn(n) * 2
        df["high"] = df[["open","close"]].max(axis=1) + np.random.rand(n) * 2
        df["low"] = df[["open","close"]].min(axis=1) - np.random.rand(n) * 2
        df["tick_volume"] = df["volume"]
        return df


