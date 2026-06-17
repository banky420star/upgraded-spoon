import numpy as np
import pandas as pd
from loguru import logger


class WalkForwardBuilder:
    """Create overlapping walk-forward windows for time-series cross-validation."""

    def __init__(
        self,
        df: pd.DataFrame,
        train_bars: int = 2000,
        test_bars: int = 500,
        step_bars: int = 250,
        min_train_bars: int = 500,
    ):
        self.df = df
        self.train_bars = train_bars
        self.test_bars = test_bars
        self.step_bars = step_bars
        self.min_train_bars = min_train_bars
        self.n = len(df)

    def generate_windows(self) -> list[dict[str, pd.DataFrame]]:
        """Return a list of {'train': df, 'test': df} dicts."""
        windows: list[dict[str, pd.DataFrame]] = []
        start = 0
        while start + self.train_bars + self.test_bars <= self.n:
            train_start = start
            train_end = start + self.train_bars
            test_end = train_end + self.test_bars
            train_df = self.df.iloc[train_start:train_end].copy()
            test_df = self.df.iloc[train_end:test_end].copy()
            if len(train_df) >= self.min_train_bars:
                windows.append({"train": train_df, "test": test_df})
            start += self.step_bars
        logger.info(f"WalkForwardBuilder: generated {len(windows)} windows")
        return windows

    def generate_windows_with_indices(self) -> list[dict[str, np.ndarray]]:
        """Return a list of {'train': idx, 'test': idx} dicts."""
        windows: list[dict[str, np.ndarray]] = []
        start = 0
        while start + self.train_bars + self.test_bars <= self.n:
            train_idx = np.arange(start, start + self.train_bars)
            test_idx = np.arange(start + self.train_bars, start + self.train_bars + self.test_bars)
            if len(train_idx) >= self.min_train_bars:
                windows.append({"train": train_idx, "test": test_idx})
            start += self.step_bars
        return windows
