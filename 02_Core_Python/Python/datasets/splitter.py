import numpy as np
import pandas as pd
from loguru import logger


class TimeSeriesSplitter:
    """Chronological train/validation/test/forward split — no random shuffle."""

    def __init__(
        self,
        df: pd.DataFrame,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
        test_frac: float = 0.10,
        forward_frac: float = 0.05,
    ):
        if not np.isclose(train_frac + val_frac + test_frac + forward_frac, 1.0):
            raise ValueError("Splits must sum to 1.0")
        self.df = df
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.forward_frac = forward_frac
        self.n = len(df)

    def split(self) -> dict[str, pd.DataFrame]:
        """Return {'train': ..., 'validation': ..., 'test': ..., 'forward': ...}."""
        train_end = int(self.n * self.train_frac)
        val_end = train_end + int(self.n * self.val_frac)
        test_end = val_end + int(self.n * self.test_frac)

        splits = {
            "train": self.df.iloc[:train_end].copy(),
            "validation": self.df.iloc[train_end:val_end].copy(),
            "test": self.df.iloc[val_end:test_end].copy(),
            "forward": self.df.iloc[test_end:].copy(),
        }
        logger.info(
            f"TimeSeriesSplitter: train={len(splits['train'])} val={len(splits['validation'])} "
            f"test={len(splits['test'])} forward={len(splits['forward'])}"
        )
        return splits

    def split_indices(self) -> dict[str, np.ndarray]:
        """Return index arrays for each split."""
        train_end = int(self.n * self.train_frac)
        val_end = train_end + int(self.n * self.val_frac)
        test_end = val_end + int(self.n * self.test_frac)
        return {
            "train": np.arange(0, train_end),
            "validation": np.arange(train_end, val_end),
            "test": np.arange(val_end, test_end),
            "forward": np.arange(test_end, self.n),
        }
