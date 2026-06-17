import numpy as np
import pandas as pd
from loguru import logger


class DatasetBuilder:
    """Assemble aligned feature + label DataFrames into ML-ready datasets."""

    def __init__(
        self,
        feature_df: pd.DataFrame,
        label_df: pd.DataFrame,
        drop_na: bool = True,
    ):
        self.feature_df = feature_df
        self.label_df = label_df
        self.drop_na = drop_na
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.meta: pd.DataFrame | None = None

    def build(
        self,
        target_column: str,
        feature_cols: list[str] | None = None,
        as_numpy: bool = True,
    ) -> tuple[np.ndarray | pd.DataFrame, np.ndarray | pd.Series] | tuple[pd.DataFrame, pd.Series]:
        """
        Parameters
        ----------
        target_column : str
            Label column to use as the prediction target.
        feature_cols : list[str] | None
            Subset of feature columns.  If None, uses all feature_df columns.
        as_numpy : bool
            If True returns (X, y) as numpy arrays; otherwise DataFrame/Series.

        Returns
        -------
        (X, y)
        """
        # Align on common index
        common_idx = self.feature_df.index.intersection(self.label_df.index)
        X = self.feature_df.loc[common_idx].copy()
        y = self.label_df.loc[common_idx, target_column].copy()

        if feature_cols is not None:
            missing = [c for c in feature_cols if c not in X.columns]
            if missing:
                raise KeyError(f"Feature columns missing from feature_df: {missing}")
            X = X[feature_cols]

        if self.drop_na:
            mask = X.notna().all(axis=1) & y.notna()
            before = len(X)
            X = X[mask]
            y = y[mask]
            after = len(X)
            if before != after:
                logger.info(f"DatasetBuilder dropped {before - after} rows with NaN")

        self.meta = pd.DataFrame({"target": y}, index=X.index)
        if as_numpy:
            self.X = X.to_numpy(dtype=np.float32)
            self.y = y.to_numpy(dtype=np.float32)
            logger.info(f"DatasetBuilder: X shape {self.X.shape}, y shape {self.y.shape}")
            return self.X, self.y
        else:
            self.X = X
            self.y = y
            logger.info(f"DatasetBuilder: X shape {X.shape}, y shape {y.shape}")
            return X, y
