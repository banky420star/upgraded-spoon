import pytest
import pandas as pd
import numpy as np

try:
    from Python.feature_pipeline import build_lstm_feature_frame as _build_feat
except ImportError:
    _build_feat = None


@pytest.mark.skipif(_build_feat is None, reason="feature_pipeline not importable")
def test_no_future_leakage():
    dates = pd.date_range("2023-01-01", periods=100, freq="1h")
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "open": rng.random(100),
        "high": rng.random(100),
        "low": rng.random(100),
        "close": rng.random(100),
        "volume": rng.random(100),
    }, index=dates)
    df_feat, _ = _build_feat(df.copy())
    # Mutate a future row
    mutation_idx = 50
    df_altered = df.copy()
    df_altered.iloc[mutation_idx, df.columns.get_loc("close")] += 1000.0
    df_feat_altered, _ = _build_feat(df_altered)
    # Features before mutation must be identical (no look-ahead)
    pd.testing.assert_frame_equal(
        df_feat.iloc[:mutation_idx],
        df_feat_altered.iloc[:mutation_idx],
    )
