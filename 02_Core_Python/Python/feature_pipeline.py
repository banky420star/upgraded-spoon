import numpy as np
import pandas as pd

# PatternDetector integration (Dreamer + Decision PPO now receive classical patterns + timing in obs)
try:
    from Python.patterns.pattern_detector import PatternDetector, PATTERN_FEATURE_NAMES
    _PATTERN_DETECTOR_AVAILABLE = True
except Exception:
    _PATTERN_DETECTOR_AVAILABLE = False
    PatternDetector = None
    PATTERN_FEATURE_NAMES = []
from Python.cross_asset import compute_cross_asset_features, TOTAL_CROSS_FEATURES
from Python.ml_signal import compute_ml_signal, ML_SIGNAL_FEATURES


ENGINEERED_V2 = "engineered_v2"
ULTIMATE_150 = "ultimate_150"
FEATURE_VERSIONS = {ENGINEERED_V2, ULTIMATE_150}

ENGINEERED_LSTM_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "ret_1",
    "ret_5",
    "ret_10",
    "rsi_14",
    "atr_14",
    "ema_12",
    "ema_26",
    "macd_line",
    "macd_signal",
    "bb_width_20",
    "stoch_k_14",
    "vol_z_20",
]


def _as_series(df: pd.DataFrame, col: str) -> pd.Series:
    obj = df[col]
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0].astype(float)
    return obj.astype(float)


def normalize_feature_version(feature_version: str | None, default: str = ENGINEERED_V2) -> str:
    version = str(feature_version or default).strip().lower()
    return version if version in FEATURE_VERSIONS else str(default)


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    if "tick_volume" in out.columns and "volume" not in out.columns:
        out = out.rename(columns={"tick_volume": "volume"})
    if "volume" not in out.columns:
        out["volume"] = 0.0
    for col in ["open", "high", "low", "close"]:
        if col not in out.columns:
            raise ValueError(f"missing required column: {col}")

    if "time" in out.columns:
        out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
        out = out.dropna(subset=["time"]).sort_values("time").set_index("time")
    elif not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.RangeIndex(len(out))
    elif out.index.tz is None:
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()
    return out


def build_lstm_feature_frame(df: pd.DataFrame, feature_version: str = ENGINEERED_V2) -> tuple[pd.DataFrame, list[str]]:
    version = normalize_feature_version(feature_version, default=ENGINEERED_V2)
    if version == ULTIMATE_150:
        features = _build_ultimate_feature_frame(df)
        return features, list(features.columns)
    features = _build_engineered_lstm_frame(df)
    return features, list(ENGINEERED_LSTM_COLUMNS)


def build_env_feature_matrix(df: pd.DataFrame, feature_version: str = ENGINEERED_V2, symbol: str = "") -> np.ndarray:
    version = normalize_feature_version(feature_version, default=ENGINEERED_V2)
    if version == ULTIMATE_150:
        features, _ = build_lstm_feature_frame(df, feature_version=ULTIMATE_150)
        return features.to_numpy(dtype=np.float32)
    return _build_engineered_env_matrix(df, symbol=symbol)


def feature_count_for_version(feature_version: str) -> int:
    version = normalize_feature_version(feature_version, default=ENGINEERED_V2)
    if version == ULTIMATE_150:
        sample = pd.DataFrame(
            {
                "time": pd.date_range("2026-01-01", periods=300, freq="5min", tz="UTC"),
                "open": np.linspace(1.0, 1.2, 300),
                "high": np.linspace(1.01, 1.21, 300),
                "low": np.linspace(0.99, 1.19, 300),
                "close": np.linspace(1.0, 1.2, 300),
                "volume": np.linspace(100, 400, 300),
            }
        )
        return int(build_env_feature_matrix(sample, feature_version=ULTIMATE_150).shape[1])
    # Dynamically compute base feature count + cross-asset features
    sample = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=300, freq="5min", tz="UTC"),
            "open": np.linspace(1.0, 1.2, 300),
            "high": np.linspace(1.01, 1.21, 300),
            "low": np.linspace(0.99, 1.19, 300),
            "close": np.linspace(1.0, 1.2, 300),
            "volume": np.linspace(100, 400, 300),
        }
    )
    base_count = int(_build_engineered_env_matrix(sample).shape[1])
    return base_count + TOTAL_CROSS_FEATURES + ML_SIGNAL_FEATURES


def expected_obs_dim(feature_version: str, window: int = 100) -> int:
    """Return the total observation dimension for a given feature version and window.

    Accounts for: window * features_per_bar + portfolio_state_features.
    This is the single source of truth for observation dimension calculations
    across training, backtesting, and inference.
    """
    from drl.trading_env import PORTFOLIO_FEATURE_COUNT  # late import avoids circular dep
    n_feat = feature_count_for_version(feature_version)
    return window * n_feat + PORTFOLIO_FEATURE_COUNT  # +12 classical patterns + timing + cross-asset + ML signal + timing + cross-asset features (doji/hammer/engulfing/flags/breakouts/double) + timing (Dreamer world model + Decision PPO now pattern+timing conditioned for rich edge)


def _build_engineered_lstm_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = _normalize_ohlcv(df)
    close = _as_series(out, "close")
    high = _as_series(out, "high")
    low = _as_series(out, "low")
    volume = _as_series(out, "volume")

    out["ret_1"] = close.pct_change().fillna(0.0)
    out["ret_5"] = close.pct_change(5).fillna(0.0)
    out["ret_10"] = close.pct_change(10).fillna(0.0)

    delta = close.diff().fillna(0.0)
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-12)
    out["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50.0)

    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    out["atr_14"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean().bfill()

    out["ema_12"] = close.ewm(span=12, adjust=False).mean()
    out["ema_26"] = close.ewm(span=26, adjust=False).mean()
    out["macd_line"] = out["ema_12"] - out["ema_26"]
    out["macd_signal"] = out["macd_line"].ewm(span=9, adjust=False).mean()

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std().fillna(0.0)
    out["bb_width_20"] = ((bb_std * 4.0) / (bb_mid.abs() + 1e-12)).fillna(0.0)

    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    out["stoch_k_14"] = (((close - low_14) / ((high_14 - low_14) + 1e-12)) * 100.0).fillna(50.0)

    vol_mean_20 = volume.rolling(20).mean()
    vol_std_20 = volume.rolling(20).std().fillna(0.0)
    out["vol_z_20"] = ((volume - vol_mean_20) / (vol_std_20 + 1e-12)).fillna(0.0)

    out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()
    return out[ENGINEERED_LSTM_COLUMNS].copy()


def _build_engineered_env_matrix(df: pd.DataFrame, symbol: str = "") -> np.ndarray:
    out = _normalize_ohlcv(df)
    o = out["open"].to_numpy(dtype=np.float64)
    h = out["high"].to_numpy(dtype=np.float64)
    l = out["low"].to_numpy(dtype=np.float64)
    c = out["close"].to_numpy(dtype=np.float64)
    v = out["volume"].to_numpy(dtype=np.float64)
    dates = out.index if isinstance(out.index, pd.DatetimeIndex) else None
    eps = 1e-12

    def shift(arr: np.ndarray, n: int) -> np.ndarray:
        if n <= 0:
            return arr.copy()
        shifted = np.empty_like(arr)
        shifted[:n] = arr[0]
        shifted[n:] = arr[:-n]
        return shifted

    range_ = np.maximum(h - l, eps)
    close_shift1 = shift(c, 1)
    close_shift5 = shift(c, 5)
    close_shift20 = shift(c, 20)

    log_ret1 = np.log(np.maximum(c, eps) / np.maximum(close_shift1, eps))
    log_ret5 = np.log(np.maximum(c, eps) / np.maximum(close_shift5, eps))
    log_ret20 = np.log(np.maximum(c, eps) / np.maximum(close_shift20, eps))

    body_ratio = (c - o) / range_
    upper_wick = (h - np.maximum(o, c)) / range_
    lower_wick = (np.minimum(o, c) - l) / range_
    range_ratio = (h - l) / (c + eps)

    rv_20 = pd.Series(log_ret1).rolling(20, min_periods=1).std().fillna(0.0).to_numpy(dtype=np.float64)
    vol_ma20 = pd.Series(np.maximum(v, 0.0)).rolling(20, min_periods=1).mean().to_numpy(dtype=np.float64)
    rel_volume = np.maximum(v, 0.0) / (vol_ma20 + eps)
    spread_est_bps = ((h - l) / (c + eps)) * 10000.0

    ma50 = pd.Series(c).rolling(50, min_periods=1).mean().to_numpy(dtype=np.float64)
    htf_trend = (c / (ma50 + eps)) - 1.0

    hour_sin = np.zeros_like(c)
    hour_cos = np.zeros_like(c)
    dow_sin = np.zeros_like(c)
    dow_cos = np.zeros_like(c)
    if dates is not None:
        hour = dates.hour.astype(np.float64)
        dow = dates.dayofweek.astype(np.float64)
        hour_sin = np.sin(2.0 * np.pi * hour / 24.0)
        hour_cos = np.cos(2.0 * np.pi * hour / 24.0)
        dow_sin = np.sin(2.0 * np.pi * dow / 7.0)
        dow_cos = np.cos(2.0 * np.pi * dow / 7.0)

    # ── NEW: Enhanced session / open / news timing features for Dreamer world model (and full ensemble)
    # Makes Dreamer (world model training) aware of market structure around opens and news.
    # These mirror the critical features from features/build_features.py used by Decision PPO / rewards.
    session_london = np.zeros_like(c)
    session_ny = np.zeros_like(c)
    major_open = np.zeros_like(c)
    news_prox = np.zeros_like(c)
    news_soon = np.zeros_like(c)
    if dates is not None:
        hour_f = dates.hour.astype(np.float64) + dates.minute.astype(np.float64) / 60.0
        session_london = ((hour_f >= 8) & (hour_f < 17)).astype(np.float64)
        session_ny = ((hour_f >= 13) & (hour_f < 22)).astype(np.float64)
        london_win = ((hour_f >= 7.5) & (hour_f <= 9.5)).astype(np.float64)
        ny_win = ((hour_f >= 12.5) & (hour_f <= 14.5)).astype(np.float64)
        major_open = np.maximum(london_win, ny_win)
    # News timing (if injected in df for training/live; else neutral 0)
    if 'news_distance_minutes' in out.columns:
        try:
            nd = pd.to_numeric(out['news_distance_minutes'], errors='coerce').fillna(999.0).to_numpy()
            news_prox = np.clip(1.0 / (1.0 + nd / 60.0), 0, 1)
            news_soon = (nd < 60).astype(np.float64)
        except Exception:
            pass

    # ── CLASSICAL PATTERNS for Dreamer observations + Decision PPO rich decisions
    # Pattern + timing state lets world model learn "engulfing at open + low news" dynamics;
    # imagination rollouts simulate pattern-conditioned outcomes; PPO learns to bias TimeExitSpec/risk/partials.
    n_pat = len(PATTERN_FEATURE_NAMES) or 11
    n = len(c)
    pattern_block = np.zeros((n, n_pat), dtype=np.float32)
    if _PATTERN_DETECTOR_AVAILABLE and n >= 5:
        try:
            detector = PatternDetector(atr_period=14)
            timing_ctx = {
                "major_open_window": float(major_open[-1]) if n > 0 else 0.0,
                "news_proximity": float(news_prox[-1]) if n > 0 else 0.0,
                "has_high_impact_news_soon": float(news_soon[-1]) if n > 0 else 0.0,
            }
            pat_vec = detector.get_pattern_feature_vector(out, timing_context=timing_ctx)
            for col_idx in range(12):
                pattern_block[:, col_idx] = pat_vec[col_idx]
        except Exception:
            pass  # neutral patterns on failure

    valid_rv = rv_20[np.isfinite(rv_20)]
    if len(valid_rv) > 10:
        q1 = np.quantile(valid_rv, 0.33)
        q2 = np.quantile(valid_rv, 0.66)
        vol_bucket = np.where(rv_20 <= q1, 0.0, np.where(rv_20 <= q2, 0.5, 1.0))
    else:
        vol_bucket = np.zeros_like(c)

    close_rel = (c / (close_shift1 + eps)) - 1.0
    open_rel = (o / (c + eps)) - 1.0
    high_rel = (h / (c + eps)) - 1.0
    low_rel = (l / (c + eps)) - 1.0
    log_vol = np.log1p(np.maximum(v, 0.0))

    matrix = np.column_stack(
        [
            open_rel,
            high_rel,
            low_rel,
            close_rel,
            log_vol,
            log_ret1,
            log_ret5,
            log_ret20,
            body_ratio,
            upper_wick,
            lower_wick,
            range_ratio,
            rv_20,
            rel_volume,
            spread_est_bps,
            hour_sin,
            hour_cos,
            dow_sin,
            dow_cos,
            htf_trend,
            vol_bucket,
            # NEW timing columns (8) + 12 classical patterns (doji/hammer/engulfing/double/flag/breakout) for pattern+timing state
            # This gives Dreamer world model pattern-conditioned dynamics and Decision PPO the edge for rich autonomous TradeDecisions (TimeExitSpec bias etc)
            session_london,
            session_ny,
            major_open,
            news_prox,
            news_soon,
            np.zeros_like(c),  # session_overlap placeholder
            np.zeros_like(c),  # mins_since_london placeholder
            np.zeros_like(c),  # news_avoidance placeholder
            # Classical pattern indicators (n_pat)
            *[pattern_block[:, i] for i in range(n_pat)],
        ]
    )
    # Cross-asset features (DXY, US10Y, USDJPY correlations with primary)
    if symbol:
        try:
            cross_feat = compute_cross_asset_features(symbol, out)
            if cross_feat.shape[1] > 0:
                matrix = np.column_stack([matrix, cross_feat])
        except Exception:
            pass  # cross-asset features unavailable; continue without

    # ML directional signal (XGBoost predicts next-bar direction from features)
    if symbol and matrix.shape[1] >= 40:
        try:
            ml_signal = compute_ml_signal(matrix, out["close"].values)
            if ml_signal.shape[1] > 0:
                matrix = np.column_stack([matrix, ml_signal])
        except Exception:
            pass  # ML signal unavailable; continue without
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _build_ultimate_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = _normalize_ohlcv(df)
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    open_ = out["open"].astype(float)
    volume = out["volume"].astype(float)
    eps = 1e-12

    feats: dict[str, pd.Series] = {}
    feats["open_rel"] = open_ / (close + eps) - 1.0
    feats["high_rel"] = high / (close + eps) - 1.0
    feats["low_rel"] = low / (close + eps) - 1.0
    feats["close_ret_1"] = close.pct_change().fillna(0.0)
    feats["body_ratio"] = (close - open_) / ((high - low).abs() + eps)
    feats["upper_wick_ratio"] = (high - np.maximum(open_, close)) / ((high - low).abs() + eps)
    feats["lower_wick_ratio"] = (np.minimum(open_, close) - low) / ((high - low).abs() + eps)
    feats["range_ratio"] = (high - low) / (close.abs() + eps)
    feats["log_volume"] = np.log1p(np.maximum(volume, 0.0))
    feats["gap_ratio"] = open_ / (close.shift(1).fillna(close.iloc[0]) + eps) - 1.0

    windows = [3, 5, 8, 13, 21, 34, 55]
    for win in windows:
        ret = close.pct_change(win).fillna(0.0)
        logret = np.log(close / (close.shift(win).fillna(close.iloc[0]) + eps)).fillna(0.0)
        range_mean = ((high - low) / (close.abs() + eps)).rolling(win, min_periods=1).mean()
        range_std = ((high - low) / (close.abs() + eps)).rolling(win, min_periods=1).std().fillna(0.0)
        ma = close.rolling(win, min_periods=1).mean()
        ema = close.ewm(span=max(2, win), adjust=False).mean()
        vol_mean = volume.rolling(win, min_periods=1).mean()
        vol_std = volume.rolling(win, min_periods=1).std().fillna(0.0)
        price_std = close.rolling(win, min_periods=1).std().fillna(0.0)
        atr = pd.concat(
            [(high - low).abs(), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
            axis=1,
        ).max(axis=1).rolling(win, min_periods=1).mean()
        delta = close.diff().fillna(0.0)
        gain = delta.clip(lower=0).rolling(win, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(win, min_periods=1).mean()
        rs = gain / (loss + eps)
        rsi = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)
        bb_width = ((close.rolling(win, min_periods=1).std().fillna(0.0) * 4.0) / (ma.abs() + eps)).fillna(0.0)
        highest = high.rolling(win, min_periods=1).max()
        lowest = low.rolling(win, min_periods=1).min()
        slope = ma.diff(win).fillna(0.0) / (ma.shift(win).abs() + eps)

        feats[f"ret_{win}"] = ret
        feats[f"logret_{win}"] = logret
        feats[f"range_mean_{win}"] = range_mean
        feats[f"range_std_{win}"] = range_std
        feats[f"close_ma_rel_{win}"] = close / (ma + eps) - 1.0
        feats[f"volume_rel_{win}"] = volume / (vol_mean + eps) - 1.0
        feats[f"realized_vol_{win}"] = close.pct_change().rolling(win, min_periods=1).std().fillna(0.0)
        feats[f"close_z_{win}"] = (close - ma) / (price_std + eps)
        feats[f"momentum_{win}"] = close.diff(win).fillna(0.0) / (close.shift(win).abs() + eps)
        feats[f"ema_rel_{win}"] = close / (ema + eps) - 1.0
        feats[f"rsi_{win}"] = (rsi / 100.0) * 2.0 - 1.0
        feats[f"atr_rel_{win}"] = atr / (close.abs() + eps)
        feats[f"bb_width_{win}"] = bb_width
        feats[f"breakout_high_{win}"] = close / (highest + eps) - 1.0
        feats[f"breakout_low_{win}"] = close / (lowest + eps) - 1.0
        feats[f"slope_{win}"] = slope.fillna(0.0)

    if isinstance(out.index, pd.DatetimeIndex):
        idx = out.index
        hour = pd.Series(idx.hour.astype(np.float32), index=out.index)
        dow = pd.Series(idx.dayofweek.astype(np.float32), index=out.index)
        month = pd.Series(idx.month.astype(np.float32), index=out.index)
        feats["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        feats["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        feats["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
        feats["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
        feats["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
        feats["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
    else:
        for name in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]:
            feats[name] = pd.Series(0.0, index=out.index)

    if isinstance(out.index, pd.DatetimeIndex):
        resamples = [
            ("15min", "m15"),
            ("1h", "h1"),
            ("4h", "h4"),
            ("1D", "d1"),
        ]
        for rule, label in resamples:
            htf = (
                out[["open", "high", "low", "close", "volume"]]
                .resample(rule)
                .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
                .ffill()
            )
            htf = htf.reindex(out.index, method="ffill")
            htf_ma = htf["close"].rolling(8, min_periods=1).mean()
            htf_std = htf["close"].rolling(8, min_periods=1).std().fillna(0.0)
            htf_delta = htf["close"].diff().fillna(0.0)
            htf_gain = htf_delta.clip(lower=0).rolling(8, min_periods=1).mean()
            htf_loss = (-htf_delta.clip(upper=0)).rolling(8, min_periods=1).mean()
            htf_rs = htf_gain / (htf_loss + eps)
            htf_rsi = (100.0 - (100.0 / (1.0 + htf_rs))).fillna(50.0)
            feats[f"{label}_close_rel"] = htf["close"] / (close + eps) - 1.0
            feats[f"{label}_range_rel"] = (htf["high"] - htf["low"]) / (close.abs() + eps)
            feats[f"{label}_volume_rel"] = htf["volume"] / (volume.rolling(20, min_periods=1).mean() + eps) - 1.0
            feats[f"{label}_trend"] = htf["close"] / (htf_ma + eps) - 1.0
            feats[f"{label}_rsi"] = (htf_rsi / 100.0) * 2.0 - 1.0
            feats[f"{label}_bb_width"] = ((htf_std * 4.0) / (htf_ma.abs() + eps)).fillna(0.0)
    else:
        for label in ["m15", "h1", "h4", "d1"]:
            for suffix in ["close_rel", "range_rel", "volume_rel", "trend", "rsi", "bb_width"]:
                feats[f"{label}_{suffix}"] = pd.Series(0.0, index=out.index)

    feats["cross_trend_h1_h4"] = feats["h1_trend"] - feats["h4_trend"]
    feats["cross_trend_m15_h1"] = feats["m15_trend"] - feats["h1_trend"]
    feats["cross_rsi_h1_d1"] = feats["h1_rsi"] - feats["d1_rsi"]
    feats["cross_rsi_m15_h4"] = feats["m15_rsi"] - feats["h4_rsi"]
    feats["cross_volume_h1_d1"] = feats["h1_volume_rel"] - feats["d1_volume_rel"]
    feats["cross_range_h1_h4"] = feats["h1_range_rel"] - feats["h4_range_rel"]
    feats["cross_close_h1_d1"] = feats["h1_close_rel"] - feats["d1_close_rel"]
    feats["cross_bb_h1_d1"] = feats["h1_bb_width"] - feats["d1_bb_width"]
    feats["cross_bb_m15_h1"] = feats["m15_bb_width"] - feats["h1_bb_width"]
    feats["cross_ret_5_21"] = feats["ret_5"] - feats["ret_21"]
    feats["cross_ret_13_55"] = feats["ret_13"] - feats["ret_55"]
    feats["cross_vol_8_34"] = feats["realized_vol_8"] - feats["realized_vol_34"]

    feature_df = pd.DataFrame(feats, index=out.index)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    return feature_df.astype(np.float32)

# ============================================================
# NEW STANDARD: Multi-Timeframe per Symbol (1m + 5m + 15m + 1h)
# ============================================================
from Python.features.multitimeframe_builder import (
    build_multitimeframe_features,
    load_best_feature_params,
    get_multitimeframe_feature_count,
)

def build_multitimeframe_feature_matrix(
    dfs: dict,
    symbol: str,
    feature_version: str = "engineered_v2",
) -> np.ndarray:
    """
    Convenience wrapper that builds the standard multi-timeframe feature matrix
    (1m + 5m + 15m + 1h) using the best known parameters for the symbol.
    
    dfs should be a dict like:
        {"1m": df1, "5m": df5, "15m": df15, "1h": df1h}
    """
    df1 = dfs.get("1m") or dfs.get("1min")
    df5 = dfs.get("5m") or dfs.get("5min")
    df15 = dfs.get("15m") or dfs.get("15min")
    df60 = dfs.get("1h") or dfs.get("60min") or dfs.get("h1")
    
    if df1 is None or (hasattr(df1, 'empty') and df1.empty):
        raise ValueError("At minimum a 1m DataFrame must be provided for MTF features")
    
    # Pass through (builder now handles None/empty higher TFs via graceful degradation)
    feat_df = build_multitimeframe_features(df1, df5, df15, df60, symbol, feature_version)
    return feat_df.to_numpy(dtype=np.float32)


# Backwards-compatible alias so existing code can opt-in easily
build_standard_multitimeframe_features = build_multitimeframe_feature_matrix

# ============================================================
# Dispatch for the new standard multi-timeframe mode
# ============================================================
def _is_multitimeframe_best_request(feature_version: str | None) -> bool:
    if not feature_version:
        return False
    fv = str(feature_version).lower().strip()
    return fv in {"multitimeframe", "multitimeframe_best", "mtf_best", "standard_mtf"}

# Patch the two main builders so they can delegate to the new multi-TF builder
_original_build_env = build_env_feature_matrix
_original_build_lstm = build_lstm_feature_frame


def _apply_artifact_feature_importance_scaling(matrix: np.ndarray, feature_version: str = "") -> np.ndarray:
    """
    ARTIFACT-DRIVEN FEATURE IMPORTANCE SCALING (from next_training_overrides.json via launch).
    Scales columns corresponding to 'patterns', 'timing', 'news_proximity' groups by the multipliers
    (e.g. patterns=1.29, timing=1.38). This makes the Decision PPO policy "pay more attention"
    to the high-evidence features identified in the validation artifact (double_bottom etc + timing).
    Non-destructive; only affects the obs vector seen by PPO during this training run.
    Heuristic column layout for engineered_v2 + patterns+timing (last ~20 cols contain them).
    """
    if matrix is None or matrix.size == 0:
        return matrix
    try:
        fi_patterns = float(os.environ.get("AGI_FI_PATTERNS", "1.0"))
        fi_timing = float(os.environ.get("AGI_FI_TIMING", "1.0"))
        fi_news = float(os.environ.get("AGI_FI_NEWS_PROXIMITY", "1.0"))
        if fi_patterns == 1.0 and fi_timing == 1.0 and fi_news == 1.0:
            return matrix  # no-op when no overrides

        out = matrix.astype(np.float32).copy()
        n_cols = out.shape[1]
        # Heuristic: in current engineered + pattern+timing layout (~41 cols)
        # patterns occupy the final ~11-12 columns; timing ~8 before them; news is one of the timing cols.
        # Conservative: boost the tail (patterns) and a timing window.
        if n_cols >= 12:
            # patterns tail
            pat_start = max(0, n_cols - 12)
            out[:, pat_start:] *= fi_patterns
        if n_cols >= 20:
            # timing block (rough window before patterns)
            tim_start = max(0, n_cols - 22)
            tim_end = max(tim_start + 1, n_cols - 12)
            out[:, tim_start:tim_end] *= fi_timing
            # news proximity is typically one of the explicit timing cols near the end of timing block
            news_col = max(0, n_cols - 18)  # approximate
            if news_col < n_cols:
                out[:, news_col] *= fi_news
        # clip to avoid explosion
        out = np.clip(out, -50.0, 50.0)
        if abs(fi_patterns - 1.0) > 0.01 or abs(fi_timing - 1.0) > 0.01:
            logger.info(f"[artifact-overrides] Applied FI scaling: patterns={fi_patterns:.2f}, timing={fi_timing:.2f}, news={fi_news:.2f} on matrix shape {matrix.shape}")
        return out.astype(np.float32)
    except Exception as _e:
        return matrix


def build_env_feature_matrix(df: pd.DataFrame, feature_version: str = ENGINEERED_V2, symbol: str = "") -> np.ndarray:
    if _is_multitimeframe_best_request(feature_version):
        # Expect the caller to have passed a properly prepared multi-TF df
        # or we fall back to normal behavior on single df
        logger.info("Multi-timeframe best feature path requested in build_env_feature_matrix")
        # For now, if a single df is passed we still build normally.
        # Full multi-TF path is used via the explicit build_multitimeframe_feature_matrix
        base = _original_build_env(df, ENGINEERED_V2, symbol=symbol)
        return _apply_artifact_feature_importance_scaling(base, feature_version)
    base = _original_build_env(df, feature_version, symbol=symbol)
    return _apply_artifact_feature_importance_scaling(base, feature_version)


def build_lstm_feature_frame(df: pd.DataFrame, feature_version: str = ENGINEERED_V2) -> tuple[pd.DataFrame, list[str]]:
    if _is_multitimeframe_best_request(feature_version):
        logger.info("Multi-timeframe best feature path requested in build_lstm_feature_frame")
        # Similar fallback
        base_df, cols = _original_build_lstm(df, ENGINEERED_V2)
        # Note: for DataFrame path we scale the underlying values if numeric
        try:
            arr = base_df.to_numpy(dtype=np.float32)
            scaled = _apply_artifact_feature_importance_scaling(arr, feature_version)
            base_df = pd.DataFrame(scaled, columns=base_df.columns, index=base_df.index)
        except Exception:
            pass
        return base_df, cols
    base_df, cols = _original_build_lstm(df, feature_version)
    try:
        arr = base_df.to_numpy(dtype=np.float32)
        scaled = _apply_artifact_feature_importance_scaling(arr, feature_version)
        base_df = pd.DataFrame(scaled, columns=base_df.columns, index=base_df.index)
    except Exception:
        pass
    return base_df, cols
