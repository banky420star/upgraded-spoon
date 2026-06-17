import numpy as np
import pandas as pd
from loguru import logger


class FeatureBuilder:
    """Compute ~150 features from OHLCV data for the Chain Gambler trading system."""

    FIB_WINDOWS = [3, 5, 8, 13, 21, 34, 55]

    def __init__(self):
        self.feature_names: list[str] = []

    def build(self, df: pd.DataFrame, trade_memory: pd.DataFrame | None = None) -> pd.DataFrame:
        """Return a DataFrame of engineered features (no future data)."""
        out = self._normalize_ohlcv(df)
        close = out["close"].astype(float)
        high = out["high"].astype(float)
        low = out["low"].astype(float)
        open_ = out["open"].astype(float)
        volume = out["volume"].astype(float)
        eps = 1e-12

        feats: dict[str, pd.Series] = {}

        # ── Base relatives ──
        feats["open_rel"] = open_ / (close + eps) - 1.0
        feats["high_rel"] = high / (close + eps) - 1.0
        feats["low_rel"] = low / (close + eps) - 1.0
        feats["close_ret_1"] = close.pct_change().fillna(0.0)
        feats["gap_ratio"] = open_ / (close.shift(1).fillna(close.iloc[0]) + eps) - 1.0

        # ── Price action ──
        for bar in [1, 3, 5, 8, 13, 21]:
            feats[f"ret_{bar}"] = close.pct_change(bar).fillna(0.0)
            feats[f"log_ret_{bar}"] = np.log(close / (close.shift(bar).fillna(close.iloc[0]) + eps)).fillna(0.0)

        range_ = (high - low).abs() + eps
        feats["candle_body_pct"] = (close - open_) / range_
        feats["upper_wick_pct"] = (high - np.maximum(open_, close)) / range_
        feats["lower_wick_pct"] = (np.minimum(open_, close) - low) / range_
        feats["range_ratio"] = (high - low) / (close.abs() + eps)

        # ── Trend ──
        ema_20 = close.ewm(span=20, adjust=False).mean()
        ema_50 = close.ewm(span=50, adjust=False).mean()
        feats["ema_20"] = ema_20
        feats["ema_50"] = ema_50
        feats["ema_slope_20"] = (ema_20 - ema_20.shift(5).fillna(ema_20.iloc[0])) / (ema_20.shift(5).abs() + eps)
        feats["ema_slope_50"] = (ema_50 - ema_50.shift(5).fillna(ema_50.iloc[0])) / (ema_50.shift(5).abs() + eps)

        # ── Momentum ──
        delta = close.diff().fillna(0.0)
        gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
        rs = gain / (loss + eps)
        rsi_14 = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)
        feats["rsi_14"] = rsi_14

        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        feats["macd"] = macd
        feats["macd_signal"] = macd.ewm(span=9, adjust=False).mean()

        # ── Volatility ──
        tr1 = (high - low).abs()
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        atr_14 = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14, min_periods=1).mean()
        feats["atr_14"] = atr_14
        feats["atr_pct"] = atr_14 / (close.abs() + eps)

        bb_mid = close.rolling(20, min_periods=1).mean()
        bb_std = close.rolling(20, min_periods=1).std().fillna(0.0)
        feats["bb_width"] = ((bb_std * 4.0) / (bb_mid.abs() + eps)).fillna(0.0)
        feats["realized_vol_20"] = close.pct_change().rolling(20, min_periods=1).std().fillna(0.0)

        # ── Volume ──
        vol_ma20 = volume.rolling(20, min_periods=1).mean()
        vol_std20 = volume.rolling(20, min_periods=1).std().fillna(0.0)
        feats["volume_zscore"] = ((volume - vol_ma20) / (vol_std20 + eps)).fillna(0.0)
        feats["volume_ma_ratio"] = volume / (vol_ma20 + eps)
        feats["log_volume"] = np.log1p(np.maximum(volume, 0.0))

        # ── Spread ──
        spread = high - low
        spread_ma20 = spread.rolling(20, min_periods=1).mean()
        spread_std20 = spread.rolling(20, min_periods=1).std().fillna(0.0)
        feats["spread"] = spread
        feats["spread_zscore"] = ((spread - spread_ma20) / (spread_std20 + eps)).fillna(0.0)

        # ── Enhanced Session & Market Timing (critical for Decision PPO around opens/news) ──
        if isinstance(out.index, pd.DatetimeIndex):
            hour = out.index.hour.astype(int)
            minute = out.index.minute.astype(int)
            hour_frac = hour + minute / 60.0

            # Basic session flags (kept for compatibility)
            feats["session_london"] = ((hour >= 8) & (hour < 17)).astype(float)
            feats["session_new_york"] = ((hour >= 13) & (hour < 22)).astype(float)
            feats["session_london_ny_overlap"] = ((hour >= 13) & (hour < 17)).astype(float)

            # Cyclical time features (better for neural nets / Decision PPO)
            feats["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24)
            feats["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24)

            # Minutes since major session opens (very useful for "how bot deals with market open")
            feats["mins_since_london_open"] = np.maximum(0, (hour_frac - 8) * 60)
            feats["mins_since_ny_open"] = np.maximum(0, (hour_frac - 13) * 60)

            # High-volatility windows around opens (London 7:30-9:30 UTC, NY 12:30-14:30)
            london_open = ((hour_frac >= 7.5) & (hour_frac <= 9.5)).astype(float)
            ny_open = ((hour_frac >= 12.5) & (hour_frac <= 14.5)).astype(float)
            feats["london_open_window"] = london_open
            feats["ny_open_window"] = ny_open
            feats["major_open_window"] = ((london_open + ny_open) > 0).astype(float)
        else:
            for k in ["session_london", "session_new_york", "session_london_ny_overlap",
                      "hour_sin", "hour_cos", "mins_since_london_open", "mins_since_ny_open",
                      "london_open_window", "ny_open_window", "major_open_window"]:
                feats[k] = pd.Series(0.0, index=out.index)

        # ── Market structure ──
        feats["distance_to_swing_high"] = self._distance_to_swing(close, high, low, kind="high")
        feats["distance_to_swing_low"] = self._distance_to_swing(close, high, low, kind="low")

        # ── News & Event Proximity (injected when available during training / live) ──
        # These are critical for the user's requirement: "how the bot deals with ... new based events"
        # During training we can backfill from trade_journal or ForexFactory calendar.
        # Values default to neutral (999 = far away) if no news data is attached.
        if "news_distance_minutes" in out.columns:
            nd = out["news_distance_minutes"].fillna(999.0)
            feats["news_proximity"] = np.clip(1.0 / (1.0 + nd / 60.0), 0, 1)  # closer = higher
            feats["has_high_impact_news_soon"] = (nd < 60).astype(float)     # within 1 hour
            feats["news_avoidance_zone"] = (nd < 30).astype(float)           # within 30 min
        else:
            feats["news_proximity"] = 0.0
            feats["has_high_impact_news_soon"] = 0.0
            feats["news_avoidance_zone"] = 0.0

        # ── Fibonacci rolling features ──
        for win in self.FIB_WINDOWS:
            ret = close.pct_change(win).fillna(0.0)
            logret = np.log(close / (close.shift(win).fillna(close.iloc[0]) + eps)).fillna(0.0)
            range_mean = ((high - low) / (close.abs() + eps)).rolling(win, min_periods=1).mean()
            range_std = ((high - low) / (close.abs() + eps)).rolling(win, min_periods=1).std().fillna(0.0)
            ma = close.rolling(win, min_periods=1).mean()
            ema = close.ewm(span=max(2, win), adjust=False).mean()
            vol_mean = volume.rolling(win, min_periods=1).mean()
            vol_std = volume.rolling(win, min_periods=1).std().fillna(0.0)
            price_std = close.rolling(win, min_periods=1).std().fillna(0.0)
            atr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(win, min_periods=1).mean()
            delta_win = close.diff().fillna(0.0)
            gain_win = delta_win.clip(lower=0).rolling(win, min_periods=1).mean()
            loss_win = (-delta_win.clip(upper=0)).rolling(win, min_periods=1).mean()
            rs_win = gain_win / (loss_win + eps)
            rsi_win = (100.0 - (100.0 / (1.0 + rs_win))).fillna(50.0)
            bb_width_win = ((close.rolling(win, min_periods=1).std().fillna(0.0) * 4.0) / (ma.abs() + eps)).fillna(0.0)
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
            feats[f"rsi_{win}"] = (rsi_win / 100.0) * 2.0 - 1.0
            feats[f"atr_rel_{win}"] = atr / (close.abs() + eps)
            feats[f"bb_width_{win}"] = bb_width_win
            feats[f"breakout_high_{win}"] = close / (highest + eps) - 1.0
            feats[f"breakout_low_{win}"] = close / (lowest + eps) - 1.0
            feats[f"slope_{win}"] = slope.fillna(0.0)
            feats[f"volume_z_{win}"] = ((volume - vol_mean) / (vol_std + eps)).fillna(0.0)

        # ── Calendar features ──
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

        # ── Higher-timeframe features ──
        if isinstance(out.index, pd.DatetimeIndex):
            for rule, label in [("15min", "m15"), ("1h", "h1"), ("4h", "h4"), ("1d", "d1")]:
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

        # ── Cross-timeframe / cross-window diffs ──
        if isinstance(out.index, pd.DatetimeIndex):
            feats["cross_trend_h1_h4"] = feats["h1_trend"] - feats["h4_trend"]
            feats["cross_trend_m15_h1"] = feats["m15_trend"] - feats["h1_trend"]
            feats["cross_rsi_h1_d1"] = feats["h1_rsi"] - feats["d1_rsi"]
            feats["cross_rsi_m15_h4"] = feats["m15_rsi"] - feats["h4_rsi"]
            feats["cross_volume_h1_d1"] = feats["h1_volume_rel"] - feats["d1_volume_rel"]
            feats["cross_range_h1_h4"] = feats["h1_range_rel"] - feats["h4_range_rel"]
            feats["cross_close_h1_d1"] = feats["h1_close_rel"] - feats["d1_close_rel"]
            feats["cross_bb_h1_d1"] = feats["h1_bb_width"] - feats["d1_bb_width"]
            feats["cross_bb_m15_h1"] = feats["m15_bb_width"] - feats["h1_bb_width"]
        else:
            for name in [
                "cross_trend_h1_h4", "cross_trend_m15_h1", "cross_rsi_h1_d1",
                "cross_rsi_m15_h4", "cross_volume_h1_d1", "cross_range_h1_h4",
                "cross_close_h1_d1", "cross_bb_h1_d1", "cross_bb_m15_h1",
            ]:
                feats[name] = pd.Series(0.0, index=out.index)

        feats["cross_ret_5_21"] = feats["ret_5"] - feats["ret_21"]
        feats["cross_ret_13_55"] = feats["ret_13"] - feats["ret_55"]
        feats["cross_vol_8_34"] = feats["realized_vol_8"] - feats["realized_vol_34"]

        # ── Trade memory (optional) ──
        if trade_memory is not None and not trade_memory.empty:
            aligned = trade_memory.reindex(out.index, method="ffill")
            feats["recent_loss_streak"] = aligned.get("loss_streak", pd.Series(0.0, index=out.index))
            feats["recent_win_rate"] = aligned.get("win_rate", pd.Series(0.0, index=out.index))
            feats["recent_slippage_avg"] = aligned.get("slippage_avg", pd.Series(0.0, index=out.index))
        else:
            feats["recent_loss_streak"] = pd.Series(0.0, index=out.index)
            feats["recent_win_rate"] = pd.Series(0.0, index=out.index)
            feats["recent_slippage_avg"] = pd.Series(0.0, index=out.index)

        feature_df = pd.DataFrame(feats, index=out.index)
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        self.feature_names = list(feature_df.columns)
        logger.info(f"FeatureBuilder produced {len(self.feature_names)} features")
        return feature_df.astype(np.float32)

    @staticmethod
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

    @staticmethod
    def _distance_to_swing(close: pd.Series, high: pd.Series, low: pd.Series, kind: str, window: int = 20) -> pd.Series:
        """Bars since the most recent local swing high/low (backward-looking only)."""
        if kind == "high":
            local_ext = high.rolling(window, min_periods=1).max()
            is_swing = (high == local_ext).astype(int)
        else:
            local_ext = low.rolling(window, min_periods=1).min()
            is_swing = (low == local_ext).astype(int)
        # cumulative counter of bars since last swing
        bars_since = is_swing.groupby((is_swing == 0).cumsum()).cumcount()
        return bars_since.astype(float)
