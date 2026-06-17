import numpy as np
import pandas as pd
from loguru import logger


class LabelBuilder:
    """Factory for future-looking labels.  These must NEVER be fed as live features."""

    DIRECTION_THRESHOLD = 1e-6

    def build(self, df: pd.DataFrame, tp_pct: float = 0.001, sl_pct: float = 0.001) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : pd.DataFrame
            Must contain 'close', 'high', 'low'.  DatetimeIndex optional.
        tp_pct / sl_pct : float
            Take-profit / stop-loss distance used for target_hit_tp_before_sl.

        Returns
        -------
        pd.DataFrame
            All columns prefixed with ``target_`` so they are easy to blacklist
            from the feature set.
        """
        out = df.copy()
        out.columns = [str(c).lower() for c in out.columns]
        for col in ["close", "high", "low"]:
            if col not in out.columns:
                raise ValueError(f"missing required column: {col}")
        if "time" in out.columns:
            out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
            out = out.dropna(subset=["time"]).sort_values("time").set_index("time")
        out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()

        close = out["close"].astype(float)
        high = out["high"].astype(float)
        low = out["low"].astype(float)
        eps = 1e-12

        labels: dict[str, pd.Series] = {}

        # ── Future returns ──
        for horizon in [1, 3, 5, 10]:
            future_close = close.shift(-horizon)
            ret = future_close / (close + eps) - 1.0
            labels[f"target_return_{horizon}"] = ret

            # Direction: up / down / flat
            direction = np.where(
                ret > self.DIRECTION_THRESHOLD,
                1,                               # up
                np.where(ret < -self.DIRECTION_THRESHOLD, -1, 0)   # down / flat
            )
            labels[f"target_direction_{horizon}"] = pd.Series(direction, index=out.index)

        # ── Volatility class (based on future realized vol) ──
        future_ret1 = close.shift(-1) / (close + eps) - 1.0
        future_ret3 = close.shift(-3) / (close + eps) - 1.0
        future_ret5 = close.shift(-5) / (close + eps) - 1.0
        fut_vol = pd.concat([future_ret1, future_ret3, future_ret5], axis=1).std(axis=1)
        vol_q33 = fut_vol.quantile(0.33)
        vol_q66 = fut_vol.quantile(0.66)
        vol_class = np.where(
            fut_vol <= vol_q33, 0,
            np.where(fut_vol <= vol_q66, 1, 2)
        )
        labels["target_volatility_class"] = pd.Series(vol_class, index=out.index)

        # ── Max Favourable / Adverse Excursion over next 5 bars ──
        mfe_list: list[float] = []
        mae_list: list[float] = []
        for i in range(len(close)):
            end = min(i + 5 + 1, len(close))
            c_now = float(close.iloc[i])
            window_high = float(high.iloc[i:end].max()) if end > i else c_now
            window_low = float(low.iloc[i:end].min()) if end > i else c_now
            mfe_list.append((window_high - c_now) / (c_now + eps))
            mae_list.append((c_now - window_low) / (c_now + eps))
        labels["target_mfe_5"] = pd.Series(mfe_list, index=out.index)
        labels["target_mae_5"] = pd.Series(mae_list, index=out.index)

        # ── Hit TP before SL (binary) ──
        hit_tp_list: list[float] = []
        for i in range(len(close)):
            c_now = float(close.iloc[i])
            tp_price = c_now * (1.0 + tp_pct)
            sl_price = c_now * (1.0 - sl_pct)
            end = min(i + 5 + 1, len(close))
            if end <= i:
                hit_tp_list.append(0.0)
                continue
            window_high = float(high.iloc[i:end].max())
            window_low = float(low.iloc[i:end].min())
            # If both hit in same window, decide by which comes first (heuristic: TP if high touches first)
            hit_tp = float(window_high >= tp_price)
            hit_sl = float(window_low <= sl_price)
            if hit_tp and hit_sl:
                # tie-break: compare distance to current price
                hit_tp_list.append(1.0 if tp_price >= sl_price else 0.0)
            elif hit_tp:
                hit_tp_list.append(1.0)
            else:
                hit_tp_list.append(0.0)
        labels["target_hit_tp_before_sl"] = pd.Series(hit_tp_list, index=out.index)

        label_df = pd.DataFrame(labels, index=out.index)
        label_df = label_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        logger.info(f"LabelBuilder produced {len(label_df.columns)} labels")
        return label_df.astype(np.float32)
