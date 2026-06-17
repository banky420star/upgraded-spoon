import numpy as np
import pandas as pd
from loguru import logger


class FeatureAuditor:
    """Audit feature quality: leakage, correlation, importance, stability, live readiness."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.report: dict = {}

    def run_full_audit(
        self,
        feature_df: pd.DataFrame,
        label_series: pd.Series | None = None,
        label_df: pd.DataFrame | None = None,
        regimes: pd.Series | None = None,
        live_feature_names: list[str] | None = None,
    ) -> dict:
        """Run all checks and return a composite report."""
        self.report = {
            "leakage": self.check_leakage(feature_df, label_df),
            "correlation": self.check_correlation(feature_df),
            "predictive_importance": (
                self.check_predictive_importance(feature_df, label_series)
                if label_series is not None
                else None
            ),
            "stability_time": self.check_stability(feature_df, n_splits=5),
            "stability_regime": (
                self.check_stability_across_regimes(feature_df, regimes)
                if regimes is not None
                else None
            ),
            "live_availability": self.check_live_availability(feature_df, live_feature_names),
            "missing_rate": self.check_missing_rate(feature_df),
            "outlier_rate": self.check_outlier_rate(feature_df),
        }
        logger.info("FeatureAuditor: full audit complete")
        return self.report

    def check_leakage(self, feature_df: pd.DataFrame, label_df: pd.DataFrame | None = None) -> dict:
        """Flag if any known label-like column exists in features."""
        forbidden = {c.lower() for c in feature_df.columns if c.startswith("target_")}
        ok = len(forbidden) == 0
        if label_df is not None:
            overlap = set(feature_df.columns) & set(label_df.columns)
            if overlap:
                forbidden |= {c.lower() for c in overlap}
                ok = False
        result = {"ok": ok, "forbidden_columns_found": sorted(forbidden)}
        if not ok:
            logger.warning(f"FeatureAuditor: leakage detected! forbidden={forbidden}")
        return result

    def check_correlation(self, feature_df: pd.DataFrame, threshold: float = 0.98) -> dict:
        """Return pairs of features with absolute correlation above threshold."""
        corr = feature_df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        pairs = [
            (col, row, float(upper.loc[row, col]))
            for col in upper.columns
            for row in upper.index
            if pd.notna(upper.loc[row, col]) and upper.loc[row, col] >= threshold
        ]
        logger.info(f"FeatureAuditor: {len(pairs)} highly-correlated pairs (>={threshold})")
        return {"threshold": threshold, "pairs": pairs, "count": len(pairs)}

    def check_predictive_importance(
        self, feature_df: pd.DataFrame, label_series: pd.Series, method: str = "mutual_info"
    ) -> dict:
        """Estimate per-feature predictive power via mutual information or RF importance."""
        try:
            from sklearn.feature_selection import mutual_info_regression
            from sklearn.ensemble import RandomForestRegressor
        except ImportError as exc:
            logger.warning(f"FeatureAuditor: sklearn unavailable for importance check: {exc}")
            return {"method": method, "error": str(exc)}

        X = feature_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        y = label_series.reindex(X.index).fillna(0.0)

        if method == "mutual_info":
            scores = mutual_info_regression(X, y, random_state=self.random_state)
            ranking = sorted(zip(X.columns, scores), key=lambda t: t[1], reverse=True)
            return {"method": method, "ranking": ranking}

        if method == "rf_importance":
            rf = RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1)
            rf.fit(X, y)
            ranking = sorted(zip(X.columns, rf.feature_importances_), key=lambda t: t[1], reverse=True)
            return {"method": method, "ranking": ranking}

        return {"method": method, "error": "unknown method"}

    def check_stability(self, feature_df: pd.DataFrame, n_splits: int = 5) -> dict:
        """Measure per-feature mean and std across time splits (walk-forward stability)."""
        n = len(feature_df)
        split_size = max(1, n // n_splits)
        stats: list[dict] = []
        for i in range(n_splits):
            start = i * split_size
            end = min((i + 1) * split_size, n)
            chunk = feature_df.iloc[start:end]
            stats.append({"mean": chunk.mean().to_dict(), "std": chunk.std().to_dict()})

        means = pd.DataFrame([s["mean"] for s in stats])
        stds = pd.DataFrame([s["std"] for s in stats])
        mean_cv = (means.std() / (means.abs().mean() + 1e-12)).fillna(0.0)
        unstable = mean_cv[mean_cv > 0.5].index.tolist()
        return {"mean_cv": mean_cv.to_dict(), "unstable_features": unstable, "split_count": n_splits}

    def check_stability_across_regimes(
        self, feature_df: pd.DataFrame, regimes: pd.Series
    ) -> dict:
        """Compare per-regime means to detect regime-dependent instability."""
        regime_groups = feature_df.groupby(regimes.reindex(feature_df.index).fillna("unknown"))
        regime_means = regime_groups.mean()
        global_mean = feature_df.mean()
        max_dev = ((regime_means - global_mean).abs() / (global_mean.abs() + 1e-12)).max().fillna(0.0)
        unstable = max_dev[max_dev > 1.0].index.tolist()
        return {
            "regime_count": int(regime_means.shape[0]),
            "max_relative_deviation": max_dev.to_dict(),
            "unstable_features": unstable,
        }

    def check_live_availability(
        self, feature_df: pd.DataFrame, live_feature_names: list[str] | None = None
    ) -> dict:
        """Check whether features expected in production are actually present."""
        if live_feature_names is None:
            live_feature_names = list(feature_df.columns)
        missing = [f for f in live_feature_names if f not in feature_df.columns]
        return {"expected": live_feature_names, "missing": missing, "ok": len(missing) == 0}

    def check_missing_rate(self, feature_df: pd.DataFrame) -> dict:
        """Fraction of NaN / inf per feature."""
        missing = feature_df.isna().mean().to_dict()
        inf_mask = np.isinf(feature_df.to_numpy(dtype=np.float32))
        inf_rate = pd.Series(inf_mask.mean(axis=0), index=feature_df.columns).to_dict()
        return {"nan_rate": missing, "inf_rate": inf_rate}

    def check_outlier_rate(self, feature_df: pd.DataFrame, z_thresh: float = 5.0) -> dict:
        """Fraction of observations beyond z_thresh standard deviations."""
        z = ((feature_df - feature_df.mean()) / (feature_df.std() + 1e-12)).abs()
        outlier = (z > z_thresh).mean().to_dict()
        return {"z_threshold": z_thresh, "outlier_rate": outlier}
