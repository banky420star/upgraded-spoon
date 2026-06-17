"""Canonical feature registry for ENGINEERED_V2 columns and feature groups.

Single source of truth. All feature-column code must import from here.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional

# --- ENGINEERED_V2: 43-column environment matrix (canonical ordering) ---

ENGINEERED_V2_COLUMNS: List[str] = [
    "open_rel", "high_rel", "low_rel", "close_rel",
    "log_vol",
    "log_ret1", "log_ret5", "log_ret20",
    "body_ratio", "upper_wick", "lower_wick", "range_ratio",
    "rv_20",
    "rel_volume", "spread_est_bps",
    "htf_trend", "vol_bucket",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "session_london", "session_ny", "major_open",
    "news_prox", "news_soon", "session_overlap",
    "mins_since_london", "news_avoid",
    *[f"pattern_{i}" for i in range(11)],
    *[f"cross_asset_{i}" for i in range(18)],
    "ml_signal",
]

N_FEATURES: int = len(ENGINEERED_V2_COLUMNS)

# --- Active feature groups ---

FEATURE_GROUPS_BY_NAME: Dict[str, List[str]] = {
    "trend":       ["htf_trend", "vol_bucket"],
    "momentum":    ["log_ret1", "log_ret5", "log_ret20"],
    "volatility":  ["rv_20"],
    "volume":      ["rel_volume", "spread_est_bps"],
    "cross_asset": [f"cross_asset_{i}" for i in range(6)],
    "ml_signal":   ["ml_signal"],
}

# --- Paused/deprecated groups ---

PAUSED_GROUPS: Dict[str, str] = {
    "pattern":               "pattern detector module missing; all pattern_i cols always zero",
    "trend_momentum_first":  "AdaptiveLSTM bias experiment; not validated",
    "bias_saturation":       "bias_fixed_temperature experiment; not validated",
}

KNOWN_DEAD_COLUMNS: Dict[str, str] = {
    **{f"pattern_{i}": "pattern detector missing" for i in range(11)},
    **{f"cross_asset_{i}": "no live data for this cross-asset slot" for i in range(6, 18)},
}

# --- Ablation groups ---

ABLATION_GROUPS: List[str] = [
    "ALL", "NO_TREND", "NO_MOMENTUM", "NO_TREND_MOMENTUM",
    "NO_VOLATILITY", "NO_VOLUME", "NO_CROSS_ASSET", "NO_ML_SIGNAL", "NO_REGIME",
]


# --- Derived FEATURE_GROUPS with indices ---

def _derive_feature_groups() -> Dict[str, dict]:
    """Build FEATURE_GROUPS from FEATURE_GROUPS_BY_NAME, resolving name -> index."""
    groups: Dict[str, dict] = {}
    for group_name, col_names in FEATURE_GROUPS_BY_NAME.items():
        indices = []
        for name in col_names:
            try:
                indices.append(ENGINEERED_V2_COLUMNS.index(name))
            except ValueError:
                pass
        groups[group_name] = {
            "indices": sorted(indices),
            "columns": col_names,
            "description": ", ".join(col_names),
        }
    return groups

FEATURE_GROUPS: Dict[str, dict] = _derive_feature_groups()


# --- Helper functions ---

def get_feature_index(name: str) -> Optional[int]:
    """Return column index for a feature name, or None."""
    try:
        return ENGINEERED_V2_COLUMNS.index(name)
    except ValueError:
        return None


def get_group_indices(group_name: str) -> List[int]:
    """Return column indices for an active feature group."""
    g = FEATURE_GROUPS.get(group_name)
    return g["indices"] if g else []


def get_ablation_indices(ablation_group: str) -> List[int]:
    """Return indices to zero out for an ablation group name."""
    if ablation_group == "ALL" or not ablation_group.startswith("NO_"):
        return []
    group_key = ablation_group[3:].lower()
    if group_key == "trend_momentum":
        return sorted(set(get_group_indices("trend") + get_group_indices("momentum")))
    return get_group_indices(group_key)


# --- Audit utilities ---

def detect_dead_columns(
    features: np.ndarray,
    zero_threshold: float = 1e-8,
) -> Tuple[List[int], Dict[int, str]]:
    """Detect zero-variance or all-NaN columns. Returns (dead_indices, {index: reason})."""
    n_cols = features.shape[1]
    stds = np.nanstd(features, axis=0)
    # all-NaN columns produce NaN std which would NOT satisfy <= zero_threshold
    dead_indices = [
        i for i in range(n_cols)
        if (not np.isfinite(stds[i])) or stds[i] <= zero_threshold
    ]
    reasons: Dict[int, str] = {}
    for i in dead_indices:
        name = ENGINEERED_V2_COLUMNS[i] if i < len(ENGINEERED_V2_COLUMNS) else f"<unnamed_{i}>"
        known = KNOWN_DEAD_COLUMNS.get(name)
        reasons[i] = known or "zero variance (std=%.2e)" % stds[i]
    return dead_indices, reasons


def audit_column(
    features: np.ndarray,
    col_idx: int,
    forward_returns: Optional[np.ndarray] = None,
) -> dict:
    """Audit a single column, returning all diagnostic stats."""
    col = features[:, col_idx]
    finite = np.isfinite(col)
    nonzero = np.abs(col) > 1e-12
    name = ENGINEERED_V2_COLUMNS[col_idx] if col_idx < len(ENGINEERED_V2_COLUMNS) else f"<unnamed_{col_idx}>"

    stats: dict = {
        "index": col_idx,
        "name": name,
        "mean": float(np.nanmean(col)),
        "std": float(np.nanstd(col)),
        "min": float(np.nanmin(col)) if finite.any() else float("nan"),
        "max": float(np.nanmax(col)) if finite.any() else float("nan"),
        "nonzero_pct": float(nonzero.mean() * 100),
        "nan_count": int((~finite).sum()),
        "inf_count": int(np.isinf(col).sum()),
    }

    if forward_returns is not None and len(forward_returns) == len(col):
        mask = finite & np.isfinite(forward_returns)
        if mask.sum() > 1:
            corr = np.corrcoef(col[mask], forward_returns[mask])[0, 1]
            stats["corr_fwd_return"] = float(corr) if np.isfinite(corr) else 0.0
        else:
            stats["corr_fwd_return"] = 0.0

    return stats


def print_column_audit(
    features: np.ndarray,
    forward_returns: Optional[np.ndarray] = None,
    title: str = "COLUMN AUDIT",
) -> None:
    """Print a full human-readable column audit table."""
    n_cols = features.shape[1]
    dead_indices, dead_reasons = detect_dead_columns(features)

    print()
    print("=" * 95)
    print(f"  [{title}]  n_features={n_cols}  n_samples={features.shape[0]}  dead={len(dead_indices)}/{n_cols}")
    print("=" * 95)
    header = f"{'idx':>3s}  {'name':<22s}  {'mean':>8s}  {'std':>8s}  {'nonzero%':>8s}  {'NaN':>5s}  {'inf':>5s}  {'corr_fwd':>8s}  status"
    print(header)
    print("-" * 95)

    for i in range(n_cols):
        s = audit_column(features, i, forward_returns)
        corr_val = s.get('corr_fwd_return')
        corr_str = f"{corr_val:+.4f}" if corr_val is not None else "     N/A"
        if i in dead_indices:
            status = f"[DEAD] {dead_reasons[i]}"
        else:
            status = f"[LIVE] std={s['std']:.4f}"
        print(
            f"{s['index']:3d}  {s['name']:<22s}  {s['mean']:8.4f}  {s['std']:8.4f}  "
            f"{s['nonzero_pct']:7.1f}%  {s['nan_count']:4d}  {s['inf_count']:4d}  {corr_str:>8s}  {status}"
        )

    print("=" * 95)

    # Group verification
    print()
    print("[GROUP_VERIFY]  Active groups and indices:")
    for group_name, group_info in FEATURE_GROUPS.items():
        indices = group_info["indices"]
        dead_in_group = [i for i in indices if i in dead_indices]
        tag = "" if not dead_in_group else f"  !! {len(dead_in_group)} dead cols"
        print(f"  {group_name:<15s}  indices={str(indices):<30s}  {group_info['columns']}{tag}")

    print()
    print("[PAUSED_GROUPS]")
    for group_name, reason in PAUSED_GROUPS.items():
        print(f"  {group_name:<22s}  {reason}")
    print()


# --- Quick sanity check ---
if __name__ == "__main__":
    print(f"ENGINEERED_V2_COLUMNS: {N_FEATURES} columns")
    print(f"Active groups: {list(FEATURE_GROUPS.keys())}")
    print(f"Paused groups: {list(PAUSED_GROUPS.keys())}")
    print(f"Ablation groups: {ABLATION_GROUPS}")
    for name, info in FEATURE_GROUPS.items():
        print(f"  {name}: indices={info['indices']}  columns={info['columns']}")
