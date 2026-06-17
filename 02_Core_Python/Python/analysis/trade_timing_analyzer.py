"""
Trade Timing Analyzer
Helps answer: "When were the most profitable trades taken?"
and surfaces patterns around market opens and news events.

Intended to be used:
- During/after Decision PPO training runs
- On the trade_journal to feed better signals back into feature engineering and reward shaping
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

# Optional PatternDetector for on-the-fly detection when journal has OHLCV
try:
    from Python.patterns.pattern_detector import PatternDetector, PATTERN_FEATURE_NAMES
    _HAS_PATTERN_DETECTOR = True
except Exception:
    _HAS_PATTERN_DETECTOR = False
    PatternDetector = None
    PATTERN_FEATURE_NAMES = []


def analyze_profitable_trade_timing(
    journal_path: str | Path = "logs/trade_journal/trade_journal.jsonl",
    top_n: int = 50,
) -> Dict:
    """
    Load the trade journal and analyze when the most profitable trades occurred.
    Returns insights on:
    - Best hours / session windows
    - Performance relative to news events
    - Recommendations for the Decision PPO (e.g. stronger news avoidance)
    """
    path = Path(journal_path)
    if not path.exists():
        return {"error": f"Journal not found at {path}"}

    try:
        df = pd.read_json(path, lines=True)
    except Exception as e:
        return {"error": f"Failed to read journal: {e}"}

    if df.empty or "pnl" not in df.columns:
        return {"error": "No valid P&L data in journal"}

    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce").fillna(0)
    df["was_profitable"] = df["pnl"] > 0

    # Time features
    if "timestamp" in df.columns:
        df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["hour"] = df["ts"].dt.hour
        df["day_of_week"] = df["ts"].dt.dayofweek
    else:
        df["hour"] = 0
        df["day_of_week"] = 0

    results = {}

    # Top profitable trades
    top_trades = df.nlargest(top_n, "pnl")[["pnl", "hour", "symbol", "news_distance_minutes", "session"]].copy()
    results["top_profitable_trades"] = top_trades.to_dict(orient="records")

    # Best hours overall
    hourly = df.groupby("hour")["pnl"].agg(["sum", "mean", "count"]).reset_index()
    hourly = hourly.sort_values("sum", ascending=False)
    results["best_hours_by_pnl"] = hourly.head(8).to_dict(orient="records")

    # Session performance
    if "session" in df.columns:
        session_perf = df.groupby("session")["pnl"].agg(["sum", "mean", "count"]).reset_index()
        results["session_performance"] = session_perf.sort_values("sum", ascending=False).to_dict(orient="records")

    # News proximity analysis
    if "news_distance_minutes" in df.columns:
        df["news_bucket"] = pd.cut(
            df["news_distance_minutes"].fillna(999),
            bins=[-1, 15, 30, 60, 120, 999],
            labels=["<15min", "15-30min", "30-60min", "1-2h", ">2h or none"]
        )
        news_perf = df.groupby("news_bucket")["pnl"].agg(["sum", "mean", "count"]).reset_index()
        results["news_proximity_performance"] = news_perf.to_dict(orient="records")

        # Recommendation signal
        near_news = df[df["news_distance_minutes"] < 30]["pnl"].sum()
        far_news = df[df["news_distance_minutes"] >= 60]["pnl"].sum()
        results["news_avoidance_recommendation"] = {
            "pnl_near_high_impact_news": float(near_news),
            "pnl_away_from_news": float(far_news),
            "suggestion": "Strongly avoid entries <30min before high-impact news" if near_news < 0 else "News proximity neutral or positive"
        }

    # Market open windows (rough)
    open_window = df[(df["hour"].isin([7,8,13,14])) & (df["pnl"] > 0)]
    results["profitable_trades_in_open_windows"] = len(open_window)
    results["total_profitable_trades"] = int(df["was_profitable"].sum())

    return results


def analyze_by_patterns_and_timing(
    journal_path: str | Path = "logs/trade_journal/trade_journal.jsonl",
    min_samples: int = 5,
) -> Dict:
    """
    NEW: Profitability breakdown by detected classical patterns + timing context.
    Gives the "edge" visibility for Decision PPO + Dreamer + Rainforest.
    Groups trades by dominant pattern (or has_* flags if present) crossed with session/news/open windows.
    """
    path = Path(journal_path)
    if not path.exists():
        return {"error": f"Journal not found at {path}"}

    try:
        df = pd.read_json(path, lines=True)
    except Exception as e:
        return {"error": f"Failed to read journal: {e}"}

    if df.empty or "pnl" not in df.columns:
        return {"error": "No valid P&L data in journal"}

    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce").fillna(0)
    df["was_profitable"] = df["pnl"] > 0

    # Try to extract or synthesize pattern labels
    pattern_col = None
    for cand in ["dominant_pattern", "pattern", "detected_pattern", "pattern_name"]:
        if cand in df.columns:
            pattern_col = cand
            break

    if pattern_col is None and _HAS_PATTERN_DETECTOR:
        # Attempt reconstruction if OHLC present in journal (rare but for E2E)
        ohlc_cols = {"open", "high", "low", "close"}
        if ohlc_cols.issubset(set(df.columns)):
            try:
                detector = PatternDetector()
                patterns = []
                for _, row in df.iterrows():
                    try:
                        sub = pd.DataFrame([row[list(ohlc_cols)]])
                        st = detector.detect(sub)
                        dom = st.dominant_pattern.name if st.dominant_pattern else "none"
                        patterns.append(dom)
                    except Exception:
                        patterns.append("unknown")
                df["derived_pattern"] = patterns
                pattern_col = "derived_pattern"
            except Exception:
                pass

    # Fallback: use any has_* columns if present (from enriched features in journal)
    has_pattern_cols = [c for c in df.columns if c.startswith("has_") and c in (PATTERN_FEATURE_NAMES or [])]
    if pattern_col is None and has_pattern_cols:
        # Pick strongest has_ per row as label
        def pick_pat(r):
            best, val = "no_pattern", 0.0
            for c in has_pattern_cols:
                if float(r.get(c, 0)) > val:
                    val = float(r[c])
                    best = c.replace("has_", "")
            return best
        df["pattern_from_has"] = df.apply(pick_pat, axis=1)
        pattern_col = "pattern_from_has"

    if pattern_col is None:
        df["pattern"] = "unknown"
        pattern_col = "pattern"

    results: Dict = {"pattern_column_used": pattern_col}

    # Pattern-level profitability
    pat_perf = df.groupby(pattern_col)["pnl"].agg(["sum", "mean", "count", "std"]).reset_index()
    pat_perf = pat_perf[pat_perf["count"] >= min_samples].sort_values("sum", ascending=False)
    results["pattern_profitability"] = pat_perf.to_dict(orient="records")

    # Cross with timing (hour/session/news)
    if "hour" not in df.columns and "timestamp" in df.columns:
        df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["hour"] = df["ts"].dt.hour

    # Pattern x session interaction (if session present)
    if "session" in df.columns:
        cross = df.groupby([pattern_col, "session"])["pnl"].agg(["sum", "mean", "count"]).reset_index()
        cross = cross[cross["count"] >= max(3, min_samples // 2)]
        results["pattern_by_session"] = cross.sort_values("sum", ascending=False).head(20).to_dict(orient="records")

    # Pattern x news proximity buckets
    if "news_distance_minutes" in df.columns:
        df["news_bucket"] = pd.cut(
            df["news_distance_minutes"].fillna(999),
            bins=[-1, 15, 45, 120, 999],
            labels=["<15min_news", "15-45min", "45-120min", "far_from_news"]
        )
        cross_news = df.groupby([pattern_col, "news_bucket"])["pnl"].agg(["sum", "mean", "count"]).reset_index()
        results["pattern_by_news_proximity"] = cross_news[cross_news["count"] >= 3].sort_values("sum", ascending=False).head(15).to_dict(orient="records")

    # Open window specific (high edge potential)
    open_mask = df.get("hour", pd.Series(0)).isin([7, 8, 13, 14])
    open_pat = df[open_mask].groupby(pattern_col)["pnl"].agg(["sum", "mean", "count"]).reset_index()
    results["pattern_profit_in_open_windows"] = open_pat.sort_values("sum", ascending=False).to_dict(orient="records")

    # Recommendation for DecisionPPO / ensemble
    best_pats = pat_perf.head(3)[pattern_col].tolist() if not pat_perf.empty else []
    results["recommendations_for_rich_decisions"] = {
        "favor_patterns": best_pats,
        "timing_note": "Bias TimeExitSpec longer holds + aggressive partials on top patterns near opens / away from news; tighten on doji or high-news windows.",
        "dreamer_rainforest_note": "World model + Rainforest now condition on these pattern+timing states for superior imagination and regime forecasts."
    }

    return results


# Back-compat: main analyze now also calls pattern analysis
_original_analyze = analyze_profitable_trade_timing

def analyze_profitable_trade_timing(
    journal_path: str | Path = "logs/trade_journal/trade_journal.jsonl",
    top_n: int = 50,
) -> Dict:
    base = _original_analyze(journal_path, top_n)
    try:
        pat = analyze_by_patterns_and_timing(journal_path)
        base["patterns_and_timing"] = pat
    except Exception as exc:
        base["patterns_and_timing"] = {"error": str(exc)}
    return base


if __name__ == "__main__":
    insights = analyze_profitable_trade_timing()
    import json
    print(json.dumps(insights, indent=2, default=str))
