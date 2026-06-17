"""
WalkForwardValidator — Overlapping walk-forward analysis.
"""
from __future__ import annotations

import uuid
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from Python.validation.backtest import BacktestCourt


class WalkForwardValidator:
    """
    Creates overlapping train/validate windows.
    Pattern: train A -> validate B, train A+B -> validate C, etc.
    """

    def __init__(
        self,
        court: Optional[BacktestCourt] = None,
        min_windows: int = 5,
        min_passed_windows: int = 3,
        train_fraction: float = 0.6,
        window_overlap: float = 0.3,
    ):
        self.court = court or BacktestCourt()
        self.min_windows = int(min_windows)
        self.min_passed_windows = int(min_passed_windows)
        self.train_fraction = float(train_fraction)
        self.window_overlap = float(window_overlap)

    def run(
        self,
        df: pd.DataFrame,
        symbol: str,
        bundle_id: str = "",
        signals: Optional[pd.Series] = None,
        policy_factory: Optional[Callable[[pd.DataFrame], Callable[[pd.DataFrame, int, Dict[str, Any]], Dict[str, Any]]]] = None,
        walk_forward_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run walk-forward validation.

        Args:
            df: full historical DataFrame
            symbol: trading symbol
            bundle_id: model bundle identifier
            signals: optional precomputed signals for the entire df
            policy_factory: callable(train_df) -> policy callable for validation df
            walk_forward_id: optional unique id

        Returns:
            JSON-serializable walk-forward artifact.
        """
        walk_forward_id = walk_forward_id or str(uuid.uuid4())
        n = len(df)
        if n < self.min_windows * 200:
            # Too little data; create synthetic windows by slicing
            return self._run_synthetic(
                df, symbol, bundle_id, signals, policy_factory, walk_forward_id
            )

        windows = self._build_windows(n)
        results: List[Dict[str, Any]] = []

        for idx, (train_start, train_end, val_start, val_end) in enumerate(windows):
            train_df = df.iloc[train_start:train_end].copy().reset_index(drop=True)
            val_df = df.iloc[val_start:val_end].copy().reset_index(drop=True)

            if len(val_df) < 50:
                continue

            val_signals = None
            if signals is not None:
                val_signals = signals.iloc[val_start:val_end].reset_index(drop=True)

            val_policy = None
            if policy_factory is not None:
                try:
                    val_policy = policy_factory(train_df)
                except Exception:
                    val_policy = None

            result = self.court.run(
                df=val_df,
                symbol=symbol,
                bundle_id=bundle_id,
                signals=val_signals,
                policy=val_policy,
                backtest_id=f"{walk_forward_id}_window_{idx}",
            )
            results.append({
                "window_index": idx,
                "train_start": int(train_start),
                "train_end": int(train_end),
                "val_start": int(val_start),
                "val_end": int(val_end),
                "backtest": result,
            })

        return self._build_artifact(walk_forward_id, bundle_id, results)

    def _run_synthetic(
        self,
        df: pd.DataFrame,
        symbol: str,
        bundle_id: str,
        signals: Optional[pd.Series],
        policy_factory: Optional[Callable],
        walk_forward_id: str,
    ) -> Dict[str, Any]:
        """If data is too short, create sequential chunks."""
        n = len(df)
        chunk = max(50, n // self.min_windows)
        results: List[Dict[str, Any]] = []
        for idx in range(self.min_windows):
            start = idx * chunk
            end = min(start + chunk, n)
            if start >= n:
                break
            val_df = df.iloc[start:end].copy().reset_index(drop=True)
            val_signals = None
            if signals is not None:
                val_signals = signals.iloc[start:end].reset_index(drop=True)

            val_policy = None
            if policy_factory is not None:
                try:
                    val_policy = policy_factory(val_df)
                except Exception:
                    val_policy = None

            result = self.court.run(
                df=val_df,
                symbol=symbol,
                bundle_id=bundle_id,
                signals=val_signals,
                policy=val_policy,
                backtest_id=f"{walk_forward_id}_window_{idx}",
            )
            results.append({
                "window_index": idx,
                "train_start": start,
                "train_end": end,
                "val_start": start,
                "val_end": end,
                "backtest": result,
            })

        return self._build_artifact(walk_forward_id, bundle_id, results)

    def _build_windows(self, n: int) -> List[tuple]:
        windows: List[tuple] = []
        step = int(n * (1.0 - self.window_overlap) / self.min_windows)
        if step < 1:
            step = 1
        for i in range(self.min_windows):
            val_start = i * step
            val_end = min(val_start + max(step, int(n * (1.0 - self.train_fraction))), n)
            train_end = val_start
            train_start = max(0, train_end - int(n * self.train_fraction))
            if val_end <= val_start:
                continue
            windows.append((train_start, train_end, val_start, val_end))
        return windows

    def _build_artifact(
        self,
        walk_forward_id: str,
        bundle_id: str,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        windows_total = len(results)
        windows_passed = sum(1 for r in results if r["backtest"].get("passed"))
        windows_failed = windows_total - windows_passed

        returns = [r["backtest"].get("net_return_after_costs", 0.0) for r in results]
        mean_return = float(np.mean(returns)) if returns else 0.0
        worst_return = float(min(returns)) if returns else 0.0

        drawdowns = [r["backtest"].get("max_drawdown", 0.0) for r in results]
        max_drawdown = float(max(drawdowns)) if drawdowns else 0.0

        passed = windows_passed >= self.min_passed_windows and windows_total >= self.min_windows

        return {
            "walk_forward_id": walk_forward_id,
            "bundle_id": bundle_id,
            "windows_total": windows_total,
            "windows_passed": windows_passed,
            "windows_failed": windows_failed,
            "mean_return_after_costs": mean_return,
            "worst_window_return": worst_return,
            "max_drawdown": max_drawdown,
            "passed": bool(passed),
            "windows": results,
        }
