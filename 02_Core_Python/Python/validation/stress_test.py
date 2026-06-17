"""
StressTester — Run backtests under adverse conditions.
"""
from __future__ import annotations

import copy
import uuid
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from Python.validation.backtest import BacktestCourt
from Python.validation.cost_model import CostModel


class StressTester:
    """
    Runs backtests under stressed market and cost conditions.
    Promotion requires passing the stress test.
    """

    def __init__(
        self,
        base_court: Optional[BacktestCourt] = None,
        stress_spread_multiplier: float = 2.0,
        stress_slippage_multiplier: float = 2.0,
        stress_volatility_multiplier: float = 1.5,
        regime_shift_prob: float = 0.05,
        min_passed_scenarios: int = 3,
    ):
        self.base_court = base_court or BacktestCourt()
        self.stress_spread_multiplier = float(stress_spread_multiplier)
        self.stress_slippage_multiplier = float(stress_slippage_multiplier)
        self.stress_volatility_multiplier = float(stress_volatility_multiplier)
        self.regime_shift_prob = float(regime_shift_prob)
        self.min_passed_scenarios = int(min_passed_scenarios)

    def run(
        self,
        df: pd.DataFrame,
        symbol: str,
        bundle_id: str = "",
        signals: Optional[pd.Series] = None,
        policy: Optional[Callable[[pd.DataFrame, int, Dict[str, Any]], Dict[str, Any]]] = None,
        stress_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run 4 stress scenarios:
        1. doubled spread
        2. doubled slippage
        3. 50% higher volatility (synthetic price noise)
        4. random regime shifts (gapped reversals)

        Returns:
            JSON-serializable stress test artifact.
        """
        stress_id = stress_id or str(uuid.uuid4())
        scenarios: List[Dict[str, Any]] = []

        # Scenario 1: doubled spread
        s1 = self._run_scenario(
            df, symbol, bundle_id, signals, policy,
            spread_mult=self.stress_spread_multiplier,
            slippage_mult=1.0,
            vol_mult=1.0,
            regime_shift=False,
            name="doubled_spread",
        )
        scenarios.append(s1)

        # Scenario 2: doubled slippage
        s2 = self._run_scenario(
            df, symbol, bundle_id, signals, policy,
            spread_mult=1.0,
            slippage_mult=self.stress_slippage_multiplier,
            vol_mult=1.0,
            regime_shift=False,
            name="doubled_slippage",
        )
        scenarios.append(s2)

        # Scenario 3: higher volatility
        s3 = self._run_scenario(
            df, symbol, bundle_id, signals, policy,
            spread_mult=1.0,
            slippage_mult=1.0,
            vol_mult=self.stress_volatility_multiplier,
            regime_shift=False,
            name="high_volatility",
        )
        scenarios.append(s3)

        # Scenario 4: random regime shifts
        s4 = self._run_scenario(
            df, symbol, bundle_id, signals, policy,
            spread_mult=1.0,
            slippage_mult=1.0,
            vol_mult=1.0,
            regime_shift=True,
            name="regime_shifts",
        )
        scenarios.append(s4)

        passed = sum(1 for s in scenarios if s.get("passed")) >= self.min_passed_scenarios
        worst_return = min(s.get("net_return_after_costs", 0.0) for s in scenarios)
        mean_return = float(np.mean([s.get("net_return_after_costs", 0.0) for s in scenarios]))
        worst_drawdown = max(s.get("max_drawdown", 0.0) for s in scenarios)

        return {
            "stress_id": stress_id,
            "bundle_id": bundle_id,
            "symbol": symbol,
            "scenarios": scenarios,
            "scenarios_total": len(scenarios),
            "scenarios_passed": sum(1 for s in scenarios if s.get("passed")),
            "scenarios_failed": sum(1 for s in scenarios if not s.get("passed")),
            "mean_return_after_costs": mean_return,
            "worst_return": worst_return,
            "worst_drawdown": worst_drawdown,
            "passed": bool(passed),
        }

    def _run_scenario(
        self,
        df: pd.DataFrame,
        symbol: str,
        bundle_id: str,
        signals: Optional[pd.Series],
        policy: Optional[Callable],
        spread_mult: float,
        slippage_mult: float,
        vol_mult: float,
        regime_shift: bool,
        name: str,
    ) -> Dict[str, Any]:
        df_stressed = self._apply_stress(
            df.copy(), vol_mult=vol_mult, regime_shift=regime_shift
        )

        # Build stressed cost model
        overrides = {symbol: {
            "spread_multiplier": spread_mult,
            "slippage_multiplier": slippage_mult,
        }}
        cost_model = CostModel(overrides=overrides)

        # Build court with stressed cost model
        court = BacktestCourt(
            cost_model=cost_model,
            initial_equity=self.base_court.initial_equity,
            position_size_fraction=self.base_court.position_size_fraction,
            max_positions=self.base_court.max_positions,
            risk_halt_drawdown=self.base_court.risk_halt_drawdown,
            sl_atr_multiplier=self.base_court.sl_atr_multiplier,
            tp_atr_multiplier=self.base_court.tp_atr_multiplier,
            trailing_trigger_pct=self.base_court.trailing_trigger_pct,
            trailing_distance_pct=self.base_court.trailing_distance_pct,
            max_hold_bars=self.base_court.max_hold_bars,
            min_profit_factor=self.base_court.min_profit_factor,
            max_drawdown_threshold=self.base_court.max_drawdown_threshold,
            min_trade_count=self.base_court.min_trade_count,
            max_single_trade_profit_share=self.base_court.max_single_trade_profit_share,
        )

        result = court.run(
            df=df_stressed,
            symbol=symbol,
            bundle_id=bundle_id,
            signals=signals,
            policy=policy,
        )
        return {
            "name": name,
            "backtest_id": result["backtest_id"],
            "net_return_after_costs": result["net_return_after_costs"],
            "max_drawdown": result["max_drawdown"],
            "profit_factor": result["profit_factor"],
            "sharpe": result["sharpe"],
            "trade_count": result["trade_count"],
            "passed": result["passed"],
        }

    def _apply_stress(
        self,
        df: pd.DataFrame,
        vol_mult: float,
        regime_shift: bool,
    ) -> pd.DataFrame:
        df = df.copy()
        if vol_mult != 1.0:
            # Inject noise proportional to existing volatility
            noise = np.random.normal(0, df["close"].std() * (vol_mult - 1.0), size=len(df))
            for col in ("open", "high", "low", "close"):
                if col in df.columns:
                    df[col] = df[col] + noise
            # Ensure high >= low
            df["high"] = np.maximum(df["high"], df["low"])
            df["close"] = np.clip(df["close"], df["low"], df["high"])

        if regime_shift:
            # Randomly flip close direction with small probability
            mask = np.random.rand(len(df)) < self.regime_shift_prob
            if mask.any():
                shifts = np.where(mask, df["close"] * np.random.choice([-1, 1], size=len(df)) * 0.005, 0.0)
                df["close"] = df["close"] + shifts
                df["open"] = df["open"] + shifts * 0.5
                df["high"] = np.maximum(df["high"], df["close"])
                df["low"] = np.minimum(df["low"], df["close"])

        return df.reset_index(drop=True)
