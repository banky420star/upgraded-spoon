"""
BaselineComparator — Compare candidate strategy against baseline policies.
"""
from __future__ import annotations

import uuid
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from Python.validation.backtest import BacktestCourt


class BaselineComparator:
    """
    Compares a candidate strategy against baseline policies.
    Must beat random, buy-and-hold, and previous champion to pass.
    """

    def __init__(
        self,
        court: Optional[BacktestCourt] = None,
        random_seed: int = 42,
    ):
        self.court = court or BacktestCourt()
        self.random_seed = int(random_seed)

    def run(
        self,
        df: pd.DataFrame,
        symbol: str,
        bundle_id: str = "",
        candidate_signals: Optional[pd.Series] = None,
        candidate_policy: Optional[Callable[[pd.DataFrame, int, Dict[str, Any]], Dict[str, Any]]] = None,
        previous_champion_signals: Optional[pd.Series] = None,
        previous_champion_policy: Optional[Callable] = None,
        comparison_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run candidate and all baselines on the same data.

        Returns:
            JSON-serializable baseline comparison artifact.
        """
        comparison_id = comparison_id or str(uuid.uuid4())
        rng = np.random.default_rng(self.random_seed)

        baselines: Dict[str, pd.Series] = {}

        # 1. Random policy
        random_actions = rng.choice([-1, 0, 1], size=len(df))
        baselines["random"] = pd.Series(random_actions, index=df.index)

        # 2. Always flat
        baselines["flat"] = pd.Series(np.zeros(len(df), dtype=int), index=df.index)

        # 3. Buy-and-hold: buy at first bar, hold forever
        ba = np.zeros(len(df), dtype=int)
        ba[0] = 1
        baselines["buy_hold"] = pd.Series(ba, index=df.index)

        # 4. Moving-average crossover
        short = df["close"].rolling(window=20, min_periods=1).mean()
        long = df["close"].rolling(window=50, min_periods=1).mean()
        ma = np.zeros(len(df), dtype=int)
        ma[short > long] = 1
        ma[short < long] = -1
        baselines["ma_crossover"] = pd.Series(ma, index=df.index)

        # 5. Previous champion (provided)
        if previous_champion_signals is not None:
            baselines["previous_champion"] = previous_champion_signals
        elif previous_champion_policy is not None:
            baselines["previous_champion"] = None  # will run via policy
        else:
            # Default to flat if no champion provided
            baselines["previous_champion"] = baselines["flat"]

        # 6. LSTM-only signal (synthetic: smoother version of MA)
        smooth = df["close"].rolling(window=10, min_periods=1).mean()
        lstm = np.zeros(len(df), dtype=int)
        lstm[smooth > smooth.shift(1)] = 1
        lstm[smooth < smooth.shift(1)] = -1
        baselines["lstm_only"] = pd.Series(lstm, index=df.index)

        # 7. PPO-without-Dreamer (synthetic: noiser MA)
        ppo_nd = np.zeros(len(df), dtype=int)
        mask_long = short > long
        mask_short = short < long
        ppo_nd[mask_long] = np.where(rng.random(mask_long.sum()) > 0.3, 1, 0)
        ppo_nd[mask_short] = np.where(rng.random(mask_short.sum()) > 0.3, -1, 0)
        baselines["ppo_without_dreamer"] = pd.Series(ppo_nd, index=df.index)

        # 8. PPO-without-Rainforest (synthetic: delayed MA)
        ppo_nr = np.zeros(len(df), dtype=int)
        delayed_short = short.shift(3).bfill()
        ppo_nr[delayed_short > long] = 1
        ppo_nr[delayed_short < long] = -1
        baselines["ppo_without_rainforest"] = pd.Series(ppo_nr, index=df.index)

        # Run candidate
        candidate_result = self.court.run(
            df=df,
            symbol=symbol,
            bundle_id=bundle_id,
            signals=candidate_signals,
            policy=candidate_policy,
            backtest_id=f"{comparison_id}_candidate",
        )

        # Run baselines
        baseline_results: Dict[str, Dict[str, Any]] = {}
        for name, sig in baselines.items():
            if sig is None:
                # Run via policy if signals not provided
                if name == "previous_champion" and previous_champion_policy is not None:
                    baseline_results[name] = self.court.run(
                        df=df,
                        symbol=symbol,
                        bundle_id=bundle_id,
                        policy=previous_champion_policy,
                        backtest_id=f"{comparison_id}_{name}",
                    )
                continue
            baseline_results[name] = self.court.run(
                df=df,
                symbol=symbol,
                bundle_id=bundle_id,
                signals=sig,
                backtest_id=f"{comparison_id}_{name}",
            )

        candidate_return = candidate_result["net_return_after_costs"]
        random_return = baseline_results.get("random", {}).get("net_return_after_costs", 0.0)
        buy_hold_return = baseline_results.get("buy_hold", {}).get("net_return_after_costs", 0.0)
        previous_champion_return = baseline_results.get("previous_champion", {}).get("net_return_after_costs", 0.0)

        beats_random = candidate_return > random_return
        beats_buy_hold = candidate_return > buy_hold_return
        beats_previous_champion = candidate_return > previous_champion_return

        passed = beats_random and beats_buy_hold and beats_previous_champion

        return {
            "baseline_comparison_id": comparison_id,
            "bundle_id": bundle_id,
            "candidate_return": float(candidate_return),
            "random_return": float(random_return),
            "buy_hold_return": float(buy_hold_return),
            "previous_champion_return": float(previous_champion_return),
            "beats_random": bool(beats_random),
            "beats_buy_hold": bool(beats_buy_hold),
            "beats_previous_champion": bool(beats_previous_champion),
            "passed": bool(passed),
            "candidate": candidate_result,
            "baselines": {k: v for k, v in baseline_results.items()},
        }
