"""
TradeCoroner — post-mortem analyzer for every closed trade.

Compares expected vs actual on:
  - return, direction, drawdown
  - Dreamer expected risk vs actual
  - Rainforest regime vs observed
  - PPO confidence vs result
  - spread / slippage assumptions vs actual

Outputs a coroner report used by the feedback loop.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

from loguru import logger


@dataclass
class CoronerReport:
    trade_id: str
    expected_action: str
    actual_outcome: str
    mistake_label: str
    root_causes: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    eligible_for_retraining: bool = False


class TradeCoroner:
    """
    Analyse a single closed trade against model expectations and
    market context. Produces a CoronerReport.
    """

    def __init__(
        self,
        data_dir: str = "logs",
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def analyse(
        self,
        trade: Dict[str, Any],
        expected_return_pct: Optional[float] = None,
        expected_direction: Optional[str] = None,
        expected_max_drawdown_pct: Optional[float] = None,
        dreamer_expected_risk: Optional[float] = None,
        rainforest_regime_at_entry: Optional[str] = None,
        rainforest_regime_at_exit: Optional[str] = None,
        ppo_confidence: Optional[float] = None,
        assumed_spread_bps: Optional[float] = None,
        assumed_slippage_bps: Optional[float] = None,
        outcome_label: Optional[str] = None,
        mistake_label: Optional[str] = None,
    ) -> CoronerReport:
        """
        Run the post-mortem and return a CoronerReport.
        """
        trade_id = str(trade.get("trade_id", "unknown"))
        actual_pnl_pct = float(trade.get("pnl_pct", 0.0))
        actual_direction = str(trade.get("side", "")).upper()
        actual_mae = float(trade.get("mae", 0.0))
        actual_slippage = float(trade.get("slippage", 0.0))
        actual_spread_paid = float(trade.get("spread_paid", 0.0))
        actual_fees = float(trade.get("fees", 0.0))
        exit_reason = str(trade.get("exit_reason", "")).lower()

        root_causes: List[str] = []
        recommendations: List[str] = []
        eligible = False

        # 1. Expected vs actual return
        if expected_return_pct is not None:
            deviation = actual_pnl_pct - expected_return_pct
            if abs(deviation) > 1.0:
                root_causes.append(
                    f"return deviation: expected {expected_return_pct:.2f}% vs actual {actual_pnl_pct:.2f}%"
                )
                recommendations.append("Recalibrate return target for this regime")
                eligible = True

        # 2. Expected vs actual direction
        if expected_direction and expected_direction.upper() != actual_direction:
            root_causes.append(
                f"direction mismatch: expected {expected_direction.upper()} vs actual {actual_direction}"
            )
            recommendations.append("Review signal feature set for directional bias")
            eligible = True

        # 3. Expected vs actual drawdown
        if expected_max_drawdown_pct is not None:
            actual_dd = abs(actual_mae)
            if actual_dd > expected_max_drawdown_pct:
                root_causes.append(
                    f"drawdown exceeded: expected max {expected_max_drawdown_pct:.2f}% vs actual {actual_dd:.2f}%"
                )
                recommendations.append("Tighten stop-loss or reduce position size")
                eligible = True

        # 4. Dreamer expected risk vs actual
        if dreamer_expected_risk is not None:
            actual_risk = abs(actual_mae) + actual_slippage + actual_spread_paid + actual_fees
            if actual_risk > dreamer_expected_risk * 1.5:
                root_causes.append(
                    f"Dreamer risk underestimate: expected {dreamer_expected_risk:.4f} vs actual {actual_risk:.4f}"
                )
                recommendations.append("Increase Dreamer risk margin by 20%")
                eligible = True

        # 5. Rainforest regime vs observed
        if (
            rainforest_regime_at_entry
            and rainforest_regime_at_exit
            and rainforest_regime_at_entry != rainforest_regime_at_exit
        ):
            root_causes.append(
                f"regime shift during trade: {rainforest_regime_at_entry} -> {rainforest_regime_at_exit}"
            )
            recommendations.append("Add regime-shift exit rule or widen regime buffer")
            eligible = True

        # 6. PPO confidence vs result
        if ppo_confidence is not None:
            if ppo_confidence > 0.8 and actual_pnl_pct < 0:
                root_causes.append(
                    f"high-confidence loss: PPO confidence {ppo_confidence:.2f} but lost {actual_pnl_pct:.2f}%"
                )
                recommendations.append("Flag for model recalibration — confidence calibration drift")
                eligible = True
            elif ppo_confidence < 0.5 and actual_pnl_pct > 0:
                root_causes.append(
                    f"low-confidence winner: PPO confidence {ppo_confidence:.2f} but won {actual_pnl_pct:.2f}%"
                )
                recommendations.append("Investigate whether feature set is incomplete")
                eligible = True

        # 7. Spread / slippage assumptions vs actual
        if assumed_spread_bps is not None:
            actual_spread_bps = actual_spread_paid * 100  # rough conversion if stored as price units
            if actual_spread_bps > assumed_spread_bps * 1.5:
                root_causes.append(
                    f"spread underestimate: assumed {assumed_spread_bps:.2f} bps vs actual {actual_spread_bps:.2f} bps"
                )
                recommendations.append("Update spread assumptions in risk engine")
                eligible = True

        if assumed_slippage_bps is not None:
            actual_slippage_bps = actual_slippage * 100
            if actual_slippage_bps > assumed_slippage_bps * 2.0:
                root_causes.append(
                    f"slippage blowout: assumed {assumed_slippage_bps:.2f} bps vs actual {actual_slippage_bps:.2f} bps"
                )
                recommendations.append("Check execution mode / broker latency or widen slippage buffer")
                eligible = True

        # 8. Use outcome / mistake labels if caller already computed them
        if outcome_label:
            if outcome_label.startswith("loser_"):
                eligible = True
        if mistake_label and mistake_label != "none":
            if mistake_label not in root_causes:
                root_causes.append(f"labelled mistake: {mistake_label}")
            recommendations.append(f"Address {mistake_label} in next training cycle")
            eligible = True

        report = CoronerReport(
            trade_id=trade_id,
            expected_action=expected_direction or "unknown",
            actual_outcome=f"{actual_direction} {actual_pnl_pct:.3f}%",
            mistake_label=mistake_label or "none",
            root_causes=root_causes,
            recommended_actions=recommendations,
            eligible_for_retraining=eligible,
        )

        self._save_report(report)
        return report

    def _save_report(self, report: CoronerReport) -> None:
        path = self.data_dir / f"coroner_{report.trade_id}.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(asdict(report), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save coroner report {report.trade_id}: {e}")
