"""
OutcomeLabeler — deterministic rule engine that tags trades with
outcome_label and mistake_label based on realised result vs model expectation.
"""
from __future__ import annotations

from typing import Dict, Any, Optional

from loguru import logger


class OutcomeLabeler:
    """
    Assigns labels used by TradeCoroner and ReplayBuilder.
    """

    def label(
        self,
        trade: Dict[str, Any],
        predicted_direction: Optional[str] = None,
        predicted_return: Optional[float] = None,
        model_confidence: Optional[float] = None,
        execution_slippage_bps: Optional[float] = None,
        spread_assumed_bps: Optional[float] = None,
        regime_at_entry: Optional[str] = None,
        regime_at_exit: Optional[str] = None,
        news_hit_within_minutes: Optional[int] = None,
    ) -> Dict[str, str]:
        """
        Return {"outcome_label": ..., "mistake_label": ...} for a closed trade.

        Args:
            trade: closed trade dict with at least pnl, pnl_pct, mfe, mae, exit_reason.
            predicted_direction: "BUY" / "SELL" / None.
            predicted_return: model predicted return (pct).
            model_confidence: 0.0-1.0.
            execution_slippage_bps: actual slippage in basis points.
            spread_assumed_bps: spread assumed by risk engine.
            regime_at_entry: regime label at entry.
            regime_at_exit: regime label at exit.
            news_hit_within_minutes: if a news event occurred within N min of entry/exit.
        """
        pnl = float(trade.get("pnl", 0.0))
        pnl_pct = float(trade.get("pnl_pct", 0.0))
        side = str(trade.get("side", "")).upper()
        mfe = float(trade.get("mfe", 0.0))
        mae = float(trade.get("mae", 0.0))
        exit_reason = str(trade.get("exit_reason", "")).lower()

        outcome = "flat_noise"
        mistake = "none"

        # Helper: did model predict direction correctly?
        direction_correct: Optional[bool] = None
        if predicted_direction and side:
            direction_correct = predicted_direction.upper() == side

        # ------------------------------------------------------------------
        # Winners
        # ------------------------------------------------------------------
        if pnl > 0:
            if direction_correct is True:
                # Good prediction + good execution = clean winner
                outcome = "winner_clean"
                mistake = "none"
            elif direction_correct is False:
                # Predicted wrong direction but still won = lucky
                outcome = "winner_lucky"
                mistake = "model_miscalibration"
            else:
                outcome = "winner_clean"
                mistake = "none"

            # Override if slippage/spread ate most of the edge
            if execution_slippage_bps is not None and spread_assumed_bps is not None:
                total_cost_bps = execution_slippage_bps + spread_assumed_bps
                if pnl_pct > 0 and total_cost_bps > abs(pnl_pct) * 100:
                    outcome = "winner_lucky"
                    mistake = "ignored_spread"

            return {"outcome_label": outcome, "mistake_label": mistake}

        # ------------------------------------------------------------------
        # Losers
        # ------------------------------------------------------------------
        if pnl <= 0:
            if direction_correct is True:
                # Model was right but we lost anyway → expected loss (noise)
                outcome = "loser_expected"
                mistake = "none"
            elif direction_correct is False:
                # Model predicted wrong and we lost
                outcome = "loser_bad_entry"
                mistake = "overfit_signal"
            else:
                outcome = "loser_expected"
                mistake = "none"

            # Execution quality overrides
            if execution_slippage_bps is not None and spread_assumed_bps is not None:
                actual_cost = execution_slippage_bps + (float(trade.get("spread_paid", 0.0)) * 100)
                if actual_cost > (spread_assumed_bps * 2):
                    outcome = "loser_execution_slippage"
                    mistake = "ignored_slippage"

            # Spread-specific death
            if exit_reason in ("stop_loss", "sl") and mae < 0 and abs(mae) < abs(pnl) * 0.5:
                if float(trade.get("spread_paid", 0.0)) > abs(pnl * 0.3):
                    outcome = "loser_spread"
                    mistake = "ignored_spread"

            # Regime shift after entry
            if regime_at_entry and regime_at_exit and regime_at_entry != regime_at_exit:
                outcome = "loser_regime_shift"
                mistake = "regime_miss"

            # News spike
            if news_hit_within_minutes is not None and news_hit_within_minutes <= 5:
                outcome = "loser_news_spike"
                mistake = "news_miss"

            # Overfit: tiny adverse move stopped us out despite "correct" prediction
            if (
                direction_correct is True
                and model_confidence is not None
                and model_confidence > 0.85
                and abs(pnl_pct) < 0.05
            ):
                outcome = "loser_overfit_signal"
                mistake = "overfit_signal"

            # Bad exit: MFE was positive but we exited at a loss
            if mfe > 0 and pnl < 0:
                if exit_reason in ("stop_loss", "sl", "timeout"):
                    outcome = "loser_bad_exit"
                    mistake = "bad_exit_timing"

            return {"outcome_label": outcome, "mistake_label": mistake}

        # Fallback (should not reach here)
        return {"outcome_label": outcome, "mistake_label": mistake}
