"""
CanaryMonitor — hourly monitor that auto-stops a demo canary when
performance guard-rails are breached.

Logs every check to logs/canary_monitor.jsonl.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

from loguru import logger

from Python.canary.demo_canary import DemoCanary


class CanaryMonitor:
    """
    Polls a DemoCanary on a cadence (expected hourly) and forces shutdown
    if daily loss, drawdown, or risk-violation thresholds are crossed.
    """

    def __init__(
        self,
        canary: DemoCanary,
        log_path: str = "logs/canary_monitor.jsonl",
        max_drawdown_pct: float = 5.0,
    ):
        self.canary = canary
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_drawdown_pct = max_drawdown_pct
        self.stopped: bool = False
        self.stop_reason: Optional[str] = None

    def check(self) -> Dict[str, Any]:
        """Run one monitoring cycle and append a JSONL record."""
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()

        reasons: list[str] = []

        # 1. Daily loss vs notional balance
        today = now.strftime("%Y-%m-%d")
        daily_net = sum(
            float(t.get("pnl", 0.0))
            - float(t.get("fees", 0.0))
            - float(t.get("spread_paid", 0.0))
            - float(t.get("slippage", 0.0))
            for t in self.canary.trades
            if str(t.get("exit_time", "")).startswith(today)
        )
        daily_loss_pct = abs(daily_net) / self.canary.notional_balance * 100.0
        if daily_loss_pct >= self.canary.cfg.max_daily_loss_pct:
            reasons.append(
                f"daily_loss_pct {daily_loss_pct:.2f}% >= max {self.canary.cfg.max_daily_loss_pct}%"
            )

        # 2. Max drawdown vs threshold
        drawdown_pct = self.canary.max_drawdown / self.canary.notional_balance * 100.0
        if drawdown_pct > self.max_drawdown_pct:
            reasons.append(
                f"max_drawdown {drawdown_pct:.2f}% > threshold {self.max_drawdown_pct}%"
            )

        # 3. Any risk violation
        if self.canary.risk_violations > 0:
            reasons.append(f"risk_violations={self.canary.risk_violations} > 0")

        # 4. Timing degradation rollback trigger (rich Decision PPO news/open window behavior)
        # If too many trades in news proximity relative to avoided, or negative timing score, rollback
        prox = getattr(self.canary, "timing_news_prox_trades", 0)
        avoided = getattr(self.canary, "timing_news_avoided_trades", 0)
        timing_score = getattr(self.canary, "timing_news_avoidance_score", 0.0) if hasattr(self.canary, "timing_news_avoidance_score") else 0.0
        total_timed = prox + avoided
        if total_timed >= 5 and (prox / max(1, total_timed) > 0.6 or timing_score < -0.5):
            reasons.append(f"timing_degradation: news_prox_ratio={prox/max(1,total_timed):.2f} score={timing_score:.2f} (poor news avoidance / open window perf from rich decisions)")

        if reasons:
            self.stopped = True
            self.stop_reason = "; ".join(reasons)
            logger.warning(
                f"CanaryMonitor STOP canary {self.canary.canary_id}: {self.stop_reason}"
            )

        record: Dict[str, Any] = {
            "timestamp": now_iso,
            "canary_id": self.canary.canary_id,
            "bundle_id": self.canary.bundle_id,
            "trades": len(self.canary.trades),
            "net_return": round(self.canary.net_return_after_costs, 4),
            "daily_loss_pct": round(daily_loss_pct, 4),
            "max_drawdown_pct": round(drawdown_pct, 4),
            "profit_factor": round(self.canary.profit_factor, 4),
            "risk_violations": self.canary.risk_violations,
            "stopped": self.stopped,
            "stop_reason": self.stop_reason,
            # Timing-specific for rich Decision PPO TimeExitSpec (open vs news avoidance)
            "timing_open_window_trades": getattr(self.canary, "timing_open_window_trades", 0),
            "timing_news_avoided_trades": getattr(self.canary, "timing_news_avoided_trades", 0),
            "timing_news_prox_trades": getattr(self.canary, "timing_news_prox_trades", 0),
            "timing_news_avoidance_score": getattr(self.canary, "timing_news_avoidance_score", 0.0) if hasattr(self.canary, "timing_news_avoidance_score") else round((getattr(self.canary, "timing_news_avoid_pnl", 0.0)) / max(1.0, self.canary.notional_balance) * 100, 4),
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return record

    def reset_stop(self) -> None:
        """Manual reset (e.g. after operator review)."""
        self.stopped = False
        self.stop_reason = None
        logger.info(f"CanaryMonitor reset for {self.canary.canary_id}")
