"""
DemoCanary — lightweight demo-account canary for model-validation gate.

Enforces strict demo-only trading limits, tracks performance, and produces
a canary artifact that gates promotion to champion / real-live.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
from loguru import logger


@dataclass
class CanaryConfig:
    enabled: bool = True
    require_account_type: str = "demo"
    max_lot_per_trade: float = 0.01
    max_open_positions: int = 1
    max_trades_per_hour: int = 3
    max_daily_loss_pct: float = 2.0
    max_spread_zscore: float = 2.0
    allow_auto_restart: bool = False
    allow_model_auto_promotion: bool = False


@dataclass
class CanaryArtifact:
    canary_id: str
    bundle_id: str
    system_mode: str
    account_type: str
    trades: int
    days_active: int
    net_return: float
    max_drawdown: float
    profit_factor: float
    risk_violations: int
    passed: bool
    approved_for_champion: bool
    approved_for_real_live: bool
    # Timing-aware metrics for Decision PPO rich TimeExitSpec monitoring (production hardening)
    timing_open_window_trades: int = 0
    timing_news_avoided_trades: int = 0
    timing_news_prox_trades: int = 0
    timing_window_pnl: float = 0.0
    timing_news_avoid_pnl: float = 0.0
    timing_news_avoidance_score: float = 0.0  # positive good (avoid pnl - prox pnl normalized)


class DemoCanary:
    """
    Demo-only canary that accumulates trades, enforces guard-rails,
    and emits a promotion-gating artifact on evaluation.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        data_dir: str = "logs",
        notional_balance: float = 10_000.0,
    ):
        self.cfg = CanaryConfig(**(config or {}))
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.canary_id: str = f"canary_{uuid.uuid4().hex[:8]}"
        self.bundle_id: str = "unknown"
        self.system_mode: str = "demo"
        self.account_type: str = "demo"

        # Equity tracking
        self.notional_balance: float = notional_balance
        self.current_balance: float = notional_balance
        self.peak_balance: float = notional_balance
        self.daily_start_balance: float = notional_balance
        self.max_drawdown: float = 0.0
        self.net_return_after_costs: float = 0.0

        # Counters
        self.trades: List[Dict[str, Any]] = []
        self.days_active: int = 0
        self.profit_factor: float = 0.0
        self.risk_violations: int = 0

        self._start_date: datetime = datetime.now(timezone.utc)
        self._hourly_trade_counts: Dict[str, int] = {}
        self._daily_trade_counts: Dict[str, int] = {}

        # Timing-specific metrics for rich Decision PPO + TimeExitSpec (production hardening)
        self.timing_open_window_trades: int = 0
        self.timing_news_avoided_trades: int = 0
        self.timing_news_prox_trades: int = 0  # within 30min high impact (should be low/negative for good policy)
        self.timing_window_pnl: float = 0.0
        self.timing_news_avoid_pnl: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def set_bundle(
        self,
        bundle_id: str,
        system_mode: str = "demo",
        account_type: str = "demo",
    ) -> None:
        self.bundle_id = bundle_id
        self.system_mode = system_mode
        self.account_type = account_type

    # ------------------------------------------------------------------
    # Pre-trade permission gate
    # ------------------------------------------------------------------
    def check_permission(
        self,
        account_type: str,
        proposed_lot: float,
        open_positions: int,
        spread_zscore: float,
    ) -> bool:
        if not self.cfg.enabled:
            logger.warning("DemoCanary disabled — trade blocked")
            return False
        if account_type != self.cfg.require_account_type:
            self._violation(
                f"account_type mismatch: {account_type} != {self.cfg.require_account_type}"
            )
            return False
        if proposed_lot > self.cfg.max_lot_per_trade:
            self._violation(
                f"proposed_lot {proposed_lot} > max {self.cfg.max_lot_per_trade}"
            )
            return False
        if open_positions >= self.cfg.max_open_positions:
            self._violation(
                f"open_positions {open_positions} >= max {self.cfg.max_open_positions}"
            )
            return False
        if spread_zscore > self.cfg.max_spread_zscore:
            self._violation(
                f"spread_zscore {spread_zscore} > max {self.cfg.max_spread_zscore}"
            )
            return False

        hour_key = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H")
        if self._hourly_trade_counts.get(hour_key, 0) >= self.cfg.max_trades_per_hour:
            self._violation(
                f"hourly trades {self._hourly_trade_counts[hour_key]} >= max {self.cfg.max_trades_per_hour}"
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Post-trade ingestion
    # ------------------------------------------------------------------
    def record_trade(self, trade: Dict[str, Any]) -> None:
        """Ingest a closed or opened trade dict."""
        self.trades.append(trade)

        now = datetime.now(timezone.utc)
        hour_key = now.strftime("%Y-%m-%d-%H")
        day_key = now.strftime("%Y-%m-%d")

        self._hourly_trade_counts[hour_key] = self._hourly_trade_counts.get(hour_key, 0) + 1
        self._daily_trade_counts[day_key] = self._daily_trade_counts.get(day_key, 0) + 1

        # Only update equity on closed trades that carry realised PnL
        if "pnl" in trade:
            pnl = float(trade.get("pnl", 0.0))
            costs = float(trade.get("fees", 0.0)) + float(trade.get("spread_paid", 0.0)) + float(trade.get("slippage", 0.0))
            net = pnl - costs
            self.current_balance += net
            self.net_return_after_costs += net

            self.peak_balance = max(self.peak_balance, self.current_balance)
            dd = self.peak_balance - self.current_balance
            if dd > self.max_drawdown:
                self.max_drawdown = dd

            self._update_profit_factor()
            self._check_daily_loss(day_key)

            # Timing metrics extension (Decision PPO rich timing-aware)
            nd = float(trade.get("news_distance_minutes", trade.get("news_proximity", 999)))
            sess = str(trade.get("session", trade.get("timing_session", ""))).lower()
            is_open_win = any(h in sess or ("open" in sess) for h in ["london", "ny", "open"]) or (7 <= datetime.now(timezone.utc).hour <= 10) or (13 <= datetime.now(timezone.utc).hour <= 16)
            if is_open_win:
                self.timing_open_window_trades += 1
                self.timing_window_pnl += net
            if nd < 30:
                self.timing_news_prox_trades += 1
            else:
                self.timing_news_avoided_trades += 1
                self.timing_news_avoid_pnl += net

        self.days_active = max(1, (now - self._start_date).days)

    def _update_profit_factor(self) -> None:
        gross_profit = sum(max(float(t.get("pnl", 0.0)), 0.0) for t in self.trades)
        gross_loss = abs(sum(min(float(t.get("pnl", 0.0)), 0.0) for t in self.trades))
        self.profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
        )

    def _check_daily_loss(self, day_key: str) -> None:
        """Flag a risk violation if daily drawdown exceeds the configured pct of notional."""
        daily_pnl = sum(
            float(t.get("pnl", 0.0))
            for t in self.trades
            if t.get("exit_time", "").startswith(day_key)
        )
        daily_costs = sum(
            float(t.get("fees", 0.0)) + float(t.get("spread_paid", 0.0)) + float(t.get("slippage", 0.0))
            for t in self.trades
            if t.get("exit_time", "").startswith(day_key)
        )
        daily_net = daily_pnl - daily_costs
        daily_loss_pct = abs(daily_net) / self.notional_balance * 100.0
        if daily_loss_pct >= self.cfg.max_daily_loss_pct:
            self._violation(
                f"daily_loss_pct {daily_loss_pct:.2f}% >= max {self.cfg.max_daily_loss_pct}%"
            )

    def _violation(self, reason: str) -> None:
        self.risk_violations += 1
        logger.warning(f"DemoCanary {self.canary_id} risk violation #{self.risk_violations}: {reason}")

    # ------------------------------------------------------------------
    # Evaluation / artifact
    # ------------------------------------------------------------------
    def evaluate(self) -> CanaryArtifact:
        """Produce the promotion-gating canary artifact."""
        passed = (
            self.risk_violations == 0
            and self.net_return_after_costs > 0
            and self.max_drawdown < (self.notional_balance * self.cfg.max_daily_loss_pct / 100.0)
        )

        # Champion gate: modest track record, positive edge
        approved_for_champion = (
            passed
            and len(self.trades) >= 10
            and self.profit_factor > 1.0
            and self.days_active >= 3
        )

        # Real-live gate: stricter — deeper track record, tighter drawdown, stronger edge
        approved_for_real_live = (
            approved_for_champion
            and len(self.trades) >= 30
            and self.days_active >= 7
            and self.max_drawdown < (self.notional_balance * 1.0 / 100.0)
            and self.profit_factor > 1.2
        )

        # Timing safety gate: if poor news avoidance (high prox ratio or negative score), block live approval
        timing_prox_ratio = getattr(self, "timing_news_prox_trades", 0) / max(1, (getattr(self, "timing_news_prox_trades", 0) + getattr(self, "timing_news_avoided_trades", 0)))
        timing_score = getattr(self, "timing_news_avoidance_score", 0.0)
        if timing_prox_ratio > 0.5 or timing_score < -0.3:
            approved_for_real_live = False
            approved_for_champion = approved_for_champion and timing_prox_ratio < 0.4  # still allow champ review with caution

        # Compute timing avoidance score (higher = better news avoidance / open window edge from rich decisions)
        avoid_pnl = getattr(self, "timing_news_avoid_pnl", 0.0)
        prox_pnl = sum(max(0.0, -float(t.get("pnl", 0))) for t in self.trades if float(t.get("news_distance_minutes", 999)) < 30)  # proxy penalty
        timing_score = round((avoid_pnl - prox_pnl) / max(1.0, self.notional_balance) * 100.0, 4)

        artifact = CanaryArtifact(
            canary_id=self.canary_id,
            bundle_id=self.bundle_id,
            system_mode=self.system_mode,
            account_type=self.account_type,
            trades=len(self.trades),
            days_active=self.days_active,
            net_return=round(self.net_return_after_costs, 4),
            max_drawdown=round(self.max_drawdown, 4),
            profit_factor=round(self.profit_factor, 4),
            risk_violations=self.risk_violations,
            passed=passed,
            approved_for_champion=approved_for_champion,
            approved_for_real_live=approved_for_real_live,
            timing_open_window_trades=getattr(self, "timing_open_window_trades", 0),
            timing_news_avoided_trades=getattr(self, "timing_news_avoided_trades", 0),
            timing_news_prox_trades=getattr(self, "timing_news_prox_trades", 0),
            timing_window_pnl=round(getattr(self, "timing_window_pnl", 0.0), 4),
            timing_news_avoid_pnl=round(getattr(self, "timing_news_avoid_pnl", 0.0), 4),
            timing_news_avoidance_score=timing_score,
        )

        path = self.data_dir / f"canary_{self.canary_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(artifact), f, indent=2)

        logger.info(
            f"DemoCanary {self.canary_id} evaluated: passed={passed}, "
            f"champion={approved_for_champion}, live={approved_for_real_live}"
        )
        return artifact
