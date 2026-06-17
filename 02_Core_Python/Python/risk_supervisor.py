from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
from dataclasses import dataclass, field

from loguru import logger

# Production audit trail for risk decisions (used by harness + monitoring)
_RISK_AUDIT_PATH = None  # set lazily to logs/risk_audit.jsonl


@dataclass
class RiskDecision:
    allowed: bool
    reason: str
    rollback_recommended: bool = False  # For harness: signal auto-flatten + pause
    severity: str = "info"  # "info" | "warn" | "critical" for monitoring/alerts


class RiskSupervisor:
    """
    Deterministic live-trade circuit breaker layer. This complements RiskEngine
    with portfolio and market-state checks before a new exposure is sent.
    """

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        risk_cfg = cfg.get("risk", {}) if isinstance(cfg, dict) else {}
        trading_cfg = cfg.get("trading", {}) if isinstance(cfg, dict) else {}
        supervisor_cfg = risk_cfg.get("supervisor", {}) if isinstance(risk_cfg.get("supervisor", {}), dict) else {}
        self.symbol_profiles = trading_cfg.get("symbol_profiles", {}) or {}

        self.enabled = bool(supervisor_cfg.get("enabled", True))
        self.max_daily_loss = float(supervisor_cfg.get("max_daily_loss", risk_cfg.get("max_daily_loss", 100.0)))
        self.max_drawdown_pct = float(
            supervisor_cfg.get("max_drawdown_pct", risk_cfg.get("max_drawdown_pct_guard", trading_cfg.get("max_drawdown", 8.0)))
        )
        self.max_symbol_exposure = float(supervisor_cfg.get("max_symbol_exposure", risk_cfg.get("max_symbol_exposure", 0.35)))
        self.max_total_exposure = float(supervisor_cfg.get("max_total_exposure", risk_cfg.get("max_total_exposure", 1.2)))
        self.max_open_positions = int(supervisor_cfg.get("max_open_positions", risk_cfg.get("max_open_positions", 6)))
        self.max_positions_per_symbol = int(
            supervisor_cfg.get("max_positions_per_symbol", risk_cfg.get("max_positions_per_symbol", 3))
        )
        self.min_trade_interval_sec = int(supervisor_cfg.get("min_trade_interval_sec", 45))
        self.max_spread_bps = float(supervisor_cfg.get("max_spread_bps", trading_cfg.get("max_spread_bps", 25.0)))
        self.max_confidence_gap = float(supervisor_cfg.get("max_confidence_gap", 1.0))

        self.last_trade_at_by_symbol: dict[str, dt.datetime] = {}
        self.halt_until: dt.datetime | None = None
        self._load_state()

    def _state_path(self) -> str:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(project_root, "logs", "risk_supervisor_state.json")

    def _save_state(self) -> None:
        try:
            state_path = self._state_path()
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            state = {
                "halt_until": self.halt_until.isoformat() if self.halt_until else None,
                "last_trade_at": {k: v.isoformat() for k, v in self.last_trade_at_by_symbol.items()},
            }
            dir_name = os.path.dirname(state_path)
            with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, suffix=".tmp") as tmp:
                json.dump(state, tmp)
                tmp_path = tmp.name
            os.replace(tmp_path, state_path)
        except Exception as exc:
            logger.warning(f"RiskSupervisor state save failed (halt persistence at risk): {exc}")

    @staticmethod
    def _is_demo_mode() -> bool:
        return os.environ.get("CHAIN_GAMBLER_EXECUTION_MODE", "").strip().lower() == "demo"

    def _load_state(self) -> None:
        try:
            state_path = self._state_path()
            if not os.path.exists(state_path):
                return
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            # Never load a persisted halt in demo mode — user expects continuous trading
            if not self._is_demo_mode():
                halt_until_str = state.get("halt_until")
                if halt_until_str:
                    self.halt_until = dt.datetime.fromisoformat(halt_until_str)
            for k, v in (state.get("last_trade_at") or {}).items():
                self.last_trade_at_by_symbol[str(k)] = dt.datetime.fromisoformat(v)
        except Exception as exc:
            logger.warning(f"RiskSupervisor state load failed (halts may not persist across restarts): {exc}")

    def _symbol_profile(self, symbol: str) -> dict:
        profile = self.symbol_profiles.get(str(symbol), {})
        return profile if isinstance(profile, dict) else {}

    def _now(self) -> dt.datetime:
        return dt.datetime.now(dt.timezone.utc)

    def _audit(self, decision: RiskDecision, symbol: str, context: dict) -> None:
        """Append structured audit record for every allow_trade decision (production reliability)."""
        global _RISK_AUDIT_PATH
        if _RISK_AUDIT_PATH is None:
            _base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            _RISK_AUDIT_PATH = os.path.join(_base, "logs", "risk_audit.jsonl")
            os.makedirs(os.path.dirname(_RISK_AUDIT_PATH), exist_ok=True)
        try:
            rec = {
                "ts": self._now().isoformat(),
                "symbol": str(symbol),
                "allowed": decision.allowed,
                "reason": decision.reason,
                "context": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in (context or {}).items()},
            }
            with open(_RISK_AUDIT_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except Exception as exc:
            logger.debug(f"Risk audit write skipped: {exc}")

    def mark_trade(self, symbol: str):
        self.last_trade_at_by_symbol[str(symbol)] = self._now()
        self._save_state()

    def enforce_halt(self, minutes: int, reason: str) -> RiskDecision:
        self.halt_until = self._now() + dt.timedelta(minutes=max(1, int(minutes)))
        if not self._is_demo_mode():
            self._save_state()
        return RiskDecision(False, reason, rollback_recommended=True, severity="critical")

    def allow_trade(
        self,
        *,
        symbol: str,
        target_exposure: float,
        confidence: float,
        spread_bps: float | None,
        snapshot: dict,
        symbol_positions: int,
        total_positions: int,
        current_symbol_exposure: float,
        total_exposure: float,
        drawdown_pct: float,
        equity: float | None = None,  # NEW: for % based guards in paper harness
    ) -> RiskDecision:
        if not self.enabled:
            return RiskDecision(True, "disabled")

        if abs(float(target_exposure)) <= abs(float(current_symbol_exposure)):
            return RiskDecision(True, "risk_reduction")

        symbol_profile = self._symbol_profile(symbol)
        max_positions_per_symbol = int(symbol_profile.get("max_positions_per_symbol", self.max_positions_per_symbol))
        max_symbol_exposure = float(symbol_profile.get("max_symbol_exposure", self.max_symbol_exposure))
        max_spread_bps = float(symbol_profile.get("max_spread_bps", self.max_spread_bps))
        min_trade_interval_sec = int(symbol_profile.get("min_trade_interval_sec", self.min_trade_interval_sec))

        now = self._now()
        # Demo mode runs continuously; halts should not block trading
        if not self._is_demo_mode() and self.halt_until and now < self.halt_until:
            dec = RiskDecision(False, f"halt_until {self.halt_until.isoformat()}", rollback_recommended=True, severity="warn")
            self._audit(dec, symbol, {"equity": equity, "target_exposure": target_exposure})
            return dec

        pnl_today = float(snapshot.get("pnl_today", 0.0) or 0.0)
        # Support both absolute and % daily loss (critical for variable-size paper/demo accounts)
        daily_loss_triggered = False
        if pnl_today <= -abs(self.max_daily_loss):
            daily_loss_triggered = True
        if equity and equity > 0:
            daily_loss_pct = abs(pnl_today) / float(equity) * 100.0
            max_daily_loss_pct = float(symbol_profile.get("max_daily_loss_pct", 1.5))  # tight default for harness
            if daily_loss_pct >= max_daily_loss_pct:
                daily_loss_triggered = True
        if daily_loss_triggered:
            dec = self.enforce_halt(24 * 60, f"daily_loss {pnl_today:.2f} <= -{abs(self.max_daily_loss):.2f} (or % breach)")
            self._audit(dec, symbol, {"equity": equity, "pnl_today": pnl_today, "target_exposure": target_exposure})
            return dec

        if drawdown_pct >= self.max_drawdown_pct:
            dec = self.enforce_halt(24 * 60, f"drawdown_pct {drawdown_pct:.2f} >= {self.max_drawdown_pct:.2f}")
            self._audit(dec, symbol, {"drawdown_pct": drawdown_pct})
            return dec

        if total_positions >= self.max_open_positions and abs(target_exposure) > 0.0:
            dec = RiskDecision(False, f"max_open_positions {total_positions} >= {self.max_open_positions}", severity="warn")
            self._audit(dec, symbol, {"total_positions": total_positions})
            return dec

        if symbol_positions >= max_positions_per_symbol and abs(target_exposure) > abs(current_symbol_exposure):
            dec = RiskDecision(False, f"max_positions_per_symbol {symbol_positions} >= {max_positions_per_symbol}", severity="warn")
            self._audit(dec, symbol, {"symbol_positions": symbol_positions})
            return dec

        projected_symbol_exposure = max(abs(current_symbol_exposure), abs(target_exposure))
        if projected_symbol_exposure > max_symbol_exposure:
            dec = RiskDecision(False, f"symbol_exposure {projected_symbol_exposure:.3f} > {max_symbol_exposure:.3f}", severity="warn")
            self._audit(dec, symbol, {"projected_symbol_exposure": projected_symbol_exposure})
            return dec

        projected_total_exposure = total_exposure - abs(current_symbol_exposure) + abs(target_exposure)
        if projected_total_exposure > self.max_total_exposure:
            dec = RiskDecision(False, f"total_exposure {projected_total_exposure:.3f} > {self.max_total_exposure:.3f}", severity="warn")
            self._audit(dec, symbol, {"projected_total_exposure": projected_total_exposure})
            return dec

        if spread_bps is not None and float(spread_bps) > max_spread_bps:
            dec = RiskDecision(False, f"spread_bps {float(spread_bps):.2f} > {max_spread_bps:.2f}", severity="warn")
            self._audit(dec, symbol, {"spread_bps": spread_bps})
            return dec

        conf = float(confidence)
        if conf < 0.0 or conf > self.max_confidence_gap:
            dec = RiskDecision(False, f"confidence {conf:.3f} outside range", severity="warn")
            self._audit(dec, symbol, {"confidence": conf})
            return dec

        last_trade_at = self.last_trade_at_by_symbol.get(str(symbol))
        if last_trade_at is not None:
            elapsed = (now - last_trade_at).total_seconds()
            if elapsed < min_trade_interval_sec:
                dec = RiskDecision(False, f"cooldown {elapsed:.0f}s < {min_trade_interval_sec}s", severity="info")
                self._audit(dec, symbol, {"elapsed_sec": elapsed})
                return dec

        ok_dec = RiskDecision(True, "ok")
        self._audit(ok_dec, symbol, {"target_exposure": target_exposure, "equity": equity})
        return ok_dec

    def trigger_rollback(self, reason: str = "harness_initiated") -> RiskDecision:
        """Explicit rollback signal for paper trading harness (closes positions + pauses)."""
        self.halt_until = self._now() + dt.timedelta(hours=4)  # conservative pause
        if not self._is_demo_mode():
            self._save_state()
        dec = RiskDecision(False, f"ROLLBACK: {reason}", rollback_recommended=True, severity="critical")
        self._audit(dec, "ALL", {"action": "trigger_rollback"})
        logger.critical(f"RiskSupervisor ROLLBACK triggered: {reason}")
        return dec

    def clear_rollback(self) -> None:
        """Operator or harness recovery after review."""
        self.halt_until = None
        if not self._is_demo_mode():
            self._save_state()
        logger.info("RiskSupervisor rollback cleared by operator/harness")

    # --- Timing-aware production hardening for rich Decision PPO TimeExitSpec (news/open windows) ---
    def is_high_impact_news_window(self, symbol: str = None) -> bool:
        """Delegates to RiskEngine timing or basic heuristic. Used by daily loss / flatten for rich decisions."""
        try:
            # Prefer underlying engine if wired (some paths construct with engine)
            if hasattr(self, "_engine") and self._engine and hasattr(self._engine, "is_high_impact_news_window"):
                return self._engine.is_high_impact_news_window(symbol)
        except Exception:
            pass
        utc_h = dt.datetime.now(dt.timezone.utc).hour
        return utc_h in (8, 9, 13, 14, 15, 20, 21)

    def should_respect_time_exit_for_loss_limit(self, active_time_exits: list = None) -> bool:
        """True if loss-triggered actions should defer to TimeExitSpec close_before_news etc."""
        if active_time_exits:
            for te in active_time_exits:
                try:
                    if getattr(te, "close_before_high_impact_news", False) and self.is_high_impact_news_window():
                        return True
                except Exception:
                    pass
        return self.is_high_impact_news_window()

    def record_pnl_with_equity_timing_aware(self, pnl: float, equity: float | None = None, active_time_exits: list = None) -> bool:
        """Wrapper that records PnL but may defer halt for news windows per rich spec."""
        # Simple record (full impl in RiskEngine)
        if hasattr(self, "record_pnl"):
            self.record_pnl(pnl)
        if equity and equity > 50:
            loss_pct = abs(min(0.0, getattr(self, "realized_pnl_today", pnl))) / equity * 100.0
            if loss_pct >= 1.5:
                if self.should_respect_time_exit_for_loss_limit(active_time_exits):
                    logger.warning("RiskSupervisor: daily loss breach but respecting TimeExitSpec/news (defer)")
                    return False
                self.trigger_rollback("daily_loss_pct_timing_aware")
                return True
        return False
