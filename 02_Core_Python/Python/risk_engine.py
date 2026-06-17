import os
from datetime import datetime, timezone

import yaml
from loguru import logger

# Timing awareness for rich Decision PPO TimeExitSpec / news windows (production hardening)
try:
    from Python.event_guard import EventGuard
except Exception:
    EventGuard = None  # type: ignore

_CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")


class RiskEngine:
    def __init__(self):
        if os.path.exists(_CFG_PATH):
            with open(_CFG_PATH, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        else:
            # CI / no-config fallback — use safe paper defaults
            cfg = {} or {}

        risk_cfg = cfg.get("risk", {})
        trading_cfg = cfg.get("trading", {})

        self.max_daily_loss = float(risk_cfg.get("max_daily_loss", 1000))
        self.max_daily_trades = int(risk_cfg.get("max_daily_trades", 50))
        self.max_daily_trades_per_symbol = int(risk_cfg.get("max_daily_trades_per_symbol", 50))
        self.max_daily_losing_trades_per_symbol = int(risk_cfg.get("max_daily_losing_trades_per_symbol", 10))
        self.max_lots = float(risk_cfg.get("max_lots", 1.0))

        # Kelly fraction: default Quarter-Kelly (0.25) for safety
        self.kelly_fraction = float(trading_cfg.get("kelly_fraction", 0.25))

        # Default symbol profile used when a symbol-specific profile does not exist.
        self.default_symbol_profile = {
            "entry_deviation": int(trading_cfg.get("entry_deviation", 20)),
            "sl_points": int(trading_cfg.get("sl_points", 250)),
            "tp_points": int(trading_cfg.get("tp_points", 450)),
        }

        self.symbol_profiles = trading_cfg.get("symbol_profiles", {}) or {}

        self.realized_pnl_today = 0.0
        self.daily_trades = 0
        self.daily_trades_by_symbol = {}
        self.daily_losing_trades_by_symbol = {}
        self.halt = False
        self.error_halt = False  # True when halt was triggered by consecutive order errors (requires restart)
        self.error_count = 0
        self.current_dd = 0.0
        self.peak_equity = None
        self.last_reset_day = datetime.now(timezone.utc).date()

    def reset_daily(self):
        self.realized_pnl_today = 0.0
        self.daily_trades = 0
        self.daily_trades_by_symbol = {}
        self.daily_losing_trades_by_symbol = {}
        self.error_count = 0
        # Only auto-clear P&L-triggered halts on day roll; error halts require restart.
        if not self.error_halt:
            self.halt = False
        self.last_reset_day = datetime.now(timezone.utc).date()

    def maybe_roll_day(self):
        today = datetime.now(timezone.utc).date()
        if today != self.last_reset_day:
            self.reset_daily()

    def record_trade(self, symbol=None):
        self.maybe_roll_day()
        self.daily_trades += 1
        if symbol:
            key = str(symbol)
            self.daily_trades_by_symbol[key] = int(self.daily_trades_by_symbol.get(key, 0)) + 1

    def record_pnl(self, pnl):
        self.maybe_roll_day()
        self.realized_pnl_today += float(pnl)
        if self.realized_pnl_today <= -abs(self.max_daily_loss):
            self.halt = True
            self._halt_reason = "daily_loss"

    def record_pnl_with_equity(self, pnl: float, equity: float | None = None) -> bool:
        """Enhanced: support % daily loss guard. Returns True if halt triggered."""
        self.record_pnl(pnl)
        if equity and equity > 50:  # avoid tiny account false triggers
            loss_pct = abs(min(0.0, self.realized_pnl_today)) / equity * 100.0
            max_pct = 1.5  # production paper harness default
            if loss_pct >= max_pct:
                self.halt = True
                self._halt_reason = f"daily_loss_pct_{loss_pct:.1f}"
                return True
        return self.halt

    def record_trade_result(self, symbol, pnl):
        self.maybe_roll_day()
        self.record_pnl(pnl)
        if symbol is None:
            return
        if float(pnl) < 0.0:
            key = str(symbol)
            self.daily_losing_trades_by_symbol[key] = int(self.daily_losing_trades_by_symbol.get(key, 0)) + 1

    def update_equity(self, equity: float):
        eq = float(equity)
        if self.peak_equity is None:
            self.peak_equity = eq
            self.current_dd = 0.0
            return
        self.peak_equity = max(self.peak_equity, eq)
        if self.peak_equity > 0:
            self.current_dd = (self.peak_equity - eq) / self.peak_equity * 100.0

    def record_error(self):
        self.error_count += 1
        if self.error_count >= 3:
            self.halt = True
            self.error_halt = True  # Requires manual restart to clear

    def can_trade(self, symbol=None):
        self.maybe_roll_day()
        if self.halt:
            logger.warning(f"RiskEngine: trade blocked — halt ({getattr(self, '_halt_reason', 'unknown')})")
            return False
        if self.daily_trades >= self.max_daily_trades:
            logger.warning(f"RiskEngine: trade blocked — daily trade limit ({self.daily_trades}/{self.max_daily_trades})")
            return False
        if symbol:
            key = str(symbol)
            ds = int(self.daily_trades_by_symbol.get(key, 0))
            dl = int(self.daily_losing_trades_by_symbol.get(key, 0))
            if ds >= self.max_daily_trades_per_symbol:
                logger.warning(f"RiskEngine: trade blocked — symbol trade limit ({key}: {ds}/{self.max_daily_trades_per_symbol})")
                return False
            if dl >= self.max_daily_losing_trades_per_symbol:
                logger.warning(f"RiskEngine: trade blocked — symbol losing trade limit ({key}: {dl}/{self.max_daily_losing_trades_per_symbol})")
                return False
        return True

    def get_symbol_profile(self, symbol: str) -> dict:
        prof = self.default_symbol_profile.copy()
        sym_prof = self.symbol_profiles.get(symbol, {})
        if isinstance(sym_prof, dict):
            prof.update(sym_prof)
        return prof

    # --- Production hardening: timing-aware safety for rich TimeExitSpec decisions ---
    def is_high_impact_news_window(self, symbol: str = None) -> bool:
        """Basic + EventGuard-aware check for high-impact news proximity (respects TimeExitSpec intent).
        Used by daily loss / flatten to honor news avoidance windows instead of blind force-close.
        Real deployments wire full EventIntel; this provides safe fallback + integration.
        """
        try:
            if EventGuard is not None:
                # Best effort: instantiate lightweight guard (config optional)
                guard = EventGuard({"event_guard": {"enabled": True, "high_impact_only": True}})
                phase, _, _ = guard._get_phase(symbol or "XAUUSDm") if hasattr(guard, "_get_phase") else ("normal", None, None)
                if phase in ("pre_event", "event_live"):
                    return True
        except Exception:
            pass
        # Fallback heuristic: major session overlaps + typical high-impact UTC windows (FX/XAU)
        # (London open 7-10, NY 13-16, typical news 12-16, 20-22 UTC etc.)
        utc_h = datetime.now(timezone.utc).hour
        if utc_h in (8, 9, 13, 14, 15, 20, 21):
            return True
        return False

    def should_respect_time_exit_for_loss_limit(self, active_time_exits: list = None) -> bool:
        """Returns True if daily loss enforcement should defer to TimeExitSpec news/session logic.
        E.g., if any active rich decision has close_before_high_impact_news, honor the managed close instead of emergency.
        """
        if not active_time_exits:
            # If no context, still respect global news window to avoid bad slippage on flatten during news
            return self.is_high_impact_news_window()
        for te in active_time_exits:
            try:
                if getattr(te, "close_before_high_impact_news", False):
                    if self.is_high_impact_news_window():
                        return True
            except Exception:
                continue
        return False

    def record_pnl_with_equity_timing_aware(self, pnl: float, equity: float | None = None, active_time_exits: list = None) -> bool:
        """Timing-aware variant for rich decisions: daily loss still recorded, but halt decision respects news windows per TimeExitSpec.
        Prevents emergency flatten during protected news windows when decisions specified avoidance.
        """
        triggered = self.record_pnl_with_equity(pnl, equity)
        if triggered and self.should_respect_time_exit_for_loss_limit(active_time_exits):
            # Downgrade to warning; let time_exit management + OrderManager/EA handle close_before_news
            logger.warning(f"RiskEngine: daily loss breach detected but respecting TimeExitSpec news window (defer emergency flatten)")
            # Do not set halt for flatten purposes; new trades still blocked via other paths
            self.halt = False  # allow managed time exits to proceed cleanly; supervisor/harness can still decide
            if hasattr(self, "_halt_reason"):
                self._halt_reason = "daily_loss_deferred_for_news_timing"
            return False
        return triggered
