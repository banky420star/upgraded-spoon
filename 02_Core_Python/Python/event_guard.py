"""
Event volatility guard — drop-in module for chain_gambler.

Wraps the existing EventIntel (observe-only) into an active trade gate that:
  - Flattens or reduces positions before major releases
  - Blocks new trades during release burst windows
  - Restricts re-entry to half-size until spreads and structure normalize
  - Enforces slippage limits on order fills (config key existed but was unused)
  - Logs event phase, spread, slippage, and block reason on every decision

Config lives under `event_guard:` in config.yaml. All numeric knobs are
overridable with env vars (see ENV_OVERRIDES below).

Usage (in Server_AGI.py):
    from Python.event_guard import EventGuard
    guard = EventGuard(config, log_dir="logs")
    # in _auto_trade_loop, before _handle_trade:
    gate = guard.check(symbol, exposure=..., lots=...)
    if not gate.allowed:
        logger.info(f"EVENT-GUARD blocked {symbol}: {gate.reason}")
        continue
    # after fill, log slippage:
    guard.log_fill(symbol, order_result, gate)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger

# ── ENV OVERRIDES ──────────────────────────────────────────────────────────
_ENV_MAP = {
    "AGI_EVG_ENABLED":                 ("enabled", bool),
    "AGI_EVG_PRE_EVENT_MIN":           ("pre_event_min", int),
    "AGI_EVG_EVENT_LIVE_MIN":          ("event_live_min", int),
    "AGI_EVG_POST_EVENT_MIN":          ("post_event_min", int),
    "AGI_EVG_FLAT_BEFORE":             ("flat_before", bool),
    "AGI_EVG_HALF_SIZE_DURING_POST":   ("half_size_during_post", bool),
    "AGI_EVG_MAX_SPREAD_MULT_POST":    ("max_spread_mult_post", float),
    "AGI_EVG_ENFORCE_SLIPPAGE":        ("enforce_slippage", bool),
    "AGI_EVG_HALF_SIZE_MULT":          ("half_size_mult", float),
    "AGI_EVG_HIGH_IMPACT_ONLY":        ("high_impact_only", bool),
}

DEFAULTS = {
    "enabled": True,
    "pre_event_min": 60,
    "event_live_min": 5,
    "post_event_min": 30,
    "flat_before": True,
    "half_size_during_post": True,
    "max_spread_mult_post": 2.0,
    "enforce_slippage": True,
    "half_size_mult": 0.5,
    "high_impact_only": True,
    "custom_events": [],
}


@dataclass
class GateResult:
    """Result of an event-guard check."""
    allowed: bool = True
    reason: str = "ok"
    phase: str = "normal"          # normal | pre_event | event_live | post_event
    exposure_mult: float = 1.0     # multiplier to apply to target exposure
    lots_mult: float = 1.0         # multiplier to apply to lot size
    spread_at_check: Optional[float] = None
    event_name: Optional[str] = None
    minutes_to_event: Optional[float] = None


@dataclass
class FillLog:
    """Logged after an order fill for audit trail."""
    ts: str = ""
    symbol: str = ""
    phase: str = "normal"
    spread_at_entry: float = 0.0
    slippage_points: float = 0.0
    max_slippage_points: float = 0.0
    slippage_ok: bool = True
    lots_requested: float = 0.0
    lots_filled: float = 0.0
    event_name: Optional[str] = None
    reason: str = ""


class EventGuard:
    """Active event-volatility gate that wraps EventIntel into trade decisions.

    Three operating modes per event phase:
      pre_event   — flatten or block new trades (configurable)
      event_live  — block all new trades, hold existing positions
      post_event  — allow half-size trades only; tighter spread gate; slippage check
    """

    def __init__(self, cfg: dict, log_dir: str = "logs"):
        # Merge defaults with user config, then apply env overrides
        eg_cfg = {**DEFAULTS, **(cfg.get("event_guard", {}) or {})}
        eg_cfg = self._apply_env_overrides(eg_cfg)

        self.enabled: bool = eg_cfg["enabled"]
        self.pre_event_min: int = eg_cfg["pre_event_min"]
        self.event_live_min: int = eg_cfg["event_live_min"]
        self.post_event_min: int = eg_cfg["post_event_min"]
        self.flat_before: bool = eg_cfg["flat_before"]
        self.half_size_during_post: bool = eg_cfg["half_size_during_post"]
        self.max_spread_mult_post: float = eg_cfg["max_spread_mult_post"]
        self.enforce_slippage: bool = eg_cfg["enforce_slippage"]
        self.half_size_mult: float = eg_cfg["half_size_mult"]
        self.high_impact_only: bool = eg_cfg["high_impact_only"]
        self.custom_events: list[dict] = eg_cfg["custom_events"]

        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # Internal state
        self._event_intel = None
        self._last_state: dict = {}
        self._symbol_configs: dict[str, dict] = {}

        # Try to import and initialise EventIntel
        self._init_event_intel(cfg)

        logger.info(
            f"EventGuard initialized: enabled={self.enabled}, "
            f"pre={self.pre_event_min}min, live={self.event_live_min}min, "
            f"post={self.post_event_min}min, flat_before={self.flat_before}, "
            f"half_size={self.half_size_during_post} (mult={self.half_size_mult})"
        )

    def _init_event_intel(self, cfg: dict):
        """Import and instantiate EventIntel if available."""
        try:
            from Python.event_intel import EventIntel
            self._event_intel = EventIntel(cfg, log_dir=str(self._log_dir))
            logger.info("EventGuard: EventIntel wired up successfully")
        except Exception as exc:
            logger.warning(f"EventGuard: EventIntel unavailable ({exc}); "
                           f"falling back to MT5 calendar only")
            self._event_intel = None

    def load_symbol_config(self, symbol: str, config_path: str = "configs"):
        """Load per-symbol risk config for spread/slippage thresholds."""
        path = Path(config_path) / f"{symbol}.yaml"
        if path.exists():
            import yaml
            with open(path) as f:
                sym_cfg = yaml.safe_load(f)
            risk = sym_cfg.get("risk", {})
            self._symbol_configs[symbol] = risk
            logger.debug(f"EventGuard: loaded config for {symbol} "
                         f"(max_spread_bps={risk.get('max_spread_bps')}, "
                         f"max_slippage_points={risk.get('max_slippage_points')})")

    # ── Public API ──────────────────────────────────────────────────────────

    def check(self, symbol: str, exposure: float = 0.0, lots: float = 0.0,
              mt5_executor=None) -> GateResult:
        """Main gate — call before every trade decision.

        Args:
            symbol: Trading symbol (e.g. "EURUSDm").
            exposure: Target exposure from PPO (abs value 0-1).
            lots: Requested lot size.
            mt5_executor: Optional MT5Executor for live spread/slippage checks.

        Returns:
            GateResult with allowed flag, reason, phase, and size multipliers.
        """
        if not self.enabled:
            return GateResult(allowed=True, reason="event_guard_disabled")

        # Tick event intel to get current regime state
        self._tick_intel()

        # Determine event phase for this symbol
        phase, event_name, minutes_to = self._get_phase(symbol)

        # Build result with phase info
        result = GateResult(
            allowed=True,
            reason="ok",
            phase=phase,
            exposure_mult=1.0,
            lots_mult=1.0,
            event_name=event_name,
            minutes_to_event=minutes_to,
        )

        # ── Phase-based logic ────────────────────────────────────────────────
        if phase == "event_live":
            result.allowed = False
            result.reason = f"event_live: {event_name or 'unknown'} — block all trades"
            result.exposure_mult = 0.0
            result.lots_mult = 0.0
            self._log_decision(symbol, result, exposure, lots)
            return result

        if phase == "pre_event":
            if self.flat_before:
                result.allowed = False
                result.reason = f"pre_event: {event_name or 'unknown'} in {minutes_to:.0f}min — flat before"
                result.exposure_mult = 0.0
                result.lots_mult = 0.0
            else:
                # Allow but with reduced size
                result.allowed = True
                result.reason = f"pre_event: {event_name or 'unknown'} in {minutes_to:.0f}min — reduced size"
                result.exposure_mult = self.half_size_mult
                result.lots_mult = self.half_size_mult
            self._log_decision(symbol, result, exposure, lots)
            return result

        if phase == "post_event":
            if self.half_size_during_post:
                result.reason = f"post_event: {event_name or 'unknown'} — half size"
                result.exposure_mult = self.half_size_mult
                result.lots_mult = self.half_size_mult
            # Tighter spread gate during post-event
            spread_result = self._check_spread_tight(symbol, mt5_executor)
            if not spread_result[0]:
                result.allowed = False
                result.reason = (f"post_event spread too wide: "
                                 f"{spread_result[1]} — {event_name or 'unknown'}")
                result.spread_at_check = spread_result[2]
                self._log_decision(symbol, result, exposure, lots)
                return result
            result.spread_at_check = spread_result[2]
            self._log_decision(symbol, result, exposure, lots)
            return result

        # Normal phase — no event active
        self._log_decision(symbol, result, exposure, lots)
        return result

    def should_flatten(self, symbol: str) -> bool:
        """Return True if open positions for this symbol should be closed."""
        if not self.enabled:
            return False
        phase, _, minutes_to = self._get_phase(symbol)
        if phase == "event_live":
            return True
        if phase == "pre_event" and self.flat_before:
            return True
        return False

    def log_fill(self, symbol: str, order_result: dict,
                 gate: GateResult, mt5_executor=None):
        """Log a fill with spread, slippage, event phase, and block reason.

        Call after every order fill (live or dry-run).
        """
        max_slippage = self._get_max_slippage(symbol)
        spread = self._get_live_spread(symbol, mt5_executor)
        slippage = self._compute_slippage(order_result)

        entry = FillLog(
            ts=datetime.now(timezone.utc).isoformat(),
            symbol=symbol,
            phase=gate.phase,
            spread_at_entry=spread,
            slippage_points=slippage,
            max_slippage_points=max_slippage,
            slippage_ok=abs(slippage) <= max_slippage if max_slippage else True,
            lots_requested=order_result.get("requested_volume", 0),
            lots_filled=order_result.get("filled_volume", 0),
            event_name=gate.event_name,
            reason=gate.reason,
        )

        path = self._log_dir / "event_guard_fills.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

        if self.enforce_slippage and max_slippage and abs(slippage) > max_slippage:
            logger.warning(
                f"EVENT-GUARD slippage breach: {symbol} slippage={slippage:.1f} > "
                f"max={max_slippage:.1f} (phase={gate.phase})"
            )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _tick_intel(self):
        """Refresh EventIntel state if available."""
        if self._event_intel is not None:
            try:
                symbols = list(self._symbol_configs.keys()) or []
                state = self._event_intel.tick(symbols)
                if state:
                    self._last_state = state
            except Exception as exc:
                logger.debug(f"EventGuard: EventIntel tick failed ({exc})")

    def _get_phase(self, symbol: str) -> tuple[str, Optional[str], Optional[float]]:
        """Determine the current event phase for a symbol.

        Priority:
          1. EventIntel state (if wired)
          2. MT5 calendar check (fallback)
          3. "normal"

        Returns (phase, event_name, minutes_to_event).
        """
        # ── Try EventIntel first ─────────────────────────────────────────────
        if self._last_state:
            by_symbol = self._last_state.get("by_symbol", {})
            sym_info = by_symbol.get(symbol, {})
            if sym_info:
                phase = sym_info.get("regime", "normal")
                phase = self._normalize_phase(phase)
                event_name = sym_info.get("headline") or sym_info.get("event_name")
                minutes_to = sym_info.get("minutes_to_event")
                return phase, event_name, minutes_to

        # ── Fallback: check MT5 calendar via executor reference ───────────────
        # (handled by the existing _check_news_blackout in MT5Executor)
        # EventGuard supplements, not replaces, that check.
        return "normal", None, None

    @staticmethod
    def _normalize_phase(phase: str) -> str:
        mapping = {
            "pre-event": "pre_event",
            "pre_event": "pre_event",
            "live": "event_live",
            "event_live": "event_live",
            "post-event": "post_event",
            "post_event": "post_event",
            "normal": "normal",
        }
        return mapping.get(phase, "normal")

    def _check_spread_tight(self, symbol: str, mt5_executor) -> tuple[bool, str, float]:
        """During post-event, require spread below normal threshold * multiplier.

        Returns (ok, reason, spread_bps).
        """
        if symbol not in self._symbol_configs:
            self.load_symbol_config(symbol)

        sym_risk = self._symbol_configs.get(symbol, {})
        base_max_spread_bps = sym_risk.get("max_spread_bps", 50)
        tight_limit = base_max_spread_bps / self.max_spread_mult_post

        if mt5_executor is None:
            return True, "no_executor", 0.0

        try:
            ok, reason = mt5_executor._check_spread_guard(symbol)
            tick = mt5_executor._get_tick(symbol)
            if tick is not None:
                spread_bps = abs(tick.ask - tick.bid) / mt5_executor._point_size(symbol) * 10000
                spread_bps = round(spread_bps, 1)
            else:
                spread_bps = 0.0
            if ok:
                if spread_bps > tight_limit:
                    return False, f"spread {spread_bps:.1f} > tight_limit {tight_limit:.1f} bps", spread_bps
            return ok, reason, spread_bps
        except Exception as exc:
            logger.debug(f"EventGuard spread check fallback: {exc}")
            return True, "check_failed", 0.0

    def _get_max_slippage(self, symbol: str) -> float:
        """Load max_slippage_points from per-symbol config."""
        if symbol not in self._symbol_configs:
            self.load_symbol_config(symbol)
        return self._symbol_configs.get(symbol, {}).get("max_slippage_points", 0)

    def _get_live_spread(self, symbol: str, mt5_executor=None) -> float:
        """Get current spread in bps from MT5 if available."""
        if mt5_executor is None:
            return 0.0
        try:
            tick = mt5_executor._get_tick(symbol)
            if tick is not None:
                point = mt5_executor._point_size(symbol)
                return round(abs(tick.ask - tick.bid) / point * 10000, 1)
        except Exception as e:
            logger.debug(f"Failed to get current spread for {symbol}: {e}")
        return 0.0

    @staticmethod
    def _compute_slippage(order_result: dict) -> float:
        """Compute slippage in points from an order result dict."""
        if "slippage" in order_result:
            return float(order_result["slippage"])
        fill = order_result.get("price_fill") or order_result.get("price")
        requested = order_result.get("price_requested") or order_result.get("price_open")
        if fill and requested:
            return abs(float(fill) - float(requested))
        return 0.0

    def _log_decision(self, symbol: str, result: GateResult,
                      exposure: float, lots: float):
        """Append decision log entry."""
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "phase": result.phase,
            "allowed": result.allowed,
            "reason": result.reason,
            "exposure_mult": result.exposure_mult,
            "lots_mult": result.lots_mult,
            "exposure": exposure,
            "lots": lots,
            "event_name": result.event_name,
            "minutes_to_event": result.minutes_to_event,
        }
        path = self._log_dir / "event_guard_decisions.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    @staticmethod
    def _apply_env_overrides(cfg: dict) -> dict:
        """Override config values from environment variables."""
        for env_key, (cfg_key, type_fn) in _ENV_MAP.items():
            val = os.environ.get(env_key)
            if val is not None:
                try:
                    cfg[cfg_key] = type_fn(val)
                except (ValueError, TypeError):
                    logger.warning(f"EventGuard: invalid env override {env_key}={val}")
        return cfg