import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger
from Python.mt5_compat import mt5 as _mt5

try:
    from Python import paper_trading as _paper
except Exception:
    _paper = None

try:
    from Python import live_safety
except Exception:
    live_safety = None


def _is_paper_mode() -> bool:
    return _paper is not None and _paper.get_mode() == "paper"


MAGIC_BY_SYMBOL = {
    "BTCUSDm": 51000,
    "XAUUSDm": 52000,
}

LANE_MAGIC_OFFSET = {
    "champion": 0,
    "canary": 100,
    "history": 200,
    "unknown": 900,
}


class MT5Executor:
    """Live MT5 execution (Windows only)."""

    # Magic number ranges: base 505000 + symbol offset * 100 + lane offset
    # Symbol offsets: EURUSD=0, GBPUSD=1, XAUUSD=2, BTCUSD=3
    # Lane offsets: champion=0, canary=1
    _SYMBOL_MAGIC_OFFSET = {
        "EURUSDm": 0, "EURUSD": 0,
        "GBPUSDm": 1, "GBPUSD": 1,
        "XAUUSDm": 2, "XAUUSD": 2,
        "BTCUSDm": 3, "BTCUSD": 3,
        "USDCADm": 4, "USDCAD": 4,
        "USDJPYm": 5, "USDJPY": 5,
        "AUDUSDm": 6, "AUDUSD": 6,
    }
    _MAGIC_BASE = 505000
    _DEFAULT_MAGIC = 505

    def __init__(self, risk):
        self.risk = risk
        self._is_live = _mt5 is not None
        self._last_order_meta = {}  # carry context for close orders
        self._last_sl_hit_time = {}  # symbol -> timestamp of last SL hit (cooldown tracking)
        self._last_spread_spike_time = {}  # symbol -> timestamp of last spread spike rejection
        self._last_failed_signal_time = {}  # symbol -> timestamp of last failed signal

        # ── Half-Kelly position sizing state ──
        self._kelly_win_rate = {}     # symbol -> recent win rate (0-1)
        self._kelly_avg_win = {}      # symbol -> average winning trade PnL
        self._kelly_avg_loss = {}     # symbol -> average losing trade PnL (positive)
        self._kelly_last_update = {}  # symbol -> timestamp of last stats update

        # ── Cached broker parameters (read once, not per-trade) ──────────────
        self._min_lots = float(os.environ.get("AGI_MIN_LOTS", "0.01"))
        self._default_lot_step = 0.02  # fallback if broker doesn't provide volume_step
        self._mt5_has_calendar = hasattr(_mt5, "calendar_country") if _mt5 else False

        if self._is_live:
            try:
                # PASS PATH EXPLICITLY: the venv launcher re-execs system Python
                # which does NOT have the MetaTrader5 module on its path, so the
                # no-arg mt5.initialize() fails with -10003 'IPC initialize
                # failed, MetaTrader 5 x64 not found'. Explicit path forces the
                # MT5 terminal64.exe to be used.
                _mt5_path = os.environ.get(
                    "AGI_MT5_TERMINAL_PATH",
                    r"C:\Program Files\MetaTrader 5\terminal64.exe",
                )
                if not _mt5.initialize(path=_mt5_path):
                    logger.error("MT5 initialize() failed - falling back to dry-run")
                    self._is_live = False
            except Exception as e:
                logger.error(f"MT5 init error: {e}")
                self._is_live = False

        if self._is_live:
            logger.success("MT5Executor: LIVE mode — connected to MetaTrader 5")
        else:
            logger.info("MT5Executor: DRY-RUN mode — trades will be logged only")

        # ── Execution log ────────────────────────────────────────────────
        _base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._exec_log_path = os.path.join(_base, "logs", "executions.jsonl")
        os.makedirs(os.path.dirname(self._exec_log_path), exist_ok=True)

        # ── Reference to AGIServer for live_armed check (set externally) ──
        self._server_ref = None

    def set_server_ref(self, server):
        """Set reference to AGIServer for live_armed and other runtime checks."""
        self._server_ref = server

    def _assert_live_gate(self, action_desc: str) -> tuple[bool, str]:
        """Return (allowed, reason) for live order_send. Paper/demo always pass."""
        if _is_paper_mode():
            return True, "paper_mode"

        mode = os.environ.get("CHAIN_GAMBLER_EXECUTION_MODE", "paper").strip().lower()
        if mode == "demo":
            return True, "demo_mode"

        if live_safety is not None:
            gate = live_safety.live_trading_allowed()
            if not gate["allowed"]:
                return False, f"live_gate_blocked:{gate['gates']}"
        else:
            return False, "live_safety_module_missing"
        return True, "live_allowed"

    def _preflight_check(self, symbol: str, side: str, lots: float) -> tuple:
        """
        Pre-flight validation before sending any order.
        Returns (allowed: bool, reason: str).
        """
        # 1. Check live mode is armed
        if self._server_ref and hasattr(self._server_ref, "live"):
            if self._server_ref.live and not getattr(self._server_ref, "live_armed", False):
                return False, "live_mode_not_armed"

        # 2. Verify symbol is in allowed list
        if self._server_ref and hasattr(self._server_ref, "symbols"):
            if symbol not in self._server_ref.symbols:
                return False, f"symbol_not_allowed ({symbol})"

        # 2b. Check trading session filter
        session_ok, session_reason = self._check_session_filter(symbol)
        if not session_ok:
            return False, session_reason

        # 3. Check spread under threshold (per-symbol config)
        spread_ok, spread_reason = self._check_spread_guard(symbol)
        if not spread_ok:
            self._last_spread_spike_time[symbol] = time.time()
            return False, spread_reason

        # 3b. Spread spike cooldown: wait 2 min after a spread rejection
        last_spread_spike = self._last_spread_spike_time.get(symbol, 0)
        if last_spread_spike > 0 and (time.time() - last_spread_spike) < 120:
            remaining = 120 - (time.time() - last_spread_spike)
            return False, f"spread_spike_cooldown ({remaining:.0f}s remaining)"

        # 3c. Failed signal cooldown: wait 5 min after a preflight failure
        last_failed = self._last_failed_signal_time.get(symbol, 0)
        if last_failed > 0 and (time.time() - last_failed) < 300:
            remaining = 300 - (time.time() - last_failed)
            return False, f"failed_signal_cooldown ({remaining:.0f}s remaining)"

        # 4. Verify lot size within symbol cap
        try:
            import yaml
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "configs", f"{symbol}.yaml"
            )
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    sym_cfg = yaml.safe_load(f)
                max_lots = sym_cfg.get("risk", {}).get("max_lots", 1.0)
                if lots > max_lots:
                    return False, f"lot_size_exceeds_cap ({lots} > {max_lots})"
        except Exception as exc:
            logger.warning(f"Per-symbol lot cap check failed for {symbol} (using default): {exc}")

        # 5. Verify margin is sufficient (live mode only)
        if self._is_live and _mt5 is not None:
            try:
                account = _mt5.account_info()
                if account:
                    # Use MT5's actual margin calculation for the symbol/lot size
                    symbol_info = _mt5.symbol_info(symbol)
                    if symbol_info:
                        # MT5 provides margin_required for 1 lot; scale by actual lots
                        # margin_initial is for 1 lot in the account currency
                        margin_per_lot = getattr(symbol_info, 'margin_initial', 0) or getattr(symbol_info, 'margin_maintenance', 0)
                        if margin_per_lot > 0:
                            required_margin = margin_per_lot * lots
                        else:
                            # Fallback: use trade_mode and contract_size to estimate
                            # For forex with 1:100 leverage, ~$1000 per standard lot
                            # For micro lots (0.01), ~$10 — but broker may offer higher leverage
                            contract_size = getattr(symbol_info, 'trade_contract_size', 100000)
                            leverage_ratio = getattr(account, 'leverage', 100)
                            # Margin = contract_size * lots / leverage
                            required_margin = (contract_size * lots) / leverage_ratio

                        if account.margin_free < required_margin:
                            return False, f"insufficient_margin (free={account.margin_free:.2f}, required={required_margin:.2f})"
            except Exception as exc:
                logger.warning(f"Margin sufficiency check failed for {symbol} (allowing with caution): {exc}")

        return True, "ok"

    def _check_session_filter(self, symbol: str) -> tuple:
        """Check if current UTC hour falls within allowed trading sessions.

        Sessions (UTC):
        - Asian: 00:00-08:00
        - London: 07:00-16:00
        - New York: 12:00-21:00

        Returns (ok: bool, reason: str).
        """
        try:
            import yaml
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "configs", f"{symbol}.yaml"
            )
            if not os.path.exists(config_path):
                return True, "ok"  # No config = all sessions allowed

            with open(config_path, "r") as f:
                sym_cfg = yaml.safe_load(f) or {}

            sessions = sym_cfg.get("trading_sessions", {})
            if not sessions:
                return True, "ok"  # No session filter = all allowed

            utc_hour = datetime.now(timezone.utc).hour

            # Determine which sessions COVER this hour (regardless of enabled/disabled)
            # Overlaps: 0-8=Asian, 7-16=London, 12-21=New York
            covering_sessions = []
            if 0 <= utc_hour < 8:
                covering_sessions.append("asian")
            if 7 <= utc_hour < 16:
                covering_sessions.append("london")
            if 12 <= utc_hour < 21:
                covering_sessions.append("new_york")

            # Hours 21-24 UTC: no session covers — allow trading (edge case)
            if not covering_sessions:
                return True, "ok"

            # Check if at least one covering session is enabled
            for session in covering_sessions:
                if sessions.get(session, True):
                    return True, "ok"

            # All covering sessions are disabled — block trading
            return False, f"outside_trading_session (utc_hour={utc_hour}, covering={covering_sessions})"

        except Exception as e:
            logger.debug(f"Session filter check failed for {symbol}: {e}")
            return True, "ok"

    def _check_spread_guard(self, symbol: str) -> tuple:
        """
        Check if current spread is within the per-symbol threshold.
        Returns (ok: bool, reason: str).
        """
        max_spread_bps = 50  # default
        try:
            import yaml
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "configs", f"{symbol}.yaml"
            )
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    sym_cfg = yaml.safe_load(f)
                max_spread_bps = sym_cfg.get("risk", {}).get("max_spread_bps", 50)
        except Exception as exc:
            logger.debug(f"Symbol config spread read failed for {symbol} (using default): {exc}")

        if not self._is_live or _mt5 is None:
            return True, "ok"

        try:
            tick = _mt5.symbol_info_tick(symbol)
            if tick is None:
                return False, f"no_tick_data ({symbol})"
            spread = tick.ask - tick.bid
            # For FX: spread in points (5th decimal for 5-digit brokers)
            # For XAU/BTC: spread in raw price
            point = _mt5.symbol_info(symbol)
            if point:
                spread_bps = (spread / point.point) if point.point > 0 else 0
                if spread_bps > max_spread_bps:
                    return False, f"spread_too_wide ({spread_bps:.1f} > {max_spread_bps} bps)"
        except Exception as e:
            logger.warning(f"Spread guard check failed for {symbol}: {e}")

        return True, "ok"

    def _log_execution(self, record: dict):
        """Append execution intent record to logs/executions.jsonl."""
        try:
            with open(self._exec_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write execution log: {e}")

    @staticmethod
    def _play_trade_alert():
        """Play a loud beep when a trade is executed (Windows only)."""
        if _ALERT_ENABLED:
            try:
                # Two-tone alert: 800Hz for 300ms, then 1000Hz for 300ms
                winsound.Beep(800, 300)
                winsound.Beep(1000, 300)
            except Exception as exc:
                logger.debug(f"Trade alert beep failed (non-critical on VPS): {exc}")

    @staticmethod
    def _pip_value_per_lot(symbol: str) -> float:
        """Approximate dollar value per 0.01 lot for SL distance calculation."""
        sym = symbol.replace("m", "")
        if sym == "XAUUSD":
            return 1.0  # ~$1 per pip per 0.01 lot
        elif sym == "BTCUSD":
            return 1.0  # ~$1 per point per 0.01 lot
        else:
            # FX pairs: ~$0.10 per pip per 0.01 lot
            return 0.10

    def compute_risk_adjusted_lots(self, symbol, exposure, confidence_scale=1.0):
        """Compute lots based on risk-per-trade percentage and ATR stop distance.

        Formula: lots = (equity * risk_pct / 100) / (SL_distance * pip_value)
        This scales automatically from $50 to $250K+.

        Falls back to Kelly sizing if ATR is unavailable.
        """
        risk_pct = getattr(self.risk, 'risk_per_trade_pct', 1.0)
        equity = getattr(self.risk, '_current_equity', 0)
        if equity <= 0:
            equity = 50.0
        # Read fresh equity from MT5 (more accurate than cached value)
        if _mt5:
            try:
                _info = _mt5.account_info()
                if _info and _info.equity > 0:
                    equity = float(_info.equity)
            except Exception as exc:
                logger.debug(f"MT5 account_info equity refresh failed (using cached): {exc}")

        # Get ATR-based SL distance
        atr = self._get_raw_atr(symbol)
        if atr <= 0:
            # Fall back to Kelly
            min_lots = self._min_lots
            sym_max_lots = self._get_symbol_max_lots(symbol)
            return self._kelly_lot_size(symbol, exposure, min_lots, sym_max_lots)

        # Load per-symbol ATR multiplier for SL
        sl_mult = self._get_symbol_sl_mult(symbol)
        sl_distance = atr * sl_mult

        # Dollar risk for this trade
        max_risk_dollars = equity * risk_pct / 100.0

        # Get pip value from MT5 tick data
        pip_value = self._get_tick_pip_value(symbol)
        if pip_value <= 0:
            pip_value = self._pip_value_per_lot(symbol)

        # Lots = risk_dollars / (sl_distance_in_pips * pip_value_per_pip_per_lot)
        # Convert SL distance to pips, then multiply by pip value
        min_lots = self._min_lots
        tick_size = self._get_tick_size(symbol)

        # For small accounts: cap SL distance to what equity can afford
        # (same logic as open_position — prevents gold/BTC ATR SLs from blocking trades)
        if equity < 100 and sl_distance > 0:
            max_sl_equity_pct_local = 15.0
            max_risk_dollars_local = equity * max_sl_equity_pct_local / 100.0
            if pip_value > 0 and tick_size > 0:
                max_sl_dist = max_risk_dollars_local / (min_lots * pip_value / tick_size) if min_lots > 0 else sl_distance
                min_sl_floor = max(0.00005 * 2, self._min_sl_for_symbol(symbol) * 0.04)
                if max_sl_dist > min_sl_floor and sl_distance > max_sl_dist:
                    sl_distance = max_sl_dist

        if tick_size > 0:
            sl_pips = sl_distance / tick_size
            lots = max_risk_dollars / (sl_pips * pip_value)
        else:
            lots = min_lots  # safe fallback

        # Scale by confidence
        lots *= confidence_scale

        # Clamp to per-symbol max
        sym_max_lots = self._get_symbol_max_lots(symbol)
        lots = max(min_lots, min(lots, sym_max_lots))

        # ── Max SL equity cap: no single trade should risk more than X% of equity ──
        max_sl_equity_pct = getattr(self.risk, 'max_sl_equity_pct', 10.0)
        # For small accounts (<$100), raise the cap to 15% to allow trading
        if equity < 100:
            max_sl_equity_pct = max(max_sl_equity_pct, 15.0)
        max_sl_dollars = equity * max_sl_equity_pct / 100.0
        if tick_size > 0 and pip_value > 0 and lots > 0:
            actual_risk_dollars = lots * sl_pips * pip_value
            if actual_risk_dollars > max_sl_dollars * 1.01 and actual_risk_dollars > 0:
                # Calculate the safe lot size for this equity cap
                safe_lots = max_sl_dollars / (sl_pips * pip_value)
                if safe_lots < min_lots:
                    # Even minimum lots exceeds the equity risk cap — SKIP this trade entirely
                    logger.warning(
                        f"ATR sizing {symbol}: SKIP — min_lots {min_lots} risks "
                        f"${min_lots * sl_pips * pip_value:.2f} > {max_sl_equity_pct}% equity "
                        f"(${max_sl_dollars:.2f}). Safe lots={safe_lots:.4f}. "
                        f"Account too small for this symbol."
                    )
                    return 0.0
                lots = safe_lots
                lots = min(lots, sym_max_lots)
                logger.info(
                    f"ATR sizing {symbol}: SL equity cap triggered — "
                    f"risk ${actual_risk_dollars:.2f} > {max_sl_equity_pct}% (${max_sl_dollars:.2f}), "
                    f"reduced to {lots:.2f} lots"
                )

        # Round to lot step (read from broker if available)
        lot_step = self._get_lot_step(symbol)
        lots = round(lots / lot_step) * lot_step
        lots = max(min_lots, min(lots, sym_max_lots))

        logger.info(
            f"ATR sizing {symbol}: equity=${equity:.2f} risk_pct={risk_pct}% "
            f"atr={atr:.5f} sl_dist={sl_distance:.5f} risk=${max_risk_dollars:.2f} "
            f"-> {lots:.2f} lots"
        )
        return lots

    def _get_symbol_max_lots(self, symbol: str) -> float:
        """Get max lots from per-symbol config."""
        try:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "configs", f"{symbol}.yaml"
            )
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    sym_cfg = yaml.safe_load(f)
                return float(sym_cfg.get("risk", {}).get("max_lots", 1.0))
        except Exception as exc:
            logger.debug(f"_get_max_lots config read failed for {symbol} (default 1.0): {exc}")
        return 1.0

    def _get_lot_step(self, symbol: str) -> float:
        """Get lot step from broker symbol info, fallback to default."""
        try:
            if self._is_live:
                info = _mt5.symbol_info(symbol)
                if info and getattr(info, "volume_step", 0) > 0:
                    return float(info.volume_step)
        except Exception as exc:
            logger.debug(f"_get_lot_step failed for {symbol}: {exc}")
        return self._default_lot_step

    def _get_symbol_sl_mult(self, symbol: str) -> float:
        """Get SL ATR multiplier from per-symbol config."""
        try:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "configs", f"{symbol}.yaml"
            )
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    sym_cfg = yaml.safe_load(f)
                return float(sym_cfg.get("risk", {}).get("sl_atr_mult", 2.0))
        except Exception as exc:
            logger.debug(f"_get_sl_atr_mult config failed for {symbol} (default 2.0): {exc}")
        return 2.0

    def _get_tick_pip_value(self, symbol: str) -> float:
        """Get tick value per lot from MT5 symbol info."""
        if not self._is_live or _mt5 is None:
            return 0.0
        try:
            info = _mt5.symbol_info(symbol)
            if info is None:
                return 0.0
            tick_value = getattr(info, 'trade_tick_value', 0)
            return float(tick_value) if tick_value else 0.0
        except Exception as exc:
            logger.debug(f"_get_tick_pip_value failed for {symbol}: {exc}")
            return 0.0

    def _get_tick_size(self, symbol: str) -> float:
        """Get tick size from MT5 symbol info."""
        if not self._is_live or _mt5 is None:
            return 0.0
        try:
            info = _mt5.symbol_info(symbol)
            if info is None:
                return 0.0
            tick_size = getattr(info, 'trade_tick_size', 0)
            return float(tick_size) if tick_size else 0.0
        except Exception:
            return 0.0

    def _update_kelly_stats(self, symbol):
        """Refresh per-symbol win rate and PnL stats from MT5 trade history."""
        now = time.time()
        # Only refresh every 5 minutes to avoid hammering MT5
        if now - self._kelly_last_update.get(symbol, 0) < 300:
            return
        self._kelly_last_update[symbol] = now

        if not self._is_live:
            return

        try:
            # Get last 30 days of deals
            to_dt = datetime.now()
            from_dt = to_dt - timedelta(days=30)
            deals = _mt5.history_deals_get(from_dt, to_dt)
            if not deals:
                return

            wins_pnl = []
            losses_pnl = []
            for d in deals:
                if d.symbol != symbol or d.entry != 1:  # entry=1 means deal out (closed)
                    continue
                if d.profit > 0:
                    wins_pnl.append(d.profit)
                elif d.profit < 0:
                    losses_pnl.append(abs(d.profit))

            total = len(wins_pnl) + len(losses_pnl)
            if total >= 5:  # Need minimum 5 trades for meaningful stats
                self._kelly_win_rate[symbol] = len(wins_pnl) / total
                self._kelly_avg_win[symbol] = sum(wins_pnl) / len(wins_pnl) if wins_pnl else 0
                self._kelly_avg_loss[symbol] = sum(losses_pnl) / len(losses_pnl) if losses_pnl else 0
                logger.debug(
                    f"Kelly stats {symbol}: WR={self._kelly_win_rate[symbol]:.2%} "
                    f"avg_win=${self._kelly_avg_win[symbol]:.2f} "
                    f"avg_loss=${self._kelly_avg_loss[symbol]:.2f}"
                )
        except Exception as e:
            logger.debug(f"Kelly stats update failed for {symbol}: {e}")

    def _kelly_lot_size(self, symbol, exposure, min_lots, max_lots):
        """Calculate lot size using Fractional Kelly criterion.

        Kelly fraction: f* = (p*b - q) / b
        Where: p = win probability, q = 1-p, b = avg_win / avg_loss
        Fractional Kelly: multiply by kelly_fraction config (default 0.25 = Quarter-Kelly)
                          to dampen volatility and protect small accounts.

        Exposure magnitude scales confidence: higher |exposure| = higher conviction.
        """
        self._update_kelly_stats(symbol)

        # Default to minimum lots if no stats available
        wr = self._kelly_win_rate.get(symbol)
        avg_win = self._kelly_avg_win.get(symbol, 0)
        avg_loss = self._kelly_avg_loss.get(symbol, 0)

        if wr is None or avg_win <= 0 or avg_loss <= 0:
            return min(min_lots, max_lots)

        # Kelly fraction: f* = (p * b - q) / b
        b = avg_win / avg_loss  # reward-to-risk ratio
        q = 1.0 - wr
        kelly_full = (wr * b - q) / b

        # Clamp to [0, 1] — negative Kelly means don't trade
        kelly_full = max(0.0, min(1.0, kelly_full))

        # Fractional Kelly: read from RiskEngine config, fallback to env var, then safe default
        kelly_fraction = getattr(self.risk, "kelly_fraction", None)
        if kelly_fraction is None:
            kelly_fraction = float(os.environ.get("AGI_KELLY_FRACTION", "0.25"))
        kelly_fraction = max(0.0, min(1.0, kelly_fraction))
        kelly_used = kelly_full * kelly_fraction

        # Scale by conviction (|exposure| as signal strength)
        # exposure is typically 0.001-0.05, normalize to 0.3-1.0 range
        conviction = min(1.0, max(0.3, abs(exposure) * 20))

        # Risk budget: fraction of account we're willing to risk per trade
        # Use BALANCE (realized) not equity (includes floating P&L) for sizing.
        # Floating profits from one symbol should NOT inflate sizing on others.
        # E.g. XAUUSDm up $200 floating -> Kelly shouldn't size EURUSDm at 1.0 lots
        balance = getattr(self.risk, "_mt5_balance", None)
        if balance is None or balance <= 0:
            # Fall back to equity only if balance unavailable
            balance = getattr(self.risk, "_current_equity", None)
        if balance is None or balance <= 0:
            balance = 50.0  # fallback for dry-run or missing account info

        # Hard cap: never trade more than balance/500 lots to prevent over-leveraging
        # e.g. $500 → 1.0 lots max, $50 → 0.1 lots max
        max_lots = min(max_lots, max(balance / 500.0, min_lots))

        # Compound growth: scale ramps up as balance grows
        # Under $50: very conservative (0.3x)
        # $50-$200: linear ramp from 0.3x to 0.8x
        # $200-$1000: linear ramp from 0.8x to 1.0x
        # Above $1000: full equity scale (1.0x) — still dampened by kelly_fraction
        if balance < 50:
            equity_scale = 0.3
        elif balance < 200:
            equity_scale = 0.3 + 0.5 * ((balance - 50) / 150)  # 0.3 to 0.8
        elif balance < 1000:
            equity_scale = 0.8 + 0.2 * ((balance - 200) / 800)  # 0.8 to 1.0
        else:
            equity_scale = 1.0
        kelly_used = kelly_used * equity_scale

        risk_pct = float(os.environ.get("AGI_RISK_PERCENT", "2.0")) / 100.0
        risk_budget = balance * risk_pct * kelly_used * conviction

        # Convert risk budget to lots using per-symbol contract size
        # Each 0.01 lot risks different amounts depending on the symbol:
        #   XAUUSDm: ~$7.68 per 0.01 lot on SL hit
        #   BTCUSDm: ~$5.00 per 0.01 lot on SL hit
        #   EURUSDm: ~$0.24 per 0.01 lot on SL hit (pip value ~$0.10)
        #   GBPUSDm: ~$0.27 per 0.01 lot on SL hit (pip value ~$0.10)
        # Use avg_loss per 0.01 lot from trade history as the risk-per-lot
        if avg_loss > 0:
            lots_from_risk = risk_budget / avg_loss
        else:
            # No loss history — use conservative pip-value estimate
            pip_value_per_lot = self._pip_value_per_lot(symbol)
            sl_distance_atr = 2.0  # assume 2x ATR SL
            lots_from_risk = risk_budget / (sl_distance_atr * pip_value_per_lot) if pip_value_per_lot > 0 else min_lots

        # Start from Kelly-risk-derived lot size
        lot_size = lots_from_risk

        # Hard cap: max risk per trade = max_sl_equity_pct% of balance (regardless of Kelly)
        # This prevents disasters like 1.0 lots on a $40 account
        max_sl_equity_pct = getattr(self.risk, 'max_sl_equity_pct', 10.0)
        if balance < 100:
            max_sl_equity_pct = max(max_sl_equity_pct, 15.0)
        max_risk_dollars = balance * max_sl_equity_pct / 100.0
        if avg_loss > 0 and lot_size > 0:
            max_lots_by_risk = max_risk_dollars / avg_loss
            lot_size = min(lot_size, max_lots_by_risk)

        # Apply broker minimum and maximum
        lot_size = max(min_lots, min(lot_size, max_lots))

        # Round to broker lot step
        lot_step = self._get_lot_step(symbol)
        lot_size = round(lot_size / lot_step) * lot_step
        lot_size = max(min_lots, min(lot_size, max_lots))

        logger.info(
            f"Kelly sizing {symbol}: f*={kelly_full:.3f} frac={kelly_fraction:.3f} "
            f"used={kelly_used:.3f} conviction={conviction:.2f} balance=${balance:.2f} "
            f"scale={equity_scale:.2f} risk_budget=${risk_budget:.2f} "
            f"max_risk=${max_risk_dollars:.2f} -> {lot_size:.2f} lots"
        )

        return lot_size

    def _symbol_info(self, symbol):
        return _mt5.symbol_info(symbol)

    def _symbol_tick(self, symbol):
        return _mt5.symbol_info_tick(symbol)

    def get_tick(self, symbol):
        return self._symbol_tick(symbol)

    def get_mid_price(self, symbol):
        tick = self._symbol_tick(symbol)
        if tick is None:
            return None
        return float((tick.bid + tick.ask) / 2.0)

    def _symbol_magic_base(self, symbol: str) -> int:
        profile = {}
        try:
            profile = self.risk.get_symbol_profile(symbol) or {}
        except Exception as exc:
            logger.debug(f"risk.get_symbol_profile failed for magic: {exc}")
            profile = {}
        if "magic_base" in profile:
            try:
                return int(profile.get("magic_base"))
            except Exception as exc:
                logger.debug(f"magic_base parse failed: {exc}")
        if "magic" in profile:
            try:
                return int(profile.get("magic"))
            except Exception as exc:
                logger.debug(f"magic parse failed: {exc}")
        return int(MAGIC_BY_SYMBOL.get(str(symbol), 59000))

    def _lane_for_order(self, order_meta: dict | None) -> str:
        lane = str((order_meta or {}).get("lane", "unknown") or "unknown").strip().lower()
        if lane in LANE_MAGIC_OFFSET:
            return lane
        return "unknown"

    def _symbol_tag(self, symbol: str) -> str:
        symbol_str = str(symbol or "").upper()
        if symbol_str.startswith("BTC"):
            return "BTC"
        if symbol_str.startswith("XAU"):
            return "XAU"
        return symbol_str[:4] or "UNK"

    def _magic_for_order(self, symbol: str, order_meta: dict | None, request_kind: str = "open") -> int:
        base = self._symbol_magic_base(symbol)
        lane = self._lane_for_order(order_meta)
        kind_offset = {"open": 0, "close": 10, "manage": 20}.get(str(request_kind), 90)
        return int(base + int(LANE_MAGIC_OFFSET.get(lane, 900)) + int(kind_offset))

    def _order_comment(self, symbol: str, order_meta: dict | None, request_kind: str = "open") -> str:
        meta = order_meta or {}
        sym = self._symbol_tag(symbol)
        lane = self._lane_for_order(meta)
        lane_tag = {"champion": "CH", "canary": "CA", "history": "HI", "unknown": "UN"}.get(lane, "UN")
        family = str(meta.get("model_family", "P") or "P").upper()[:1]
        version = str(meta.get("model_version", "") or "")
        version_tag = version[-6:] if version else "000000"
        ppo_target = float(meta.get("ppo_target", meta.get("exposure", 0.0)) or 0.0)
        ppo_tag = int(round(ppo_target * 100.0))
        req_tag = {"open": "O", "close": "C", "manage": "M"}.get(str(request_kind), "U")
        comment = f"AGI|{sym}|{lane_tag}|{req_tag}|{family}{version_tag}|P{ppo_tag:+03d}"
        return comment[:31]

    def _result_ticket(self, result):
        if result is None:
            return None
        for attr in ("order", "deal"):
            value = getattr(result, attr, None)
            if value not in (None, 0):
                return int(value)
        return None

    def _safe_order_send(self, request: dict, max_retries: int = 1) -> Any:
        """Production retry for transient MT5 retcodes (requote/price change). Refreshes price on retry."""
        if not self._is_live or _mt5 is None:
            return None
        for attempt in range(max_retries + 1):
            try:
                res = _mt5.order_send(request)
                if res is None:
                    return None
                rc = int(getattr(res, "retcode", 0) or 0)
                # Transient retcodes worth one retry with fresh price
                if rc in (10004, 10006, 10018, 10021) and attempt < max_retries:
                    # refresh price for market order
                    sym = request.get("symbol")
                    if sym:
                        tick = self._symbol_tick(sym)
                        if tick:
                            if request.get("type") in (_mt5.ORDER_TYPE_BUY, 0):
                                request["price"] = tick.ask
                            else:
                                request["price"] = tick.bid
                    time.sleep(0.15)
                    continue
                return res
            except Exception as exc:
                if attempt == max_retries:
                    logger.warning(f"_safe_order_send final fail: {exc}")
                    return None
                time.sleep(0.1)
        return None

    def _log_order_send(self, symbol: str, request_action: str, request: dict, result, order_meta: dict | None):
        meta = order_meta or {}
        requested_price = float(request.get("price", 0.0) or 0.0)
        executed_price = 0.0
        slippage_points = 0.0
        if result is not None:
            try:
                executed_price = float(getattr(result, "price", 0.0) or 0.0)
                if requested_price > 0 and executed_price > 0:
                    # Approximate slippage in points (broker dependent, best effort)
                    point = 0.00001
                    try:
                        sinfo = self._symbol_info(symbol)
                        if sinfo and getattr(sinfo, "point", 0):
                            point = float(sinfo.point)
                    except Exception:
                        pass
                    slippage_points = abs(executed_price - requested_price) / max(point, 1e-10)
            except Exception:
                pass
        payload = {
            "action": str(request_action),
            "request_action": str(request_action),
            "side": str(meta.get("order_type") or request.get("type") or ""),
            "lots": float(request.get("volume", 0.0) or 0.0),
            "executed_lots": float(request.get("volume", 0.0) or 0.0),
            "target": float(meta.get("exposure", 0.0) or 0.0),
            "ppo": float(meta.get("ppo_target", 0.0) or 0.0),
            "dreamer": float(meta.get("dreamer_target", 0.0) or 0.0),
            "agi": float(meta.get("agi_bias", 0.0) or 0.0),
            "magic": request.get("magic"),
            "comment": request.get("comment"),
            "retcode": getattr(result, "retcode", None) if result is not None else None,
            "ticket": self._result_ticket(result),
            "requested_price": requested_price,
            "executed_price": executed_price,
            "slippage_points": round(slippage_points, 1),
        }
        logger.info(
            "ORDER_SEND {} | action={} side={} lots={:.2f} target={:.4f} ppo={:.4f} dreamer={:.4f} agi={:.4f} magic={} comment={} retcode={} ticket={} req_price={} exec_price={} slip_pts={}",
            symbol,
            payload["action"],
            payload["side"],
            payload["lots"],
            payload["target"],
            payload["ppo"],
            payload["dreamer"],
            payload["agi"],
            payload["magic"],
            payload["comment"],
            payload["retcode"],
            payload["ticket"],
            payload["requested_price"],
            payload["executed_price"],
            payload["slippage_points"],
        )
        # Also append to dedicated slippage audit for harness/monitoring
        try:
            _base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            slip_log = os.path.join(_base, "logs", "slippage_audit.jsonl")
            os.makedirs(os.path.dirname(slip_log), exist_ok=True)
            with open(slip_log, "a", encoding="utf-8") as f:
                rec = {k: payload[k] for k in ["action", "symbol", "lots", "requested_price", "executed_price", "slippage_points", "retcode", "ticket"] if k in payload}
                rec["symbol"] = symbol
                rec["ts"] = datetime.now(timezone.utc).isoformat()
                f.write(json.dumps(rec, default=str) + "\n")
        except Exception:
            pass
        return payload

    def _select_filling_mode(self, symbol):
        info = self._symbol_info(symbol)
        if info is None:
            return _mt5.ORDER_FILLING_RETURN

        fm = int(getattr(info, "filling_mode", _mt5.ORDER_FILLING_RETURN))

        # Some brokers expose bitmask-like values (e.g. 3 => FOK/IOC allowed).
        if fm == 3:
            return _mt5.ORDER_FILLING_IOC

        if fm in (_mt5.ORDER_FILLING_FOK, _mt5.ORDER_FILLING_IOC, _mt5.ORDER_FILLING_RETURN):
            return fm

        # Safe fallback for market execution when RETURN is unsupported.
        return _mt5.ORDER_FILLING_IOC

    def _min_stop_distance(self, symbol):
        info = self._symbol_info(symbol)
        if info is None:
            return 0.0, 5

        point = float(info.point) if info.point else 0.0001
        stops_level = max(int(getattr(info, "trade_stops_level", 0)), 0)
        freeze_level = max(int(getattr(info, "trade_freeze_level", 0)), 0)
        min_points = max(stops_level, freeze_level) + 2
        return min_points * point, min_points

    def _sanitize_sl_tp(self, symbol, order_type, sl, tp, tick):
        info = self._symbol_info(symbol)
        if info is None or tick is None:
            return sl, tp

        digits = int(info.digits) if info.digits is not None else 5
        min_dist, _ = self._min_stop_distance(symbol)

        bid = float(tick.bid)
        ask = float(tick.ask)

        new_sl = float(sl) if sl else None
        new_tp = float(tp) if tp else None

        if order_type == _mt5.ORDER_TYPE_BUY:
            # Buy SL must stay below bid, TP must stay above ask.
            max_sl = bid - min_dist
            min_tp = ask + min_dist

            if new_sl is not None:
                if new_sl >= max_sl:
                    new_sl = max_sl
                new_sl = round(new_sl, digits)
                if new_sl <= 0:
                    new_sl = None

            if new_tp is not None:
                if new_tp <= min_tp:
                    new_tp = min_tp
                new_tp = round(new_tp, digits)
        else:
            # Sell SL must stay above ask, TP must stay below bid.
            min_sl = ask + min_dist
            max_tp = bid - min_dist

            if new_sl is not None:
                if new_sl <= min_sl:
                    new_sl = min_sl
                new_sl = round(new_sl, digits)

            if new_tp is not None:
                if new_tp >= max_tp:
                    new_tp = max_tp
                new_tp = round(new_tp, digits)
                if new_tp <= 0:
                    new_tp = None

        return new_sl, new_tp

    def get_positions(self, symbol):
        longs = []
        shorts = []

        if not self._is_live:
            return longs, shorts

        if _is_paper_mode():
            positions = _paper.paper_positions_get(symbol=symbol)
        else:
            positions = _mt5.positions_get(symbol=symbol)
        if positions:
            for p in positions:
                if isinstance(p, dict):
                    pos_type = p.get("type")
                    is_buy = (pos_type == "BUY")
                else:
                    try:
                        is_buy = (p.type == _mt5.ORDER_TYPE_BUY)
                    except AttributeError:
                        is_buy = False
                if is_buy:
                    longs.append(p)
                else:
                    shorts.append(p)
        return longs, shorts

    def reconcile_exposure(self, symbol, target_exposure, max_lots, order_meta=None, execution_context=None):
        if not self.risk.can_trade(symbol):
            return {"request_action": "blocked", "executed": False}

        longs, shorts = self.get_positions(symbol)

        long_lots = sum(p.volume for p in longs)
        short_lots = sum(p.volume for p in shorts)

        target_lots = round(float(target_exposure) * float(max_lots), 2)
        result_meta = {
            "request_action": "noop",
            "executed": False,
            "target_lots": float(target_lots),
        }
        if abs(target_lots) < 0.01:
            if long_lots > 0:
                result_meta = self.close_positions(longs, order_meta=order_meta, execution_context=execution_context)
            if short_lots > 0:
                result_meta = self.close_positions(shorts, order_meta=order_meta, execution_context=execution_context)
            return result_meta

        if target_lots > 0:
            if short_lots > 0:
                result_meta = self.close_positions(shorts, order_meta=order_meta, execution_context=execution_context)
                short_lots = 0.0
            if long_lots > target_lots + 0.01:
                result_meta = self.close_positions(longs, order_meta=order_meta, execution_context=execution_context)
                long_lots = 0.0
            add_lots = round(target_lots - long_lots, 2)
            if add_lots >= 0.01:
                result_meta = self.open_position(
                    symbol,
                    _mt5.ORDER_TYPE_BUY,
                    add_lots,
                    order_meta=order_meta,
                    execution_context=execution_context,
                )
        else:
            desired_short_lots = abs(target_lots)
            if long_lots > 0:
                result_meta = self.close_positions(longs, order_meta=order_meta, execution_context=execution_context)
                long_lots = 0.0
            if short_lots > desired_short_lots + 0.01:
                result_meta = self.close_positions(shorts, order_meta=order_meta, execution_context=execution_context)
                short_lots = 0.0
            add_lots = round(desired_short_lots - short_lots, 2)
            if add_lots >= 0.01:
                result_meta = self.open_position(
                    symbol,
                    _mt5.ORDER_TYPE_SELL,
                    add_lots,
                    order_meta=order_meta,
                    execution_context=execution_context,
                )

        if isinstance(result_meta, dict) and result_meta.get("executed"):
            self.risk.record_trade(symbol)
        return result_meta

    def close_positions(self, positions, order_meta=None, execution_context=None):
        last_meta = {"request_action": "close", "executed": False}
        for p in positions:
            tick = self._symbol_tick(p.symbol)
            if tick is None:
                logger.warning(f"Tick unavailable for {p.symbol} — skipping close for ticket {p.ticket}")
                continue

            close_type = _mt5.ORDER_TYPE_SELL if p.type == _mt5.ORDER_TYPE_BUY else _mt5.ORDER_TYPE_BUY
            close_price = tick.bid if close_type == _mt5.ORDER_TYPE_SELL else tick.ask

            request = {
                "action": _mt5.TRADE_ACTION_DEAL,
                "symbol": p.symbol,
                "volume": p.volume,
                "type": close_type,
                "position": p.ticket,
                "price": close_price,
                "deviation": 20,
                "type_time": _mt5.ORDER_TIME_GTC,
                "type_filling": self._select_filling_mode(p.symbol),
            }
            request["magic"] = self._magic_for_order(p.symbol, order_meta, request_kind="close")
            request["comment"] = self._order_comment(p.symbol, order_meta, request_kind="close")
            if _is_paper_mode():
                result = _paper.paper_close_position(p.ticket, close_price, p.volume)
            else:
                allowed, reason = self._assert_live_gate("close")
                if not allowed:
                    logger.error(f"Live gate blocked close for {p.symbol}: {reason}")
                    self.risk.record_error()
                    result = None
                else:
                    result = self._safe_order_send(request)
            last_meta = self._log_order_send(p.symbol, "close", request, result, order_meta)
            last_meta["executed"] = bool(result is not None and result.retcode == _mt5.TRADE_RETCODE_DONE)
            if result is None:
                self.risk.record_error()
            elif result.retcode != _mt5.TRADE_RETCODE_DONE:
                logger.warning(f"Close order failed for {p.symbol}: retcode={result.retcode} ticket={p.ticket}")
        return last_meta

    def _atr_points(self, symbol, bars=120, period=14):
        rates = _mt5.copy_rates_from_pos(symbol, _mt5.TIMEFRAME_M5, 0, bars)
        info = self._symbol_info(symbol)
        if rates is None or len(rates) < period + 2 or info is None:
            return None

        point = float(info.point) if info.point else 0.0001
        highs = [float(r[2]) for r in rates]
        lows = [float(r[3]) for r in rates]
        closes = [float(r[4]) for r in rates]

        trs = []
        for i in range(1, len(rates)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            trs.append(tr)

        if len(trs) < period:
            return None

        atr = sum(trs[-period:]) / float(period)
        atr_points = int(max(1, round(atr / max(point, 1e-12))))
        return atr_points

    def _dynamic_points(self, symbol):
        profile = self.risk.get_symbol_profile(symbol)
        base_sl = int(profile.get("sl_points", 250))
        base_tp = int(profile.get("tp_points", 450))

        atr_points = self._atr_points(symbol)
        if atr_points is None:
            return base_sl, base_tp

        dyn_sl = max(base_sl, int(atr_points * 1.4))
        dyn_tp = max(base_tp, int(atr_points * 2.2))
        return dyn_sl, dyn_tp

    def _get_sl_tp(self, symbol, order_type, entry_price):
        info = self._symbol_info(symbol)
        tick = self._symbol_tick(symbol)
        if info is None or tick is None:
            return None, None, 20

        point = float(info.point) if info.point else 0.0001
        digits = int(info.digits) if info.digits is not None else 5
        _, min_pts = self._min_stop_distance(symbol)
        deviation = int(self.risk.get_symbol_profile(symbol).get("entry_deviation", 20))

        dyn_sl, dyn_tp = self._dynamic_points(symbol)
        sl_points = max(int(dyn_sl), min_pts)
        tp_points = max(int(dyn_tp), min_pts)

        sl_dist = sl_points * point
        tp_dist = tp_points * point

        if order_type == _mt5.ORDER_TYPE_BUY:
            sl = round(entry_price - sl_dist, digits)
            tp = round(entry_price + tp_dist, digits)
        else:
            sl = round(entry_price + sl_dist, digits)
            tp = round(entry_price - tp_dist, digits)

        sl, tp = self._sanitize_sl_tp(symbol, order_type, sl, tp, tick)
        return sl, tp, deviation

    def open_position(self, symbol, order_type, volume, order_meta=None, execution_context=None):
        # Harness / paper controlled mode override: force tiny fixed lots for safety (ignores Kelly/ATR)
        fixed_lot_env = os.environ.get("AGI_PAPER_FIXED_LOT", "").strip()
        if fixed_lot_env:
            try:
                fixed = float(fixed_lot_env)
                if fixed > 0:
                    volume = fixed
                    logger.info(f"[HARNESS] Using AGI_PAPER_FIXED_LOT={fixed} for {symbol} (bypassed dynamic sizing)")
            except Exception:
                pass
        tick = self._symbol_tick(symbol)
        if tick is None:
            logger.warning(f"Tick unavailable for {symbol} — skipping open")
            return {"request_action": "open", "executed": False, "reason": "tick_unavailable"}

        price = tick.ask if order_type == _mt5.ORDER_TYPE_BUY else tick.bid

        # Margin check — skip if account can't afford even a 20% buffer on required margin
        try:
            acc = _mt5.account_info()
            margin_req = _mt5.order_calc_margin(order_type, symbol, float(volume), price)
            if acc is not None and margin_req is not None:
                free = float(getattr(acc, "margin_free", 0) or 0)
                # Fallback: if broker reports zero margin requirement, estimate from contract size
                req = float(margin_req)
                if req <= 0:
                    info = _mt5.symbol_info(symbol)
                    if info is not None:
                        contract = float(getattr(info, "trade_contract_size", 1) or 1)
                        leverage = float(getattr(acc, "leverage", 100) or 100)
                        if leverage <= 0 or leverage > 1e6:
                            leverage = 100
                        req = float(volume) * price * contract / leverage
                if free < req * 1.2:
                    logger.warning(
                        f"Insufficient margin for {symbol}: need ~{req:.2f}, free={free:.2f}, volume={volume}"
                    )
                    return {"request_action": "open", "executed": False, "reason": "insufficient_margin"}
        except Exception as exc:
            logger.error(f"Margin/execute precheck failed for {symbol} (proceeding to order with risk): {exc}")

        sl, tp, deviation = self._get_sl_tp(symbol, order_type, price)

        request = {
            "action": _mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": price,
            "deviation": deviation,
            "type_time": _mt5.ORDER_TIME_GTC,
            "type_filling": self._select_filling_mode(symbol),
        }
        request["magic"] = self._magic_for_order(symbol, order_meta, request_kind="open")
        request["comment"] = self._order_comment(symbol, order_meta, request_kind="open")

        if sl is not None:
            request["sl"] = sl
        if tp is not None:
            request["tp"] = tp

        if _is_paper_mode():
            result = _paper.paper_order_send(request)
        else:
            allowed, reason = self._assert_live_gate("open")
            if not allowed:
                logger.error(f"Live gate blocked open for {symbol}: {reason}")
                self.risk.record_error()
                result = None
            else:
                result = self._safe_order_send(request)
        meta = self._log_order_send(symbol, "open", request, result, order_meta)
        meta["executed"] = bool(result is not None and result.retcode == _mt5.TRADE_RETCODE_DONE)
        if result is None:
            self.risk.record_error()
        elif result.retcode != _mt5.TRADE_RETCODE_DONE:
            logger.warning(f"Order send failed for {symbol}: retcode={result.retcode} volume={volume}")
        return meta

    def manage_open_positions(self, symbol):
        if _is_paper_mode():
            positions = _paper.paper_positions_get(symbol=symbol)
        else:
            positions = _mt5.positions_get(symbol=symbol)
        info = self._symbol_info(symbol)
        tick = self._symbol_tick(symbol)
        if not positions or info is None or tick is None:
            return

        point = float(info.point) if info.point else 0.0001
        digits = int(info.digits) if info.digits is not None else 5
        profile = self.risk.get_symbol_profile(symbol)

        dyn_sl, dyn_tp = self._dynamic_points(symbol)
        breakeven_trigger = int(profile.get("breakeven_points", max(25, dyn_sl // 3)))
        trailing_trigger = int(profile.get("trailing_trigger_points", max(40, dyn_sl // 2)))
        trailing_step = int(profile.get("trailing_step_points", max(10, dyn_sl // 8)))

        for p in positions:
            current_price = tick.bid if p.type == _mt5.ORDER_TYPE_BUY else tick.ask
            profit_points = (
                (current_price - p.price_open) / point
                if p.type == _mt5.ORDER_TYPE_BUY
                else (p.price_open - current_price) / point
            )

            new_sl = float(p.sl) if p.sl else None
            new_tp = float(p.tp) if p.tp else None

            # Add TP if missing
            if new_tp is None:
                tp_dist = dyn_tp * point
                new_tp = (
                    p.price_open + tp_dist
                    if p.type == _mt5.ORDER_TYPE_BUY
                    else p.price_open - tp_dist
                )

            # Break-even promotion
            if profit_points >= breakeven_trigger:
                be_buffer = 5 * point
                be_sl = (
                    p.price_open + be_buffer
                    if p.type == _mt5.ORDER_TYPE_BUY
                    else p.price_open - be_buffer
                )
                if (p.type == _mt5.ORDER_TYPE_BUY and (new_sl is None or be_sl > new_sl)) or (
                    p.type == _mt5.ORDER_TYPE_SELL and (new_sl is None or be_sl < new_sl)
                ):
                    new_sl = be_sl

            # Trailing after trigger
            if profit_points >= trailing_trigger:
                trail_dist = trailing_step * point
                trail_sl = (
                    current_price - trail_dist
                    if p.type == _mt5.ORDER_TYPE_BUY
                    else current_price + trail_dist
                )
                if (p.type == _mt5.ORDER_TYPE_BUY and (new_sl is None or trail_sl > new_sl)) or (
                    p.type == _mt5.ORDER_TYPE_SELL and (new_sl is None or trail_sl < new_sl)
                ):
                    new_sl = trail_sl

            new_sl, new_tp = self._sanitize_sl_tp(symbol, p.type, new_sl, new_tp, tick)

            if new_sl is not None:
                new_sl = round(new_sl, digits)
            if new_tp is not None:
                new_tp = round(new_tp, digits)

            sl_changed = (new_sl is not None) and (p.sl is None or abs(float(new_sl) - float(p.sl)) > (0.5 * point))
            tp_changed = (new_tp is not None) and (p.tp is None or abs(float(new_tp) - float(p.tp)) > (0.5 * point))

            if not sl_changed and not tp_changed:
                continue

            req = {
                "action": _mt5.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "position": p.ticket,
            }
            if new_sl is not None:
                req["sl"] = new_sl
            if new_tp is not None:
                req["tp"] = new_tp
            req["magic"] = self._magic_for_order(symbol, None, request_kind="manage")
            req["comment"] = self._order_comment(symbol, None, request_kind="manage")

            if _is_paper_mode():
                result = _paper.paper_order_send(req)
            else:
                allowed, reason = self._assert_live_gate("manage")
                if not allowed:
                    logger.error(f"Live gate blocked manage for {symbol}: {reason}")
                    self.risk.record_error()
                    result = None
                else:
                    result = self._safe_order_send(req)
            self._log_order_send(symbol, "manage", req, result, {"symbol": symbol})
            if result is None or result.retcode != _mt5.TRADE_RETCODE_DONE:
                self.risk.record_error()

    def force_flatten_all(self, reason: str = "harness_rollback") -> dict:
        """Production safety: close every open position across all symbols. Used by paper harness rollback triggers.
        Returns summary of actions.
        """
        logger.critical(f"MT5Executor FORCE_FLATTEN_ALL triggered: {reason}")
        summary = {"closed": 0, "failed": 0, "symbols": []}
        if not self._is_live or _mt5 is None:
            # In paper or dry, delegate to paper layer if present
            try:
                if _is_paper_mode():
                    # best effort: nothing to do or paper handles
                    pass
            except Exception:
                pass
            return summary
        try:
            positions = _mt5.positions_get() or []
            for p in positions:
                try:
                    tick = self._symbol_tick(p.symbol)
                    if tick is None:
                        summary["failed"] += 1
                        continue
                    close_type = _mt5.ORDER_TYPE_SELL if p.type == _mt5.ORDER_TYPE_BUY else _mt5.ORDER_TYPE_BUY
                    close_price = tick.bid if close_type == _mt5.ORDER_TYPE_SELL else tick.ask
                    req = {
                        "action": _mt5.TRADE_ACTION_DEAL,
                        "symbol": p.symbol,
                        "volume": p.volume,
                        "type": close_type,
                        "position": p.ticket,
                        "price": close_price,
                        "deviation": 50,  # wider for emergency close
                        "type_time": _mt5.ORDER_TIME_GTC,
                        "type_filling": self._select_filling_mode(p.symbol),
                        "magic": self._magic_for_order(p.symbol, {"lane": "history"}, request_kind="close"),
                        "comment": f"FLATTEN|{reason}"[:31],
                    }
                    allowed, gate_reason = self._assert_live_gate("flatten")
                    if not allowed:
                        logger.error(f"Flatten blocked by gate for {p.symbol}: {gate_reason}")
                        summary["failed"] += 1
                        continue
                    res = _mt5.order_send(req)
                    self._log_order_send(p.symbol, "flatten_close", req, res, {"reason": reason})
                    if res is not None and res.retcode == _mt5.TRADE_RETCODE_DONE:
                        summary["closed"] += 1
                        summary["symbols"].append(p.symbol)
                    else:
                        summary["failed"] += 1
                except Exception as e:
                    logger.warning(f"Flatten error on {p.symbol}: {e}")
                    summary["failed"] += 1
        except Exception as exc:
            logger.error(f"force_flatten_all top level error: {exc}")
        return summary
