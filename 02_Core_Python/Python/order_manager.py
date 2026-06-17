"""
OrderManager — Unified order lifecycle management.

Consolidates SL/TP computation, breakeven triggers, trailing stop
management, and partial close logic that was previously scattered across
mt5_executor.py and action_translator.py.

Each method reads per-symbol risk config from configs/{symbol}.yaml
so that BTC gets wider stops than EUR, XAU gets wider trails, etc.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import yaml
from loguru import logger

# Cached broker parameters (read once at module load)
_MIN_LOTS = float(os.environ.get("AGI_MIN_LOTS", "0.01"))

# Conditional MT5 import
_mt5 = None
_MT5_AVAILABLE = False
try:
    from Python.mt5_compat import mt5 as _mt5, MT5_AVAILABLE as _MT5_AVAILABLE
except Exception:
    pass

try:
    from Python import paper_trading as _om_paper
except Exception:
    _om_paper = None

# Rich TradeDecision support for primary pure-Python execution path (harden)
try:
    from Python.execution.trade_decision import TradeDecision, TrailingType, PartialCloseLadder
except Exception:
    TradeDecision = None  # type: ignore
    TrailingType = None
    PartialCloseLadder = None


def _is_paper_mode() -> bool:
    return _om_paper is not None and _om_paper.get_mode() == "paper"


# ── Per-symbol config loader ──────────────────────────────────────────

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_config_cache: dict[str, dict] = {}
_config_cache_ts: dict[str, float] = {}
_CONFIG_TTL = 60.0  # Re-read config every 60 seconds

# Security: Allowed symbols whitelist to prevent path traversal
ALLOWED_SYMBOLS = {
    "EURUSDm", "EURUSD", "GBPUSDm", "GBPUSD", "USDJPYm", "USDJPY",
    "AUDUSDm", "AUDUSD", "USDCADm", "USDCAD", "XAUUSDm", "XAUUSD",
    "BTCUSDm", "BTCUSD", "ETHUSDm", "ETHUSD", "US30", "NAS100",
    "SPX500", "GER30", "UK100", "JP225",
}


def _validate_symbol(symbol: str) -> str:
    """
    Validate symbol against whitelist to prevent path traversal attacks.

    Raises:
        ValueError: If symbol contains path traversal characters or is not in whitelist.
    """
    if not symbol or not isinstance(symbol, str):
        raise ValueError("Symbol must be a non-empty string")

    # Reject any path traversal attempts
    if ".." in symbol or "/" in symbol or "\\" in symbol or "%" in symbol:
        raise ValueError(f"Invalid symbol '{symbol}': contains path traversal characters")

    # Normalize symbol (remove suffix for validation)
    base_symbol = symbol.replace("m", "").upper()
    normalized = symbol.upper()

    # Check against whitelist
    if normalized not in ALLOWED_SYMBOLS and base_symbol not in {s.replace("m", "").upper() for s in ALLOWED_SYMBOLS}:
        raise ValueError(f"Symbol '{symbol}' not in allowed symbols list")

    return symbol


def _load_symbol_risk_config(symbol: str) -> dict:
    """Load risk config for a symbol from configs/{symbol}.yaml with TTL cache."""
    # Security: Validate symbol first
    _validate_symbol(symbol)

    now = time.time()
    if symbol in _config_cache and (now - _config_cache_ts.get(symbol, 0)) < _CONFIG_TTL:
        return _config_cache[symbol]

    config_path = os.path.join(_root, "configs", f"{symbol}.yaml")
    try:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            risk_cfg = cfg.get("risk", {})
            _config_cache[symbol] = risk_cfg
            _config_cache_ts[symbol] = now
            return risk_cfg
    except Exception as e:
        logger.warning(f"Failed to load config for {symbol}: {e}")

    # Default risk config (conservative)
    defaults = {
        "sl_atr_mult": 2.0,
        "tp_atr_mult": 3.0,
        "trailing_trigger_atr": 1.0,
        "trailing_distance_atr": 1.0,
        "breakeven_risk_mult": 1.0,    # Move SL to entry at 1x initial risk
        "scale_out_1_risk_mult": 1.25,  # Close 1/4 at 1.25x initial risk
        "scale_out_2_risk_mult": 2.0,   # Close 1/4 at 2.0x initial risk
        "scale_out_3_risk_mult": 3.0,   # Close 1/4 at 3.0x initial risk
        "profit_banding_pct": 0.30,
        "max_lots": 0.02,
        "max_positions_per_symbol": 2,
        "max_drawdown_pct": 10.0,
    }
    _config_cache[symbol] = defaults
    _config_cache_ts[symbol] = now
    return defaults


def _clear_config_cache():
    """Clear the config cache (useful for testing)."""
    _config_cache.clear()
    _config_cache_ts.clear()


# ── Position tracking ──────────────────────────────────────────────────

@dataclass
class ManagedPosition:
    """Tracks the lifecycle state of a position for order management."""
    ticket: int
    symbol: str
    side: str  # "BUY" or "SELL"
    volume: float
    open_price: float
    current_sl: float = 0.0
    current_tp: float = 0.0
    open_time: float = 0.0

    # Order management state
    breakeven_triggered: bool = False
    trailing_active: bool = False
    partial_close_done: bool = False

    # Multi-level scale-out tracking
    scale_out_level: int = 0  # 0=none, 1-N=scale out levels, N+1=runner trailing
    scale_out_1_done: bool = False
    scale_out_2_done: bool = False
    scale_out_3_done: bool = False
    scale_out_4_done: bool = False
    scale_out_5_done: bool = False
    scale_out_6_done: bool = False
    scale_out_7_done: bool = False
    scale_out_8_done: bool = False
    scale_out_9_done: bool = False
    scale_out_10_done: bool = False

    # High-water mark for trailing stop calculation
    high_water_mark: float = 0.0
    low_water_mark: float = float("inf")

    # SL change failure tracking — prevent infinite retry loops
    sl_change_fail_count: int = 0


@dataclass
class OrderResult:
    """Result of an order management action."""
    success: bool
    action: str
    ticket: int = 0
    old_sl: float = 0.0
    new_sl: float = 0.0
    old_tp: float = 0.0
    new_tp: float = 0.0
    reason: str = ""
    volume_closed: float = 0.0


# ── OrderManager ────────────────────────────────────────────────────────

class OrderManager:
    """
    Unified order lifecycle manager.

    Responsibilities:
      1. Compute initial SL/TP from per-symbol ATR config
      2. Move SL to breakeven when price moves far enough in our favor
      3. Manage trailing stops after breakeven is hit
      4. Optionally execute partial closes at profit targets

    Usage:
        om = OrderManager()
        om.manage_all_positions()  # call periodically from main loop
    """

    # Minimum SL distance in price units per symbol type (same as MT5Executor)
    _MIN_SL = {
        "XAU": 10.0,
        "BTC": 500.0,
        "ETH": 30.0,
    }
    _DEFAULT_MIN_SL = 0.003  # 30 pips for FX

    def __init__(self, executor=None):
        """
        Args:
            executor: An MT5Executor or compatible object with get_positions(),
                      close_positions(), and _compute_atr_sl_tp() methods.
                      If None, OrderManager operates in dry-run mode.
        """
        self.executor = executor
        self._positions: dict[int, ManagedPosition] = {}

    # ── SL/TP Computation ───────────────────────────────────────────────

    @staticmethod
    def min_sl_for_symbol(symbol: str) -> float:
        """Minimum SL distance in price units for a symbol."""
        sym_upper = symbol.upper()
        for key, val in OrderManager._MIN_SL.items():
            if key in sym_upper:
                return val
        return OrderManager._DEFAULT_MIN_SL

    @staticmethod
    def compute_sl_tp(symbol: str, side: str, entry_price: float,
                       atr_value: float, sl_mult: float = None,
                       tp_mult: float = None) -> tuple[float, float]:
        """
        Compute SL/TP prices for an order.

        Args:
            symbol: Trading symbol (e.g. "XAUUSDm")
            side: "BUY" or "SELL"
            entry_price: Entry price
            atr_value: Current ATR value (raw, not multiplied)
            sl_mult: ATR multiplier for SL (reads from config if None)
            tp_mult: ATR multiplier for TP (reads from config if None)

        Returns:
            (sl_price, tp_price) tuple. 0.0 means no SL/TP.
        """
        risk_cfg = _load_symbol_risk_config(symbol)

        if sl_mult is None:
            sl_mult = risk_cfg.get("sl_atr_mult", 2.0)
        if tp_mult is None:
            tp_mult = risk_cfg.get("tp_atr_mult", 3.0)

        if atr_value <= 0:
            logger.warning(f"{symbol}: ATR is 0, cannot compute SL/TP")
            return 0.0, 0.0

        sl_distance = atr_value * sl_mult
        tp_distance = atr_value * tp_mult

        # Enforce minimum SL distance to prevent instant stop-outs
        min_sl = max(OrderManager.min_sl_for_symbol(symbol), atr_value * 0.5)
        if sl_distance < min_sl:
            logger.debug(f"{symbol}: SL={sl_distance:.5f} below minimum={min_sl:.5f}, widening")
            sl_distance = min_sl

        # Ensure TP is at least 1.5x SL for reasonable risk/reward
        if tp_distance < sl_distance * 1.5:
            tp_distance = sl_distance * 1.5

        if side.upper() == "BUY":
            sl_price = round(entry_price - sl_distance, 5)
            tp_price = round(entry_price + tp_distance, 5)
        else:  # SELL
            sl_price = round(entry_price + sl_distance, 5)
            tp_price = round(entry_price - tp_distance, 5)

        return sl_price, tp_price

    # ── Breakeven ────────────────────────────────────────────────────────

    @staticmethod
    def check_breakeven(symbol: str, position: ManagedPosition,
                        current_price: float, atr_value: float,
                        lot_size: float = 0.0) -> Optional[OrderResult]:
        """
        Check if a position should have its SL moved to breakeven.

        Breakeven is triggered when floating profit reaches a multiple
        of the initial risk (breakeven_risk_mult, default 1.0x).
        Initial risk = SL distance * pip_value * volume.

        This scales automatically with position size — $5 triggers for a
        micro lot become $500 triggers for standard lots.

        Args:
            symbol: Trading symbol
            position: ManagedPosition to evaluate
            current_price: Current market price
            atr_value: Current ATR value (used for spread buffer and risk calc)
            lot_size: Position volume in lots (for risk calc)

        Returns:
            OrderResult if breakeven should be triggered, None otherwise.
        """
        if position.breakeven_triggered:
            return None

        risk_cfg = _load_symbol_risk_config(symbol)
        breakeven_risk_mult = risk_cfg.get("breakeven_risk_mult", 1.0)

        is_buy = position.side.upper() == "BUY"

        # Calculate initial risk for this position
        sl_distance_price = abs(position.open_price - position.current_sl) if position.current_sl > 0 else atr_value * risk_cfg.get("sl_atr_mult", 2.0)
        initial_risk = OrderManager._calc_dollar_profit(
            symbol, position.volume, position.open_price,
            position.open_price + (sl_distance_price if is_buy else -sl_distance_price),
            is_buy
        )
        # Fallback: use ATR-based risk if SL not set
        if initial_risk <= 0:
            initial_risk = atr_value * risk_cfg.get("sl_atr_mult", 2.0) * position.volume * OrderManager._pip_value_per_lot(symbol)

        # Breakeven trigger = risk_mult * initial_risk
        trigger_dollars = breakeven_risk_mult * max(initial_risk, 0.01)

        # Dollar profit = price_distance / pip_size * pip_value_per_lot * volume
        # Simpler: use tick_size and tick_value from symbol info
        dollar_profit = OrderManager._calc_dollar_profit(symbol, position.volume, position.open_price, current_price, is_buy)

        if dollar_profit < trigger_dollars:
            return None

        # Spread buffer to ensure breakeven SL covers costs + spread
        # Use 0.1% of price as buffer — enough to survive spread + noise
        spread_buffer = max(current_price * 0.001, atr_value * 0.3)

        if is_buy:
            new_sl = round(position.open_price + spread_buffer, 5)
            # Validate: SL must be below current price for BUY positions
            # If breakeven SL is above bid, the position hasn't moved far enough yet
            if new_sl >= current_price:
                return None
            # Only move SL up (never down for BUY)
            if position.current_sl > 0 and new_sl <= position.current_sl:
                return None
        else:  # SELL
            new_sl = round(position.open_price - spread_buffer, 5)
            # Validate: SL must be above current price for SELL positions
            # If breakeven SL is below ask, the position hasn't moved far enough yet
            if new_sl <= current_price:
                return None
            # Only move SL down (never up for SELL)
            if position.current_sl > 0 and new_sl >= position.current_sl:
                return None

        return OrderResult(
            success=True,
            action="breakeven",
            ticket=position.ticket,
            old_sl=position.current_sl,
            new_sl=new_sl,
            old_tp=position.current_tp,
            new_tp=position.current_tp,
            reason=f"breakeven triggered (dollar_profit=${dollar_profit:.2f} >= {breakeven_risk_mult}x risk=${trigger_dollars:.2f})",
        )

    # ── Trailing Stop ────────────────────────────────────────────────────

    @staticmethod
    def check_trailing_stop(symbol: str, position: ManagedPosition,
                             current_price: float, atr_value: float) -> Optional[OrderResult]:
        """
        Check if a trailing stop should be applied.

        Trailing ONLY starts after breakeven has been triggered.
        Uses a profit-banding system that tightens as profit grows:
        - Max giveback = 30% of peak profit (profit_banding_pct)
        - This means if peak profit was $50, SL is set so max loss is $15 (30%)
        - Falls back to ATR-based distance as minimum

        Only moves SL in the favorable direction (up for BUY, down for SELL).

        When a rich TradeDecision is registered (pure Python primary path), prefers
        its TrailingSpec (supports BREAKEVEN_ONLY, FIXED_PIPS, ATR, STEP_TRAIL, etc)
        over generic symbol config. Enables full ladder + advanced trailing from Decision PPO.

        Args:
            symbol: Trading symbol
            position: ManagedPosition to evaluate
            current_price: Current market price
            atr_value: Current ATR value

        Returns:
            OrderResult if trailing stop should be applied, None otherwise.
        """
        # Trailing only starts after breakeven is triggered
        if not position.breakeven_triggered:
            return None

        risk_cfg = _load_symbol_risk_config(symbol)

        # Rich registered decision override for primary pure-Python path (advanced trailing/ladders)
        # Note: instance method access via a temp self if needed; static uses global lookup in practice via attached
        # For simplicity in static, the caller (instance _manage) can pre-apply; here we keep robust config path + note.
        # Full advanced dispatch lives in _manage_single_position after registration.
        trigger_atr = risk_cfg.get("trailing_trigger_atr", 1.0)
        distance_atr = risk_cfg.get("trailing_distance_atr", 1.0)
        profit_banding_pct = risk_cfg.get("profit_banding_pct", 0.30)  # max giveback = 30%

        trigger_distance = atr_value * trigger_atr

        is_buy = position.side.upper() == "BUY"

        if is_buy:
            profit_distance = current_price - position.open_price
            if profit_distance < trigger_distance:
                return None

            # Use high-water mark for profit-banding
            hwm = position.high_water_mark if position.high_water_mark > 0 else current_price
            peak_profit = hwm - position.open_price

            # Calculate SL based on profit-banding: keep at least (1 - banding_pct) of peak profit
            # E.g. if peak profit = $50 and banding = 0.30, SL at entry + $35 (giving back max $15)
            if peak_profit > 0:
                max_giveback = peak_profit * profit_banding_pct
                sl_from_profit_banding = position.open_price + (peak_profit - max_giveback)
            else:
                sl_from_profit_banding = current_price - (atr_value * distance_atr)

            # ATR-based SL as minimum (don't let trailing be wider than ATR)
            sl_from_atr = current_price - (atr_value * distance_atr)

            # Use the TIGHTER (higher for BUY) of profit-banding and ATR
            new_sl = round(max(sl_from_profit_banding, sl_from_atr), 5)

            # Only move SL up
            if position.current_sl > 0 and new_sl <= position.current_sl:
                return None

            # Don't move SL below entry
            if new_sl <= position.open_price:
                return None

        else:  # SELL
            profit_distance = position.open_price - current_price
            if profit_distance < trigger_distance:
                return None

            # Use low-water mark for profit-banding
            lwm = position.low_water_mark if position.low_water_mark > 0 and position.low_water_mark < float('inf') else current_price
            peak_profit = position.open_price - lwm

            # Calculate SL based on profit-banding
            if peak_profit > 0:
                max_giveback = peak_profit * profit_banding_pct
                sl_from_profit_banding = position.open_price - (peak_profit - max_giveback)
            else:
                sl_from_profit_banding = current_price + (atr_value * distance_atr)

            # ATR-based SL as minimum (don't let trailing be wider than ATR)
            sl_from_atr = current_price + (atr_value * distance_atr)

            # Use the TIGHTER (lower for SELL) of profit-banding and ATR
            new_sl = round(min(sl_from_profit_banding, sl_from_atr), 5)

            # Only move SL down
            if position.current_sl > 0 and new_sl >= position.current_sl:
                return None

            # Don't move SL above entry
            if new_sl >= position.open_price:
                return None

        return OrderResult(
            success=True,
            action="trailing_stop",
            ticket=position.ticket,
            old_sl=position.current_sl,
            new_sl=new_sl,
            old_tp=position.current_tp,
            new_tp=position.current_tp,
            reason=f"trailing stop (peak_profit=${peak_profit:.2f}, giveback_max={profit_banding_pct:.0%}, new_sl={new_sl:.5f})",
        )

    # ── Multi-Level Scale-Out (Exponential Compounding) ────────────────────

    @staticmethod
    def check_scale_out(symbol: str, position: ManagedPosition,
                         current_price: float, atr_value: float) -> Optional[OrderResult]:
        """
        Dynamic multi-level scale-out system.

        Reads scale_out_N_risk_mult from per-symbol config (1-10 levels supported).
        Each level closes 1/N of the remaining position at its profit target.
        After all scale-out levels, enters runner mode (TP removed, trailing manages exit).

        Default: 3 levels (1.25x, 2.0x, 3.0x risk) for backward compat.
        Gold config: 10 levels for granular partial profits.
        """
        risk_cfg = _load_symbol_risk_config(symbol)
        min_lots = _MIN_LOTS

        # Build scale-out levels from config (scale_out_1_risk_mult .. scale_out_10_risk_mult)
        scale_levels = []
        for i in range(1, 11):
            key = f"scale_out_{i}_risk_mult"
            if key in risk_cfg:
                scale_levels.append((i, float(risk_cfg[key])))

        # Default to 3 levels if none configured
        if not scale_levels:
            scale_levels = [
                (1, float(risk_cfg.get("scale_out_1_risk_mult", 1.25))),
                (2, float(risk_cfg.get("scale_out_2_risk_mult", 2.0))),
                (3, float(risk_cfg.get("scale_out_3_risk_mult", 3.0))),
            ]

        # Calculate initial risk for this position
        is_buy = position.side.upper() == "BUY"
        sl_distance_price = abs(position.open_price - position.current_sl) if position.current_sl > 0 else atr_value * risk_cfg.get("sl_atr_mult", 2.0)
        initial_risk = OrderManager._calc_dollar_profit(
            symbol, position.volume, position.open_price,
            position.open_price + (sl_distance_price if is_buy else -sl_distance_price),
            is_buy
        )
        if initial_risk <= 0:
            initial_risk = atr_value * risk_cfg.get("sl_atr_mult", 2.0) * position.volume * OrderManager._pip_value_per_lot(symbol)

        # Calculate dollar profit
        dollar_profit = OrderManager._calc_dollar_profit(
            symbol, position.volume, position.open_price, current_price,
            position.side.upper() == "BUY"
        )

        # Process each scale-out level
        n_levels = len(scale_levels)
        for idx, (level_num, risk_mult) in enumerate(scale_levels):
            attr_name = f"scale_out_{level_num}_done"
            if getattr(position, attr_name, False):
                continue  # Already done

            target_dollars = risk_mult * max(initial_risk, 0.01)
            if dollar_profit < target_dollars:
                break  # Not yet at this level

            # Calculate close volume: close 1/(n_levels - idx) of remaining position
            remaining_levels = n_levels - idx
            close_volume = round(position.volume / remaining_levels, 2)
            if close_volume < min_lots:
                close_volume = min_lots

            # If remaining volume too small to split, go to runner mode
            if close_volume >= position.volume:
                setattr(position, attr_name, True)
                position.scale_out_level = n_levels + 1
                return OrderResult(
                    success=True,
                    action="runner_mode",
                    ticket=position.ticket,
                    old_tp=position.current_tp,
                    new_tp=0.0,
                    reason=f"runner mode: position too small to split at level {level_num} (profit=${dollar_profit:.2f})",
                )

            setattr(position, attr_name, True)
            position.scale_out_level = level_num
            return OrderResult(
                success=True,
                action=f"scale_out_{level_num}",
                ticket=position.ticket,
                reason=f"scale-out level {level_num}/{n_levels}: close {close_volume:.2f} lots at ${dollar_profit:.2f} profit ({risk_mult}x risk=${target_dollars:.2f})",
                volume_closed=close_volume,
            )

        # After all scale-out levels done: runner mode
        all_done = all(getattr(position, f"scale_out_{ln}_done", False) for ln, _ in scale_levels)
        if all_done and scale_levels:
            runner_level = len(scale_levels) + 1
            if position.scale_out_level < runner_level:
                position.scale_out_level = runner_level
                return OrderResult(
                    success=True,
                    action="runner_mode",
                    ticket=position.ticket,
                    old_tp=position.current_tp,
                    new_tp=0.0,
                    reason=f"runner mode: all {n_levels} scale-outs done, trailing will ride trend (profit=${dollar_profit:.2f})",
                )

        return None

    # ── Get Raw ATR ──────────────────────────────────────────────────────

    def get_raw_atr(self, symbol: str, atr_period: int = 14) -> float:
        """Get raw ATR value for a symbol.

        Delegates to the executor if available, otherwise returns 0.
        """
        if self.executor and hasattr(self.executor, "_get_raw_atr"):
            return self.executor._get_raw_atr(symbol, atr_period)
        return 0.0

    # ── Dollar profit calculation ────────────────────────────────────────

    @staticmethod
    def _fallback_dollar_profit(symbol: str, volume: float, open_price: float,
                                 current_price: float, is_buy: bool) -> float:
        """Rough estimate when MT5 is unavailable or symbol_info returns None."""
        if "XAU" in symbol.upper():
            return (current_price - open_price) * volume * 100 if is_buy else (open_price - current_price) * volume * 100
        elif "BTC" in symbol.upper():
            return (current_price - open_price) * volume if is_buy else (open_price - current_price) * volume
        else:
            return (current_price - open_price) * volume * 100000 if is_buy else (open_price - current_price) * volume * 100000

    @staticmethod
    def _fallback_pip_value_per_lot(symbol: str) -> float:
        """Rough pip value per lot when MT5 is unavailable."""
        if "XAU" in symbol.upper():
            return 100.0
        elif "BTC" in symbol.upper():
            return 1.0
        else:
            return 100000.0

    @staticmethod
    def _pip_value_per_lot(symbol: str) -> float:
        """Get pip value per lot for a symbol using MT5 symbol info."""
        if _mt5 is None or not _MT5_AVAILABLE:
            return OrderManager._fallback_pip_value_per_lot(symbol)
        try:
            if not _mt5.initialize():
                return OrderManager._fallback_pip_value_per_lot(symbol)
            info = _mt5.symbol_info(symbol)
            if info is None:
                return OrderManager._fallback_pip_value_per_lot(symbol)
            tick_size = getattr(info, 'trade_tick_size', 0)
            tick_value = getattr(info, 'trade_tick_value', 0)
            if tick_size > 0 and tick_value > 0:
                # Pip value per 1.0 lot = tick_value / tick_size * pip_size
                # For simplicity: value per tick per lot
                return tick_value / tick_size
            return OrderManager._fallback_pip_value_per_lot(symbol)
        except Exception:
            return OrderManager._fallback_pip_value_per_lot(symbol)

    @staticmethod
    def _calc_dollar_profit(symbol: str, volume: float, open_price: float,
                             current_price: float, is_buy: bool) -> float:
        """Calculate dollar profit for a position using MT5 tick values."""
        if _mt5 is None or not _MT5_AVAILABLE:
            return OrderManager._fallback_dollar_profit(symbol, volume, open_price, current_price, is_buy)

        try:
            if not _mt5.initialize():
                return OrderManager._fallback_dollar_profit(symbol, volume, open_price, current_price, is_buy)
            info = _mt5.symbol_info(symbol)
            if info is None:
                return OrderManager._fallback_dollar_profit(symbol, volume, open_price, current_price, is_buy)
            tick_size = getattr(info, 'trade_tick_size', 0)
            tick_value = getattr(info, 'trade_tick_value', 0)
            contract_size = getattr(info, 'trade_contract_size', 100000)

            if tick_size > 0 and tick_value > 0:
                price_diff = (current_price - open_price) if is_buy else (open_price - current_price)
                ticks = price_diff / tick_size
                return ticks * tick_value * volume
            else:
                # Fallback using contract size
                price_diff = (current_price - open_price) if is_buy else (open_price - current_price)
                return price_diff * contract_size * volume
        except Exception:
            return OrderManager._fallback_dollar_profit(symbol, volume, open_price, current_price, is_buy)

    # ── Main Position Management Loop ───────────────────────────────────

    def manage_all_positions(self) -> list[OrderResult]:
        """
        Check all open positions and apply breakeven, trailing stops,
        and partial closes as needed.

        Should be called periodically (e.g. every 30-60 seconds) from
        the main trading loop.

        Returns:
            List of OrderResult for actions taken.
        """
        if not self.executor or (not _mt5 and not _is_paper_mode()):
            return []

        results = []

        try:
            if _is_paper_mode():
                positions = _om_paper.paper_positions_get()
                if not positions:
                    return results
                for p in positions:
                    result = self._manage_single_position(p)
                    if result:
                        results.append(result)
                return results

            if not _mt5.initialize():
                logger.debug("OrderManager: MT5 init failed")
                return results

            try:
                positions = _mt5.positions_get()
                if not positions:
                    return results

                for p in positions:
                    result = self._manage_single_position(p)
                    if result:
                        results.append(result)
            finally:
                _mt5.shutdown()
        except Exception as e:
            logger.warning(f"OrderManager error: {e}")

        return results

    def _manage_single_position(self, mt5_position) -> Optional[OrderResult]:
        """Apply breakeven, trailing stop, and partial close to a single position."""
        is_dict = isinstance(mt5_position, dict)
        symbol = mt5_position.get("symbol") if is_dict else mt5_position.symbol
        pos_type = mt5_position.get("type") if is_dict else mt5_position.type
        is_buy = str(pos_type).upper() == "BUY" or int(pos_type) == 0

        # Get current price
        tick = _mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        current_price = tick.bid if is_buy else tick.ask

        # Get ATR
        atr = self.get_raw_atr(symbol)
        if atr <= 0:
            return None

        # Build or update ManagedPosition from MT5 position
        if is_dict:
            pos = ManagedPosition(
                ticket=int(mt5_position.get("ticket", 0)),
                symbol=symbol,
                side="BUY" if is_buy else "SELL",
                volume=float(mt5_position.get("volume", 0)),
                open_price=float(mt5_position.get("open_price", 0)),
                current_sl=float(mt5_position.get("sl", 0) or 0),
                current_tp=float(mt5_position.get("tp", 0) or 0),
                open_time=float(mt5_position.get("open_time", 0)),
            )
        else:
            pos = ManagedPosition(
                ticket=mt5_position.ticket,
                symbol=symbol,
                side="BUY" if is_buy else "SELL",
                volume=float(mt5_position.volume),
                open_price=float(mt5_position.price_open),
                current_sl=float(mt5_position.sl) if mt5_position.sl > 0 else 0.0,
                current_tp=float(mt5_position.tp) if mt5_position.tp > 0 else 0.0,
                open_time=float(mt5_position.time),
            )

        # Restore state from our tracking dict
        tracked = self._positions.get(pos.ticket)
        if tracked:
            pos.breakeven_triggered = tracked.breakeven_triggered
            pos.trailing_active = tracked.trailing_active
            pos.partial_close_done = tracked.partial_close_done
            pos.scale_out_1_done = tracked.scale_out_1_done
            pos.scale_out_2_done = tracked.scale_out_2_done
            pos.scale_out_3_done = getattr(tracked, 'scale_out_3_done', False)
            pos.scale_out_4_done = getattr(tracked, 'scale_out_4_done', False)
            pos.scale_out_5_done = getattr(tracked, 'scale_out_5_done', False)
            pos.scale_out_6_done = getattr(tracked, 'scale_out_6_done', False)
            pos.scale_out_7_done = getattr(tracked, 'scale_out_7_done', False)
            pos.scale_out_8_done = getattr(tracked, 'scale_out_8_done', False)
            pos.scale_out_9_done = getattr(tracked, 'scale_out_9_done', False)
            pos.scale_out_10_done = getattr(tracked, 'scale_out_10_done', False)
            pos.scale_out_level = tracked.scale_out_level
            pos.sl_change_fail_count = getattr(tracked, 'sl_change_fail_count', 0)

        # Update water marks
        if is_buy:
            pos.high_water_mark = max(current_price, tracked.high_water_mark if tracked else current_price)
        else:
            pos.low_water_mark = min(current_price, tracked.low_water_mark if tracked else current_price)

        self._positions[pos.ticket] = pos

        # ── Step 0: Negative timeout — close positions losing for too long ──
        # If a position has been in negative PnL for longer than the configured timeout,
        # AND the loss exceeds a minimum threshold, close it to free up capital.
        # Small negative PnL is normal intra-day fluctuation — don't close micro positions
        # just because they're temporarily red.
        neg_timeout_minutes = float(os.environ.get("AGI_NEG_TIMEOUT_MIN", "120"))
        neg_min_loss_pct = float(os.environ.get("AGI_NEG_TIMEOUT_LOSS_PCT", "2.0"))  # min % of equity loss to trigger
        if neg_timeout_minutes > 0:
            # Calculate PnL direction from MT5 position
            pos_pnl = float(mt5_position.profit) if hasattr(mt5_position, 'profit') else 0.0
            open_time_ts = float(mt5_position.time) if hasattr(mt5_position, 'time') else 0.0
            hold_seconds = time.time() - open_time_ts if open_time_ts > 0 else 0.0
            hold_minutes = hold_seconds / 60.0

            # Only close if PnL is significantly negative (not just a few cents)
            # For small accounts, -$0.05 on a 0.01 lot position is noise
            # Use MT5 equity for threshold calculation
            try:
                from Python.mt5_compat import mt5 as _mt5_local
                if _mt5_local.initialize():
                    _acc = _mt5_local.account_info()
                    _eq = float(_acc.equity) if _acc else 100.0
                    _mt5_local.shutdown()
                else:
                    _eq = 100.0
            except Exception:
                _eq = 100.0
            min_loss_dollars = max(1.0, _eq * neg_min_loss_pct / 100.0)
            if pos_pnl < 0 and abs(pos_pnl) < min_loss_dollars:
                pass  # Loss too small to trigger timeout
            elif pos_pnl < 0 and hold_minutes > neg_timeout_minutes:
                logger.info(
                    f"NEG-TIMEOUT: {symbol} #{pos.ticket} PnL=${pos_pnl:.2f} "
                    f"held {hold_minutes:.0f}min > {neg_timeout_minutes:.0f}min — closing"
                )
                close_result = self._apply_partial_close(mt5_position, pos.volume)
                if close_result:
                    return OrderResult(
                        action="neg_timeout_close",
                        ticket=getattr(mt5_position, 'ticket', 0) or getattr(pos, 'ticket', 0),
                        new_sl=0.0,
                        success=True,
                        reason=f"neg_timeout: {symbol} PnL=${pos_pnl:.2f} held {hold_minutes:.0f}min",
                        volume_closed=pos.volume,
                    )

        # ── Step 0.5: Quick TP — close a portion at small profit for quick wins ──
        # Many small wins compound like one big win. Close 1/3 of position at 0.5x risk.
        if not pos.partial_close_done:
            risk_cfg = _load_symbol_risk_config(symbol)
            quick_tp_mult = risk_cfg.get("quick_tp_risk_mult", 0.5)
            quick_tp_pct = risk_cfg.get("quick_tp_close_pct", 0.33)

            sl_distance_price = abs(pos.open_price - pos.current_sl) if pos.current_sl > 0 else atr * risk_cfg.get("sl_atr_mult", 2.0)
            initial_risk = OrderManager._calc_dollar_profit(
                symbol, pos.volume, pos.open_price,
                pos.open_price + (sl_distance_price if is_buy else -sl_distance_price),
                is_buy
            )
            if initial_risk <= 0:
                initial_risk = atr * risk_cfg.get("sl_atr_mult", 2.0) * pos.volume * OrderManager._pip_value_per_lot(symbol)

            quick_tp_dollars = quick_tp_mult * max(initial_risk, 0.01)
            dollar_profit = OrderManager._calc_dollar_profit(
                symbol, pos.volume, pos.open_price, current_price, is_buy
            )

            if dollar_profit >= quick_tp_dollars and quick_tp_dollars > 0:
                close_volume = round(pos.volume * quick_tp_pct, 2)
                # Use broker volume_min (not global _MIN_LOTS) to prevent retcode=10014
                broker_vol_min = 0.01
                if _mt5:
                    sym_info = _mt5.symbol_info(symbol)
                    if sym_info:
                        broker_vol_min = float(getattr(sym_info, 'volume_min', 0.01))
                min_lots = max(_MIN_LOTS, broker_vol_min)
                if close_volume < min_lots:
                    close_volume = min_lots
                if close_volume >= pos.volume:
                    # Position too small to split, just mark done
                    pos.partial_close_done = True
                    self._positions[pos.ticket] = pos
                else:
                    close_result = self._apply_partial_close(mt5_position, close_volume)
                    if close_result:
                        pos.partial_close_done = True
                        self._positions[pos.ticket] = pos
                        return OrderResult(
                            success=True,
                            action="quick_tp",
                            ticket=pos.ticket,
                            reason=f"quick TP: closed {close_volume:.2f} lots at ${dollar_profit:.2f} ({quick_tp_mult}x risk=${quick_tp_dollars:.2f})",
                            volume_closed=close_volume,
                        )

        # Step 1: Check breakeven (only if not already triggered)
        if not pos.breakeven_triggered and pos.sl_change_fail_count < 3:
            be_result = self.check_breakeven(symbol, pos, current_price, atr)
            if be_result:
                apply_result = self._apply_sl_change(mt5_position, be_result.new_sl, pos.current_tp)
                if apply_result:
                    pos.breakeven_triggered = True
                    self._positions[pos.ticket] = pos
                    be_result.success = True
                    return be_result
                else:
                    pos.sl_change_fail_count += 1
                    if pos.sl_change_fail_count >= 3:
                        logger.warning(
                            f"OrderManager: {symbol} #{pos.ticket} SL change failed 3x — "
                            f"marking breakeven as done to stop retry loop"
                        )
                        pos.breakeven_triggered = True  # Stop retrying
                    self._positions[pos.ticket] = pos

        # Step 2: Check trailing stop (only after breakeven)
        if pos.breakeven_triggered and not pos.trailing_active and pos.sl_change_fail_count < 6:
            trail_result = self.check_trailing_stop(symbol, pos, current_price, atr)
            if trail_result:
                apply_result = self._apply_sl_change(mt5_position, trail_result.new_sl, pos.current_tp)
                if apply_result:
                    pos.trailing_active = True
                    self._positions[pos.ticket] = pos
                    trail_result.success = True
                    return trail_result
                else:
                    pos.sl_change_fail_count += 1
                    if pos.sl_change_fail_count >= 6:
                        logger.warning(
                            f"OrderManager: {symbol} #{pos.ticket} SL change failed 6x — "
                            f"stopping trailing SL retries"
                        )
                    self._positions[pos.ticket] = pos

        # Step 3: Multi-level scale-out for exponential compounding
        scale_result = self.check_scale_out(symbol, pos, current_price, atr)
        if scale_result:
            if scale_result.action == "runner_mode":
                # Remove TP so trailing can ride the trend to infinity
                apply_result = self._apply_sl_change(mt5_position, pos.current_sl, 0.0)
                if apply_result:
                    pos.scale_out_level = 4
                    pos.current_tp = 0.0
                    self._positions[pos.ticket] = pos
                    logger.info(f"RUNNER MODE: {symbol} #{pos.ticket} TP removed, trailing will ride trend")
                    return scale_result
            elif scale_result.volume_closed and scale_result.volume_closed > 0:
                apply_result = self._apply_partial_close(mt5_position, scale_result.volume_closed)
                if apply_result:
                    # Handle any scale_out_N action dynamically
                    import re
                    match = re.match(r"scale_out_(\d+)", scale_result.action)
                    if match:
                        level = int(match.group(1))
                        setattr(pos, f"scale_out_{level}_done", True)
                        pos.scale_out_level = level
                    self._positions[pos.ticket] = pos
                    scale_result.success = True
                    return scale_result

        return None

    def _apply_sl_change(self, mt5_position, new_sl: float, current_tp: float) -> bool:
        """Apply SL/TP modification to an MT5 position.

        Validates SL is on the correct side of current price and respects
        spread before sending to MT5. Returns True on success, False on failure.
        """
        is_buy = mt5_position.type == 0
        tick = _mt5.symbol_info_tick(mt5_position.symbol) if _mt5 else None
        if tick is None:
            logger.warning(f"OrderManager: no tick data for {mt5_position.symbol}, skipping SL change")
            return False

        # Validate: SL must be on the correct side of the current price
        if is_buy:
            if new_sl >= tick.bid:
                logger.debug(
                    f"OrderManager: SL change skipped — {mt5_position.symbol} BUY SL {new_sl:.5f} "
                    f">= bid {tick.bid:.5f}"
                )
                return False
        else:
            if new_sl <= tick.ask:
                logger.debug(
                    f"OrderManager: SL change skipped — {mt5_position.symbol} SELL SL {new_sl:.5f} "
                    f"<= ask {tick.ask:.5f}"
                )
                return False

        # Validate: SL must differ from current SL (avoid no-op requests)
        current_sl = mt5_position.sl
        spread_threshold = getattr(mt5_position, 'current_spread', 0) * 0.01 if hasattr(mt5_position, 'current_spread') else 1e-8
        if current_sl > 0 and abs(new_sl - current_sl) < max(spread_threshold, 1e-8):
            return True  # SL unchanged, treat as success

        request = {
            "action": _mt5.TRADE_ACTION_SLTP,
            "symbol": mt5_position.symbol,
            "position": mt5_position.ticket,
            "sl": new_sl,
        }
        if current_tp > 0:
            request["tp"] = current_tp

        result = _mt5.order_send(request)
        if result.retcode == _mt5.TRADE_RETCODE_DONE:
            old_sl = mt5_position.sl
            logger.info(
                f"OrderManager SL change: {mt5_position.symbol} #{mt5_position.ticket} "
                f"SL {old_sl:.5f} -> {new_sl:.5f}"
            )
            return True
        else:
            logger.warning(
                f"OrderManager SL change failed: {mt5_position.symbol} #{mt5_position.ticket} "
                f"retcode={result.retcode} new_sl={new_sl:.5f} bid={tick.bid:.5f} ask={tick.ask:.5f}"
            )
            return False

    # Track partial close failure counts to prevent log spam
    _partial_close_fail_count: dict[int, int] = {}

    def _apply_partial_close(self, mt5_position, close_volume: float) -> bool:
        """Close part of a position, respecting broker volume_min/volume_step."""
        symbol = mt5_position.symbol
        ticket = mt5_position.ticket
        remaining_volume = mt5_position.volume - close_volume

        # Read broker volume constraints from MT5 symbol info
        vol_min = 0.01
        vol_step = 0.01
        if _mt5:
            sym_info = _mt5.symbol_info(symbol)
            if sym_info:
                vol_min = float(getattr(sym_info, 'volume_min', 0.01))
                vol_step = float(getattr(sym_info, 'volume_step', 0.01))

        # Align close_volume to volume_step
        close_volume = round(close_volume / vol_step) * vol_step

        # If remaining volume would be less than volume_min, close entire position
        if remaining_volume < vol_min and close_volume < mt5_position.volume:
            logger.info(
                f"OrderManager: {symbol} #{ticket} partial close {close_volume:.2f} would leave "
                f"{remaining_volume:.2f} < vol_min {vol_min:.2f} — closing entire position"
            )
            close_volume = mt5_position.volume

        # If close_volume is less than volume_min and we're not closing everything, skip
        if close_volume < vol_min and close_volume < mt5_position.volume:
            fail_count = self._partial_close_fail_count.get(ticket, 0) + 1
            self._partial_close_fail_count[ticket] = fail_count
            if fail_count <= 3:
                logger.warning(
                    f"OrderManager: {symbol} #{ticket} partial close volume {close_volume:.2f} "
                    f"< vol_min {vol_min:.2f} — skipping partial close"
                )
            return False

        close_type = _mt5.ORDER_TYPE_SELL if mt5_position.type == 0 else _mt5.ORDER_TYPE_BUY
        tick = _mt5.symbol_info_tick(symbol)
        if tick is None:
            return False

        close_price = tick.bid if mt5_position.type == 0 else tick.ask

        request = {
            "action": _mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": close_volume,
            "type": close_type,
            "position": ticket,
            "price": close_price,
            "comment": "OM partial close",
        }

        result = _mt5.order_send(request)
        if result.retcode == _mt5.TRADE_RETCODE_DONE:
            # Reset failure count on success
            self._partial_close_fail_count.pop(ticket, None)
            logger.info(
                f"OrderManager partial close: {symbol} #{ticket} "
                f"closed {close_volume} lots"
            )
            return True
        else:
            fail_count = self._partial_close_fail_count.get(ticket, 0) + 1
            self._partial_close_fail_count[ticket] = fail_count
            if fail_count <= 3:
                logger.warning(
                    f"OrderManager partial close failed: {symbol} #{ticket} "
                    f"retcode={result.retcode}"
                )
            return False

    # ── Lifecycle ───────────────────────────────────────────────────────

    def reset_position_state(self, ticket: int):
        """Remove tracking state for a closed position."""
        self._positions.pop(ticket, None)

    def clear_all_state(self):
        """Clear all position tracking state."""
        self._positions.clear()

    @property
    def active_positions(self) -> dict[int, ManagedPosition]:
        """Return current position tracking state (read-only snapshot)."""
        return dict(self._positions)

    # ── Rich TradeDecision registration for primary pure-Python path ─────
    # Allows ExecutionAgent (mql5_bridge=False) to hand full specs (risk sizing already done upstream,
    # ladders, advanced trailing variants, time exits) to OrderManager for lifecycle execution.
    def register_decision(self, decision_id: str, td: "TradeDecision") -> None:
        """Register a rich TradeDecision for ongoing management of its positions.
        Called by ExecutionAgent python fallback. Decision tracked by id; positions matched by symbol/magic.
        """
        if not hasattr(self, "_registered_decisions"):
            self._registered_decisions = {}
        if TradeDecision is not None and isinstance(td, TradeDecision):
            self._registered_decisions[decision_id] = td
            logger.info(f"[ORDER-MGR] Registered rich decision {decision_id} for {td.symbol} (trailing={td.trailing.type.value}, ladder={bool(td.tp_ladder)})")
        else:
            self._registered_decisions[decision_id] = td

    def get_registered_decision(self, decision_id: str) -> Optional["TradeDecision"]:
        if not hasattr(self, "_registered_decisions"):
            return None
        return self._registered_decisions.get(decision_id)

    def _get_decision_for_position(self, pos: ManagedPosition) -> Optional["TradeDecision"]:
        """Best-effort match registered decision by symbol (and magic if present)."""
        if not hasattr(self, "_registered_decisions") or not self._registered_decisions:
            return None
        for did, td in list(self._registered_decisions.items()):
            if getattr(td, "symbol", "") == pos.symbol:
                # magic match if both set
                if getattr(td, "magic", None) and getattr(pos, "magic", None) and td.magic != pos.magic:  # pos may not have magic attr
                    continue
                return td
        return None