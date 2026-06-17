from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from Python.mt5_compat import mt5 as _mt5

_STATE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "live_state.json")
_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
PAPER_CLOSED_LOG = os.path.join(_LOG_DIR, "paper_closed_trades.jsonl")

PAPER_DEFAULT_BALANCE = 100_000.0


def _load_state() -> dict:
    try:
        with open(_STATE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state(state: dict):
    try:
        with open(_STATE_PATH, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as exc:
        logger.warning(f"Failed to save live_state.json: {exc}")


def get_mode() -> str:
    env_mode = os.environ.get("CHAIN_GAMBLER_EXECUTION_MODE", "").strip().lower()
    if env_mode in ("paper", "demo", "live"):
        return env_mode
    state = _load_state()
    return state.get("trading", {}).get("mode", "paper")


def set_mode(mode: str) -> dict:
    state = _load_state()
    if "trading" not in state:
        state["trading"] = {}
    state["trading"]["mode"] = mode
    if mode == "paper":
        _ensure_paper_account(state)
    _save_state(state)
    return {"mode": mode}


def _ensure_paper_account(state: dict):
    trading = state.setdefault("trading", {})
    if "paper_account" not in trading:
        trading["paper_account"] = {
            "balance": PAPER_DEFAULT_BALANCE,
            "equity": PAPER_DEFAULT_BALANCE,
            "free_margin": PAPER_DEFAULT_BALANCE,
            "profit": 0.0,
            "open_positions": 0,
            "positions": [],
            "realized_today": 0.0,
            "drawdown_pct": 0.0,
        }


def get_paper_account() -> dict:
    state = _load_state()
    _ensure_paper_account(state)
    return state["trading"]["paper_account"]


def _next_ticket() -> int:
    state = _load_state()
    tid = state.get("trading", {}).get("paper_next_ticket", 900_000)
    state.setdefault("trading", {})["paper_next_ticket"] = tid + 1
    _save_state(state)
    return tid


def _get_current_price(symbol: str, side: str) -> float | None:
    tick = _mt5.symbol_info_tick(symbol)
    if tick is None:
        return None
    return float(tick.ask if side == "buy" else tick.bid)


def _calc_pnl(position: dict, current_price: float) -> float:
    if position["type"] == "BUY":
        return (current_price - position["open_price"]) * position["volume"] * 100_000
    else:
        return (position["open_price"] - current_price) * position["volume"] * 100_000


def paper_order_send(request: dict) -> dict | None:
    """Simulate an MT5 order_send in paper mode.

    Returns a fake result object with `.retcode = TRADE_RETCODE_DONE` on success.
    """
    symbol = request.get("symbol")
    action = request.get("action")
    order_type = request.get("type")
    volume = float(request.get("volume", 0))
    price = request.get("price")
    sl = request.get("sl")
    tp = request.get("tp")
    magic = request.get("magic", 0)
    comment = request.get("comment", "")

    side = "buy" if order_type == _mt5.ORDER_TYPE_BUY else "sell"

    if action == _mt5.TRADE_ACTION_DEAL:
        fill_price = price or _get_current_price(symbol, side)
        if fill_price is None:
            logger.warning(f"[PAPER] No price for {symbol}, skipping simulated fill")
            return None

        # Build fake position
        pos = {
            "ticket": _next_ticket(),
            "symbol": symbol,
            "type": "BUY" if order_type == _mt5.ORDER_TYPE_BUY else "SELL",
            "volume": volume,
            "open_price": fill_price,
            "current_price": fill_price,
            "profit": 0.0,
            "sl": sl,
            "tp": tp,
            "comment": comment,
            "magic": magic,
            "open_time": time.time(),
        }

        state = _load_state()
        _ensure_paper_account(state)
        acc = state["trading"]["paper_account"]
        acc["positions"].append(pos)
        acc["open_positions"] = len(acc["positions"])
        _save_state(state)

        logger.info(f"[PAPER] Simulated open {side.upper()} {volume} {symbol} @ {fill_price}")

        class _FakeResult:
            retcode = _mt5.TRADE_RETCODE_DONE
            deal = pos["ticket"]
            order = pos["ticket"]
            volume = volume
            price = fill_price

        return _FakeResult()

    elif action == _mt5.TRADE_ACTION_SLTP:
        # Modify SL/TP on an existing paper position
        state = _load_state()
        _ensure_paper_account(state)
        acc = state["trading"]["paper_account"]
        for pos in acc["positions"]:
            if pos["symbol"] == symbol and pos.get("ticket") == request.get("position"):
                if sl is not None:
                    pos["sl"] = sl
                if tp is not None:
                    pos["tp"] = tp
                _save_state(state)
                class _FakeResult:
                    retcode = _mt5.TRADE_RETCODE_DONE
                return _FakeResult()
        logger.warning(f"[PAPER] Position not found for SLTP modify: {request}")
        return None

    return None


def paper_close_position(position_ticket: int, price: float | None = None, volume: float | None = None) -> dict | None:
    """Simulate closing a paper position."""
    state = _load_state()
    _ensure_paper_account(state)
    acc = state["trading"]["paper_account"]
    positions = acc["positions"]

    for i, pos in enumerate(positions):
        if pos["ticket"] == position_ticket:
            close_price = price or _get_current_price(pos["symbol"], "buy" if pos["type"] == "SELL" else "sell")
            if close_price is None:
                return None
            pnl = _calc_pnl(pos, close_price)
            acc["balance"] += pnl
            acc["equity"] = acc["balance"]
            acc["free_margin"] = acc["balance"]
            acc["profit"] += pnl
            acc["realized_today"] = acc.get("realized_today", 0.0) + pnl
            positions.pop(i)
            acc["open_positions"] = len(positions)
            _save_state(state)

            logger.info(f"[PAPER] Simulated close #{position_ticket} {pos['symbol']} @ {close_price} PnL={pnl:.2f}")

            # Persist closed trade for dashboard equity curve
            try:
                os.makedirs(_LOG_DIR, exist_ok=True)
                with open(PAPER_CLOSED_LOG, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "ticket": pos["ticket"],
                                "symbol": pos["symbol"],
                                "side": pos["type"],
                                "volume": pos["volume"],
                                "open_price": pos["open_price"],
                                "close_price": close_price,
                                "profit": round(pnl, 2),
                                "close_time": datetime.now(timezone.utc).isoformat(),
                                "bot_lane": pos.get("bot_lane", "paper"),
                                "comment": pos.get("comment", ""),
                            }
                        )
                        + "\n"
                    )
            except Exception as exc:
                logger.warning(f"Failed to log paper closed trade: {exc}")

            class _FakeResult:
                retcode = _mt5.TRADE_RETCODE_DONE
                deal = _next_ticket()
                order = _next_ticket()
                volume = volume or pos["volume"]
                price = close_price

            return _FakeResult()

    logger.warning(f"[PAPER] Position #{position_ticket} not found for close")
    return None


def paper_positions_get(symbol: str | None = None) -> list[dict]:
    """Return simulated paper positions, optionally filtered by symbol."""
    acc = get_paper_account()
    positions = acc.get("positions", [])
    # Update floating PnL for each position
    for pos in positions:
        price = _get_current_price(pos["symbol"], "buy" if pos["type"] == "SELL" else "sell")
        if price is not None:
            pos["current_price"] = price
            pos["profit"] = round(_calc_pnl(pos, price), 2)

    if symbol:
        positions = [p for p in positions if p["symbol"] == symbol]
    return positions


def paper_account_info() -> dict | None:
    """Return simulated account info for paper mode."""
    acc = get_paper_account()
    positions = acc.get("positions", [])
    floating = sum(p.get("profit", 0.0) for p in positions)
    equity = acc["balance"] + floating

    class _FakeAccountInfo:
        balance = acc["balance"]
        equity = equity
        margin_free = equity  # simplified
        profit = floating
        login = 999999
        server = "PAPER-CHAIN"
        name = "Paper Trader"
        currency = "USD"
        leverage = 100

    return _FakeAccountInfo()


def reset_paper_account(balance: float = PAPER_DEFAULT_BALANCE):
    state = _load_state()
    state.setdefault("trading", {})["paper_account"] = {
        "balance": balance,
        "equity": balance,
        "free_margin": balance,
        "profit": 0.0,
        "open_positions": 0,
        "positions": [],
        "realized_today": 0.0,
        "drawdown_pct": 0.0,
    }
    _save_state(state)
    logger.info(f"[PAPER] Account reset to ${balance:,.2f}")
