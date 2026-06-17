import csv
import datetime as dt
import json
import os
from collections import defaultdict

try:
    from Python.mt5_compat import mt5
except Exception:
    mt5 = None


def _parse_ts(value):
    try:
        return dt.datetime.fromisoformat(str(value))
    except Exception:
        return None


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _iter_jsonl(path):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _compute_core_metrics(rows):
    trades = len(rows)
    pnl_values = [r["pnl"] for r in rows]
    wins = [v for v in pnl_values if v > 0]
    losses = [v for v in pnl_values if v <= 0]
    total_pnl = sum(pnl_values)
    win_rate = (len(wins) / trades) * 100.0 if trades else 0.0
    avg_win = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss = (sum(losses) / len(losses)) if losses else 0.0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses and abs(sum(losses)) > 1e-12 else (999.0 if wins else 0.0)
    expectancy = total_pnl / trades if trades else 0.0
    hold_minutes = [r["hold_minutes"] for r in rows if r["hold_minutes"] is not None]
    avg_hold_minutes = (sum(hold_minutes) / len(hold_minutes)) if hold_minutes else 0.0
    max_loss_streak = 0
    recent_loss_streak = 0
    cur_loss = 0
    for v in pnl_values:
        if v <= 0:
            cur_loss += 1
            max_loss_streak = max(max_loss_streak, cur_loss)
        else:
            cur_loss = 0
    for v in reversed(pnl_values):
        if v <= 0:
            recent_loss_streak += 1
        else:
            break
    return {
        "trades": trades,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 2),
        "total_pnl": round(total_pnl, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 4),
        "expectancy": round(expectancy, 4),
        "avg_hold_minutes": round(avg_hold_minutes, 2),
        "max_loss_streak": int(max_loss_streak),
        "recent_loss_streak": int(recent_loss_streak),
    }


def build_trade_learning(log_dir, out_dir, lookback_days=30):
    trade_events_path = os.path.join(log_dir, "trade_events.jsonl")
    os.makedirs(out_dir, exist_ok=True)

    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=max(1, int(lookback_days)))
    opened_by_ticket = {}
    closed_rows = []

    for row in _iter_jsonl(trade_events_path):
        event = str(row.get("event", "")).strip()
        payload = row.get("payload", {}) or {}
        ts = _parse_ts(row.get("ts"))
        if ts is None or ts < cutoff:
            continue

        if event == "trade_open":
            ticket = int(payload.get("ticket", 0) or 0)
            if ticket <= 0:
                continue
            opened_by_ticket[ticket] = {
                "opened_at": ts,
                "symbol": str(payload.get("symbol", "?")),
                "side": str(payload.get("side", "UNKNOWN")).upper(),
                "open_price": _safe_float(payload.get("open_price")),
                "volume": _safe_float(payload.get("volume")),
            }
            continue

        if event != "trade_closed":
            continue

        ticket = int(payload.get("ticket", 0) or 0)
        symbol = str(payload.get("symbol", "?"))
        pnl = _safe_float(payload.get("profit", 0.0))
        volume = _safe_float(payload.get("volume", 0.0))
        close_price = _safe_float(payload.get("price", 0.0))
        side = "UNKNOWN"
        hold_minutes = None
        open_price = 0.0
        opened = opened_by_ticket.get(ticket)
        if opened:
            side = opened["side"]
            open_price = opened["open_price"]
            if opened["opened_at"] is not None:
                hold_minutes = (ts - opened["opened_at"]).total_seconds() / 60.0

        closed_rows.append(
            {
                "ts": ts,
                "ticket": ticket,
                "symbol": symbol,
                "side": side,
                "pnl": pnl,
                "volume": volume,
                "open_price": open_price,
                "close_price": close_price,
                "hold_minutes": hold_minutes,
                "hour_utc": int(ts.astimezone(dt.timezone.utc).hour),
            }
        )

    # Fallback to MT5 closed deals if JSONL close events are unavailable.
    if not closed_rows and mt5 is not None:
        try:
            if mt5.initialize():
                now_utc = dt.datetime.now(dt.timezone.utc)
                deals = mt5.history_deals_get(cutoff, now_utc) or []
                for d in deals:
                    try:
                        if int(getattr(d, "entry", -1)) != int(mt5.DEAL_ENTRY_OUT):
                            continue
                        pnl = _safe_float(
                            _safe_float(getattr(d, "profit", 0.0))
                            + _safe_float(getattr(d, "commission", 0.0))
                            + _safe_float(getattr(d, "swap", 0.0))
                        )
                        ts = dt.datetime.fromtimestamp(int(getattr(d, "time", 0)), tz=dt.timezone.utc)
                        side = "BUY" if int(getattr(d, "type", 1)) == int(mt5.ORDER_TYPE_SELL) else "SELL"
                        closed_rows.append(
                            {
                                "ts": ts,
                                "ticket": int(getattr(d, "position_id", 0) or 0),
                                "symbol": str(getattr(d, "symbol", "?")),
                                "side": side,
                                "pnl": pnl,
                                "volume": _safe_float(getattr(d, "volume", 0.0)),
                                "open_price": 0.0,
                                "close_price": _safe_float(getattr(d, "price", 0.0)),
                                "hold_minutes": None,
                                "hour_utc": int(ts.hour),
                            }
                        )
                    except Exception:
                        continue
        except Exception:
            pass

    overall = _compute_core_metrics(closed_rows)

    by_symbol = []
    grouped = defaultdict(list)
    by_hour = defaultdict(list)
    for r in closed_rows:
        grouped[r["symbol"]].append(r)
        by_hour[r["hour_utc"]].append(r)

    for symbol, rows in grouped.items():
        m = _compute_core_metrics(rows)
        by_symbol.append({"symbol": symbol, **m})
    by_symbol.sort(key=lambda x: x["total_pnl"], reverse=True)

    hour_rows = []
    for hour in sorted(by_hour.keys()):
        m = _compute_core_metrics(by_hour[hour])
        hour_rows.append({"hour_utc": hour, **m})

    best_symbols = [x for x in by_symbol if x["trades"] >= 3 and x["total_pnl"] > 0][:5]
    worst_symbols = [x for x in sorted(by_symbol, key=lambda x: x["total_pnl"]) if x["trades"] >= 3 and x["total_pnl"] < 0][:5]

    summary = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "lookback_days": int(lookback_days),
        **overall,
        "best_symbols": best_symbols,
        "worst_symbols": worst_symbols,
        "by_symbol": by_symbol,
        "by_hour_utc": hour_rows,
    }

    json_path = os.path.join(out_dir, "trade_learning_latest.json")
    csv_path = os.path.join(out_dir, "trade_learning_by_symbol.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "symbol",
                "trades",
                "wins",
                "losses",
                "win_rate",
                "total_pnl",
                "avg_win",
                "avg_loss",
                "profit_factor",
                "expectancy",
                "avg_hold_minutes",
                "max_loss_streak",
                "recent_loss_streak",
            ],
        )
        writer.writeheader()
        for row in by_symbol:
            writer.writerow(row)

    return summary


def load_trade_memory(out_dir: str, symbol: str | None = None) -> dict:
    """
    Returns normalized per-symbol trade-memory metrics for training-time feedback.
    Falls back to neutral values when no history exists.
    """
    neutral = {
        "trades": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 50.0,
        "expectancy": 0.0,
        "profit_factor": 1.0,
        "avg_loss": 0.0,
        "max_loss_streak": 0,
        "recent_loss_streak": 0,
    }
    path = os.path.join(out_dir, "trade_learning_latest.json")
    if not os.path.exists(path):
        return dict(neutral)
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f) or {}
    except Exception:
        return dict(neutral)

    rows = payload.get("by_symbol", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list) or not rows:
        return dict(neutral)
    if symbol:
        for row in rows:
            if str((row or {}).get("symbol", "")).upper() == str(symbol).upper():
                out = dict(neutral)
                out.update({k: row.get(k, out.get(k)) for k in out.keys()})
                return out

    # Fallback: weighted aggregate across symbols.
    trades = sum(int((r or {}).get("trades", 0) or 0) for r in rows)
    if trades <= 0:
        return dict(neutral)
    win_rate = sum(float((r or {}).get("win_rate", 0.0) or 0.0) * int((r or {}).get("trades", 0) or 0) for r in rows) / trades
    expectancy = sum(float((r or {}).get("expectancy", 0.0) or 0.0) * int((r or {}).get("trades", 0) or 0) for r in rows) / trades
    profit_factor = sum(float((r or {}).get("profit_factor", 1.0) or 1.0) * int((r or {}).get("trades", 0) or 0) for r in rows) / trades
    avg_loss = sum(float((r or {}).get("avg_loss", 0.0) or 0.0) * int((r or {}).get("trades", 0) or 0) for r in rows) / trades
    losses = sum(int((r or {}).get("losses", 0) or 0) for r in rows)
    wins = sum(int((r or {}).get("wins", 0) or 0) for r in rows)
    max_loss_streak = max(int((r or {}).get("max_loss_streak", 0) or 0) for r in rows)
    recent_loss_streak = max(int((r or {}).get("recent_loss_streak", 0) or 0) for r in rows)
    return {
        "trades": int(trades),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": float(win_rate),
        "expectancy": float(expectancy),
        "profit_factor": float(profit_factor),
        "avg_loss": float(avg_loss),
        "max_loss_streak": int(max_loss_streak),
        "recent_loss_streak": int(recent_loss_streak),
    }
