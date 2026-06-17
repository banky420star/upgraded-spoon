"""
Trade Review — Post-trade analysis, annotation, and feedback loop.

Compiles executed trades with their decision context, analyzes outcomes,
creates human-readable annotations, and feeds results back into training.

Pipeline:
  1. Gather trade executions from MT5 + decision logs
  2. Match trades with their decision rationale (from decisions.jsonl)
  3. Analyze outcomes: P/L, SL/TP effectiveness, signal quality
  4. Annotate each trade with reason tags
  5. Store enriched trade records for retraining feedback
  6. Report summary metrics
"""
import os
import json
import time
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from loguru import logger

try:
    from Python.mt5_compat import mt5
except ImportError:
    mt5 = None

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(_BASE, "logs")
REVIEW_DIR = os.path.join(_BASE, "logs", "trade_reviews")
os.makedirs(REVIEW_DIR, exist_ok=True)


# ── Reason Tags ──────────────────────────────────────────────────────
# Tags that explain WHY a trade won or lost
TAG_SL_TOO_TIGHT = "sl_too_tight"
TAG_TP_HIT = "tp_hit"
TAG_SIGNAL_CORRECT = "signal_correct"
TAG_SIGNAL_WRONG = "signal_wrong"
TAG_LOW_CONFIDENCE = "low_confidence"
TAG_HIGH_VOLATILITY = "high_volatility_regime"
TAG_BUY_BIAS = "buy_bias"
TAG_REVERSAL = "market_reversal"
TAG_SPREAD_WIDENED = "spread_widened"
TAG_NEWS_EVENT = "news_event_impact"
TAG_UNKNOWN = "unknown"

REASON_MAP = {
    4: "sl_hit",
    5: "tp_hit",
    6: "margin_call",
    7: "closed_by_dealer",
    8: "partial_close",
}


def _ts_to_utc(ts) -> datetime:
    """Convert MT5 timestamp (int or datetime) to UTC datetime."""
    if isinstance(ts, datetime):
        return ts
    return datetime.fromtimestamp(int(ts), tz=timezone.utc)


def gather_closed_trades(days_back: int = 7) -> list[dict]:
    """Fetch closed trade deals from MT5 with full context."""
    if mt5 is None:
        logger.warning("MT5 not available — cannot gather trades")
        return []

    if not mt5.initialize():
        logger.error("MT5 init failed")
        return []

    try:
        since = datetime.now(timezone.utc) - timedelta(days=days_back)
        deals = mt5.history_deals_get(since, datetime.now(timezone.utc))
        if not deals:
            return []

        # Also get orders to find open times for hold duration
        orders = mt5.history_orders_get(since, datetime.now(timezone.utc))
        order_open_times = {}
        if orders:
            for o in orders:
                if o.type in (0, 1):  # BUY or SELL market order
                    order_open_times[o.ticket] = _ts_to_utc(o.time_setup)

        trades = []
        for d in deals:
            if d.entry != 1:  # Only closing deals
                continue
            if d.type == 2:  # Skip balance operations
                continue

            close_reason = REASON_MAP.get(d.reason, f"reason_{d.reason}")
            is_sl = d.reason == 4
            is_tp = d.reason == 5

            # Compute hold duration
            open_time = order_open_times.get(d.order)
            hold_minutes = None
            if open_time:
                close_dt = _ts_to_utc(d.time)
                hold_minutes = int((close_dt - open_time).total_seconds() / 60)

            # Extract model version from comment (format: {SYM6}{OP/CL}{CH/CA}{VERSION6})
            model_version = ""
            comment = d.comment or ""
            if len(comment) >= 12:
                model_version = comment[6:]  # Last part is version

            trades.append({
                "ticket": d.ticket,
                "order": d.order,
                "symbol": d.symbol,
                "side": "BUY" if d.type == 0 else "SELL",
                "volume": d.volume,
                "price": d.price,
                "profit": round(d.profit, 2),
                "commission": round(d.commission, 2),
                "swap": round(d.swap, 6),
                "close_time": _ts_to_utc(d.time).isoformat(),
                "open_time": open_time.isoformat() if open_time else "",
                "hold_minutes": hold_minutes,
                "close_reason": close_reason,
                "is_sl": is_sl,
                "is_tp": is_tp,
                "comment": comment,
                "model_version": model_version,
            })
        return trades
    except Exception as e:
        logger.error(f"Error gathering closed trades: {e}")
        return []


def load_decision_log(hours_back: int = 168) -> list[dict]:
    """Load recent decisions from the JSONL log.

    Default 168 hours (7 days) to match the trade review window.
    """
    path = os.path.join(LOG_DIR, "decisions.jsonl")
    if not os.path.exists(path):
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)
    decisions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line.strip())
                ts = d.get("timestamp", "")
                if ts:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if dt >= cutoff:
                        decisions.append(d)
            except (json.JSONDecodeError, ValueError):
                continue
    return decisions


def match_trade_to_decision(trade: dict, decisions: list[dict]) -> dict | None:
    """Find the decision that led to a trade by matching symbol, direction, and time proximity.

    Matches on: (a) symbol, (b) side/direction alignment (BUY trade matches BUY
    decision, SELL matches SELL), (c) closest prior timestamp within 24 hours
    of the trade's open time.  Filters out mismatched directions.  Returns the
    matched decision record or None.
    """
    symbol = trade["symbol"]
    trade_side = trade.get("side", "")  # "BUY" or "SELL"

    # Prefer open_time (when the position was actually entered) over close_time.
    # Fall back to close_time if open_time is missing.
    time_str = trade.get("open_time") or trade.get("close_time", "")
    if not time_str:
        return None
    try:
        ref_dt = datetime.fromisoformat(time_str)
    except (ValueError, TypeError):
        return None

    best = None
    best_dt_diff = float("inf")

    for d in decisions:
        if d.get("symbol") != symbol:
            continue

        # Direction alignment: BUY trade matches BUY decision, SELL matches SELL
        decision_action = d.get("action", "")
        if trade_side and decision_action and decision_action != trade_side:
            continue

        ts = d.get("timestamp", "")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            # Decision must be BEFORE the trade opened, and within 24 hours
            diff = (ref_dt - dt).total_seconds()
            if 0 < diff < 86400 and diff < best_dt_diff:
                best = d
                best_dt_diff = diff
        except (ValueError, TypeError):
            continue

    return best


# ── Config helper ─────────────────────────────────────────────────────

def _load_max_spread_bps() -> float:
    """Load max_spread_bps from project config.yaml (risk section)."""
    try:
        import yaml
        cfg_path = os.path.join(_BASE, "config.yaml")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            return float(cfg.get("risk", {}).get("max_spread_bps", 50))
    except Exception:
        pass
    return 50.0


# ── Outcome Tagging ───────────────────────────────────────────────────

def assign_outcome_tags(trade: dict, decision: dict = None) -> list[str]:
    """Analyze a closed trade and return a list of outcome tags.

    Tags describe *why* a trade turned out the way it did, using both the
    trade result and the original decision context.  This is a focused,
    complementary set to the existing annotate_trade() tags.
    """
    tags = []
    profit = trade.get("profit", 0)
    trade_side = trade.get("side", "")

    # --- signal_correct / signal_wrong ---
    if decision is not None:
        decision_action = decision.get("action", "")
        direction_matched = (
            trade_side == decision_action
            if (trade_side and decision_action)
            else False
        )
        if profit > 0 and direction_matched:
            tags.append("signal_correct")
        if profit < 0 and not direction_matched and decision_action not in ("HOLD", ""):
            tags.append("signal_wrong")

    # --- market_reversal ---
    # Heuristic: trade hit SL but had been in profit at some point.
    # MT5 deals don't carry intraday high/low, so we approximate by
    # checking if SL was hit while the trade ended with a net profit
    # (trailing SL locked gains) or if the close reason is SL with a
    # very small loss relative to hold time (price went up then reversed).
    if trade.get("is_sl"):
        if profit > 0:
            # Trailing SL locked in gains — price went favorably then reversed
            tags.append("market_reversal")
        elif profit < 0:
            hold_min = trade.get("hold_minutes")
            # If the trade was held for a reasonable time before SL, likely a reversal
            if hold_min is not None and hold_min > 30:
                tags.append("market_reversal")

    # --- sl_too_tight ---
    hold_min = trade.get("hold_minutes")
    if trade.get("is_sl") and hold_min is not None and hold_min <= 5:
        tags.append("sl_too_tight")

    # --- tp_hit ---
    if trade.get("is_tp"):
        tags.append("tp_hit")

    # --- low_confidence ---
    if decision is not None:
        confidence = decision.get("confidence", 0)
        if confidence < 0.65:
            tags.append("low_confidence")

    # --- high_volatility_regime ---
    if decision is not None:
        regime = decision.get("lstm_regime", "") or decision.get("regime", "")
        if regime == "HIGH_VOLATILITY":
            tags.append("high_volatility_regime")

    # --- spread_too_wide ---
    entry_spread = trade.get("entry_spread_bps")
    if entry_spread is not None:
        try:
            if float(entry_spread) > _load_max_spread_bps():
                tags.append("spread_too_wide")
        except (ValueError, TypeError):
            pass

    # --- entered_late ---
    if decision is not None:
        dec_ts = decision.get("timestamp", "")
        entry_ts = trade.get("open_time", "")
        if dec_ts and entry_ts:
            try:
                dec_dt = datetime.fromisoformat(dec_ts.replace("Z", "+00:00"))
                entry_dt = datetime.fromisoformat(entry_ts)
                latency = (entry_dt - dec_dt).total_seconds()
                if latency > 60:
                    tags.append("entered_late")
            except (ValueError, TypeError):
                pass

    # --- overtraded ---
    # Requires external context (list of all trades); set by enrich_reviews().
    if trade.get("_overtraded"):
        tags.append("overtraded")

    return tags


def enrich_reviews(trades: list, decisions: list) -> list[dict]:
    """Match each trade to its decision, assign outcome tags, return enriched records.

    Each record contains the original trade dict plus:
      - "decision_context": matched decision or None
      - "outcome_tags": list of outcome tag strings from assign_outcome_tags
    """
    # Pre-compute per-symbol trade counts in 1-hour windows to detect overtrading.
    # For each trade, count how many other trades for the same symbol fell within
    # the prior hour of that trade's close time.
    trade_close_times: dict[str, list[datetime]] = {}
    for t in trades:
        sym = t.get("symbol", "")
        ts_str = t.get("close_time", "")
        if not ts_str:
            continue
        try:
            dt = datetime.fromisoformat(ts_str)
            trade_close_times.setdefault(sym, []).append(dt)
        except (ValueError, TypeError):
            continue

    enriched = []
    for trade in trades:
        decision = match_trade_to_decision(trade, decisions)

        # Detect overtrading: >5 trades for the same symbol in the hour before close
        sym = trade.get("symbol", "")
        ts_str = trade.get("close_time", "")
        is_overtraded = False
        if sym in trade_close_times and ts_str:
            try:
                close_dt = datetime.fromisoformat(ts_str)
                one_hour_ago = close_dt - timedelta(hours=1)
                count_in_hour = sum(
                    1 for dt in trade_close_times[sym]
                    if one_hour_ago <= dt <= close_dt
                )
                if count_in_hour > 5:
                    is_overtraded = True
            except (ValueError, TypeError):
                pass
        trade_copy = dict(trade)
        trade_copy["_overtraded"] = is_overtraded

        outcome_tags = assign_outcome_tags(trade_copy, decision)

        # Remove the internal flag before storing
        trade_copy.pop("_overtraded", None)

        record = {
            **trade_copy,
            "decision_context": decision,
            "outcome_tags": outcome_tags,
        }
        enriched.append(record)

    return enriched


def annotate_trade(trade: dict, decision: dict | None) -> list[str]:
    """Generate reason tags for a trade based on outcome and context."""
    tags = []

    if trade["is_sl"]:
        # Differentiate SL at loss vs SL in profit
        profit = trade.get("profit", 0)
        commission = trade.get("commission", 0)
        swap = trade.get("swap", 0)
        gross = profit - commission - swap
        hold_min = trade.get("hold_minutes")
        sl_distance = trade.get("sl_distance", 0)  # Set by executor: entry - SL
        if profit > 0:
            # SL hit but trade was in profit — trailing stop locked gains
            tags.append("sl_in_profit")
            # Tag very short holds as quick trailing stop capture
            if hold_min is not None and hold_min < 30:
                tags.append("quick_trail_capture")
        elif abs(gross) < 0.5 and profit < 0:
            # Gross profit near zero, net negative — spread killed it
            tags.append(TAG_SPREAD_WIDENED)
        elif hold_min is not None and hold_min < 15:
            # SL hit very quickly — likely a bad entry signal
            tags.append(TAG_SIGNAL_WRONG)
        elif sl_distance > 0 and sl_distance < 100:
            # SL distance was very small (for crypto < $100, for FX < 0.001)
            # This indicates the SL was placed too close to entry
            tags.append(TAG_SL_TOO_TIGHT)
        elif profit < 0:
            # SL hit at loss with reasonable distance — market moved against us
            tags.append(TAG_SIGNAL_WRONG)
        else:
            tags.append(TAG_SL_TOO_TIGHT)
    elif trade["is_tp"]:
        tags.append(TAG_TP_HIT)
        tags.append(TAG_SIGNAL_CORRECT)
    else:
        # Not SL or TP — closed by signal reversal or reconciliation
        if trade["profit"] > 0:
            tags.append(TAG_SIGNAL_CORRECT)
            tags.append("signal_reversal_close")

    if decision:
        confidence = decision.get("confidence", 0)
        regime = decision.get("lstm_regime", "UNKNOWN")
        action = decision.get("action", "HOLD")
        ppo_action = decision.get("ppo_primary_action", 0)
        ppo_bias = decision.get("ppo_bias", 0)
        reason = decision.get("reason", "")

        if confidence < 0.5:
            tags.append(TAG_LOW_CONFIDENCE)

        if regime == "HIGH_VOLATILITY":
            tags.append(TAG_HIGH_VOLATILITY)

        # Detect BUY bias: if PPO is always positive
        if abs(ppo_bias) > 0.001 and action == "BUY":
            tags.append(TAG_BUY_BIAS)

        # If trade lost and signal was BUY but PPO output was tiny
        if trade["profit"] < 0 and abs(ppo_action) < 0.01:
            tags.append(TAG_SIGNAL_WRONG)

        # Tag if trade executed during HIGH_VOL regime (gate was in play)
        # A trade that passed the gate legitimately has ppo_action >= threshold
        # A bypass means the trade happened in HIGH_VOL but shouldn't have
        _high_vol_min = float(os.environ.get("AGI_HIGH_VOL_MIN_ACTION", "0.01"))
        if regime == "HIGH_VOLATILITY" and action != "HOLD":
            if abs(ppo_action) < _high_vol_min:
                # PPO action was below gate threshold but trade still executed — true bypass
                tags.append("bypassed_high_vol_gate")
            else:
                # Trade legitimately passed the gate in HIGH_VOL regime
                tags.append("high_vol_entry_passed_gate")

    # If no decision matched, flag as unknown
    if decision is None:
        tags.append(TAG_UNKNOWN)

    # If trade lost but wasn't SL (manual close or other)
    if trade["profit"] < 0 and not trade["is_sl"] and not trade["is_tp"]:
        tags.append(TAG_REVERSAL)

    # Tag long holds
    hold_min = trade.get("hold_minutes")
    if hold_min is not None and hold_min > 240:
        tags.append("long_hold")

    return tags


def analyze_trades(trades: list[dict], decisions: list[dict]) -> dict:
    """Full trade analysis with annotations. Returns enriched trades + summary."""
    enriched = []
    by_symbol = defaultdict(list)
    tag_counts = defaultdict(int)

    for trade in trades:
        decision = match_trade_to_decision(trade, decisions)
        tags = annotate_trade(trade, decision)

        record = {
            **trade,
            "decision_context": decision,
            "tags": tags,
            "tags_str": ", ".join(tags),
        }
        enriched.append(record)
        by_symbol[trade["symbol"]].append(record)
        for t in tags:
            tag_counts[t] += 1

    # Compute summary
    total = len(trades)
    if total == 0:
        return {"enriched": [], "summary": {}}

    wins = [t for t in enriched if t["profit"] > 0]
    losses = [t for t in enriched if t["profit"] <= 0]
    sl_trades = [t for t in enriched if t["is_sl"]]
    tp_trades = [t for t in enriched if t["is_tp"]]

    summary = {
        "total_trades": total,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / total * 100, 1),
        "total_pnl": round(sum(t["profit"] for t in trades), 2),
        "avg_win": round(sum(t["profit"] for t in wins) / max(len(wins), 1), 2),
        "avg_loss": round(sum(t["profit"] for t in losses) / max(len(losses), 1), 2),
        "profit_factor": round(
            abs(sum(t["profit"] for t in wins)) / max(abs(sum(t["profit"] for t in losses)), 0.01), 2
        ),
        "sl_hits": len(sl_trades),
        "tp_hits": len(tp_trades),
        "sl_rate": round(len(sl_trades) / total * 100, 1) if total else 0,
        "tp_rate": round(len(tp_trades) / total * 100, 1) if total else 0,
        "tag_distribution": dict(tag_counts),
        "by_symbol": {},
    }

    for sym, sym_trades in by_symbol.items():
        sym_wins = [t for t in sym_trades if t["profit"] > 0]
        sym_total = len(sym_trades)
        summary["by_symbol"][sym] = {
            "trades": sym_total,
            "wins": len(sym_wins),
            "win_rate": round(len(sym_wins) / max(sym_total, 1) * 100, 1),
            "pnl": round(sum(t["profit"] for t in sym_trades), 2),
            "sl_hits": len([t for t in sym_trades if t["is_sl"]]),
            "tp_hits": len([t for t in sym_trades if t["is_tp"]]),
        }

    return {"enriched": enriched, "summary": summary}


def save_review(result: dict) -> str:
    """Save trade review to a timestamped JSON file."""
    now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(REVIEW_DIR, f"review_{now}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Trade review saved: {path}")
    return path


def run_review(days_back: int = 7) -> dict:
    """Run a full trade review cycle."""
    logger.info(f"Starting trade review (last {days_back} days)...")

    trades = gather_closed_trades(days_back=days_back)
    decisions = load_decision_log(hours_back=days_back * 24)

    logger.info(f"Gathered {len(trades)} closed trades, {len(decisions)} decisions")

    result = analyze_trades(trades, decisions)
    path = save_review(result)

    summary = result["summary"]
    logger.info(
        f"Review complete: {summary.get('total_trades', 0)} trades | "
        f"Win rate: {summary.get('win_rate', 0)}% | "
        f"PnL: ${summary.get('total_pnl', 0):.2f} | "
        f"PF: {summary.get('profit_factor', 0)}"
    )
    logger.info(f"Tag distribution: {summary.get('tag_distribution', {})}")

    # ── Update scenario memory with trade outcomes ──
    try:
        from Python.scenario_memory import get_scenario_memory
        smem = get_scenario_memory()
        enriched = result.get("enriched", [])
        matched = 0
        for trade in enriched:
            profit = float(trade.get("profit", 0) or 0)
            # Try to match trade to a scenario record by symbol + time
            symbol = trade.get("symbol", "")
            side = trade.get("type", "").lower()
            hold_minutes = float(trade.get("hold_minutes", 0) or 0)
            close_reason = "SL" if trade.get("is_sl") else "TP" if trade.get("is_tp") else "manual"

            # Find matching open scenario record
            for dec_id, record in list(smem.records.items()):
                if (record.symbol == symbol and
                    record.outcome == "open" and
                    record.action.lower() == side):
                    smem.record_outcome(
                        decision_id=dec_id,
                        exit_price=float(trade.get("price_close", 0) or trade.get("price", 0) or 0),
                        pnl=profit,
                        pnl_pct=profit / max(float(trade.get("price_open", 1) or 1), 0.01) * 100,
                        hold_minutes=hold_minutes,
                        close_reason=close_reason,
                        max_drawup=float(trade.get("max_drawup", 0) or 0),
                        max_drawdown=float(trade.get("max_drawdown", 0) or abs(profit) * 0.5),
                    )
                    matched += 1
                    break
        if matched > 0:
            logger.info(f"ScenarioMemory: matched {matched} trade outcomes to entry scenarios")
            smem.save_stats()
    except Exception as e:
        logger.debug(f"Scenario memory update skipped: {e}")

    # ── Ollama advisor: auto-analyze losing trades ──
    try:
        from Python.ollama_advisor import get_advisor
        import yaml
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        ollama_cfg = cfg.get("ollama", {})
        if ollama_cfg.get("enabled", True) and ollama_cfg.get("auto_review", {}).get("losing_trades", True):
            advisor = get_advisor(cfg)
            if advisor._available:
                # Analyze top 3 worst losses
                losses = [t for t in result.get("enriched", []) if t.get("profit", 0) < 0]
                losses.sort(key=lambda t: t.get("profit", 0))
                for loss in losses[:3]:
                    analysis = advisor.analyze_trade({
                        "symbol": loss.get("symbol", "?"),
                        "side": loss.get("type", "?"),
                        "entry_price": loss.get("price_open", 0),
                        "exit_price": loss.get("price_close", 0),
                        "pnl": loss.get("profit", 0),
                        "sl_hit": loss.get("is_sl", False),
                        "tp_hit": loss.get("is_tp", False),
                        "duration_minutes": loss.get("duration_minutes", 0),
                        "atr_at_entry": loss.get("atr_at_entry", "?"),
                        "volatility_regime": loss.get("volatility_regime", "?"),
                        "confidence": loss.get("confidence", "?"),
                        "model_version": loss.get("model_version", "?"),
                    })
                    if analysis:
                        logger.info(f"Ollama analysis for {loss.get('symbol', '?')} loss: {analysis[:200]}...")
    except Exception as e:
        logger.debug(f"Ollama advisor review skipped: {e}")

    return result


def get_latest_review() -> dict | None:
    """Load the most recent trade review."""
    reviews = sorted(
        [f for f in os.listdir(REVIEW_DIR) if f.startswith("review_") and f.endswith(".json")],
        reverse=True,
    )
    if not reviews:
        return None
    path = os.path.join(REVIEW_DIR, reviews[0])
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_economic_calendar(days_ahead: int = 7) -> list[dict]:
    """
    Fetch upcoming economic calendar events from MT5.

    Tries multiple MT5 calendar API variants since availability varies
    by build. Falls back to a static high-impact event list if the
    MT5 calendar API is not available.
    """
    if mt5 is None:
        logger.debug("MT5 not available — cannot fetch economic calendar")
        return _fallback_calendar()

    if not mt5.initialize():
        logger.warning("MT5 init failed — cannot fetch economic calendar")
        return _fallback_calendar()

    try:
        events = []
        now = datetime.now(timezone.utc)
        to_date = now + timedelta(days=days_ahead)

        # Try method 1: calendar_country() → calendar_value_last_by_country()
        if hasattr(mt5, "calendar_country"):
            try:
                countries = mt5.calendar_country()
                if countries:
                    for country in countries:
                        country_code = getattr(country, "code", "")
                        country_name = getattr(country, "name", country_code)
                        currency = getattr(country, "currency", country_code)
                        try:
                            if hasattr(mt5, "calendar_value_last_by_country"):
                                country_events = mt5.calendar_value_last_by_country(country_code, now, to_date)
                            elif hasattr(mt5, "calendar_value_last"):
                                country_events = mt5.calendar_value_last(country_code, now, to_date)
                            else:
                                continue
                        except (AttributeError, TypeError):
                            continue
                        if not country_events:
                            continue
                        for ev in country_events:
                            ev_time_raw = getattr(ev, "time", None)
                            if ev_time_raw is None:
                                continue
                            try:
                                ev_dt = ev_time_raw if isinstance(ev_time_raw, datetime) else datetime.fromtimestamp(int(ev_time_raw), tz=timezone.utc)
                            except (ValueError, TypeError, OSError):
                                ev_dt = None
                            importance = getattr(ev, "importance", 0)
                            importance_label = {0: "low", 1: "medium", 2: "high"}.get(importance, "unknown")
                            events.append({
                                "country": country_code, "country_name": country_name,
                                "currency": currency, "name": getattr(ev, "name", ""),
                                "event_id": getattr(ev, "event_id", ""),
                                "time": ev_dt.isoformat() if ev_dt else str(ev_time_raw),
                                "importance": importance, "importance_label": importance_label,
                            })
            except Exception as e:
                logger.debug(f"calendar_country method failed: {e}")

        # Try method 2: calendar_value_last() directly (some builds support this without country)
        if not events and hasattr(mt5, "calendar_value_last"):
            try:
                all_events = mt5.calendar_value_last(now, to_date)
                if all_events:
                    for ev in all_events:
                        ev_time_raw = getattr(ev, "time", None)
                        if ev_time_raw is None:
                            continue
                        try:
                            ev_dt = ev_time_raw if isinstance(ev_time_raw, datetime) else datetime.fromtimestamp(int(ev_time_raw), tz=timezone.utc)
                        except (ValueError, TypeError, OSError):
                            ev_dt = None
                        importance = getattr(ev, "importance", 0)
                        importance_label = {0: "low", 1: "medium", 2: "high"}.get(importance, "unknown")
                        events.append({
                            "country": getattr(ev, "country_code", ""), "country_name": "",
                            "currency": "", "name": getattr(ev, "name", ""),
                            "event_id": getattr(ev, "event_id", ""),
                            "time": ev_dt.isoformat() if ev_dt else str(ev_time_raw),
                            "importance": importance, "importance_label": importance_label,
                        })
            except Exception as e:
                logger.debug(f"calendar_value_last method failed: {e}")

        if events:
            events.sort(key=lambda e: e["time"])
            return events

        # No MT5 calendar API available — use fallback
        logger.info("MT5 calendar API not available in this build — using fallback events")
        return _fallback_calendar()

    finally:
        mt5.shutdown()


def _fallback_calendar() -> list[dict]:
    """Return a static list of known high-impact weekly events for major currencies.

    This is used when the MT5 calendar API is not available. It provides
    a weekly schedule of typical high-impact events that traders should watch.
    Updated weekly manually or from a config file.
    """
    now = datetime.now(timezone.utc)
    events = []

    # Typical weekly high-impact events (day-of-week based)
    weekly_events = [
        {"name": "FOMC Meeting Minutes", "currency": "USD", "importance": 2,
         "day": 2, "hour": 18},  # Wednesday 18:00 UTC
        {"name": "US Initial Jobless Claims", "currency": "USD", "importance": 2,
         "day": 3, "hour": 12},  # Thursday 12:30 UTC
        {"name": "US Non-Farm Payrolls", "currency": "USD", "importance": 2,
         "day": 4, "hour": 12},  # Friday 12:30 UTC (1st Friday)
        {"name": "ECB Rate Decision", "currency": "EUR", "importance": 2,
         "day": 3, "hour": 12},  # Thursday
        {"name": "BOE Rate Decision", "currency": "GBP", "importance": 2,
         "day": 3, "hour": 11},  # Thursday
        {"name": "US CPI", "currency": "USD", "importance": 2,
         "day": 1, "hour": 12},  # Tuesday
        {"name": "China GDP / Trade Data", "currency": "CNY", "importance": 1,
         "day": 0, "hour": 2},   # Monday early hours
        {"name": "Gold Physical Demand Update", "currency": "XAU", "importance": 1,
         "day": 0, "hour": 6},
    ]

    for i in range(7):
        day = now + timedelta(days=i)
        for ev in weekly_events:
            if day.weekday() == ev["day"]:
                event_time = day.replace(hour=ev["hour"], minute=0, second=0, microsecond=0)
                events.append({
                    "country": "", "country_name": "",
                    "currency": ev["currency"],
                    "name": ev["name"],
                    "event_id": f"fallback_{ev['name'].replace(' ', '_').lower()}",
                    "time": event_time.isoformat(),
                    "importance": ev["importance"],
                    "importance_label": {0: "low", 1: "medium", 2: "high"}.get(ev["importance"], "unknown"),
                })

    events.sort(key=lambda e: e["time"])
    return events


if __name__ == "__main__":
    result = run_review()
    print(json.dumps(result["summary"], indent=2, default=str))