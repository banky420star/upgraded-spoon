"""
Scenario Memory — Trade scenario fingerprinting, tracking, and feedback loop.

This module gives the trading bot a cumulative memory of which market setups
lead to wins and losses. It:

1. Fingerprints market conditions at trade entry into scenario labels
2. Records full context: decision, entry price, SL/TP, regime, sentiment
3. Matches closed trades back to their entry scenarios
4. Computes per-scenario win rates, PnL, drawdown statistics
5. Persists to logs/scenario_memory.jsonl
6. Loads history on startup for cumulative statistics
7. Provides a feedback signal: should_trade(scenario) returns a confidence
   modifier that the decision pipeline can use to boost or suppress trades
"""
from __future__ import annotations

import json
import os
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
SCENARIO_FILE = LOG_DIR / "scenario_memory.jsonl"

# ── Regime buckets ──────────────────────────────────────────────────────────
VOL_BUCKETS = ["LOW_VOLATILITY", "MED_VOLATILITY", "HIGH_VOLATILITY"]
TREND_BUCKETS = ["bullish_align", "bearish_align", "bullish_contrarian", "bearish_flip", "flat"]
SESSION_BUCKETS = ["asian", "london", "ny_overlap", "ny", "off_hours"]
CONFIDENCE_BUCKETS = ["high", "medium", "low"]
SENTIMENT_BUCKETS = ["fear", "neutral", "greed"]


def _bucket_confidence(conf: float) -> str:
    if conf >= 0.8:
        return "high"
    if conf >= 0.5:
        return "medium"
    return "low"


def _bucket_sentiment(fgi: int) -> str:
    if fgi < 30:
        return "fear"
    if fgi > 70:
        return "greed"
    return "neutral"


def _bucket_session(hour_utc: int) -> str:
    if 0 <= hour_utc < 7:
        return "asian"
    if 7 <= hour_utc < 13:
        return "london"
    if 13 <= hour_utc < 17:
        return "ny_overlap"
    if 17 <= hour_utc < 21:
        return "ny"
    return "off_hours"


def fingerprint_scenario(
    symbol: str,
    regime: str,
    confidence: float,
    action: str,
    trend_flip: str = "flat",
    sentiment: float = 0.0,
    fgi: int = 50,
    vol_scale: float = 1.0,
    hour_utc: int | None = None,
) -> str:
    """Create a human-readable scenario fingerprint from trade context.

    Example: "EURUSDm_bearish_flip_HIGH_VOL_low_conf_fear_london_SELL"
    """
    trend = trend_flip if trend_flip and trend_flip != "flat" else "ranging"
    conf = _bucket_confidence(confidence)
    sent = _bucket_sentiment(fgi) if fgi else _bucket_sentiment(50)
    session = _bucket_session(hour_utc) if hour_utc is not None else "unknown"
    vol = regime if regime in VOL_BUCKETS else "UNKNOWN_VOL"

    return f"{symbol}_{trend}_{vol}_{conf}_conf_{sent}_{session}_{action}"


@dataclass
class ScenarioRecord:
    """A single scenario observation: entry context linked to outcome."""
    scenario: str                          # fingerprint label
    decision_id: str                       # unique ID linking decision -> execution -> outcome
    symbol: str
    action: str                            # BUY / SELL / HOLD
    regime: str
    confidence: float
    trend_flip: str
    sentiment: float
    fgi: int
    vol_scale: float
    session: str

    # Entry context
    entry_price: float = 0.0
    entry_time: str = ""
    sl: float = 0.0
    tp: float = 0.0
    lot_size: float = 0.0
    atr_at_entry: float = 0.0
    spread_at_entry: float = 0.0

    # Outcome (filled when trade closes)
    exit_price: float = 0.0
    exit_time: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0
    hold_minutes: float = 0.0
    outcome: str = ""                      # win / loss / breakeven / open / rejected
    close_reason: str = ""                 # SL / TP / manual / timeout
    max_drawup: float = 0.0               # max favorable excursion (pips)
    max_drawdown: float = 0.0             # max adverse excursion (pips)

    # Full decision context (for pattern analysis)
    ppo_raw_action: float = 0.0
    ppo_corrected: float = 0.0
    bias: float = 0.0
    exposure: float = 0.0

    # Risk context
    risk_can_trade: bool = True
    risk_dd_pct: float = 0.0
    equity_at_entry: float = 0.0


@dataclass
class ScenarioStats:
    """Aggregate statistics for a scenario fingerprint."""
    scenario: str
    count: int = 0
    wins: int = 0
    losses: int = 0
    breakevens: int = 0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    total_pnl: float = 0.0
    avg_hold_minutes: float = 0.0
    avg_max_drawup: float = 0.0
    avg_max_drawdown: float = 0.0
    best_pnl: float = 0.0
    worst_pnl: float = 0.0
    last_seen: str = ""

    def update(self, record: ScenarioRecord):
        """Update stats with a new completed record."""
        if record.outcome == "open" or record.outcome == "rejected":
            return
        self.count += 1
        self.total_pnl += record.pnl
        self.last_seen = record.exit_time or record.entry_time
        if record.outcome == "win":
            self.wins += 1
        elif record.outcome == "loss":
            self.losses += 1
        else:
            self.breakevens += 1
        self.win_rate = self.wins / max(self.count, 1)
        self.avg_pnl = self.total_pnl / max(self.count, 1)
        self.avg_hold_minutes = (
            (self.avg_hold_minutes * (self.count - 1) + record.hold_minutes)
            / max(self.count, 1)
        )
        self.avg_max_drawup = (
            (self.avg_max_drawup * (self.count - 1) + record.max_drawup)
            / max(self.count, 1)
        )
        self.avg_max_drawdown = (
            (self.avg_max_drawdown * (self.count - 1) + abs(record.max_drawdown))
            / max(self.count, 1)
        )
        self.best_pnl = max(self.best_pnl, record.pnl)
        self.worst_pnl = min(self.worst_pnl, record.pnl)


class ScenarioMemory:
    """Cumulative scenario memory that persists across bot restarts.

    Tracks trade scenarios, their outcomes, and provides feedback signals
    for the decision pipeline.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = path or SCENARIO_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # In-memory stores
        self.records: Dict[str, ScenarioRecord] = {}       # decision_id -> record
        self.stats: Dict[str, ScenarioStats] = {}           # scenario fingerprint -> stats
        self._recent: deque = deque(maxlen=500)             # last 500 records

        # Load history
        self._load()

    # ── Recording ────────────────────────────────────────────────────────

    def record_entry(
        self,
        decision: Dict[str, Any],
        entry_price: float = 0.0,
        sl: float = 0.0,
        tp: float = 0.0,
        lot_size: float = 0.0,
        atr: float = 0.0,
        spread: float = 0.0,
        equity: float = 0.0,
    ) -> str:
        """Record a trade entry. Returns decision_id for later outcome linking."""
        decision_id = decision.get("decision_id") or str(uuid.uuid4())

        symbol = decision.get("symbol", "UNKNOWN")
        action = decision.get("action", "HOLD")
        regime = decision.get("regime") or decision.get("volatility") or "UNKNOWN"
        confidence = decision.get("confidence", 0.0)
        trend_flip = decision.get("trend_flip", "flat")
        sentiment = decision.get("sentiment", 0.0)
        fgi = decision.get("fgi", 50)
        vol_scale = decision.get("vol_scale", 1.0)

        hour_utc = None
        entry_time = decision.get("timestamp", datetime.now(timezone.utc).isoformat())
        try:
            dt = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
            hour_utc = dt.hour
        except (ValueError, TypeError) as e:
            logger.debug(f"Failed to parse entry_time '{entry_time}': {e}")
            hour_utc = 12  # Default to noon UTC

        scenario = fingerprint_scenario(
            symbol=symbol,
            regime=regime,
            confidence=confidence,
            action=action,
            trend_flip=trend_flip,
            sentiment=sentiment,
            fgi=int(fgi) if fgi else 50,
            vol_scale=vol_scale,
            hour_utc=hour_utc,
        )

        record = ScenarioRecord(
            scenario=scenario,
            decision_id=decision_id,
            symbol=symbol,
            action=action,
            regime=regime,
            confidence=confidence,
            trend_flip=trend_flip,
            sentiment=sentiment,
            fgi=int(fgi) if fgi else 50,
            vol_scale=vol_scale,
            session=_bucket_session(hour_utc) if hour_utc else "unknown",
            entry_price=entry_price,
            entry_time=entry_time,
            sl=sl,
            tp=tp,
            lot_size=lot_size,
            atr_at_entry=atr,
            spread_at_entry=spread,
            equity_at_entry=equity,
            ppo_raw_action=float(decision.get("ppo_primary_action", 0) or 0),
            ppo_corrected=float(decision.get("ppo_corrected_action", 0) or decision.get("corrected_action", 0) or 0),
            bias=float(decision.get("bias", 0) or 0),
            exposure=float(decision.get("target_exposure", 0) or 0),
            risk_can_trade=decision.get("risk_can_trade", True),
            risk_dd_pct=float(decision.get("risk_dd_pct", 0) or 0),
            outcome="open",
        )

        self.records[decision_id] = record
        self._recent.append(record)

        # Ensure stats entry exists
        if scenario not in self.stats:
            self.stats[scenario] = ScenarioStats(scenario=scenario)

        self._append_to_file(record)
        return decision_id

    def record_outcome(
        self,
        decision_id: str,
        exit_price: float = 0.0,
        pnl: float = 0.0,
        pnl_pct: float = 0.0,
        hold_minutes: float = 0.0,
        close_reason: str = "",
        max_drawup: float = 0.0,
        max_drawdown: float = 0.0,
    ) -> Optional[ScenarioRecord]:
        """Record a trade outcome. Links back to entry via decision_id."""
        record = self.records.get(decision_id)
        if record is None:
            logger.warning(f"ScenarioMemory: no entry record for decision_id={decision_id}")
            return None

        record.exit_price = exit_price
        record.exit_time = datetime.now(timezone.utc).isoformat()
        record.pnl = pnl
        record.pnl_pct = pnl_pct
        record.hold_minutes = hold_minutes
        record.close_reason = close_reason
        record.max_drawup = max_drawup
        record.max_drawdown = max_drawdown

        # Classify outcome
        if pnl > 0.5:
            record.outcome = "win"
        elif pnl < -0.5:
            record.outcome = "loss"
        else:
            record.outcome = "breakeven"

        # Update stats
        stats = self.stats.get(record.scenario)
        if stats is None:
            stats = ScenarioStats(scenario=record.scenario)
            self.stats[record.scenario] = stats
        stats.update(record)

        self._append_to_file(record)
        logger.info(
            f"ScenarioMemory: {record.scenario} -> {record.outcome} "
            f"PnL=${pnl:.2f} hold={hold_minutes:.0f}min "
            f"drawup={max_drawup:.1f} drawdown={max_drawdown:.1f}"
        )
        return record

    def record_rejected(
        self,
        decision: Dict[str, Any],
        reason: str = "",
    ) -> str:
        """Record a rejected trade (preflight failed)."""
        decision_id = decision.get("decision_id") or str(uuid.uuid4())

        symbol = decision.get("symbol", "UNKNOWN")
        action = decision.get("action", "HOLD")
        regime = decision.get("regime") or decision.get("volatility") or "UNKNOWN"
        confidence = decision.get("confidence", 0.0)
        trend_flip = decision.get("trend_flip", "flat")
        sentiment = decision.get("sentiment", 0.0)
        fgi = decision.get("fgi", 50)
        vol_scale = decision.get("vol_scale", 1.0)

        hour_utc = None
        entry_time = decision.get("timestamp", datetime.now(timezone.utc).isoformat())
        try:
            dt = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
            hour_utc = dt.hour
        except (ValueError, TypeError) as e:
            logger.debug(f"Failed to parse entry_time '{entry_time}': {e}")
            hour_utc = 12  # Default to noon UTC

        scenario = fingerprint_scenario(
            symbol=symbol, regime=regime, confidence=confidence,
            action=action, trend_flip=trend_flip,
            sentiment=sentiment, fgi=int(fgi) if fgi else 50,
            vol_scale=vol_scale, hour_utc=hour_utc,
        )

        record = ScenarioRecord(
            scenario=scenario,
            decision_id=decision_id,
            symbol=symbol,
            action=action,
            regime=regime,
            confidence=confidence,
            trend_flip=trend_flip,
            sentiment=sentiment,
            fgi=int(fgi) if fgi else 50,
            vol_scale=vol_scale,
            session=_bucket_session(hour_utc) if hour_utc else "unknown",
            entry_time=entry_time,
            outcome="rejected",
            close_reason=reason,
            ppo_raw_action=float(decision.get("ppo_primary_action", 0) or 0),
            ppo_corrected=float(decision.get("ppo_corrected_action", 0) or decision.get("corrected_action", 0) or 0),
            bias=float(decision.get("bias", 0) or 0),
            exposure=float(decision.get("target_exposure", 0) or 0),
            risk_can_trade=decision.get("risk_can_trade", True),
            risk_dd_pct=float(decision.get("risk_dd_pct", 0) or 0),
        )

        self.records[decision_id] = record
        self._recent.append(record)
        self._append_to_file(record)
        return decision_id

    # ── Feedback signal ──────────────────────────────────────────────────

    def should_trade(self, scenario: str) -> Dict[str, Any]:
        """Return a confidence modifier based on historical scenario performance.

        Returns dict with:
            modifier: float (0.0-1.5) — multiply confidence_scale by this
            win_rate: float — historical win rate for this scenario
            sample_size: int — number of completed trades for this scenario
            advice: str — human-readable advice
        """
        stats = self.stats.get(scenario)
        if stats is None or stats.count < 3:
            # Not enough data — neutral modifier
            return {
                "modifier": 1.0,
                "win_rate": 0.0,
                "sample_size": 0,
                "advice": "insufficient_data",
            }

        win_rate = stats.win_rate

        if win_rate < 0.25:
            modifier = 0.3
            advice = "avoid"
        elif win_rate < 0.35:
            modifier = 0.5
            advice = "reduce_size"
        elif win_rate < 0.45:
            modifier = 0.8
            advice = "cautious"
        elif win_rate < 0.55:
            modifier = 1.0
            advice = "neutral"
        elif win_rate < 0.65:
            modifier = 1.1
            advice = "favorable"
        elif win_rate < 0.75:
            modifier = 1.2
            advice = "strong_setup"
        else:
            modifier = 1.3
            advice = "high_conviction"

        # If avg PnL is negative despite decent win rate, reduce modifier
        if stats.avg_pnl < 0 and win_rate < 0.6:
            modifier = min(modifier, 0.7)

        # If sample size is small (<10), reduce modifier impact
        if stats.count < 10:
            modifier = 1.0 + (modifier - 1.0) * (stats.count / 10)

        return {
            "modifier": round(modifier, 3),
            "win_rate": round(win_rate, 3),
            "sample_size": stats.count,
            "advice": advice,
        }

    def scenario_modifier(self, decision: Dict[str, Any]) -> float:
        """Quick lookup: return just the modifier for a decision context."""
        scenario = fingerprint_scenario(
            symbol=decision.get("symbol", "UNKNOWN"),
            regime=decision.get("regime") or decision.get("volatility") or "UNKNOWN",
            confidence=decision.get("confidence", 0.0),
            action=decision.get("action", "HOLD"),
            trend_flip=decision.get("trend_flip", "flat"),
            sentiment=decision.get("sentiment", 0.0),
            fgi=int(decision.get("fgi", 50)) if decision.get("fgi") else 50,
            vol_scale=decision.get("vol_scale", 1.0),
        )
        feedback = self.should_trade(scenario)
        return feedback["modifier"]

    # ── Query methods ────────────────────────────────────────────────────

    def get_scenario_stats(self, scenario: str) -> Optional[ScenarioStats]:
        """Return stats for a specific scenario."""
        return self.stats.get(scenario)

    def get_best_scenarios(self, min_trades: int = 5, top_n: int = 10) -> List[ScenarioStats]:
        """Return the top-performing scenarios by win rate."""
        candidates = [s for s in self.stats.values() if s.count >= min_trades]
        candidates.sort(key=lambda s: (-s.win_rate, -s.avg_pnl))
        return candidates[:top_n]

    def get_worst_scenarios(self, min_trades: int = 5, top_n: int = 10) -> List[ScenarioStats]:
        """Return the worst-performing scenarios (lowest win rate)."""
        candidates = [s for s in self.stats.values() if s.count >= min_trades]
        candidates.sort(key=lambda s: (s.win_rate, s.avg_pnl))
        return candidates[:top_n]

    def get_should_avoid(self, min_trades: int = 3) -> List[str]:
        """Return scenario fingerprints with < 30% win rate."""
        return [
            s.scenario for s in self.stats.values()
            if s.count >= min_trades and s.win_rate < 0.30
        ]

    def get_by_symbol(self, symbol: str) -> Dict[str, ScenarioStats]:
        """Return all scenario stats for a specific symbol."""
        return {
            k: v for k, v in self.stats.items()
            if k.startswith(f"{symbol}_")
        }

    def get_recent_records(self, limit: int = 50) -> List[ScenarioRecord]:
        """Return the N most recent records."""
        return list(self._recent)[-limit:]

    # ── Session review ──────────────────────────────────────────────────

    def generate_session_review(self, symbol: str | None = None) -> Dict[str, Any]:
        """Generate a comprehensive session review with scenario analysis.

        Returns a dict with:
            best_scenarios: top 5 scenarios by win rate
            worst_scenarios: bottom 5 scenarios by win rate
            should_avoid: scenarios with <30% win rate
            rules_adjustments: suggested config changes based on patterns
            by_regime: win/loss/PnL grouped by volatility regime
            by_session: win/loss/PnL grouped by trading session
            by_sentiment: win/loss/PnL grouped by sentiment bucket
            illustrations: human-readable scenario illustrations with ASCII charts
        """
        scenarios = list(self.stats.values())
        if symbol:
            scenarios = [s for s in scenarios if s.scenario.startswith(f"{symbol}_")]

        # Sort by win rate
        with_data = [s for s in scenarios if s.count >= 3]
        with_data.sort(key=lambda s: -s.win_rate)

        best = with_data[:5] if len(with_data) >= 5 else with_data
        worst = list(reversed(with_data[-5:])) if len(with_data) >= 5 else []

        avoid = self.get_should_avoid(min_trades=3)
        if symbol:
            avoid = [a for a in avoid if a.startswith(f"{symbol}_")]

        # Group by regime
        by_regime: Dict[str, Dict] = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0, "count": 0})
        by_session: Dict[str, Dict] = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0, "count": 0})
        by_sentiment: Dict[str, Dict] = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0, "count": 0})

        for record in self._recent:
            if record.outcome in ("open", "rejected"):
                continue
            if symbol and record.symbol != symbol:
                continue

            regime = record.regime or "UNKNOWN"
            session = record.session or "unknown"
            sentiment = _bucket_sentiment(record.fgi) if record.fgi else "neutral"

            bucket = by_regime[regime]
            bucket["count"] += 1
            bucket["pnl"] += record.pnl
            if record.outcome == "win":
                bucket["wins"] += 1
            elif record.outcome == "loss":
                bucket["losses"] += 1

            bucket = by_session[session]
            bucket["count"] += 1
            bucket["pnl"] += record.pnl
            if record.outcome == "win":
                bucket["wins"] += 1
            elif record.outcome == "loss":
                bucket["losses"] += 1

            bucket = by_sentiment[sentiment]
            bucket["count"] += 1
            bucket["pnl"] += record.pnl
            if record.outcome == "win":
                bucket["wins"] += 1
            elif record.outcome == "loss":
                bucket["losses"] += 1

        # Compute rules adjustments
        rules = self._suggest_rules(with_data)

        # Generate scenario illustrations
        illustrations = self._generate_illustrations(best, worst, avoid, with_data)

        return {
            "best_scenarios": [asdict(s) for s in best],
            "worst_scenarios": [asdict(s) for s in worst],
            "should_avoid": avoid,
            "rules_adjustments": rules,
            "by_regime": dict(by_regime),
            "by_session": dict(by_session),
            "by_sentiment": dict(by_sentiment),
            "total_scenarios": len(scenarios),
            "total_completed_trades": sum(s.count for s in scenarios),
            "illustrations": illustrations,
        }

    def _suggest_rules(self, scenarios: List[ScenarioStats]) -> List[Dict[str, Any]]:
        """Analyze scenario patterns and suggest rule adjustments."""
        suggestions = []

        # Check for consistently losing regimes
        regime_pnl = defaultdict(float)
        regime_count = defaultdict(int)
        for s in scenarios:
            parts = s.scenario.split("_")
            if len(parts) >= 3:
                regime = parts[2]  # LOW/MED/HIGH_VOLATILITY
                regime_pnl[regime] += s.total_pnl
                regime_count[regime] += s.count

        for regime, pnl in regime_pnl.items():
            count = regime_count[regime]
            if count >= 5 and pnl < -5.0:
                suggestions.append({
                    "rule": f"reduce_{regime.lower()}_exposure",
                    "reason": f"{regime} regime has ${pnl:.2f} total PnL over {count} trades",
                    "action": f"Consider reducing vol_scale in {regime} or adding stronger confidence threshold",
                })

        # Check for losing sentiment conditions
        for s in scenarios:
            if s.count >= 5 and s.win_rate < 0.30:
                suggestions.append({
                    "rule": f"avoid_scenario_{s.scenario}",
                    "reason": f"{s.scenario} has {s.win_rate:.0%} win rate over {s.count} trades (avg PnL ${s.avg_pnl:.2f})",
                    "action": "Consider blocking this scenario or reducing position size to 50%",
                })

        # Check for strong setups to increase size
        for s in scenarios:
            if s.count >= 10 and s.win_rate > 0.65 and s.avg_pnl > 1.0:
                suggestions.append({
                    "rule": f"boost_scenario_{s.scenario}",
                    "reason": f"{s.scenario} has {s.win_rate:.0%} win rate and ${s.avg_pnl:.2f} avg PnL",
                    "action": "Consider increasing confidence_scale for this scenario",
                })

        # Check session patterns
        session_wins = defaultdict(int)
        session_losses = defaultdict(int)
        session_pnl = defaultdict(float)
        for record in self._recent:
            if record.outcome in ("open", "rejected"):
                continue
            session_wins[record.session] += 1 if record.outcome == "win" else 0
            session_losses[record.session] += 1 if record.outcome == "loss" else 0
            session_pnl[record.session] += record.pnl

        for session, pnl in session_pnl.items():
            total = session_wins[session] + session_losses[session]
            if total >= 5 and pnl < -3.0:
                suggestions.append({
                    "rule": f"avoid_{session}_session",
                    "reason": f"{session} session has ${pnl:.2f} PnL over {total} trades",
                    "action": f"Consider disabling trading during {session} hours",
                })

        return suggestions

    def _generate_illustrations(
        self,
        best: List[ScenarioStats],
        worst: List[ScenarioStats],
        avoid: List[str],
        with_data: List[ScenarioStats] = None,
    ) -> List[Dict[str, Any]]:
        """Generate human-readable scenario illustrations with ASCII charts and explanations."""
        illustrations = []

        # ── Best scenarios ──
        for s in best[:3]:
            win_bar = "█" * int(s.win_rate * 20)
            loss_bar = "░" * (20 - int(s.win_rate * 20))
            profit_sign = "+" if s.avg_pnl >= 0 else ""
            illustration = {
                "type": "best_scenario",
                "scenario": s.scenario,
                "title": f"✅ Top Setup: {s.scenario}",
                "explanation": (
                    f"This setup has a {s.win_rate:.0%} win rate across {s.count} trades. "
                    f"Average profit: ${profit_sign}{s.avg_pnl:.2f}. "
                    f"Average hold: {s.avg_hold_minutes:.0f} min. "
                    f"Best trade: ${s.best_pnl:.2f}, worst: ${s.worst_pnl:.2f}. "
                    f"Average drawup: {s.avg_max_drawup:.1f} pips, drawdown: {s.avg_max_drawdown:.1f} pips."
                ),
                "chart": f"  Win Rate: [{win_bar}{loss_bar}] {s.win_rate:.0%}\n"
                         f"  Avg PnL:  {profit_sign}${s.avg_pnl:.2f} | "
                         f"Total: ${s.total_pnl:.2f}\n"
                         f"  Trades:    {s.count} ({s.wins}W/{s.losses}L/{s.breakevens}BE)",
                "advice": "Consider increasing position size slightly for this setup.",
                "stats": asdict(s),
            }
            illustrations.append(illustration)

        # ── Worst scenarios ──
        for s in worst[:3]:
            win_bar = "█" * int(s.win_rate * 20)
            loss_bar = "░" * (20 - int(s.win_rate * 20))
            profit_sign = "+" if s.avg_pnl >= 0 else ""
            illustration = {
                "type": "worst_scenario",
                "scenario": s.scenario,
                "title": f"❌ Worst Setup: {s.scenario}",
                "explanation": (
                    f"This setup has only a {s.win_rate:.0%} win rate across {s.count} trades. "
                    f"Average loss: ${s.avg_pnl:.2f}. "
                    f"Average drawdown: {s.avg_max_drawdown:.1f} pips. "
                    f"Worst trade: ${s.worst_pnl:.2f}. "
                    f"This setup drains capital and should be avoided or sized down."
                ),
                "chart": f"  Win Rate: [{win_bar}{loss_bar}] {s.win_rate:.0%}\n"
                         f"  Avg PnL:  {profit_sign}${s.avg_pnl:.2f} | "
                         f"Total: ${s.total_pnl:.2f}\n"
                         f"  Trades:    {s.count} ({s.wins}W/{s.losses}L/{s.breakevens}BE)\n"
                         f"  Avg DD:    {s.avg_max_drawdown:.1f} pips",
                "advice": "Avoid or reduce position size to 50% for this setup.",
                "stats": asdict(s),
            }
            illustrations.append(illustration)

        # ── Avoid list ──
        if avoid:
            avoid_illustration = {
                "type": "avoid_list",
                "scenario": "multiple",
                "title": f"🚫 Avoid These Setups ({len(avoid)} scenarios)",
                "explanation": (
                    f"The following {len(avoid)} scenarios have less than 30% win rate. "
                    f"Taking trades in these setups consistently loses money:\n"
                    + "\n".join(f"  • {s}" for s in avoid[:10])
                ),
                "chart": f"  {'Setup':<50} {'Win%':>5}\n"
                         + "\n".join(
                             f"  {s:<50} {self.stats[s].win_rate:>5.0%}" if s in self.stats else f"  {s:<50}   N/A"
                             for s in avoid[:10]
                         ),
                "advice": "Block these setups or cut position size to minimum.",
            }
            illustrations.append(avoid_illustration)

        # ── Regime summary ──
        regime_summary = []
        for regime in VOL_BUCKETS:
            stats = [s for s in self.stats.values() if regime in s.scenario and s.count >= 2]
            if stats:
                total_wins = sum(s.wins for s in stats)
                total_losses = sum(s.losses for s in stats)
                total_pnl = sum(s.total_pnl for s in stats)
                total_count = sum(s.count for s in stats)
                wr = total_wins / max(total_count, 1)
                regime_summary.append({
                    "regime": regime,
                    "trades": total_count,
                    "win_rate": round(wr, 3),
                    "total_pnl": round(total_pnl, 2),
                    "avg_pnl": round(total_pnl / max(total_count, 1), 2),
                })

        if regime_summary:
            illustration = {
                "type": "regime_summary",
                "scenario": "all",
                "title": "📊 Regime Performance Summary",
                "explanation": (
                    "How the bot performs in different volatility regimes. "
                    "LOW_VOL should be more predictable, HIGH_VOL more risky."
                ),
                "chart": "\n".join(
                    f"  {r['regime']:<20} WR={r['win_rate']:>5.0%} "
                    f"PnL=${r['total_pnl']:>8.2f} "
                    f"Avg=${r['avg_pnl']:>6.2f} "
                    f"({r['trades']} trades)"
                    for r in regime_summary
                ),
                "advice": "Focus capital on winning regimes. Reduce or avoid losing ones.",
                "details": regime_summary,
            }
            illustrations.append(illustration)

        # ── What-if analysis: which conditions produce max profit ──
        if with_data:
            best_pnl = max(with_data, key=lambda s: s.total_pnl)
            best_wr = max(with_data, key=lambda s: s.win_rate)
            max_profit_illustration = {
                "type": "max_profit_conditions",
                "scenario": "analysis",
                "title": "💰 Conditions for Maximum Profit",
                "explanation": (
                    f"Highest total PnL: {best_pnl.scenario} "
                    f"(${best_pnl.total_pnl:.2f} over {best_pnl.count} trades)\n"
                    f"Highest win rate: {best_wr.scenario} "
                    f"({best_wr.win_rate:.0%} over {best_wr.count} trades)"
                ),
                "chart": f"  Best by PnL:     {best_pnl.scenario}\n"
                         f"    Total: ${best_pnl.total_pnl:.2f} | "
                         f"Avg: ${best_pnl.avg_pnl:.2f} | "
                         f"WR: {best_pnl.win_rate:.0%}\n"
                         f"  Best by Win Rate: {best_wr.scenario}\n"
                         f"    WR: {best_wr.win_rate:.0%} | "
                         f"Avg: ${best_wr.avg_pnl:.2f} | "
                         f"Total: ${best_wr.total_pnl:.2f}",
                "advice": "Increase allocation to these setups for maximum returns.",
            }
            illustrations.append(max_profit_illustration)

        return illustrations
        """Analyze scenario patterns and suggest rule adjustments."""
        suggestions = []

        # Check for consistently losing regimes
        regime_pnl = defaultdict(float)
        regime_count = defaultdict(int)
        for s in scenarios:
            # Extract regime from scenario name (format: SYMBOL_TREND_VOL_CONF_SENT_SESSION_ACTION)
            parts = s.scenario.split("_")
            if len(parts) >= 3:
                regime = parts[2]  # LOW/MED/HIGH_VOLATILITY
                regime_pnl[regime] += s.total_pnl
                regime_count[regime] += s.count

        for regime, pnl in regime_pnl.items():
            count = regime_count[regime]
            if count >= 5 and pnl < -5.0:
                suggestions.append({
                    "rule": f"reduce_{regime.lower()}_exposure",
                    "reason": f"{regime} regime has ${pnl:.2f} total PnL over {count} trades",
                    "action": f"Consider reducing vol_scale in {regime} or adding stronger confidence threshold",
                })

        # Check for losing sentiment conditions
        for s in scenarios:
            if s.count >= 5 and s.win_rate < 0.30:
                suggestions.append({
                    "rule": f"avoid_scenario_{s.scenario}",
                    "reason": f"{s.scenario} has {s.win_rate:.0%} win rate over {s.count} trades (avg PnL ${s.avg_pnl:.2f})",
                    "action": "Consider blocking this scenario or reducing position size to 50%",
                })

        # Check for strong setups to increase size
        for s in scenarios:
            if s.count >= 10 and s.win_rate > 0.65 and s.avg_pnl > 1.0:
                suggestions.append({
                    "rule": f"boost_scenario_{s.scenario}",
                    "reason": f"{s.scenario} has {s.win_rate:.0%} win rate and ${s.avg_pnl:.2f} avg PnL",
                    "action": "Consider increasing confidence_scale for this scenario",
                })

        # Check session patterns
        session_wins = defaultdict(int)
        session_losses = defaultdict(int)
        session_pnl = defaultdict(float)
        for record in self._recent:
            if record.outcome in ("open", "rejected"):
                continue
            session_wins[record.session] += 1 if record.outcome == "win" else 0
            session_losses[record.session] += 1 if record.outcome == "loss" else 0
            session_pnl[record.session] += record.pnl

        for session, pnl in session_pnl.items():
            total = session_wins[session] + session_losses[session]
            if total >= 5 and pnl < -3.0:
                suggestions.append({
                    "rule": f"avoid_{session}_session",
                    "reason": f"{session} session has ${pnl:.2f} PnL over {total} trades",
                    "action": f"Consider disabling trading during {session} hours",
                })

        return suggestions

    # ── Persistence ──────────────────────────────────────────────────────

    def _append_to_file(self, record: ScenarioRecord):
        """Append a record to the JSONL file."""
        try:
            line = json.dumps(asdict(record), default=str)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as exc:
            logger.warning(f"ScenarioMemory: failed to append record: {exc}")

    def _load(self):
        """Load historical records from JSONL file."""
        if not self.path.exists():
            logger.info(f"ScenarioMemory: no history file at {self.path}")
            return

        count = 0
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        record = ScenarioRecord(**{k: v for k, v in data.items() if k in ScenarioRecord.__dataclass_fields__})
                        self.records[record.decision_id] = record
                        # Update stats from completed records
                        if record.outcome in ("win", "loss", "breakeven"):
                            if record.scenario not in self.stats:
                                self.stats[record.scenario] = ScenarioStats(scenario=record.scenario)
                            self.stats[record.scenario].update(record)
                        count += 1
                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        logger.debug(f"Skipping invalid record: {e}")
                        continue
        except Exception as exc:
            logger.warning(f"ScenarioMemory: failed to load history: {exc}")

        logger.info(f"ScenarioMemory: loaded {count} records, {len(self.stats)} scenarios")

    def save_stats(self) -> str:
        """Save current scenario stats to a JSON file. Returns path."""
        stats_path = LOG_DIR / "scenario_stats.json"
        data = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "total_scenarios": len(self.stats),
            "total_records": len(self.records),
            "scenarios": {k: asdict(v) for k, v in self.stats.items()},
            "should_avoid": self.get_should_avoid(),
            "best_scenarios": [s.scenario for s in self.get_best_scenarios()],
            "worst_scenarios": [s.scenario for s in self.get_worst_scenarios()],
        }
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"ScenarioMemory: saved stats to {stats_path}")
        return str(stats_path)


# ── Module-level singleton ──────────────────────────────────────────────────

_memory: Optional[ScenarioMemory] = None


def get_scenario_memory() -> ScenarioMemory:
    """Get or create the global scenario memory instance."""
    global _memory
    if _memory is None:
        _memory = ScenarioMemory()
    return _memory


def reset_scenario_memory():
    """Reset the global scenario memory (for testing)."""
    global _memory
    _memory = None