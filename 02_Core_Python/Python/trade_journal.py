"""
TradeJournal — Post-trade learning and analysis.

Records every trade with context:
  - Market regime at entry
  - Strategy used
  - News distance
  - Spread at entry
  - R-multiple achieved
  - Exit reason

Learns:
  - Which regimes are profitable
  - Which symbols perform
  - Which sessions work
  - Which exit rules protect profit

Usage:
    from Python.trade_journal import TradeJournal
    journal = TradeJournal(config)
    journal.record_entry(position_id, symbol, entry_data)
    journal.record_exit(position_id, exit_data)
    insights = journal.analyze()
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
from loguru import logger

from Python.market_guardian import MarketRegime


@dataclass
class TradeRecord:
    """Complete record of a trade."""
    trade_id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: Optional[float] = None
    volume: float = 0.0

    # Context at entry
    regime: str = "unknown"
    strategy: str = "unknown"
    atr_14: float = 0.0
    adx_14: float = 0.0
    rsi_14: float = 50.0
    spread_bps: float = 0.0
    news_distance_minutes: float = 999.0
    session: str = "unknown"
    model_confidence: float = 0.0

    # Risk/R metrics
    initial_r: float = 0.0  # Distance to stop
    stop_price: float = 0.0
    target_price: float = 0.0

    # Outcome
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    r_multiple: float = 0.0  # PnL in R units
    max_r_reached: float = 0.0  # Maximum favorable excursion in R
    exit_reason: str = ""
    exit_action: str = ""

    # Analysis
    quality_score: int = 0
    was_profitable: bool = False
    hit_stop: bool = False
    hit_target: bool = False
    partial_exits: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        if self.entry_time is None:
            self.entry_time = datetime.now(timezone.utc)
        self.was_profitable = self.pnl > 0


@dataclass
class LearningInsights:
    """Insights from trade analysis."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0

    # By regime
    regime_performance: Dict[str, Dict] = field(default_factory=dict)

    # By symbol
    symbol_performance: Dict[str, Dict] = field(default_factory=dict)

    # By strategy
    strategy_performance: Dict[str, Dict] = field(default_factory=dict)

    # By session
    session_performance: Dict[str, Dict] = field(default_factory=dict)

    # Quality score effectiveness
    quality_score_effectiveness: Dict[str, float] = field(default_factory=dict)

    # Exit effectiveness
    exit_effectiveness: Dict[str, float] = field(default_factory=dict)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)


class TradeJournal:
    """
    Journal for recording and analyzing trades.
    Learns from outcomes to improve future decisions.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.journal_config = self.config.get("trade_journal", {})

        # Storage
        self._trades: Dict[str, TradeRecord] = {}
        self._completed_trades: List[TradeRecord] = []
        self._max_completed = self.journal_config.get("max_history", 1000)

        # File storage
        self._data_dir = Path(self.journal_config.get("data_dir", "logs/trade_journal"))
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # Load existing
        self._load_trades()

    def record_entry(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        entry_price: float,
        stop_price: float,
        target_price: float,
        volume: float,
        context: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> TradeRecord:
        """
        Record trade entry with full context.

        Args:
            trade_id: Unique trade identifier
            symbol: Trading symbol
            side: "BUY" or "SELL"
            entry_price: Entry price
            stop_price: Stop price
            target_price: Target price
            volume: Trade volume
            context: Dict with regime, strategy, atr, adx, rsi, spread, news_distance, session, model_confidence
            timestamp: Entry timestamp
        """
        initial_r = abs(entry_price - stop_price)

        record = TradeRecord(
            trade_id=trade_id,
            symbol=symbol,
            side=side.upper(),
            entry_price=entry_price,
            exit_price=None,
            volume=volume,
            regime=context.get("regime", "unknown"),
            strategy=context.get("strategy", "unknown"),
            atr_14=context.get("atr_14", 0.0),
            adx_14=context.get("adx_14", 0.0),
            rsi_14=context.get("rsi_14", 50.0),
            spread_bps=context.get("spread_bps", 0.0),
            news_distance_minutes=context.get("news_distance_minutes", 999.0),
            session=context.get("session", "unknown"),
            model_confidence=context.get("model_confidence", 0.0),
            initial_r=initial_r,
            stop_price=stop_price,
            target_price=target_price,
            entry_time=timestamp or datetime.now(timezone.utc),
            quality_score=context.get("quality_score", 0)
        )

        self._trades[trade_id] = record
        logger.debug(f"Recorded entry for trade {trade_id}: {side} {volume} {symbol} @ {entry_price}")

        return record

    def record_exit(
        self,
        trade_id: str,
        exit_price: float,
        pnl: float,
        exit_reason: str,
        exit_action: str = "",
        max_r_reached: float = 0.0,
        partial_exits: Optional[List[Dict]] = None,
        timestamp: Optional[datetime] = None
    ) -> Optional[TradeRecord]:
        """
        Record trade exit and complete the record.

        Args:
            trade_id: Trade identifier
            exit_price: Exit price
            pnl: Profit/loss in currency
            exit_reason: Why trade was closed
            exit_action: Action type from ExitEngine
            max_r_reached: Maximum R multiple achieved
            partial_exits: List of partial exit records
            timestamp: Exit timestamp
        """
        if trade_id not in self._trades:
            logger.warning(f"Cannot record exit: trade {trade_id} not found")
            return None

        record = self._trades[trade_id]
        record.exit_price = exit_price
        record.exit_time = timestamp or datetime.now(timezone.utc)
        record.pnl = pnl
        record.pnl_pct = (pnl / record.entry_price) * 100 if record.entry_price > 0 else 0
        record.r_multiple = pnl / record.initial_r if record.initial_r > 0 else 0
        record.max_r_reached = max_r_reached
        record.exit_reason = exit_reason
        record.exit_action = exit_action
        record.was_profitable = pnl > 0
        record.hit_stop = abs(exit_price - record.stop_price) < record.initial_r * 0.1
        record.hit_target = exit_price is not None and abs(exit_price - record.target_price) < record.initial_r * 0.5

        if partial_exits:
            record.partial_exits = partial_exits

        # Move to completed
        self._completed_trades.append(record)
        del self._trades[trade_id]

        # Trim history
        if len(self._completed_trades) > self._max_completed:
            self._completed_trades = self._completed_trades[-self._max_completed:]

        # Save
        self._save_trade(record)

        logger.info(f"Recorded exit for trade {trade_id}: PnL ${pnl:.2f} ({record.r_multiple:.2f}R) - {exit_reason}")

        return record

    def analyze(self, lookback: int = 100) -> LearningInsights:
        """
        Analyze completed trades and generate insights.

        Args:
            lookback: Number of recent trades to analyze

        Returns:
            LearningInsights with performance breakdowns
        """
        trades = self._completed_trades[-lookback:]
        if not trades:
            return LearningInsights()

        insights = LearningInsights(total_trades=len(trades))

        # Overall stats
        winning_trades = [t for t in trades if t.was_profitable]
        losing_trades = [t for t in trades if not t.was_profitable]

        insights.winning_trades = len(winning_trades)
        insights.losing_trades = len(losing_trades)
        insights.win_rate = len(winning_trades) / len(trades) if trades else 0

        total_pnl = sum(t.pnl for t in trades)
        insights.total_pnl = total_pnl
        insights.avg_pnl = total_pnl / len(trades) if trades else 0

        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))

        insights.avg_win = total_wins / len(winning_trades) if winning_trades else 0
        insights.avg_loss = -total_losses / len(losing_trades) if losing_trades else 0
        insights.profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # By regime
        regime_stats: Dict[str, List[TradeRecord]] = {}
        for trade in trades:
            regime = trade.regime
            if regime not in regime_stats:
                regime_stats[regime] = []
            regime_stats[regime].append(trade)

        for regime, regime_trades in regime_stats.items():
            wins = sum(1 for t in regime_trades if t.was_profitable)
            pnl = sum(t.pnl for t in regime_trades)
            insights.regime_performance[regime] = {
                "trades": len(regime_trades),
                "wins": wins,
                "win_rate": wins / len(regime_trades),
                "total_pnl": round(pnl, 2),
                "avg_r": sum(t.r_multiple for t in regime_trades) / len(regime_trades),
            }

        # By symbol
        symbol_stats: Dict[str, List[TradeRecord]] = {}
        for trade in trades:
            sym = trade.symbol
            if sym not in symbol_stats:
                symbol_stats[sym] = []
            symbol_stats[sym].append(trade)

        for sym, sym_trades in symbol_stats.items():
            wins = sum(1 for t in sym_trades if t.was_profitable)
            pnl = sum(t.pnl for t in sym_trades)
            insights.symbol_performance[sym] = {
                "trades": len(sym_trades),
                "wins": wins,
                "win_rate": wins / len(sym_trades),
                "total_pnl": round(pnl, 2),
                "avg_r": sum(t.r_multiple for t in sym_trades) / len(sym_trades),
            }

        # By strategy
        strategy_stats: Dict[str, List[TradeRecord]] = {}
        for trade in trades:
            strat = trade.strategy
            if strat not in strategy_stats:
                strategy_stats[strat] = []
            strategy_stats[strat].append(trade)

        for strat, strat_trades in strategy_stats.items():
            wins = sum(1 for t in strat_trades if t.was_profitable)
            pnl = sum(t.pnl for t in strat_trades)
            insights.strategy_performance[strat] = {
                "trades": len(strat_trades),
                "wins": wins,
                "win_rate": wins / len(strat_trades),
                "total_pnl": round(pnl, 2),
            }

        # Quality score effectiveness
        high_quality = [t for t in trades if t.quality_score >= 80]
        low_quality = [t for t in trades if t.quality_score < 70]

        if high_quality and low_quality:
            high_win_rate = sum(1 for t in high_quality if t.was_profitable) / len(high_quality)
            low_win_rate = sum(1 for t in low_quality if t.was_profitable) / len(low_quality)
            insights.quality_score_effectiveness = {
                "high_quality_win_rate": round(high_win_rate, 2),
                "low_quality_win_rate": round(low_win_rate, 2),
                "edge": round(high_win_rate - low_win_rate, 2)
            }

        # Exit effectiveness
        exit_stats: Dict[str, List[TradeRecord]] = {}
        for trade in trades:
            reason = trade.exit_action or trade.exit_reason
            if reason not in exit_stats:
                exit_stats[reason] = []
            exit_stats[reason].append(trade)

        for reason, reason_trades in exit_stats.items():
            wins = sum(1 for t in reason_trades if t.was_profitable)
            insights.exit_effectiveness[reason] = round(wins / len(reason_trades), 2)

        # Generate recommendations
        insights.recommendations = self._generate_recommendations(insights)

        return insights

    def _generate_recommendations(self, insights: LearningInsights) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Regime recommendations
        best_regime = None
        best_regime_wr = 0
        for regime, stats in insights.regime_performance.items():
            if stats["trades"] >= 10 and stats["win_rate"] > best_regime_wr:
                best_regime = regime
                best_regime_wr = stats["win_rate"]

        worst_regime = None
        worst_regime_wr = 1.0
        for regime, stats in insights.regime_performance.items():
            if stats["trades"] >= 10 and stats["win_rate"] < worst_regime_wr:
                worst_regime = regime
                worst_regime_wr = stats["win_rate"]

        if best_regime and best_regime_wr > 0.6:
            recommendations.append(f"Focus on {best_regime} regime (WR: {best_regime_wr:.1%})")

        if worst_regime and worst_regime_wr < 0.4:
            recommendations.append(f"Avoid {worst_regime} regime (WR: {worst_regime_wr:.1%})")

        # Symbol recommendations
        for sym, stats in insights.symbol_performance.items():
            if stats["trades"] >= 10:
                if stats["win_rate"] < 0.4:
                    recommendations.append(f"Consider removing {sym} (WR: {stats['win_rate']:.1%})")
                elif stats["win_rate"] > 0.6:
                    recommendations.append(f"Consider increasing {sym} size (WR: {stats['win_rate']:.1%})")

        # Strategy recommendations
        for strat, stats in insights.strategy_performance.items():
            if stats["trades"] >= 10:
                if stats["win_rate"] < 0.4:
                    recommendations.append(f"Review {strat} strategy (WR: {stats['win_rate']:.1%})")

        # Exit recommendations
        if "hit_stop" in insights.exit_effectiveness:
            stop_wr = insights.exit_effectiveness["hit_stop"]
            if stop_wr < 0.3:
                recommendations.append("Too many stops being hit - widen SL or filter entries better")

        return recommendations

    def get_trade(self, trade_id: str) -> Optional[TradeRecord]:
        """Get a specific trade record."""
        if trade_id in self._trades:
            return self._trades[trade_id]
        for trade in self._completed_trades:
            if trade.trade_id == trade_id:
                return trade
        return None

    def get_recent_trades(self, n: int = 10) -> List[TradeRecord]:
        """Get recent completed trades."""
        return self._completed_trades[-n:]

    def get_stats(self) -> Dict[str, Any]:
        """Get quick stats."""
        return {
            "open_trades": len(self._trades),
            "completed_trades": len(self._completed_trades),
            "total_pnl": round(sum(t.pnl for t in self._completed_trades), 2),
            "avg_r": round(sum(t.r_multiple for t in self._completed_trades) / len(self._completed_trades), 2)
            if self._completed_trades else 0
        }

    def _save_trade(self, trade: TradeRecord):
        """Save trade to file."""
        try:
            file_path = self._data_dir / f"{trade.trade_id}.json"
            with open(file_path, 'w') as f:
                data = asdict(trade)
                # Convert datetime to string
                if data.get('entry_time'):
                    data['entry_time'] = data['entry_time'].isoformat() if isinstance(data['entry_time'], datetime) else data['entry_time']
                if data.get('exit_time'):
                    data['exit_time'] = data['exit_time'].isoformat() if isinstance(data['exit_time'], datetime) else data['exit_time']
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save trade {trade.trade_id}: {e}")

    def _load_trades(self):
        """Load historical trades from disk."""
        try:
            for file_path in self._data_dir.glob("*.json"):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        # Parse datetimes
                        if data.get('entry_time'):
                            data['entry_time'] = datetime.fromisoformat(data['entry_time'])
                        if data.get('exit_time'):
                            data['exit_time'] = datetime.fromisoformat(data['exit_time'])

                        trade = TradeRecord(**data)
                        self._completed_trades.append(trade)
                except Exception as e:
                    logger.warning(f"Failed to load trade from {file_path}: {e}")

            # Sort by exit time
            self._completed_trades.sort(key=lambda x: x.exit_time or datetime.min.replace(tzinfo=timezone.utc))

            logger.info(f"Loaded {len(self._completed_trades)} historical trades from journal")
        except Exception as e:
            logger.warning(f"Failed to load trade history: {e}")


# Convenience function
def get_journal_summary(journal: TradeJournal, n: int = 50) -> str:
    """Get a text summary of journal for logging/display."""
    insights = journal.analyze(n)

    lines = [
        f"Trade Journal Summary (last {n} trades)",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"Total Trades: {insights.total_trades}",
        f"Win Rate: {insights.win_rate:.1%}",
        f"Total PnL: ${insights.total_pnl:,.2f}",
        f"Avg PnL: ${insights.avg_pnl:,.2f}",
        f"Profit Factor: {insights.profit_factor:.2f}",
        f"",
        f"By Regime:",
    ]

    for regime, stats in sorted(insights.regime_performance.items(),
                                 key=lambda x: x[1]['win_rate'], reverse=True):
        lines.append(f"  {regime}: {stats['win_rate']:.1%} WR, ${stats['total_pnl']:,.2f}")

    if insights.recommendations:
        lines.extend(["", "Recommendations:"])
        for rec in insights.recommendations[:5]:
            lines.append(f"  • {rec}")

    return "\n".join(lines)