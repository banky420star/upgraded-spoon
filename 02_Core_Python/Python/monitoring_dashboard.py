"""
Monitoring Dashboard for Chain Gambler

Real-time monitoring of:
- Signal quality metrics
- Kelly sizing effectiveness
- Drawdown and risk metrics
- Trade distribution analysis
- Performance attribution

Usage:
    python monitoring_dashboard.py --mode live
    python monitoring_dashboard.py --mode backtest --file backtest_results/trades.csv
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class SignalQualityMonitor:
    """Monitor signal quality over time."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.quality_history: List[tuple] = []
        self.filter_breakdown: Dict[str, List[bool]] = defaultdict(list)

    def add_signal(self, timestamp: datetime, quality_score: float, filters: Dict[str, bool]):
        """Record signal quality."""
        self.quality_history.append((timestamp, quality_score))

        for filter_name, passed in filters.items():
            self.filter_breakdown[filter_name].append(passed)

        # Keep only recent history
        if len(self.quality_history) > self.window_size:
            self.quality_history.pop(0)
            for key in self.filter_breakdown:
                self.filter_breakdown[key].pop(0)

    def get_stats(self) -> Dict[str, Any]:
        """Get current signal quality stats."""
        if not self.quality_history:
            return {"avg_quality": 0, "trend": "neutral"}

        qualities = [q for _, q in self.quality_history]
        avg_quality = np.mean(qualities)

        # Calculate trend
        if len(qualities) >= 20:
            recent = np.mean(qualities[-20:])
            older = np.mean(qualities[-40:-20])
            if recent > older * 1.05:
                trend = "improving"
            elif recent < older * 0.95:
                trend = "degrading"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        # Filter pass rates
        filter_rates = {
            name: sum(passes) / len(passes) if passes else 0
            for name, passes in self.filter_breakdown.items()
        }

        return {
            "avg_quality": avg_quality,
            "trend": trend,
            "filter_pass_rates": filter_rates,
            "total_signals": len(self.quality_history),
        }


class KellyEffectivenessMonitor:
    """Monitor Kelly criterion sizing effectiveness."""

    def __init__(self):
        self.kelly_history: List[tuple] = []
        self.outcomes: List[tuple] = []  # (kelly_fraction, outcome)

    def add_trade(self, timestamp: datetime, kelly_fraction: float, pnl: float):
        """Record trade with Kelly fraction."""
        self.kelly_history.append((timestamp, kelly_fraction))
        self.outcomes.append((kelly_fraction, pnl))

    def get_stats(self) -> Dict[str, Any]:
        """Get Kelly effectiveness stats."""
        if not self.outcomes:
            return {"avg_kelly": 0, "effectiveness": "unknown"}

        avg_kelly = np.mean([k for k, _ in self.outcomes])

        # Analyze by Kelly bucket
        buckets = {
            "low": [],    # < 0.3
            "medium": [], # 0.3 - 0.7
            "high": [],   # > 0.7
        }

        for k, pnl in self.outcomes:
            if k < 0.3:
                buckets["low"].append(pnl)
            elif k < 0.7:
                buckets["medium"].append(pnl)
            else:
                buckets["high"].append(pnl)

        bucket_stats = {}
        for name, pnls in buckets.items():
            if pnls:
                bucket_stats[name] = {
                    "count": len(pnls),
                    "avg_pnl": np.mean(pnls),
                    "win_rate": len([p for p in pnls if p > 0]) / len(pnls),
                }

        return {
            "avg_kelly": avg_kelly,
            "buckets": bucket_stats,
            "total_trades": len(self.outcomes),
        }


class DrawdownMonitor:
    """Monitor drawdown and recovery."""

    def __init__(self):
        self.equity_history: List[tuple] = []
        self.peak_equity = 0
        self.drawdown_start: Optional[datetime] = None
        self.current_dd = 0.0

    def update(self, timestamp: datetime, equity: float):
        """Update equity and calculate drawdown."""
        self.equity_history.append((timestamp, equity))

        if equity > self.peak_equity:
            self.peak_equity = equity
            if self.drawdown_start is not None:
                # Recovery
                self.drawdown_start = None

        dd = (self.peak_equity - equity) / self.peak_equity * 100
        self.current_dd = dd

        if dd > 5 and self.drawdown_start is None:
            self.drawdown_start = timestamp

    def get_stats(self) -> Dict[str, Any]:
        """Get drawdown stats."""
        if not self.equity_history:
            return {"max_dd": 0, "current_dd": 0}

        max_dd = 0
        peak = self.equity_history[0][1]

        for _, equity in self.equity_history:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            max_dd = max(max_dd, dd)

        # Calculate recovery time if in drawdown
        recovery_info = ""
        if self.drawdown_start:
            days_in_dd = (datetime.now(timezone.utc) - self.drawdown_start).days
            recovery_info = f"{days_in_dd} days"

        return {
            "max_drawdown": max_dd,
            "current_drawdown": self.current_dd,
            "recovery_time": recovery_info,
            "in_drawdown": self.drawdown_start is not None,
        }


class TradeDistributionAnalyzer:
    """Analyze trade distributions and patterns."""

    def __init__(self):
        self.trades: List[Dict] = []

    def add_trade(self, trade_data: Dict):
        """Add trade to analysis."""
        self.trades.append(trade_data)

    def get_distribution(self) -> Dict[str, Any]:
        """Get trade distribution analysis."""
        if not self.trades:
            return {"message": "No trades available"}

        pnls = [t.get("pnl", 0) for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        # Time of day analysis
        hour_performance = defaultdict(list)
        for trade in self.trades:
            ts = trade.get("timestamp")
            if isinstance(ts, str):
                try:
                    hour = datetime.fromisoformat(ts).hour
                except (ValueError, TypeError) as e:
                    logger.debug(f"Failed to parse timestamp '{ts}': {e}")
                    continue
            elif isinstance(ts, datetime):
                hour = ts.hour
            else:
                continue
            hour_performance[hour].append(trade.get("pnl", 0))

        best_hours = sorted(
            [(h, sum(pnls)/len(pnls)) for h, pnls in hour_performance.items() if pnls],
            key=lambda x: x[1],
            reverse=True
        )[:3]

        return {
            "total_trades": len(self.trades),
            "win_rate": len(wins) / len(pnls) if pnls else 0,
            "avg_win": np.mean(wins) if wins else 0,
            "avg_loss": np.mean(losses) if losses else 0,
            "profit_factor": abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf'),
            "sharpe": self._calculate_sharpe(pnls),
            "best_hours": best_hours,
            "pnl_std": np.std(pnls) if len(pnls) > 1 else 0,
        }

    def _calculate_sharpe(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0
        avg = np.mean(returns)
        std = np.std(returns)
        return avg / std if std > 0 else 0


class PerformanceAttribution:
    """Attribute performance by symbol, regime, etc."""

    def __init__(self):
        self.by_symbol: Dict[str, List[float]] = defaultdict(list)
        self.by_regime: Dict[str, List[float]] = defaultdict(list)
        self.by_quality: Dict[str, List[float]] = defaultdict(list)

    def add_trade(self, symbol: str, regime: str, quality: float, pnl: float):
        """Add trade for attribution."""
        self.by_symbol[symbol].append(pnl)

        if regime:
            self.by_regime[regime].append(pnl)

        quality_bucket = "high" if quality >= 0.7 else "medium" if quality >= 0.5 else "low"
        self.by_quality[quality_bucket].append(pnl)

    def get_attribution(self) -> Dict[str, Dict]:
        """Get performance attribution."""
        attribution = {}

        # By symbol
        attribution["by_symbol"] = {
            sym: {
                "trades": len(pnls),
                "total_pnl": sum(pnls),
                "avg_pnl": np.mean(pnls),
                "win_rate": len([p for p in pnls if p > 0]) / len(pnls) if pnls else 0,
            }
            for sym, pnls in self.by_symbol.items()
        }

        # By regime
        attribution["by_regime"] = {
            regime: {
                "trades": len(pnls),
                "total_pnl": sum(pnls),
                "avg_pnl": np.mean(pnls) if pnls else 0,
            }
            for regime, pnls in self.by_regime.items()
        }

        # By quality
        attribution["by_quality"] = {
            bucket: {
                "trades": len(pnls),
                "total_pnl": sum(pnls),
                "avg_pnl": np.mean(pnls) if pnls else 0,
            }
            for bucket, pnls in self.by_quality.items()
        }

        return attribution


class MonitoringDashboard:
    """
    Main monitoring dashboard.
    """

    def __init__(self):
        self.signal_monitor = SignalQualityMonitor()
        self.kelly_monitor = KellyEffectivenessMonitor()
        self.drawdown_monitor = DrawdownMonitor()
        self.distribution_analyzer = TradeDistributionAnalyzer()
        self.attribution = PerformanceAttribution()

        self.last_update = datetime.now(timezone.utc)

    def process_trade(self, trade_data: Dict):
        """Process new trade data."""
        timestamp = trade_data.get("timestamp", datetime.now(timezone.utc))
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        # Update monitors
        self.kelly_monitor.add_trade(
            timestamp,
            trade_data.get("kelly_fraction", 0),
            trade_data.get("pnl", 0)
        )

        self.distribution_analyzer.add_trade(trade_data)

        self.attribution.add_trade(
            trade_data.get("symbol", "UNKNOWN"),
            trade_data.get("regime", ""),
            trade_data.get("signal_quality", 0),
            trade_data.get("pnl", 0)
        )

        # Update equity if available
        if "equity" in trade_data:
            self.drawdown_monitor.update(timestamp, trade_data["equity"])

    def process_signal(self, signal_data: Dict):
        """Process new signal data."""
        timestamp = signal_data.get("timestamp", datetime.now(timezone.utc))
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        self.signal_monitor.add_signal(
            timestamp,
            signal_data.get("signal_quality_score", 0),
            signal_data.get("filters", {})
        )

    def generate_report(self) -> str:
        """Generate comprehensive monitoring report."""
        lines = [
            "=" * 80,
            "CHAIN GAMBLER MONITORING DASHBOARD",
            f"Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "=" * 80,
            "",
        ]

        # Signal Quality Section
        signal_stats = self.signal_monitor.get_stats()
        lines.extend([
            "SIGNAL QUALITY METRICS",
            "-" * 40,
            f"Average Quality: {signal_stats['avg_quality']:.3f}",
            f"Trend: {signal_stats['trend']}",
            f"Total Signals: {signal_stats['total_signals']}",
            "Filter Pass Rates:",
        ])
        for filter_name, rate in signal_stats.get("filter_pass_rates", {}).items():
            lines.append(f"  {filter_name}: {rate:.1%}")
        lines.append("")

        # Kelly Effectiveness Section
        kelly_stats = self.kelly_monitor.get_stats()
        lines.extend([
            "KELLY SIZING EFFECTIVENESS",
            "-" * 40,
            f"Average Kelly: {kelly_stats['avg_kelly']:.3f}",
            f"Total Trades: {kelly_stats['total_trades']}",
            "Performance by Kelly Bucket:",
        ])
        for bucket, stats in kelly_stats.get("buckets", {}).items():
            lines.append(
                f"  {bucket}: {stats['count']} trades, "
                f"avg_pnl=${stats['avg_pnl']:.2f}, "
                f"win_rate={stats['win_rate']:.1%}"
            )
        lines.append("")

        # Drawdown Section
        dd_stats = self.drawdown_monitor.get_stats()
        lines.extend([
            "DRAWDOWN MONITORING",
            "-" * 40,
            f"Max Drawdown: {dd_stats['max_drawdown']:.2f}%",
            f"Current Drawdown: {dd_stats['current_drawdown']:.2f}%",
        ])
        if dd_stats['in_drawdown']:
            lines.append(f"In Drawdown: {dd_stats['recovery_time']}")
        lines.append("")

        # Trade Distribution Section
        dist = self.distribution_analyzer.get_distribution()
        if "message" not in dist:
            lines.extend([
                "TRADE DISTRIBUTION",
                "-" * 40,
                f"Total Trades: {dist['total_trades']}",
                f"Win Rate: {dist['win_rate']:.1%}",
                f"Profit Factor: {dist['profit_factor']:.2f}",
                f"Sharpe: {dist['sharpe']:.2f}",
                "Best Trading Hours:",
            ])
            for hour, avg_pnl in dist.get("best_hours", []):
                lines.append(f"  Hour {hour:02d}: ${avg_pnl:.2f} avg")
            lines.append("")

        # Performance Attribution Section
        attr = self.attribution.get_attribution()
        lines.extend([
            "PERFORMANCE ATTRIBUTION",
            "-" * 40,
            "By Symbol:",
        ])
        for sym, stats in sorted(attr['by_symbol'].items(), key=lambda x: x[1]['total_pnl'], reverse=True)[:5]:
            lines.append(
                f"  {sym}: ${stats['total_pnl']:.2f} total, "
                f"{stats['win_rate']:.0%} WR"
            )

        lines.extend([
            "",
            "By Quality Bucket:",
        ])
        for bucket, stats in attr['by_quality'].items():
            lines.append(
                f"  {bucket}: ${stats['total_pnl']:.2f} total, "
                f"{stats['trades']} trades"
            )

        lines.extend([
            "",
            "=" * 80,
        ])

        return "\n".join(lines)

    def save_report(self, output_path: Optional[Path] = None):
        """Save report to file."""
        if output_path is None:
            output_dir = PROJECT_ROOT / "logs" / "monitoring"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        report = self.generate_report()
        with open(output_path, "w") as f:
            f.write(report)

        return output_path


def analyze_backtest_file(filepath: Path) -> str:
    """Analyze backtest results file."""
    logger.info(f"Analyzing backtest file: {filepath}")

    # Load trades
    df = pd.read_csv(filepath)

    dashboard = MonitoringDashboard()

    # Process each trade
    for _, row in df.iterrows():
        trade_data = {
            "timestamp": row.get("timestamp", datetime.now()),
            "symbol": row.get("symbol", "UNKNOWN"),
            "pnl": row.get("pnl", 0),
            "signal_quality": row.get("signal_quality", 0),
            "kelly_fraction": row.get("kelly_fraction", 0),
            "regime": row.get("regime", ""),
        }
        dashboard.process_trade(trade_data)

    report = dashboard.generate_report()

    # Save to file
    output_path = filepath.parent / f"monitoring_analysis_{datetime.now().strftime('%Y%m%d')}.txt"
    with open(output_path, "w") as f:
        f.write(report)

    logger.success(f"Analysis saved to {output_path}")

    return report


def main():
    """Run monitoring dashboard."""
    import argparse

    parser = argparse.ArgumentParser(description="Chain Gambler Monitoring Dashboard")
    parser.add_argument("--mode", choices=["live", "backtest"], default="live")
    parser.add_argument("--file", type=str, help="Backtest trades file to analyze")
    parser.add_argument("--interval", type=int, default=60, help="Update interval in seconds")

    args = parser.parse_args()

    if args.mode == "backtest":
        if not args.file:
            print("Error: --file required for backtest mode")
            return 1

        filepath = Path(args.file)
        if not filepath.exists():
            print(f"Error: File not found: {filepath}")
            return 1

        report = analyze_backtest_file(filepath)
        print(report)

    else:  # live mode
        dashboard = MonitoringDashboard()

        try:
            while True:
                report = dashboard.generate_report()
                os.system("clear" if os.name != "nt" else "cls")
                print(report)
                time.sleep(args.interval)

        except KeyboardInterrupt:
            print("\nStopping monitoring...")
            output = dashboard.save_report()
            print(f"Final report saved to: {output}")

    return 0


if __name__ == "__main__":
    import time
    sys.exit(main())
