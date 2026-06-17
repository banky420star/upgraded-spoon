"""
Comprehensive Backtesting Engine for Chain Gambler

Features:
- Full historical backtesting with realistic execution
- Signal quality analysis
- Kelly sizing validation
- Ollama LLM review integration
- Performance attribution by symbol
- Drawdown and risk analysis
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Python.data_feed import fetch_training_data


@dataclass
class TradeRecord:
    """Record of a single trade."""
    timestamp: datetime
    symbol: str
    action: str  # BUY/SELL
    entry_price: float
    exit_price: float
    lot_size: float
    sl: float
    tp: float
    pnl: float
    pnl_pct: float
    holding_bars: int
    exit_reason: str  # TP, SL, TIMEOUT, SIGNAL

    # Optimization metrics
    signal_quality_score: float
    kelly_fraction: float
    trend_flip: str
    regime: str
    confidence: float


@dataclass
class BacktestResult:
    """Complete backtest results."""
    start_date: datetime
    end_date: datetime
    symbols: List[str]

    # Performance metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float

    # Return metrics
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float

    # Risk metrics
    volatility_annual: float
    var_95: float  # Value at Risk
    tail_ratio: float

    # Optimization metrics
    avg_signal_quality: float
    kelly_effectiveness: float
    cost_savings_pct: float

    # Trade history
    trades: List[TradeRecord] = field(default_factory=list)
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)
    drawdown_curve: List[Tuple[datetime, float]] = field(default_factory=list)


class BacktestExecutor:
    """
    Realistic backtest executor with slippage and spread modeling.
    """

    def __init__(
        self,
        spread_bps: float = 1.5,  # Average spread in basis points
        slippage_bps: float = 0.5,  # Execution slippage
        commission_per_lot: float = 7.0,  # Round-trip commission
    ):
        self.spread_bps = spread_bps
        self.slippage_bps = slippage_bps
        self.commission_per_lot = commission_per_lot

    def execute_entry(
        self,
        price: float,
        action: str,
        volatility: float,
        session_quality: float = 1.0,
    ) -> Tuple[float, float]:
        """
        Simulate entry execution.

        Returns:
            (filled_price, spread_paid)
        """
        # Base spread adjusted by volatility and session
        adjusted_spread = self.spread_bps * (1 + volatility * 10) / session_quality
        spread_decimal = adjusted_spread / 10000

        # Add slippage
        slippage_decimal = self.slippage_bps / 10000 * np.random.uniform(0.5, 1.5)

        if action == "BUY":
            filled_price = price * (1 + spread_decimal/2 + slippage_decimal)
        else:  # SELL
            filled_price = price * (1 - spread_decimal/2 - slippage_decimal)

        spread_paid = price * spread_decimal

        return filled_price, spread_paid

    def execute_exit(
        self,
        entry_price: float,
        current_price: float,
        action: str,
        exit_reason: str,
    ) -> Tuple[float, float]:
        """
        Simulate exit execution.

        Returns:
            (filled_price, pnl_before_commission)
        """
        # Exit has half the spread of entry
        spread_decimal = self.spread_bps / 10000 * 0.5

        if action == "BUY":
            filled_price = current_price * (1 - spread_decimal)
            pnl = (filled_price - entry_price) / entry_price
        else:
            filled_price = current_price * (1 + spread_decimal)
            pnl = (entry_price - filled_price) / entry_price

        return filled_price, pnl

    def calculate_commission(self, lot_size: float) -> float:
        """Calculate commission for trade."""
        return self.commission_per_lot * lot_size


class BacktestEngine:
    """
    Main backtesting engine for Chain Gambler.
    """

    def __init__(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "M5",
        initial_equity: float = 10000.0,
        max_positions: int = 5,
        ollama_review: bool = True,  # Enable Ollama LLM review
    ):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        self.initial_equity = initial_equity
        self.max_positions = max_positions
        self.ollama_review = ollama_review

        # Initialize components
        self.executor = BacktestExecutor()
        self.equity = initial_equity
        self.peak_equity = initial_equity
        self.current_drawdown = 0.0

        # Tracking
        self.positions: Dict[str, Dict] = {}  # Active positions
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.drawdown_curve: List[Tuple[datetime, float]] = []

        # Performance by symbol
        self.symbol_stats: Dict[str, Dict] = {s: {
            'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0
        } for s in symbols}

        logger.success(f"BacktestEngine initialized: {len(symbols)} symbols")

    def run_backtest(self) -> BacktestResult:
        """
        Run full backtest.
        """
        logger.info(f"Starting backtest: {self.start_date} to {self.end_date}")

        # Load data for all symbols
        data = {}
        for symbol in self.symbols:
            try:
                df = fetch_training_data(
                    symbol=symbol,
                    period=f"{(self.end_date - self.start_date).days}d",
                    interval=self.timeframe
                )
                if df is not None and len(df) > 100:
                    data[symbol] = df
                    logger.success(f"Loaded {len(df)} bars for {symbol}")
                else:
                    logger.warning(f"Insufficient data for {symbol}")
            except Exception as e:
                logger.error(f"Failed to load {symbol}: {e}")

        if not data:
            raise ValueError("No data loaded for any symbol")

        # Initialize brain
        self._init_brain()

        # Run simulation
        current_date = self.start_date
        bar_count = 0

        while current_date < self.end_date:
            # Process each symbol
            for symbol in self.symbols:
                if symbol not in data:
                    continue

                df = data[symbol]

                # Get current window
                mask = (df.index >= current_date - timedelta(days=5)) & (df.index <= current_date)
                window = df[mask]

                if len(window) < 100:
                    continue

                # Update equity tracking
                self._update_equity(current_date)

                # Check for exits
                self._check_exits(symbol, window)

                # Generate signal
                self._process_signal(symbol, window)

            # Advance time
            if self.timeframe == "M5":
                current_date += timedelta(minutes=5)
            elif self.timeframe == "M15":
                current_date += timedelta(minutes=15)
            elif self.timeframe == "H1":
                current_date += timedelta(hours=1)
            else:
                current_date += timedelta(days=1)

            bar_count += 1
            if bar_count % 1000 == 0:
                logger.info(f"Processed {bar_count} bars, equity: ${self.equity:.2f}")

        # Close any remaining positions
        for symbol in list(self.positions.keys()):
            self._close_position(symbol, data[symbol], "BACKTEST_END")

        # Generate results
        result = self._generate_result()

        # Ollama review if enabled
        if self.ollama_review:
            self._request_ollama_review(result)

        return result

    def _init_brain(self):
        """Initialize HybridBrain for backtest."""
        try:
            from Python.hybrid_brain import HybridBrain
            from Python.risk_engine import RiskEngine

            class MockExecutor:
                def reconcile_exposure(self, *args, **kwargs):
                    pass

            self.risk = RiskEngine()
            self.brain = HybridBrain(risk=self.risk, executor=MockExecutor())
            logger.success("HybridBrain initialized for backtest")

        except Exception as e:
            logger.error(f"Failed to initialize brain: {e}")
            raise

    def _update_equity(self, timestamp: datetime):
        """Update equity curve and drawdown."""
        self.equity_curve.append((timestamp, self.equity))

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        self.current_drawdown = (self.peak_equity - self.equity) / self.peak_equity * 100
        self.drawdown_curve.append((timestamp, self.current_drawdown))

    def _check_exits(self, symbol: str, df: pd.DataFrame):
        """Check if any positions should be closed."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        current_price = df['close'].iloc[-1]

        # Check SL/TP
        if pos['action'] == 'BUY':
            if current_price <= pos['sl']:
                self._close_position(symbol, df, 'SL')
            elif current_price >= pos['tp']:
                self._close_position(symbol, df, 'TP')
        else:  # SELL
            if current_price >= pos['sl']:
                self._close_position(symbol, df, 'SL')
            elif current_price <= pos['tp']:
                self._close_position(symbol, df, 'TP')

        # Check max holding period (24 hours = 288 5-min bars)
        if pos.get('bars_held', 0) > 288:
            self._close_position(symbol, df, 'TIMEOUT')
        else:
            pos['bars_held'] = pos.get('bars_held', 0) + 1

    def _process_signal(self, symbol: str, df: pd.DataFrame):
        """Process trading signal."""
        # Skip if max positions reached
        if len(self.positions) >= self.max_positions:
            return

        # Skip if already in position
        if symbol in self.positions:
            return

        # Get decision from brain
        try:
            decision = self.brain.decide(symbol, df)
        except Exception as e:
            logger.debug(f"Brain decision failed: {e}")
            return

        action = decision.get('action', 'HOLD')
        if action == 'HOLD':
            return

        # Check signal quality
        quality_score = decision.get('signal_quality_score', 0.5)
        if quality_score < 0.65:  # Minimum quality threshold
            logger.debug(f"Signal quality too low: {quality_score:.2f}")
            return

        # Calculate position size
        exposure = decision.get('exposure', 0)
        kelly_mult = decision.get('kelly_multiplier', 1.0)

        # Base lot size on exposure and equity
        lot_size = abs(exposure) * (self.equity / 10000) * kelly_mult
        lot_size = max(0.01, min(lot_size, 2.0))  # Clamp

        # Get current price
        current_price = df['close'].iloc[-1]

        # Calculate SL/TP
        atr = self._calculate_atr(df)
        if action == 'BUY':
            sl = current_price - atr * 2
            tp = current_price + atr * 3
        else:
            sl = current_price + atr * 2
            tp = current_price - atr * 3

        # Execute entry
        entry_price, spread_paid = self.executor.execute_entry(
            current_price, action, volatility=atr/current_price
        )

        # Record position
        self.positions[symbol] = {
            'action': action,
            'entry_price': entry_price,
            'lot_size': lot_size,
            'sl': sl,
            'tp': tp,
            'entry_time': df.index[-1],
            'bars_held': 0,
            'signal_quality_score': quality_score,
            'kelly_fraction': kelly_mult,
            'trend_flip': decision.get('trend_flip', 'flat'),
            'regime': decision.get('lstm_regime', 'UNKNOWN'),
            'confidence': decision.get('confidence', 0.5),
            'spread_paid': spread_paid,
        }

        logger.info(
            f"ENTER {symbol} {action} @ {entry_price:.5f} "
            f"lots={lot_size:.2f} sq={quality_score:.2f}"
        )

    def _close_position(self, symbol: str, df: pd.DataFrame, reason: str):
        """Close a position."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        current_price = df['close'].iloc[-1]

        # Execute exit
        exit_price, pnl_pct = self.executor.execute_exit(
            pos['entry_price'], current_price, pos['action'], reason
        )

        # Calculate PnL
        position_value = pos['lot_size'] * 100000  # Standard lot
        gross_pnl = position_value * pnl_pct
        commission = self.executor.calculate_commission(pos['lot_size'])
        net_pnl = gross_pnl - commission - pos['spread_paid']
        net_pnl_pct = net_pnl / self.equity * 100

        # Update equity
        self.equity += net_pnl

        # Record trade
        trade = TradeRecord(
            timestamp=df.index[-1],
            symbol=symbol,
            action=pos['action'],
            entry_price=pos['entry_price'],
            exit_price=exit_price,
            lot_size=pos['lot_size'],
            sl=pos['sl'],
            tp=pos['tp'],
            pnl=net_pnl,
            pnl_pct=net_pnl_pct,
            holding_bars=pos['bars_held'],
            exit_reason=reason,
            signal_quality_score=pos['signal_quality_score'],
            kelly_fraction=pos['kelly_fraction'],
            trend_flip=pos['trend_flip'],
            regime=pos['regime'],
            confidence=pos['confidence'],
        )
        self.trades.append(trade)

        # Update symbol stats
        self.symbol_stats[symbol]['trades'] += 1
        self.symbol_stats[symbol]['pnl'] += net_pnl
        if net_pnl > 0:
            self.symbol_stats[symbol]['wins'] += 1
        else:
            self.symbol_stats[symbol]['losses'] += 1

        logger.info(
            f"EXIT {symbol} {pos['action']} @ {exit_price:.5f} "
            f"reason={reason} pnl=${net_pnl:.2f} ({net_pnl_pct:.2f}%)"
        )

        del self.positions[symbol]

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR."""
        highs = df['high'].values[-period-1:]
        lows = df['low'].values[-period-1:]
        closes = df['close'].values[-period-1:]

        tr_list = []
        for i in range(1, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_list.append(tr)

        return np.mean(tr_list) if tr_list else closes[-1] * 0.001

    def _generate_result(self) -> BacktestResult:
        """Generate backtest results."""
        if not self.trades:
            logger.warning("No trades generated during backtest")
            return BacktestResult(
                start_date=self.start_date,
                end_date=self.end_date,
                symbols=self.symbols,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                expectancy=0.0,
                total_return_pct=0.0,
                annualized_return_pct=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown_pct=0.0,
                calmar_ratio=0.0,
                volatility_annual=0.0,
                var_95=0.0,
                tail_ratio=0.0,
                avg_signal_quality=0.0,
                kelly_effectiveness=0.0,
                cost_savings_pct=0.0,
            )

        # Calculate metrics
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t.pnl > 0)
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        wins = [t.pnl for t in self.trades if t.pnl > 0]
        losses = [t.pnl for t in self.trades if t.pnl < 0]

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss)) if total_trades > 0 else 0

        # Returns
        total_return_pct = (self.equity - self.initial_equity) / self.initial_equity * 100
        days = (self.end_date - self.start_date).days
        annualized_return_pct = total_return_pct * (365 / days) if days > 0 else 0

        # Drawdown
        max_drawdown_pct = max(dd for _, dd in self.drawdown_curve) if self.drawdown_curve else 0

        # Volatility
        returns = [t.pnl_pct for t in self.trades]
        volatility_annual = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0

        # Sharpe (assuming risk-free rate = 0)
        sharpe_ratio = annualized_return_pct / volatility_annual if volatility_annual > 0 else 0

        # Sortino (downside deviation only)
        downside_returns = [r for r in returns if r < 0]
        downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 1 else 0.001
        sortino_ratio = annualized_return_pct / downside_std if downside_std > 0 else 0

        # Calmar
        calmar_ratio = annualized_return_pct / max_drawdown_pct if max_drawdown_pct > 0 else 0

        # VaR
        var_95 = np.percentile(returns, 5) if len(returns) > 10 else 0

        # Tail ratio
        tail_95 = np.percentile(returns, 5)
        tail_5 = np.percentile(returns, 95)
        tail_ratio = abs(tail_5 / tail_95) if tail_95 != 0 else 0

        # Optimization metrics
        avg_signal_quality = np.mean([t.signal_quality_score for t in self.trades])
        kelly_effectiveness = np.mean([t.kelly_fraction for t in self.trades])

        # Cost savings estimate (comparing to no optimization)
        cost_savings_pct = (1 - avg_signal_quality) * -0.5  # Rough estimate

        return BacktestResult(
            start_date=self.start_date,
            end_date=self.end_date,
            symbols=self.symbols,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            total_return_pct=total_return_pct,
            annualized_return_pct=annualized_return_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown_pct=max_drawdown_pct,
            calmar_ratio=calmar_ratio,
            volatility_annual=volatility_annual,
            var_95=var_95,
            tail_ratio=tail_ratio,
            avg_signal_quality=avg_signal_quality,
            kelly_effectiveness=kelly_effectiveness,
            cost_savings_pct=cost_savings_pct,
            trades=self.trades,
            equity_curve=self.equity_curve,
            drawdown_curve=self.drawdown_curve,
        )

    def _request_ollama_review(self, result: BacktestResult):
        """Request Ollama LLM review of backtest results."""
        try:
            from Python.ollama_advisor import make_advisor

            advisor = make_advisor()

            # Prepare summary
            summary = {
                "total_trades": result.total_trades,
                "win_rate": f"{result.win_rate:.1%}",
                "profit_factor": f"{result.profit_factor:.2f}",
                "sharpe_ratio": f"{result.sharpe_ratio:.2f}",
                "max_drawdown": f"{result.max_drawdown_pct:.1f}%",
                "total_return": f"{result.total_return_pct:.1f}%",
                "avg_signal_quality": f"{result.avg_signal_quality:.2f}",
                "expectancy": f"${result.expectancy:.2f}",
            }

            # Get review
            review = advisor.review_risk_state({
                "backtest_summary": summary,
                "symbol_performance": self.symbol_stats,
            })

            if review.get("ok"):
                logger.success(f"Ollama Review: {review.get('status', 'unknown')}")
                if "recommended_operator_action" in review:
                    logger.info(f"Recommendation: {review['recommended_operator_action']}")
            else:
                logger.warning(f"Ollama review failed: {review.get('error', 'unknown')}")

        except Exception as e:
            logger.debug(f"Ollama review not available: {e}")


def generate_backtest_report(result: BacktestResult, output_path: Path):
    """Generate comprehensive backtest report."""

    report_lines = [
        "=" * 80,
        "CHAIN GAMBLER BACKTEST REPORT",
        "=" * 80,
        f"Generated: {datetime.now().isoformat()}",
        f"Period: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}",
        f"Symbols: {', '.join(result.symbols)}",
        "",
        "-" * 80,
        "PERFORMANCE METRICS",
        "-" * 80,
        f"Total Trades: {result.total_trades}",
        f"Win Rate: {result.win_rate:.1%} ({result.winning_trades}/{result.total_trades})",
        f"Profit Factor: {result.profit_factor:.2f}",
        f"Expectancy: ${result.expectancy:.2f} per trade",
        "",
        "-" * 80,
        "RETURN METRICS",
        "-" * 80,
        f"Total Return: {result.total_return_pct:.2f}%",
        f"Annualized Return: {result.annualized_return_pct:.2f}%",
        f"Sharpe Ratio: {result.sharpe_ratio:.2f}",
        f"Sortino Ratio: {result.sortino_ratio:.2f}",
        f"Max Drawdown: {result.max_drawdown_pct:.2f}%",
        f"Calmar Ratio: {result.calmar_ratio:.2f}",
        "",
        "-" * 80,
        "RISK METRICS",
        "-" * 80,
        f"Annual Volatility: {result.volatility_annual:.2f}%",
        f"Value at Risk (95%): {result.var_95:.2f}%",
        f"Tail Ratio: {result.tail_ratio:.2f}",
        "",
        "-" * 80,
        "OPTIMIZATION METRICS",
        "-" * 80,
        f"Avg Signal Quality: {result.avg_signal_quality:.2f}",
        f"Kelly Effectiveness: {result.kelly_effectiveness:.2f}",
        f"Cost Savings Est: {result.cost_savings_pct:.2f}%",
        "",
        "=" * 80,
    ]

    report_text = "\n".join(report_lines)

    with open(output_path, 'w') as f:
        f.write(report_text)

    logger.success(f"Backtest report saved to {output_path}")
    return report_text


def main():
    """Run backtest from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Chain Gambler Backtest Engine")
    parser.add_argument("--symbols", nargs="+", default=["EURUSDm", "GBPUSDm", "XAUUSDm"])
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--equity", type=float, default=10000.0)
    parser.add_argument("--output", type=str, default="backtest_results")
    parser.add_argument("--no-ollama", action="store_true", help="Disable Ollama review")

    args = parser.parse_args()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)

    # Run backtest
    engine = BacktestEngine(
        symbols=args.symbols,
        start_date=start_date,
        end_date=end_date,
        initial_equity=args.equity,
        ollama_review=not args.no_ollama,
    )

    result = engine.run_backtest()

    # Generate report
    output_dir = PROJECT_ROOT / "backtests" / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report = generate_backtest_report(result, report_path)

    # Print summary
    print("\n" + report)

    # Save trades to CSV
    if result.trades:
        trades_df = pd.DataFrame([
            {
                'timestamp': t.timestamp,
                'symbol': t.symbol,
                'action': t.action,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'lot_size': t.lot_size,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'exit_reason': t.exit_reason,
                'signal_quality': t.signal_quality_score,
                'kelly_fraction': t.kelly_fraction,
            }
            for t in result.trades
        ])
        trades_path = output_dir / "trades.csv"
        trades_df.to_csv(trades_path, index=False)
        logger.success(f"Trades saved to {trades_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
