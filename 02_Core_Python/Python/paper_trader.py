"""
Paper Trading Environment for Chain Gambler

Simulates live trading without real money:
- Real-time signal generation
- Mock execution with realistic fills (latency + slippage simulation)
- Portfolio tracking
- Ollama LLM oversight and review
- Daily performance reports
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Import speed simulator for realistic execution
try:
    from Python.speed_simulator import SpeedSimulator, get_speed_simulator
    _HAS_SPEED_SIM = True
except ImportError:
    _HAS_SPEED_SIM = False
    logger.warning("Speed simulator not available")


@dataclass
class PaperPosition:
    """Paper trading position."""
    symbol: str
    action: str  # BUY/SELL
    entry_price: float
    lot_size: float
    sl: float
    tp: float
    entry_time: datetime
    signal_quality: float
    kelly_fraction: float
    decision_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PaperTrade:
    """Completed paper trade."""
    symbol: str
    action: str
    entry_price: float
    exit_price: float
    lot_size: float
    pnl: float
    pnl_pct: float
    holding_time: timedelta
    exit_reason: str
    entry_time: datetime
    exit_time: datetime
    signal_quality: float
    kelly_fraction: float


class PaperPortfolio:
    """Paper trading portfolio tracker."""

    def __init__(self, initial_equity: float = 10000.0):
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        self.peak_equity = initial_equity
        self.max_drawdown = 0.0

        self.positions: Dict[str, PaperPosition] = {}
        self.trades: List[PaperTrade] = []
        self.equity_curve: List[tuple] = []

        # Daily tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset = datetime.now(timezone.utc)

        logger.success(f"PaperPortfolio initialized: ${initial_equity:.2f}")

    def open_position(self, position: PaperPosition):
        """Record position opening."""
        self.positions[position.symbol] = position
        logger.info(
            f"[PAPER] OPEN {position.symbol} {position.action} "
            f"@ {position.entry_price:.5f} lots={position.lot_size:.2f}"
        )

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str,
    ) -> Optional[PaperTrade]:
        """Close position and record trade."""
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]

        # Calculate PnL
        if pos.action == "BUY":
            pnl = (exit_price - pos.entry_price) * pos.lot_size * 100000
        else:
            pnl = (pos.entry_price - exit_price) * pos.lot_size * 100000

        pnl_pct = pnl / self.current_equity * 100

        trade = PaperTrade(
            symbol=symbol,
            action=pos.action,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            lot_size=pos.lot_size,
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_time=exit_time - pos.entry_time,
            exit_reason=exit_reason,
            entry_time=pos.entry_time,
            exit_time=exit_time,
            signal_quality=pos.signal_quality,
            kelly_fraction=pos.kelly_fraction,
        )

        # Update portfolio
        self.current_equity += pnl
        self.trades.append(trade)
        del self.positions[symbol]

        # Update stats
        self.daily_pnl += pnl
        self.daily_trades += 1

        # Update equity curve
        self.equity_curve.append((exit_time, self.current_equity))

        # Check drawdown
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
        drawdown = (self.peak_equity - self.current_equity) / self.peak_equity * 100
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        logger.info(
            f"[PAPER] CLOSE {symbol} {pos.action} @ {exit_price:.5f} "
            f"pnl=${pnl:.2f} ({pnl_pct:.2f}%) reason={exit_reason}"
        )

        return trade

    def get_stats(self) -> Dict[str, Any]:
        """Get current portfolio stats."""
        total_return = (self.current_equity - self.initial_equity) / self.initial_equity * 100

        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]

        win_rate = len(wins) / len(self.trades) if self.trades else 0

        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0

        return {
            "equity": self.current_equity,
            "total_return_pct": total_return,
            "max_drawdown_pct": self.max_drawdown,
            "total_trades": len(self.trades),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "open_positions": len(self.positions),
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
        }

    def reset_daily(self):
        """Reset daily counters."""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset = datetime.now(timezone.utc)


class OllamaOverseer:
    """
    Ollama LLM oversight for paper trading.

    Reviews trading decisions and provides guidance.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.advisor = None
        self._init_advisor()

    def _init_advisor(self):
        """Initialize Ollama advisor."""
        if not self.enabled:
            return

        try:
            from Python.ollama_advisor import make_advisor
            self.advisor = make_advisor()
            logger.success("OllamaOverseer initialized")
        except Exception as e:
            logger.warning(f"Ollama advisor not available: {e}")
            self.enabled = False

    def review_trade_decision(
        self,
        symbol: str,
        decision: Dict[str, Any],
        portfolio_stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Review a trading decision before execution.

        Returns:
            Review with approval status and comments.
        """
        if not self.enabled or self.advisor is None:
            return {"approved": True, "reason": "No oversight available"}

        try:
            review = self.advisor.explain_trade_decision({
                "symbol": symbol,
                "action": decision.get("action"),
                "exposure": decision.get("exposure"),
                "signal_quality": decision.get("signal_quality_score", 0),
                "confidence": decision.get("confidence", 0),
                "regime": decision.get("lstm_regime"),
                "portfolio_equity": portfolio_stats.get("equity"),
                "daily_pnl": portfolio_stats.get("daily_pnl"),
            })

            # Parse approval
            actionability = review.get("actionability", "monitor")
            approved = actionability in ["approve", "monitor"]

            return {
                "approved": approved,
                "actionability": actionability,
                "risk_flags": review.get("risk_flags", []),
                "notes": review.get("notes", []),
                "confidence_comment": review.get("confidence_comment", ""),
                "raw_review": review,
            }

        except Exception as e:
            logger.warning(f"Trade review failed: {e}")
            return {"approved": True, "reason": f"Review error: {e}"}

    def daily_review(self, portfolio: PaperPortfolio) -> Dict[str, Any]:
        """Generate daily review of trading activity."""
        if not self.enabled or self.advisor is None:
            return {"ok": False, "reason": "Ollama not available"}

        try:
            stats = portfolio.get_stats()

            # Get recent trades
            recent_trades = portfolio.trades[-20:] if len(portfolio.trades) > 20 else portfolio.trades

            review = self.advisor.review_risk_state({
                "equity": stats["equity"],
                "total_return": stats["total_return_pct"],
                "max_drawdown": stats["max_drawdown_pct"],
                "win_rate": stats["win_rate"],
                "daily_pnl": stats["daily_pnl"],
                "daily_trades": stats["daily_trades"],
                "recent_trades": [
                    {
                        "symbol": t.symbol,
                        "pnl": t.pnl,
                        "quality": t.signal_quality,
                    }
                    for t in recent_trades
                ],
            })

            return review

        except Exception as e:
            logger.warning(f"Daily review failed: {e}")
            return {"ok": False, "error": str(e)}

    def analyze_performance(self, trades: List[PaperTrade]) -> Dict[str, Any]:
        """Analyze performance patterns in trades."""
        if not self.enabled or self.advisor is None:
            return {"ok": False, "reason": "Ollama not available"}

        try:
            # Group by signal quality
            high_quality = [t for t in trades if t.signal_quality >= 0.7]
            low_quality = [t for t in trades if t.signal_quality < 0.7]

            analysis = {
                "high_quality_win_rate": len([t for t in high_quality if t.pnl > 0]) / len(high_quality) if high_quality else 0,
                "low_quality_win_rate": len([t for t in low_quality if t.pnl > 0]) / len(low_quality) if low_quality else 0,
                "avg_quality": sum(t.signal_quality for t in trades) / len(trades) if trades else 0,
                "kelly_avg": sum(t.kelly_fraction for t in trades) / len(trades) if trades else 0,
            }

            return {"ok": True, "analysis": analysis}

        except Exception as e:
            return {"ok": False, "error": str(e)}


class PaperTrader:
    """
    Main paper trading system.
    """

    def __init__(
        self,
        symbols: List[str],
        initial_equity: float = 10000.0,
        ollama_oversight: bool = True,
    ):
        self.symbols = symbols
        self.portfolio = PaperPortfolio(initial_equity)
        self.overseer = OllamaOverseer(enabled=ollama_oversight)

        # Initialize brain
        self._init_brain()

        # Initialize speed simulator for realistic execution
        if _HAS_SPEED_SIM:
            network_profile = os.environ.get("AGI_NETWORK_PROFILE", "good")
            broker_profile = os.environ.get("AGI_BROKER_PROFILE", "mm_premium")
            self.speed_sim = SpeedSimulator(
                network_profile=network_profile,
                broker_profile=broker_profile
            )
            logger.success(f"Speed simulator enabled: {network_profile}/{broker_profile}")
        else:
            self.speed_sim = None

        # Tracking
        self.running = False
        self.cycle_count = 0
        self.last_daily_report = datetime.now(timezone.utc)

    def _init_brain(self):
        """Initialize trading brain."""
        try:
            from Python.hybrid_brain import HybridBrain
            from Python.risk_engine import RiskEngine

            class MockExecutor:
                def reconcile_exposure(self, *args, **kwargs):
                    pass

            self.risk = RiskEngine()
            self.brain = HybridBrain(risk=self.risk, executor=MockExecutor())
            logger.success("Brain initialized for paper trading")

        except Exception as e:
            logger.error(f"Failed to initialize brain: {e}")
            raise

    def run_cycle(self):
        """Run one trading cycle."""
        try:
            from Python.data_feed import get_latest_data

            current_time = datetime.now(timezone.utc)

            # Check daily report
            if (current_time - self.last_daily_report).days >= 1:
                self._generate_daily_report()
                self.portfolio.reset_daily()
                self.last_daily_report = current_time

            # Process each symbol
            for symbol in self.symbols:
                # Get latest data
                try:
                    df = get_latest_data(symbol, timeframe="M5", bars=110)
                    if df is None or len(df) < 100:
                        continue
                except Exception as e:
                    logger.debug(f"Data fetch failed for {symbol}: {e}")
                    continue

                # Update portfolio equity
                self.risk.update_equity(self.portfolio.current_equity)

                # Check for exits
                self._check_exits(symbol, df)

                # Generate signal
                self._process_signal(symbol, df)

            self.cycle_count += 1

        except Exception as e:
            logger.error(f"Cycle error: {e}")
            import traceback
            traceback.print_exc()

    def _check_exits(self, symbol: str, df: pd.DataFrame):
        """Check if position should be closed."""
        if symbol not in self.portfolio.positions:
            return

        pos = self.portfolio.positions[symbol]
        current_price = df['close'].iloc[-1]
        current_time = datetime.now(timezone.utc)

        exit_triggered = None
        exit_reason = None

        # Check SL/TP
        if pos.action == "BUY":
            if current_price <= pos.sl:
                exit_triggered = True
                exit_reason = "SL"
            elif current_price >= pos.tp:
                exit_triggered = True
                exit_reason = "TP"
        else:  # SELL
            if current_price >= pos.sl:
                exit_triggered = True
                exit_reason = "SL"
            elif current_price <= pos.tp:
                exit_triggered = True
                exit_reason = "TP"

        # Check timeout (4 hours)
        if (current_time - pos.entry_time).total_seconds() > 4 * 3600:
            exit_triggered = True
            exit_reason = "TIMEOUT"

        if exit_triggered:
            # Simulate exit execution with slippage
            exit_price = self._simulate_exit_execution(
                symbol, pos.action, pos.lot_size, current_price, df
            )
            self._close_position(symbol, exit_price, current_time, exit_reason)

    def _simulate_exit_execution(
        self,
        symbol: str,
        action: str,
        lot_size: float,
        requested_price: float,
        df: pd.DataFrame
    ) -> float:
        """Simulate exit execution with realistic slippage."""
        if self.speed_sim is None:
            return requested_price

        # Determine market conditions from dataframe
        volatility = "MEDIUM"
        regime = "trending"

        # Simple volatility detection
        if len(df) >= 20:
            returns = df['close'].pct_change().dropna()
            vol_std = returns.std() * 100  # vol as percentage
            if vol_std > 0.5:
                volatility = "HIGH"
            elif vol_std < 0.2:
                volatility = "LOW"

        exec_result = self.speed_sim.simulate_execution(
            symbol=symbol,
            order_type="MARKET",
            side="SELL" if action == "BUY" else "BUY",  # Close position = opposite side
            size=lot_size,
            requested_price=requested_price,
            market_volatility=volatility,
            market_regime=regime,
        )

        if exec_result.success and exec_result.filled:
            logger.debug(
                f"[PAPER] Exit simulated: {symbol} @ {exec_result.fill_price:.5f} "
                f"(slip: {exec_result.slippage:.2f}pips)"
            )
            return exec_result.fill_price
        else:
            # Execution failed - use requested price as fallback
            logger.warning(f"[PAPER] Exit execution failed for {symbol}, using market price")
            return requested_price

    def _process_signal(self, symbol: str, df: pd.DataFrame):
        """Process trading signal."""
        # Check if already in position
        if symbol in self.portfolio.positions:
            return

        # Check risk
        if not self.risk.can_trade(symbol):
            return

        # Get decision
        try:
            decision = self.brain.decide(symbol, df)
        except Exception as e:
            logger.debug(f"Decision failed: {e}")
            return

        action = decision.get("action", "HOLD")
        if action == "HOLD":
            return

        # Check signal quality
        quality = decision.get("signal_quality_score", 0)
        if quality < 0.65:
            logger.debug(f"Quality too low: {quality:.2f}")
            return

        # Ollama review
        stats = self.portfolio.get_stats()
        review = self.overseer.review_trade_decision(symbol, decision, stats)

        if not review.get("approved", True):
            logger.info(
                f"Trade rejected by overseer: {review.get('actionability')} - "
                f"{review.get('notes', [])}"
            )
            return

        # Calculate position
        exposure = decision.get("exposure", 0)
        kelly_mult = decision.get("kelly_multiplier", 1.0)

        lot_size = abs(exposure) * (self.portfolio.current_equity / 10000) * kelly_mult
        lot_size = max(0.01, min(lot_size, 2.0))

        # Get prices
        current_price = df['close'].iloc[-1]

        # Simulate realistic execution with speed simulator
        if self.speed_sim is not None:
            exec_result = self.speed_sim.simulate_execution(
                symbol=symbol,
                order_type="MARKET",
                side=action,
                size=lot_size,
                requested_price=current_price,
                market_volatility=decision.get("volatility", "MEDIUM"),
                market_regime=decision.get("regime_note", "trending"),
            )

            if not exec_result.success or not exec_result.filled:
                logger.warning(
                    f"[PAPER] Execution failed for {symbol}: {exec_result.reject_reason} "
                    f"(latency: {exec_result.latency_ms}ms)"
                )
                return

            entry_price = exec_result.fill_price
            execution_slippage = exec_result.slippage
            execution_latency = exec_result.latency_ms

            logger.info(
                f"[PAPER] Execution simulated: {symbol} {action} "
                f"@ {entry_price:.5f} (slip: {execution_slippage:.2f}pips, "
                f"latency: {execution_latency}ms)"
            )
        else:
            entry_price = current_price
            execution_slippage = 0.0
            execution_latency = 0

        atr = self._calculate_atr(df)

        if action == "BUY":
            sl = entry_price - atr * 2
            tp = entry_price + atr * 3
        else:
            sl = entry_price + atr * 2
            tp = entry_price - atr * 3

        # Create position
        position = PaperPosition(
            symbol=symbol,
            action=action,
            entry_price=entry_price,
            lot_size=lot_size,
            sl=sl,
            tp=tp,
            entry_time=datetime.now(timezone.utc),
            signal_quality=quality,
            kelly_fraction=kelly_mult,
            decision_data=decision,
        )

        # Add execution metadata
        position.decision_data["execution_slippage_pips"] = execution_slippage
        position.decision_data["execution_latency_ms"] = execution_latency
        position.decision_data["requested_price"] = current_price

        self.portfolio.open_position(position)

    def _close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_time: datetime,
        reason: str,
    ):
        """Close position."""
        trade = self.portfolio.close_position(symbol, exit_price, exit_time, reason)

        if trade:
            # Log to file
            self._log_trade(trade)

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR."""
        try:
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

            return sum(tr_list) / len(tr_list) if tr_list else closes[-1] * 0.001
        except (IndexError, KeyError, ValueError) as e:
            logger.debug(f"ATR calculation failed: {e}, using fallback")
            return df['close'].iloc[-1] * 0.001

    def _log_trade(self, trade: PaperTrade):
        """Log trade to file."""
        log_dir = PROJECT_ROOT / "logs" / "paper_trades"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"{trade.exit_time.strftime('%Y%m%d')}.jsonl"

        record = {
            "timestamp": trade.exit_time.isoformat(),
            "symbol": trade.symbol,
            "action": trade.action,
            "entry": trade.entry_price,
            "exit": trade.exit_price,
            "pnl": trade.pnl,
            "pnl_pct": trade.pnl_pct,
            "reason": trade.exit_reason,
            "quality": trade.signal_quality,
            "kelly": trade.kelly_fraction,
        }

        with open(log_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _generate_daily_report(self):
        """Generate and log daily report."""
        stats = self.portfolio.get_stats()

        logger.info("=" * 60)
        logger.info("DAILY PAPER TRADING REPORT")
        logger.info("=" * 60)
        logger.info(f"Equity: ${stats['equity']:.2f}")
        logger.info(f"Daily PnL: ${stats['daily_pnl']:.2f}")
        logger.info(f"Daily Trades: {stats['daily_trades']}")
        logger.info(f"Total Return: {stats['total_return_pct']:.2f}%")
        logger.info(f"Max Drawdown: {stats['max_drawdown_pct']:.2f}%")
        logger.info(f"Win Rate: {stats['win_rate']:.1%}")
        logger.info("=" * 60)

        # Ollama review
        review = self.overseer.daily_review(self.portfolio)
        if review.get("ok"):
            logger.info(f"Ollama Status: {review.get('status', 'unknown')}")
            if "recommended_operator_action" in review:
                logger.info(f"Recommendation: {review['recommended_operator_action']}")

    def start(self):
        """Start paper trading."""
        self.running = True
        logger.success("Paper trading started")

        while self.running:
            self.run_cycle()
            time.sleep(30)  # Run every 30 seconds

    def stop(self):
        """Stop paper trading."""
        self.running = False
        logger.success("Paper trading stopped")

    def get_report(self) -> str:
        """Get current trading report."""
        stats = self.portfolio.get_stats()

        lines = [
            "=" * 60,
            "PAPER TRADING REPORT",
            "=" * 60,
            f"Equity: ${stats['equity']:.2f}",
            f"Return: {stats['total_return_pct']:.2f}%",
            f"Max DD: {stats['max_drawdown_pct']:.2f}%",
            f"Trades: {stats['total_trades']}",
            f"Win Rate: {stats['win_rate']:.1%}",
            f"Daily PnL: ${stats['daily_pnl']:.2f}",
            f"Open Pos: {stats['open_positions']}",
            "=" * 60,
        ]

        return "\n".join(lines)


def main():
    """Run paper trading."""
    import argparse

    parser = argparse.ArgumentParser(description="Chain Gambler Paper Trading")
    parser.add_argument("--symbols", nargs="+", default=["EURUSDm", "GBPUSDm", "XAUUSDm"])
    parser.add_argument("--equity", type=float, default=10000.0)
    parser.add_argument("--no-ollama", action="store_true", help="Disable Ollama oversight")
    parser.add_argument("--cycles", type=int, default=0, help="Number of cycles (0=infinite)")

    args = parser.parse_args()

    trader = PaperTrader(
        symbols=args.symbols,
        initial_equity=args.equity,
        ollama_oversight=not args.no_ollama,
    )

    try:
        if args.cycles > 0:
            for _ in range(args.cycles):
                trader.run_cycle()
                time.sleep(30)
            print(trader.get_report())
        else:
            trader.start()
    except KeyboardInterrupt:
        logger.info("Stopping...")
        trader.stop()
        print(trader.get_report())


if __name__ == "__main__":
    main()
