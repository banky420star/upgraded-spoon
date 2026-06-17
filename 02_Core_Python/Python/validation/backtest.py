"""
BacktestCourt — Realistic bar-by-bar backtest simulator.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from Python.validation.cost_model import CostModel
from Python.data.symbol_metadata import ContractSpec, get_contract, default_contract


@dataclass
class TradeRecord:
    entry_time: datetime
    exit_time: datetime
    symbol: str
    side: str  # LONG or SHORT
    entry_price: float
    exit_price: float
    volume: float
    gross_pnl: float
    cost: float
    net_pnl: float
    exit_reason: str
    hold_bars: int


class BacktestCourt:
    """
    Simulates trading with realistic costs.
    """

    def __init__(
        self,
        cost_model: Optional[CostModel] = None,
        initial_equity: float = 10_000.0,
        position_size_fraction: float = 0.01,
        max_positions: int = 1,
        risk_halt_drawdown: float = 0.15,
        sl_atr_multiplier: float = 2.0,
        tp_atr_multiplier: float = 3.0,
        trailing_trigger_pct: float = 0.003,
        trailing_distance_pct: float = 0.002,
        max_hold_bars: int = 288,
        min_profit_factor: float = 1.2,
        max_drawdown_threshold: float = 0.15,
        min_trade_count: int = 20,
        max_single_trade_profit_share: float = 0.40,
    ):
        self.cost_model = cost_model or CostModel()
        self.initial_equity = float(initial_equity)
        self.position_size_fraction = float(position_size_fraction)
        self.max_positions = int(max_positions)
        self.risk_halt_drawdown = float(risk_halt_drawdown)
        self.sl_atr_multiplier = float(sl_atr_multiplier)
        self.tp_atr_multiplier = float(tp_atr_multiplier)
        self.trailing_trigger_pct = float(trailing_trigger_pct)
        self.trailing_distance_pct = float(trailing_distance_pct)
        self.max_hold_bars = int(max_hold_bars)
        self.min_profit_factor = float(min_profit_factor)
        self.max_drawdown_threshold = float(max_drawdown_threshold)
        self.min_trade_count = int(min_trade_count)
        self.max_single_trade_profit_share = float(max_single_trade_profit_share)

    def _get_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        highs = df["high"].values[-period - 1 :]
        lows = df["low"].values[-period - 1 :]
        closes = df["close"].values[-period - 1 :]
        trs = []
        for i in range(1, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            trs.append(tr)
        return float(np.mean(trs)) if trs else float(closes[-1] * 0.001)

    def _compute_position_size(self, equity: float, price: float, symbol: str, atr: float) -> float:
        spec = get_contract(symbol) or default_contract()
        risk_amount = equity * self.position_size_fraction
        # Risk-based sizing: risk 1% of equity over the SL distance
        sl_distance = atr * self.sl_atr_multiplier
        if sl_distance <= 0:
            sl_distance = price * 0.001
        notional = risk_amount / (sl_distance / price)
        lots = notional / (price * spec.contract_size)
        lots = max(spec.min_lot, min(lots, spec.max_lot))
        # Round to lot step
        lots = round(lots / spec.lot_step) * spec.lot_step
        lots = max(spec.min_lot, min(lots, spec.max_lot))
        return float(lots)

    def _entry_cost(self, symbol: str, price: float, volume: float) -> float:
        cb = self.cost_model.compute_cost(symbol, "BUY", price, volume)
        return cb.total_cost

    def _exit_cost(self, symbol: str, price: float, volume: float) -> float:
        cb = self.cost_model.compute_cost(symbol, "SELL", price, volume)
        return cb.total_cost

    def run(
        self,
        df: pd.DataFrame,
        symbol: str,
        bundle_id: str = "",
        signals: Optional[pd.Series] = None,
        policy: Optional[Callable[[pd.DataFrame, int, Dict[str, Any]], Dict[str, Any]]] = None,
        backtest_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run backtest on a DataFrame.

        Args:
            df: OHLCV DataFrame with columns open, high, low, close, volume
            symbol: trading symbol
            bundle_id: identifier for the model/training bundle
            signals: optional pd.Series aligned to df index with values {-1, 0, 1}
            policy: optional callable(df_window, bar_index, context) -> dict
                    expected keys: action ('BUY'|'SELL'|'HOLD'|'FLAT'), size (optional float)
            backtest_id: optional unique id; generated if not provided

        Returns:
            JSON-serializable backtest artifact dict.
        """
        if signals is None and policy is None:
            raise ValueError("Provide either signals or policy")

        backtest_id = backtest_id or str(uuid.uuid4())
        df = df.copy()
        df = df.reset_index(drop=True)
        n = len(df)

        equity = float(self.initial_equity)
        peak_equity = equity
        equity_curve: List[float] = [equity]
        drawdown_curve: List[float] = [0.0]

        trades: List[TradeRecord] = []
        position: Optional[Dict[str, Any]] = None
        halted = False

        window_size = max(50, min(200, n // 10))

        for i in range(window_size, n):
            if halted:
                equity_curve.append(equity)
                dd = (peak_equity - equity) / peak_equity
                drawdown_curve.append(dd)
                continue

            row = df.iloc[i]
            price = float(row["close"])
            window = df.iloc[max(0, i - window_size) : i + 1]
            atr = self._get_atr(window)

            # Resolve signal
            action = "HOLD"
            if signals is not None:
                sig = float(signals.iloc[i])
                if sig > 0.1:
                    action = "BUY"
                elif sig < -0.1:
                    action = "SELL"
                elif position is not None and abs(sig) < 0.1:
                    action = "FLAT"
            elif policy is not None:
                ctx = {
                    "equity": equity,
                    "peak_equity": peak_equity,
                    "position": position,
                    "symbol": symbol,
                }
                try:
                    decision = policy(window, i, ctx)
                except Exception as exc:
                    logger.debug(f"Policy error at bar {i}: {exc}")
                    decision = {}
                action = str(decision.get("action", "HOLD")).upper()

            # Risk halt check
            dd = (peak_equity - equity) / peak_equity
            if dd >= self.risk_halt_drawdown:
                halted = True
                if position is not None:
                    equity = self._close_position(position, price, row, equity, trades, "RISK_HALT")
                position = None
                equity_curve.append(equity)
                drawdown_curve.append(dd)
                continue

            # Manage existing position
            if position is not None:
                pos = position
                pos["bars_held"] = pos.get("bars_held", 0) + 1

                # Trailing stop update
                if pos["side"] == "LONG":
                    mfe = (price - pos["entry_price"]) / pos["entry_price"]
                    if mfe >= self.trailing_trigger_pct:
                        new_sl = price * (1.0 - self.trailing_distance_pct)
                        if new_sl > pos["sl"]:
                            pos["sl"] = new_sl
                else:
                    mfe = (pos["entry_price"] - price) / pos["entry_price"]
                    if mfe >= self.trailing_trigger_pct:
                        new_sl = price * (1.0 + self.trailing_distance_pct)
                        if new_sl < pos["sl"]:
                            pos["sl"] = new_sl

                # Check exits
                exit_reason = None
                if pos["side"] == "LONG":
                    if price <= pos["sl"]:
                        exit_reason = "SL"
                    elif price >= pos["tp"]:
                        exit_reason = "TP"
                else:
                    if price >= pos["sl"]:
                        exit_reason = "SL"
                    elif price <= pos["tp"]:
                        exit_reason = "TP"

                if pos["bars_held"] >= self.max_hold_bars:
                    exit_reason = "TIMEOUT"

                # Flat signal
                if exit_reason is None and action == "FLAT":
                    exit_reason = "SIGNAL"

                # Opposite signal reverses position
                reverse = False
                if exit_reason is None:
                    if pos["side"] == "LONG" and action == "SELL":
                        exit_reason = "REVERSAL"
                        reverse = True
                    elif pos["side"] == "SHORT" and action == "BUY":
                        exit_reason = "REVERSAL"
                        reverse = True

                if exit_reason:
                    equity = self._close_position(pos, price, row, equity, trades, exit_reason)
                    position = None
                    if not reverse:
                        action = "HOLD"

                if reverse:
                    action = "SELL" if exit_reason == "REVERSAL" else action

            # Open new position
            if position is None and action in ("BUY", "SELL"):
                volume = self._compute_position_size(equity, price, symbol, atr)
                if volume > 0:
                    entry_cost = self._entry_cost(symbol, price, volume)
                    equity -= entry_cost
                    sl = price - atr * self.sl_atr_multiplier if action == "BUY" else price + atr * self.sl_atr_multiplier
                    tp = price + atr * self.tp_atr_multiplier if action == "BUY" else price - atr * self.tp_atr_multiplier
                    position = {
                        "side": "LONG" if action == "BUY" else "SHORT",
                        "entry_price": price,
                        "volume": volume,
                        "sl": sl,
                        "tp": tp,
                        "entry_time": row.get("time", pd.Timestamp.now()) if isinstance(row, pd.Series) else None,
                        "bars_held": 0,
                        "entry_cost": entry_cost,
                    }

            equity_curve.append(equity)
            if equity > peak_equity:
                peak_equity = equity
            dd = (peak_equity - equity) / peak_equity
            drawdown_curve.append(dd)

        # Close any remaining position at last price
        if position is not None:
            last_price = float(df.iloc[-1]["close"])
            last_row = df.iloc[-1]
            equity = self._close_position(position, last_price, last_row, equity, trades, "BACKTEST_END")
            position = None

        return self._build_artifact(
            backtest_id=backtest_id,
            bundle_id=bundle_id,
            symbol=symbol,
            equity_curve=equity_curve,
            drawdown_curve=drawdown_curve,
            trades=trades,
        )

    def _close_position(
        self,
        pos: Dict[str, Any],
        price: float,
        row: pd.Series,
        equity: float,
        trades: List[TradeRecord],
        reason: str,
    ) -> float:
        symbol = pos.get("symbol", "")
        spec = get_contract(symbol) or default_contract()
        side = pos["side"]
        entry_price = pos["entry_price"]
        volume = pos["volume"]

        if side == "LONG":
            gross_pnl = (price - entry_price) * volume * spec.contract_size
        else:
            gross_pnl = (entry_price - price) * volume * spec.contract_size

        exit_cost = self._exit_cost(symbol, price, volume)
        net_pnl = gross_pnl - exit_cost - pos.get("entry_cost", 0.0)
        equity += net_pnl

        trades.append(
            TradeRecord(
                entry_time=pos.get("entry_time", pd.Timestamp.now()),
                exit_time=row.get("time", pd.Timestamp.now()) if isinstance(row, pd.Series) else pd.Timestamp.now(),
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                exit_price=price,
                volume=volume,
                gross_pnl=float(gross_pnl),
                cost=float(exit_cost + pos.get("entry_cost", 0.0)),
                net_pnl=float(net_pnl),
                exit_reason=reason,
                hold_bars=pos.get("bars_held", 0),
            )
        )
        return float(equity)

    def _build_artifact(
        self,
        backtest_id: str,
        bundle_id: str,
        symbol: str,
        equity_curve: List[float],
        drawdown_curve: List[float],
        trades: List[TradeRecord],
    ) -> Dict[str, Any]:
        net_return = (equity_curve[-1] - self.initial_equity) / self.initial_equity if equity_curve else 0.0
        max_drawdown = float(max(drawdown_curve)) if drawdown_curve else 0.0
        trade_count = len(trades)

        wins = [t.net_pnl for t in trades if t.net_pnl > 0]
        losses = [abs(t.net_pnl) for t in trades if t.net_pnl < 0]
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = sum(losses) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Sharpe from equity returns per step
        rets = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i - 1] > 0:
                rets.append((equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1])
        sharpe = 0.0
        if len(rets) > 1:
            mean_r = np.mean(rets)
            std_r = np.std(rets, ddof=1)
            if std_r > 0:
                # Annualize assuming ~252 trading days, bars per day depends on timeframe
                # Use simple per-bar Sharpe here for generality
                sharpe = float(mean_r / std_r)

        total_net = sum(t.net_pnl for t in trades)
        single_trade_profit_share = 0.0
        if total_net > 0 and wins:
            single_trade_profit_share = max(abs(t.net_pnl) for t in trades) / total_net

        passed = True
        if net_return <= 0:
            passed = False
        if profit_factor < self.min_profit_factor:
            passed = False
        if max_drawdown > self.max_drawdown_threshold:
            passed = False
        if trade_count < self.min_trade_count:
            passed = False
        if single_trade_profit_share > self.max_single_trade_profit_share:
            passed = False

        return {
            "backtest_id": backtest_id,
            "bundle_id": bundle_id,
            "symbol": symbol,
            "net_return_after_costs": float(net_return),
            "max_drawdown": float(max_drawdown),
            "profit_factor": float(profit_factor),
            "sharpe": float(sharpe),
            "trade_count": int(trade_count),
            "single_trade_profit_share": float(single_trade_profit_share),
            "passed": bool(passed),
            "initial_equity": float(self.initial_equity),
            "final_equity": float(equity_curve[-1]) if equity_curve else float(self.initial_equity),
            "gross_profit": float(gross_profit),
            "gross_loss": float(gross_loss),
            "total_costs": float(sum(t.cost for t in trades)),
            "trades": [
                {
                    "entry_time": str(t.entry_time),
                    "exit_time": str(t.exit_time),
                    "symbol": t.symbol,
                    "side": t.side,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "volume": t.volume,
                    "gross_pnl": t.gross_pnl,
                    "cost": t.cost,
                    "net_pnl": t.net_pnl,
                    "exit_reason": t.exit_reason,
                    "hold_bars": t.hold_bars,
                }
                for t in trades
            ],
        }
