"""
Ingestor — Fetch and persist raw MT5 market data.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger

from Python.mt5_compat import mt5, MT5_AVAILABLE

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RAW_ROOT = os.path.join(_PROJECT_ROOT, "data", "raw", "mt5")
_CATEGORIES = ("candles", "ticks", "spreads", "account", "positions", "deals", "symbol_metadata")

_COPY_TICKS_ALL = 0


def _ensure_dirs():
    for cat in _CATEGORIES:
        os.makedirs(os.path.join(_RAW_ROOT, cat), exist_ok=True)


def _candle_id(symbol: str, timeframe: str, timestamp: str, source: str, broker: str) -> str:
    payload = f"{symbol}:{timeframe}:{timestamp}:{source}:{broker}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _today_file(category: str, ext: str = "jsonl") -> str:
    _ensure_dirs()
    date_stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return os.path.join(_RAW_ROOT, category, f"{date_stamp}.{ext}")


def _append_jsonl(path: str, records: list[dict]):
    if not records:
        return
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, default=str) + "\n")


class Ingestor:
    """Fetches candles, ticks, spreads, symbol metadata, account snapshots, positions, and deals from MT5."""

    def __init__(self, project_root: str | None = None):
        self.project_root = project_root or _PROJECT_ROOT
        self.raw_root = os.path.join(self.project_root, "data", "raw", "mt5")
        self.broker: str = "unknown"
        self.source: str = "mt5"
        self._ensure_dirs()
        self._init_broker()

    def _init_broker(self):
        try:
            if MT5_AVAILABLE and mt5.initialize():
                ai = mt5.account_info()
                if ai is not None:
                    self.broker = getattr(ai, "server", "unknown") or "unknown"
        except Exception as exc:
            logger.debug(f"Could not resolve broker name: {exc}")
            self.broker = "unknown"

    def _ensure_dirs(self):
        for cat in _CATEGORIES:
            os.makedirs(os.path.join(self.raw_root, cat), exist_ok=True)

    def _today_path(self, category: str, ext: str = "jsonl") -> str:
        self._ensure_dirs()
        date_stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        return os.path.join(self.raw_root, category, f"{date_stamp}.{ext}")

    def _mt5_timeframe(self, timeframe: str) -> int:
        mapping = {
            "1m": mt5.TIMEFRAME_M1,
            "5m": mt5.TIMEFRAME_M5,
            "15m": mt5.TIMEFRAME_M15,
            "30m": mt5.TIMEFRAME_M30,
            "1h": mt5.TIMEFRAME_H1,
            "4h": mt5.TIMEFRAME_H4,
            "1d": mt5.TIMEFRAME_D1,
        }
        return mapping.get(timeframe.lower(), mt5.TIMEFRAME_M5)

    def fetch_candles(self, symbol: str, timeframe: str = "5m", count: int = 1000) -> list[dict]:
        """Fetch OHLCV candles from MT5."""
        if not MT5_AVAILABLE:
            logger.warning("MT5 unavailable — returning empty candles")
            return []

        try:
            if not mt5.initialize():
                logger.warning("MT5 initialize failed — returning empty candles")
                return []
        except Exception as exc:
            logger.warning(f"MT5 initialize error: {exc}")
            return []

        tf = self._mt5_timeframe(timeframe)
        try:
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
        except Exception as exc:
            logger.warning(f"MT5 copy_rates_from_pos error: {exc}")
            return []

        if rates is None or len(rates) == 0:
            logger.debug(f"No candles returned for {symbol} {timeframe}")
            return []

        broker = self.broker
        source = self.source
        now = datetime.now(timezone.utc).isoformat()
        candles = []
        for r in rates:
            ts = datetime.fromtimestamp(int(r["time"]), tz=timezone.utc).isoformat()
            spread = float(r.get("spread", 0) or 0)
            candle = {
                "candle_id": _candle_id(symbol, timeframe, ts, source, broker),
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": ts,
                "source": source,
                "broker": broker,
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
                "volume": float(r.get("tick_volume", r.get("volume", 0)) or 0),
                "spread": spread,
                "raw_ingested_at": now,
            }
            candles.append(candle)
        logger.info(f"Ingested {len(candles)} candles for {symbol} {timeframe}")
        return candles

    def store_candles(self, candles: list[dict]) -> str:
        path = self._today_path("candles")
        _append_jsonl(path, candles)
        logger.debug(f"Stored {len(candles)} candles → {path}")
        return path

    def ingest_candles(self, symbol: str, timeframe: str = "5m", count: int = 1000) -> list[dict]:
        candles = self.fetch_candles(symbol, timeframe, count)
        if candles:
            self.store_candles(candles)
        return candles

    def fetch_ticks(self, symbol: str, count: int = 1000) -> list[dict]:
        """Fetch last N ticks from MT5."""
        if not MT5_AVAILABLE:
            logger.warning("MT5 unavailable — returning empty ticks")
            return []
        try:
            if not mt5.initialize():
                return []
        except Exception:
            return []
        try:
            now = datetime.now(timezone.utc)
            flags = getattr(mt5, "COPY_TICKS_ALL", _COPY_TICKS_ALL)
            ticks = mt5.copy_ticks_from(symbol, now - timedelta(hours=24), count, flags)
        except Exception as exc:
            logger.warning(f"MT5 tick fetch error: {exc}")
            return []
        if ticks is None or len(ticks) == 0:
            return []
        now = datetime.now(timezone.utc).isoformat()
        records = []
        for t in ticks:
            ts = datetime.fromtimestamp(int(t["time"]), tz=timezone.utc).isoformat()
            records.append({
                "symbol": symbol,
                "timestamp": ts,
                "bid": float(t.get("bid", 0)),
                "ask": float(t.get("ask", 0)),
                "last": float(t.get("last", 0)),
                "volume": float(t.get("volume", 0)),
                "flags": int(t.get("flags", 0)),
                "source": self.source,
                "broker": self.broker,
                "raw_ingested_at": now,
            })
        return records

    def store_ticks(self, ticks: list[dict]) -> str:
        path = self._today_path("ticks")
        _append_jsonl(path, ticks)
        return path

    def ingest_ticks(self, symbol: str, count: int = 1000) -> list[dict]:
        ticks = self.fetch_ticks(symbol, count)
        if ticks:
            self.store_ticks(ticks)
        return ticks

    def fetch_spreads(self, symbol: str, count: int = 1000) -> list[dict]:
        """Fetch spread samples from MT5 tick data."""
        if not MT5_AVAILABLE:
            return []
        try:
            if not mt5.initialize():
                return []
        except Exception:
            return []
        try:
            now = datetime.now(timezone.utc)
            flags = getattr(mt5, "COPY_TICKS_ALL", _COPY_TICKS_ALL)
            ticks = mt5.copy_ticks_from(symbol, now - timedelta(hours=24), count, flags)
        except Exception as exc:
            logger.warning(f"MT5 spread fetch error: {exc}")
            return []
        if ticks is None or len(ticks) == 0:
            return []
        now = datetime.now(timezone.utc).isoformat()
        records = []
        for t in ticks:
            bid = float(t.get("bid", 0))
            ask = float(t.get("ask", 0))
            spread = ask - bid if ask and bid else 0.0
            ts = datetime.fromtimestamp(int(t["time"]), tz=timezone.utc).isoformat()
            records.append({
                "symbol": symbol,
                "timestamp": ts,
                "bid": bid,
                "ask": ask,
                "spread": spread,
                "source": self.source,
                "broker": self.broker,
                "raw_ingested_at": now,
            })
        return records

    def store_spreads(self, spreads: list[dict]) -> str:
        path = self._today_path("spreads")
        _append_jsonl(path, spreads)
        return path

    def ingest_spreads(self, symbol: str, count: int = 1000) -> list[dict]:
        spreads = self.fetch_spreads(symbol, count)
        if spreads:
            self.store_spreads(spreads)
        return spreads

    def fetch_account_snapshot(self) -> dict | None:
        """Fetch current account snapshot from MT5."""
        if not MT5_AVAILABLE:
            logger.warning("MT5 unavailable — cannot fetch account snapshot")
            return None
        try:
            if not mt5.initialize():
                logger.warning("MT5 initialize failed — cannot fetch account snapshot")
                return None
        except Exception as exc:
            logger.warning(f"MT5 initialize error: {exc}")
            return None
        try:
            ai = mt5.account_info()
        except Exception as exc:
            logger.warning(f"MT5 account_info error: {exc}")
            return None
        if ai is None:
            return None
        now = datetime.now(timezone.utc).isoformat()
        return {
            "balance": float(getattr(ai, "balance", 0) or 0),
            "equity": float(getattr(ai, "equity", 0) or 0),
            "margin": float(getattr(ai, "margin", 0) or 0),
            "margin_free": float(getattr(ai, "margin_free", 0) or 0),
            "leverage": int(getattr(ai, "leverage", 0) or 0),
            "currency": getattr(ai, "currency", ""),
            "server": getattr(ai, "server", ""),
            "login": int(getattr(ai, "login", 0) or 0),
            "snapshot_at": now,
            "source": self.source,
            "broker": self.broker,
        }

    def store_account_snapshot(self, snapshot: dict | None) -> str | None:
        if snapshot is None:
            return None
        path = self._today_path("account")
        _append_jsonl(path, [snapshot])
        return path

    def ingest_account_snapshot(self) -> dict | None:
        snap = self.fetch_account_snapshot()
        self.store_account_snapshot(snap)
        return snap

    def fetch_positions(self) -> list[dict]:
        """Fetch open positions from MT5."""
        if not MT5_AVAILABLE:
            logger.warning("MT5 unavailable — returning empty positions")
            return []
        try:
            if not mt5.initialize():
                return []
        except Exception:
            return []
        try:
            positions = mt5.positions_get()
        except Exception as exc:
            logger.warning(f"MT5 positions_get error: {exc}")
            return []
        if positions is None or len(positions) == 0:
            return []
        now = datetime.now(timezone.utc).isoformat()
        records = []
        for p in positions:
            records.append({
                "ticket": int(getattr(p, "ticket", 0)),
                "symbol": getattr(p, "symbol", ""),
                "type": "BUY" if getattr(p, "type", 0) == 0 else "SELL",
                "volume": float(getattr(p, "volume", 0) or 0),
                "open_price": float(getattr(p, "price_open", 0) or 0),
                "current_price": float(getattr(p, "price_current", 0) or 0),
                "sl": float(getattr(p, "sl", 0) or 0),
                "tp": float(getattr(p, "tp", 0) or 0),
                "swap": float(getattr(p, "swap", 0) or 0),
                "profit": float(getattr(p, "profit", 0) or 0),
                "magic": int(getattr(p, "magic", 0) or 0),
                "comment": getattr(p, "comment", ""),
                "source": self.source,
                "broker": self.broker,
                "raw_ingested_at": now,
            })
        return records

    def store_positions(self, positions: list[dict]) -> str:
        path = self._today_path("positions")
        _append_jsonl(path, positions)
        return path

    def ingest_positions(self) -> list[dict]:
        positions = self.fetch_positions()
        if positions:
            self.store_positions(positions)
        return positions

    def fetch_deals(self, days_back: int = 7) -> list[dict]:
        """Fetch closed deals from MT5 history."""
        if not MT5_AVAILABLE:
            logger.warning("MT5 unavailable — returning empty deals")
            return []
        try:
            if not mt5.initialize():
                return []
        except Exception:
            return []
        try:
            since = datetime.now(timezone.utc) - timedelta(days=days_back)
            until = datetime.now(timezone.utc)
            deals = mt5.history_deals_get(since, until)
        except Exception as exc:
            logger.warning(f"MT5 history_deals_get error: {exc}")
            return []
        if deals is None or len(deals) == 0:
            return []
        now = datetime.now(timezone.utc).isoformat()
        records = []
        for d in deals:
            ts = datetime.fromtimestamp(int(getattr(d, "time", 0)), tz=timezone.utc).isoformat()
            entry = int(getattr(d, "entry", 0))
            records.append({
                "ticket": int(getattr(d, "ticket", 0)),
                "order": int(getattr(d, "order", 0)),
                "position_id": int(getattr(d, "position_id", 0)),
                "symbol": getattr(d, "symbol", ""),
                "type": "BUY" if getattr(d, "type", 0) == 0 else "SELL",
                "entry": entry,
                "reason": int(getattr(d, "reason", 0)),
                "time": ts,
                "volume": float(getattr(d, "volume", 0) or 0),
                "price": float(getattr(d, "price", 0) or 0),
                "profit": float(getattr(d, "profit", 0) or 0),
                "commission": float(getattr(d, "commission", 0) or 0),
                "swap": float(getattr(d, "swap", 0) or 0),
                "fee": float(getattr(d, "fee", 0) or 0),
                "comment": getattr(d, "comment", ""),
                "source": self.source,
                "broker": self.broker,
                "raw_ingested_at": now,
            })
        # Try to pair entry/exit prices for closing deals
        entry_prices: dict[int, float] = {}
        for r in records:
            if r["entry"] == 0 and r["position_id"]:
                entry_prices[r["position_id"]] = r["price"]
        for r in records:
            if r["entry"] == 1 and r["position_id"] in entry_prices:
                r["entry_price"] = entry_prices[r["position_id"]]
                r["exit_price"] = r["price"]
        return records

    def store_deals(self, deals: list[dict]) -> str:
        path = self._today_path("deals")
        _append_jsonl(path, deals)
        return path

    def ingest_deals(self, days_back: int = 7) -> list[dict]:
        deals = self.fetch_deals(days_back)
        if deals:
            self.store_deals(deals)
        return deals

    def fetch_symbol_metadata(self, symbol: str) -> dict | None:
        """Fetch symbol metadata from MT5."""
        if not MT5_AVAILABLE:
            logger.warning("MT5 unavailable — cannot fetch symbol metadata")
            return None
        try:
            if not mt5.initialize():
                return None
        except Exception:
            return None
        try:
            info = mt5.symbol_info(symbol)
        except Exception as exc:
            logger.warning(f"MT5 symbol_info error: {exc}")
            return None
        if info is None:
            return None
        now = datetime.now(timezone.utc).isoformat()
        return {
            "symbol": symbol,
            "digits": int(getattr(info, "digits", 0)),
            "point": float(getattr(info, "point", 0) or 0),
            "trade_tick_size": float(getattr(info, "trade_tick_size", 0) or 0),
            "trade_tick_value": float(getattr(info, "trade_tick_value", 0) or 0),
            "trade_contract_size": float(getattr(info, "trade_contract_size", 0) or 0),
            "volume_min": float(getattr(info, "volume_min", 0) or 0),
            "volume_max": float(getattr(info, "volume_max", 0) or 0),
            "volume_step": float(getattr(info, "volume_step", 0) or 0),
            "spread_floating": float(getattr(info, "spread_floating", 0) or 0),
            "swap_long": float(getattr(info, "swap_long", 0) or 0),
            "swap_short": float(getattr(info, "swap_short", 0) or 0),
            "source": self.source,
            "broker": self.broker,
            "raw_ingested_at": now,
        }

    def store_symbol_metadata(self, meta: dict | None) -> str | None:
        if meta is None:
            return None
        path = self._today_path("symbol_metadata")
        _append_jsonl(path, [meta])
        return path

    def ingest_symbol_metadata(self, symbol: str) -> dict | None:
        meta = self.fetch_symbol_metadata(symbol)
        self.store_symbol_metadata(meta)
        return meta
