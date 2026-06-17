"""
SymbolMetadata — Fetch and cache MT5 symbol properties.
Also provides static contract specs for backtesting cost models.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from loguru import logger

from Python.mt5_compat import mt5, MT5_AVAILABLE

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CACHE_DIR = os.path.join(_PROJECT_ROOT, "data", "raw", "mt5", "symbol_metadata")


def _ensure_dir():
    os.makedirs(_CACHE_DIR, exist_ok=True)


def _cache_path(symbol: str) -> str:
    _ensure_dir()
    safe = str(symbol).replace("/", "_")
    return os.path.join(_CACHE_DIR, f"{safe}.json")


@dataclass(frozen=True)
class ContractSpec:
    symbol: str
    asset_class: str  # forex, crypto, metal, index
    pip_value: float  # value of 1 pip in quote currency for 1.0 lot
    contract_size: float  # units per lot
    commission_per_lot: float  # base round-trip commission per lot
    commission_model: str  # "per_lot", "percent", "crypto_flat"
    typical_spread_bps: float  # typical spread in basis points
    avg_slippage_bps: float  # typical slippage in basis points
    min_lot: float
    max_lot: float
    lot_step: float
    margin_rate: float  # approximate margin requirement as fraction
    quote_currency: str
    base_currency: str


SYMBOL_REGISTRY: Dict[str, ContractSpec] = {
    "EURUSDm": ContractSpec(
        symbol="EURUSDm",
        asset_class="forex",
        pip_value=10.0,
        contract_size=100_000.0,
        commission_per_lot=7.0,
        commission_model="per_lot",
        typical_spread_bps=1.2,
        avg_slippage_bps=0.4,
        min_lot=0.01,
        max_lot=100.0,
        lot_step=0.01,
        margin_rate=0.01,
        quote_currency="USD",
        base_currency="EUR",
    ),
    "GBPUSDm": ContractSpec(
        symbol="GBPUSDm",
        asset_class="forex",
        pip_value=10.0,
        contract_size=100_000.0,
        commission_per_lot=7.0,
        commission_model="per_lot",
        typical_spread_bps=1.5,
        avg_slippage_bps=0.5,
        min_lot=0.01,
        max_lot=100.0,
        lot_step=0.01,
        margin_rate=0.01,
        quote_currency="USD",
        base_currency="GBP",
    ),
    "USDJPYm": ContractSpec(
        symbol="USDJPYm",
        asset_class="forex",
        pip_value=1000.0,
        contract_size=100_000.0,
        commission_per_lot=7.0,
        commission_model="per_lot",
        typical_spread_bps=1.3,
        avg_slippage_bps=0.4,
        min_lot=0.01,
        max_lot=100.0,
        lot_step=0.01,
        margin_rate=0.01,
        quote_currency="JPY",
        base_currency="USD",
    ),
    "XAUUSDm": ContractSpec(
        symbol="XAUUSDm",
        asset_class="metal",
        pip_value=10.0,
        contract_size=100.0,
        commission_per_lot=7.0,
        commission_model="per_lot",
        typical_spread_bps=3.0,
        avg_slippage_bps=1.0,
        min_lot=0.01,
        max_lot=100.0,
        lot_step=0.01,
        margin_rate=0.01,
        quote_currency="USD",
        base_currency="XAU",
    ),
    "BTCUSDm": ContractSpec(
        symbol="BTCUSDm",
        asset_class="crypto",
        pip_value=1.0,
        contract_size=1.0,
        commission_per_lot=0.0,
        commission_model="crypto_flat",
        typical_spread_bps=8.0,
        avg_slippage_bps=3.0,
        min_lot=0.01,
        max_lot=10.0,
        lot_step=0.01,
        margin_rate=0.05,
        quote_currency="USD",
        base_currency="BTC",
    ),
    "ETHUSDm": ContractSpec(
        symbol="ETHUSDm",
        asset_class="crypto",
        pip_value=1.0,
        contract_size=1.0,
        commission_per_lot=0.0,
        commission_model="crypto_flat",
        typical_spread_bps=10.0,
        avg_slippage_bps=4.0,
        min_lot=0.01,
        max_lot=20.0,
        lot_step=0.01,
        margin_rate=0.05,
        quote_currency="USD",
        base_currency="ETH",
    ),
    "US30m": ContractSpec(
        symbol="US30m",
        asset_class="index",
        pip_value=1.0,
        contract_size=1.0,
        commission_per_lot=7.0,
        commission_model="per_lot",
        typical_spread_bps=2.5,
        avg_slippage_bps=0.8,
        min_lot=0.01,
        max_lot=50.0,
        lot_step=0.01,
        margin_rate=0.02,
        quote_currency="USD",
        base_currency="US30",
    ),
    "US100m": ContractSpec(
        symbol="US100m",
        asset_class="index",
        pip_value=1.0,
        contract_size=1.0,
        commission_per_lot=7.0,
        commission_model="per_lot",
        typical_spread_bps=2.2,
        avg_slippage_bps=0.7,
        min_lot=0.01,
        max_lot=50.0,
        lot_step=0.01,
        margin_rate=0.02,
        quote_currency="USD",
        base_currency="US100",
    ),
}


def get_contract(symbol: str) -> Optional[ContractSpec]:
    """Return contract spec for a symbol, or None if unknown."""
    s = str(symbol)
    return SYMBOL_REGISTRY.get(s) or SYMBOL_REGISTRY.get(s.upper())


def default_contract() -> ContractSpec:
    """Return a generic forex default contract."""
    return SYMBOL_REGISTRY["EURUSDm"]


class SymbolMetadata:
    """Fetch and cache MT5 symbol_info for a given symbol."""

    def __init__(self, project_root: str | None = None):
        self.project_root = project_root or _PROJECT_ROOT
        self.cache_dir = os.path.join(self.project_root, "data", "raw", "mt5", "symbol_metadata")
        os.makedirs(self.cache_dir, exist_ok=True)

    def fetch(self, symbol: str) -> dict[str, Any] | None:
        """Fetch symbol metadata from MT5."""
        if not MT5_AVAILABLE:
            logger.warning("MT5 unavailable — cannot fetch symbol metadata")
            return None
        try:
            if not mt5.initialize():
                logger.warning("MT5 initialize failed")
                return None
        except Exception as exc:
            logger.warning(f"MT5 initialize error: {exc}")
            return None
        try:
            info = mt5.symbol_info(symbol)
        except Exception as exc:
            logger.warning(f"MT5 symbol_info error: {exc}")
            return None
        if info is None:
            logger.warning(f"symbol_info returned None for {symbol}")
            return None
        now = datetime.now(timezone.utc).isoformat()
        meta = {
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
            "fetched_at": now,
        }
        return meta

    def save(self, symbol: str, meta: dict[str, Any] | None) -> str | None:
        if meta is None:
            return None
        path = _cache_path(symbol)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
        return path

    def load(self, symbol: str) -> dict[str, Any] | None:
        path = _cache_path(symbol)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get(self, symbol: str) -> dict[str, Any] | None:
        """Load from cache or fetch from MT5."""
        cached = self.load(symbol)
        if cached:
            return cached
        meta = self.fetch(symbol)
        self.save(symbol, meta)
        return meta
