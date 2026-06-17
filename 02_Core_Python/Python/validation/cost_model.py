"""
CostModel — Realistic transaction cost engine per symbol.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from Python.data.symbol_metadata import ContractSpec, get_contract, default_contract


@dataclass
class CostBreakdown:
    spread_cost: float
    commission: float
    slippage_cost: float
    total_cost: float


class CostModel:
    """
    Computes spread cost, commission, and slippage per symbol.
    """

    def __init__(self, overrides: Optional[Dict[str, dict]] = None):
        self.overrides = overrides or {}

    def _get_spec(self, symbol: str) -> ContractSpec:
        spec = get_contract(symbol)
        if spec is None:
            spec = default_contract()
        return spec

    def compute_cost(
        self,
        symbol: str,
        side: str,
        price: float,
        volume: float,
        spread: Optional[float] = None,
        slippage_estimate: Optional[float] = None,
    ) -> CostBreakdown:
        """
        Compute transaction costs for a single trade.

        Args:
            symbol: trading symbol (e.g. EURUSDm, BTCUSDm)
            side: 'BUY' or 'SELL'
            price: execution price
            volume: lot size traded
            spread: optional explicit spread in price terms; if None uses typical from spec
            slippage_estimate: optional explicit slippage in price terms; if None uses typical from spec

        Returns:
            CostBreakdown dataclass
        """
        spec = self._get_spec(symbol)
        override = self.overrides.get(symbol, {})

        # Spread cost: half spread on entry, half on exit (we charge full spread once here for round-trip simplicity)
        if spread is not None:
            spread_decimal = spread / price
        else:
            spread_decimal = spec.typical_spread_bps / 10_000.0

        # Adjust spread by override
        spread_multiplier = override.get("spread_multiplier", 1.0)
        spread_decimal *= spread_multiplier

        spread_cost = price * spread_decimal * volume * spec.contract_size

        # Slippage cost
        if slippage_estimate is not None:
            slippage_decimal = slippage_estimate / price
        else:
            slippage_decimal = spec.avg_slippage_bps / 10_000.0

        slippage_multiplier = override.get("slippage_multiplier", 1.0)
        slippage_decimal *= slippage_multiplier

        slippage_cost = price * slippage_decimal * volume * spec.contract_size

        # Commission
        commission_model = override.get("commission_model", spec.commission_model)
        if commission_model == "per_lot":
            commission = spec.commission_per_lot * volume
        elif commission_model == "percent":
            notional = price * volume * spec.contract_size
            commission = notional * (override.get("commission_pct", 0.0002))
        elif commission_model == "crypto_flat":
            # Crypto: commission as a small percent of notional
            notional = price * volume * spec.contract_size
            commission = notional * (override.get("commission_pct", 0.001))
        else:
            commission = spec.commission_per_lot * volume

        total_cost = spread_cost + commission + slippage_cost
        return CostBreakdown(
            spread_cost=float(spread_cost),
            commission=float(commission),
            slippage_cost=float(slippage_cost),
            total_cost=float(total_cost),
        )

    def compute_round_trip_cost(
        self,
        symbol: str,
        price: float,
        volume: float,
        spread: Optional[float] = None,
        slippage_estimate: Optional[float] = None,
    ) -> CostBreakdown:
        """Convenience wrapper for a round-trip (entry + exit)."""
        entry = self.compute_cost(symbol, "BUY", price, volume, spread, slippage_estimate)
        # Exit costs are roughly half spread + same slippage + same commission
        spec = self._get_spec(symbol)
        exit_spread = (spread or spec.typical_spread_bps) / 2.0
        exit = self.compute_cost(symbol, "SELL", price, volume, exit_spread, slippage_estimate)
        return CostBreakdown(
            spread_cost=float(entry.spread_cost + exit.spread_cost),
            commission=float(entry.commission + exit.commission),
            slippage_cost=float(entry.slippage_cost + exit.slippage_cost),
            total_cost=float(entry.total_cost + exit.total_cost),
        )
