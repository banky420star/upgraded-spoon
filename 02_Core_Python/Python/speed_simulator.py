"""
Speed Simulator for Chain Gambler

Simulates realistic trade execution conditions:
- Network latency (round-trip time to broker)
- Order processing delays (market vs limit orders)
- Slippage (price impact based on order size and market volatility)
- Fill probability (chance of complete/partial fill)
- Spread dynamics (bid-ask spread variation)
- Market impact (price movement from large orders)

Usage:
    from Python.speed_simulator import SpeedSimulator, ExecutionScenario

    sim = SpeedSimulator()
    result = sim.simulate_execution(
        symbol="EURUSDm",
        order_type="MARKET",
        side="BUY",
        size=0.5,
        requested_price=1.0850,
        market_volatility="MED",  # LOW, MED, HIGH
        market_regime="trending",  # trending, ranging, volatile
    )
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd
from loguru import logger


class VolatilityRegime(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class MarketCondition(Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    NEWS_EVENT = "news_event"


@dataclass
class ExecutionResult:
    """Result of simulated trade execution."""
    success: bool
    filled: bool
    fill_price: float
    requested_price: float
    slippage: float  # in pips/points
    slippage_pct: float  # percentage
    latency_ms: int  # round-trip latency in milliseconds
    execution_time_ms: int  # total execution time
    fill_amount: float  # actual filled amount
    partial_fill: bool
    rejected: bool
    reject_reason: str
    spread_at_fill: float
    market_impact: float  # price movement caused by order
    timestamp: datetime
    scenario: str  # which scenario was used


@dataclass
class NetworkProfile:
    """Network latency profile for different conditions."""
    name: str
    base_latency_ms: int
    jitter_ms: int  # random variation
    packet_loss_pct: float  # chance of order timeout/retry
    reliability: float  # 0-1, chance of successful connection


@dataclass
class BrokerProfile:
    """Broker execution profile."""
    name: str
    market_order_delay_ms: int
    limit_order_delay_ms: int
    stop_order_delay_ms: int
    avg_slippage_pips: float
    slippage_std_pips: float
    fill_rate_pct: float
    partial_fill_rate_pct: float
    spread_markup_pips: float


class SpeedSimulator:
    """
    Simulates realistic trade execution speed and slippage.

    Models:
    1. Network latency (geographic distance, connection quality)
    2. Broker processing time (order type dependent)
    3. Slippage (volatility, order size, liquidity)
    4. Fill probability (market conditions, order type)
    5. Spread dynamics (time of day, volatility)
    """

    # Predefined network profiles
    NETWORK_PROFILES = {
        "excellent": NetworkProfile(
            name="excellent",
            base_latency_ms=20,
            jitter_ms=5,
            packet_loss_pct=0.0,
            reliability=0.999
        ),
        "good": NetworkProfile(
            name="good",
            base_latency_ms=50,
            jitter_ms=15,
            packet_loss_pct=0.1,
            reliability=0.995
        ),
        "average": NetworkProfile(
            name="average",
            base_latency_ms=120,
            jitter_ms=40,
            packet_loss_pct=0.5,
            reliability=0.98
        ),
        "poor": NetworkProfile(
            name="poor",
            base_latency_ms=300,
            jitter_ms=100,
            packet_loss_pct=2.0,
            reliability=0.95
        ),
        "mobile": NetworkProfile(
            name="mobile",
            base_latency_ms=200,
            jitter_ms=80,
            packet_loss_pct=1.0,
            reliability=0.97
        ),
    }

    # Predefined broker profiles
    BROKER_PROFILES = {
        "ecn_fast": BrokerProfile(
            name="ecn_fast",
            market_order_delay_ms=50,
            limit_order_delay_ms=30,
            stop_order_delay_ms=60,
            avg_slippage_pips=0.1,
            slippage_std_pips=0.05,
            fill_rate_pct=99.5,
            partial_fill_rate_pct=5.0,
            spread_markup_pips=0.0
        ),
        "ecn_standard": BrokerProfile(
            name="ecn_standard",
            market_order_delay_ms=100,
            limit_order_delay_ms=80,
            stop_order_delay_ms=120,
            avg_slippage_pips=0.2,
            slippage_std_pips=0.1,
            fill_rate_pct=98.0,
            partial_fill_rate_pct=10.0,
            spread_markup_pips=0.1
        ),
        "mm_premium": BrokerProfile(
            name="mm_premium",
            market_order_delay_ms=150,
            limit_order_delay_ms=100,
            stop_order_delay_ms=200,
            avg_slippage_pips=0.3,
            slippage_std_pips=0.15,
            fill_rate_pct=99.0,
            partial_fill_rate_pct=2.0,
            spread_markup_pips=0.2
        ),
        "mm_standard": BrokerProfile(
            name="mm_standard",
            market_order_delay_ms=250,
            limit_order_delay_ms=200,
            stop_order_delay_ms=300,
            avg_slippage_pips=0.5,
            slippage_std_pips=0.25,
            fill_rate_pct=97.0,
            partial_fill_rate_pct=8.0,
            spread_markup_pips=0.5
        ),
        "mm_slow": BrokerProfile(
            name="mm_slow",
            market_order_delay_ms=500,
            limit_order_delay_ms=400,
            stop_order_delay_ms=600,
            avg_slippage_pips=1.0,
            slippage_std_pips=0.5,
            fill_rate_pct=95.0,
            partial_fill_rate_pct=15.0,
            spread_markup_pips=1.0
        ),
    }

    # Symbol-specific pip values and typical spreads
    SYMBOL_SPECS = {
        "EURUSDm": {"pip_value": 0.0001, "typical_spread_pips": 1.0, "liquidity": "high"},
        "GBPUSDm": {"pip_value": 0.0001, "typical_spread_pips": 1.5, "liquidity": "high"},
        "USDJPYm": {"pip_value": 0.01, "typical_spread_pips": 1.2, "liquidity": "high"},
        "XAUUSDm": {"pip_value": 0.01, "typical_spread_pips": 20.0, "liquidity": "medium"},
        "BTCUSDm": {"pip_value": 1.0, "typical_spread_pips": 50.0, "liquidity": "medium"},
        "default": {"pip_value": 0.0001, "typical_spread_pips": 2.0, "liquidity": "medium"},
    }

    def __init__(
        self,
        network_profile: str = None,
        broker_profile: str = None,
        random_seed: int = None
    ):
        """
        Initialize speed simulator.

        Args:
            network_profile: Network condition (excellent/good/average/poor/mobile)
            broker_profile: Broker type (ecn_fast/ecn_standard/mm_premium/mm_standard/mm_slow)
            random_seed: Optional seed for reproducible results
        """
        # Configuration from environment or defaults
        self.network_profile_name = network_profile or os.environ.get(
            "AGI_NETWORK_PROFILE", "good"
        )
        self.broker_profile_name = broker_profile or os.environ.get(
            "AGI_BROKER_PROFILE", "mm_premium"
        )

        self.network = self.NETWORK_PROFILES.get(
            self.network_profile_name, self.NETWORK_PROFILES["good"]
        )
        self.broker = self.BROKER_PROFILES.get(
            self.broker_profile_name, self.BROKER_PROFILES["mm_premium"]
        )

        # Set random seed for reproducibility
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        # Statistics tracking
        self.execution_history: List[ExecutionResult] = []
        self.stats = {
            "total_orders": 0,
            "successful_fills": 0,
            "partial_fills": 0,
            "rejected": 0,
            "avg_slippage": 0.0,
            "avg_latency_ms": 0.0,
        }

        logger.success(
            f"SpeedSimulator initialized: network={self.network.name}, "
            f"broker={self.broker.name}"
        )

    def simulate_execution(
        self,
        symbol: str,
        order_type: str,  # MARKET, LIMIT, STOP
        side: str,  # BUY, SELL
        size: float,
        requested_price: float,
        market_volatility: str = "MEDIUM",
        market_regime: str = "trending",
        current_spread_pips: float = None,
    ) -> ExecutionResult:
        """
        Simulate a trade execution with realistic speed and slippage.

        Args:
            symbol: Trading symbol
            order_type: MARKET, LIMIT, or STOP
            side: BUY or SELL
            size: Position size in lots
            requested_price: Expected fill price
            market_volatility: LOW, MEDIUM, HIGH
            market_regime: trending, ranging, volatile, news_event
            current_spread_pips: Optional current spread

        Returns:
            ExecutionResult with simulated execution details
        """
        timestamp = datetime.now()
        self.stats["total_orders"] += 1

        # Get symbol specifications
        sym_spec = self._get_symbol_spec(symbol)
        pip_value = sym_spec["pip_value"]

        # Step 1: Simulate network latency
        network_latency = self._simulate_network_latency()

        # Check for connection failure
        if random.random() > self.network.reliability:
            return ExecutionResult(
                success=False,
                filled=False,
                fill_price=0.0,
                requested_price=requested_price,
                slippage=0.0,
                slippage_pct=0.0,
                latency_ms=network_latency,
                execution_time_ms=network_latency,
                fill_amount=0.0,
                partial_fill=False,
                rejected=True,
                reject_reason="network_connection_failed",
                spread_at_fill=0.0,
                market_impact=0.0,
                timestamp=timestamp,
                scenario="connection_failure"
            )

        # Step 2: Simulate broker processing delay
        processing_delay = self._simulate_processing_delay(order_type)

        # Step 3: Calculate current spread
        if current_spread_pips is None:
            current_spread_pips = self._simulate_spread(
                sym_spec["typical_spread_pips"], market_volatility, market_regime
            )

        spread_price = current_spread_pips * pip_value

        # Step 4: Calculate slippage
        slippage_pips, slippage_pct = self._simulate_slippage(
            symbol, size, order_type, side, market_volatility, market_regime, sym_spec
        )

        # Step 5: Calculate fill probability
        fill_prob = self._calculate_fill_probability(
            order_type, market_volatility, market_regime, size
        )

        # Step 6: Determine execution outcome
        if random.random() > fill_prob / 100:
            # Order not filled
            total_time = network_latency + processing_delay
            result = ExecutionResult(
                success=False,
                filled=False,
                fill_price=0.0,
                requested_price=requested_price,
                slippage=0.0,
                slippage_pct=0.0,
                latency_ms=network_latency,
                execution_time_ms=total_time,
                fill_amount=0.0,
                partial_fill=False,
                rejected=False,
                reject_reason="not_filled",
                spread_at_fill=spread_price,
                market_impact=0.0,
                timestamp=timestamp,
                scenario=f"{market_volatility}_{market_regime}"
            )
        else:
            # Order filled (possibly partial)
            partial_fill_prob = self.broker.partial_fill_rate_pct / 100
            is_partial = random.random() < partial_fill_prob and size > 1.0

            if is_partial:
                fill_amount = size * random.uniform(0.3, 0.9)
                self.stats["partial_fills"] += 1
            else:
                fill_amount = size
                self.stats["successful_fills"] += 1

            # Calculate actual fill price with slippage
            if side == "BUY":
                fill_price = requested_price + (slippage_pips * pip_value)
            else:  # SELL
                fill_price = requested_price - (slippage_pips * pip_value)

            # Add market impact for large orders
            market_impact = self._calculate_market_impact(size, sym_spec, market_regime)
            if side == "BUY":
                fill_price += market_impact
            else:
                fill_price -= market_impact

            total_time = network_latency + processing_delay

            result = ExecutionResult(
                success=True,
                filled=True,
                fill_price=round(fill_price, 6),
                requested_price=requested_price,
                slippage=slippage_pips,
                slippage_pct=slippage_pct,
                latency_ms=network_latency,
                execution_time_ms=total_time,
                fill_amount=round(fill_amount, 2),
                partial_fill=is_partial,
                rejected=False,
                reject_reason="",
                spread_at_fill=spread_price,
                market_impact=market_impact,
                timestamp=timestamp,
                scenario=f"{market_volatility}_{market_regime}"
            )

        # Update statistics
        self.execution_history.append(result)
        self._update_stats(result)

        return result

    def _simulate_network_latency(self) -> int:
        """Simulate network round-trip latency in milliseconds."""
        base = self.network.base_latency_ms
        jitter = random.randint(0, self.network.jitter_ms)

        # Occasionally add spike
        if random.random() < 0.05:  # 5% chance of latency spike
            jitter += random.randint(50, 200)

        return base + jitter

    def _simulate_processing_delay(self, order_type: str) -> int:
        """Simulate broker processing delay in milliseconds."""
        if order_type.upper() == "MARKET":
            base = self.broker.market_order_delay_ms
        elif order_type.upper() == "LIMIT":
            base = self.broker.limit_order_delay_ms
        elif order_type.upper() == "STOP":
            base = self.broker.stop_order_delay_ms
        else:
            base = self.broker.market_order_delay_ms

        # Add some variance
        variance = int(base * 0.2)
        return base + random.randint(-variance, variance)

    def _simulate_spread(
        self,
        typical_spread_pips: float,
        volatility: str,
        regime: str
    ) -> float:
        """Simulate current bid-ask spread."""
        base = typical_spread_pips

        # Volatility multiplier
        vol_mult = {
            "LOW": 0.8,
            "MEDIUM": 1.0,
            "HIGH": 1.5
        }.get(volatility.upper(), 1.0)

        # Regime multiplier
        regime_mult = {
            "trending": 1.0,
            "ranging": 1.2,
            "volatile": 2.0,
            "news_event": 3.0
        }.get(regime, 1.0)

        spread = base * vol_mult * regime_mult

        # Add some randomness
        spread *= random.uniform(0.9, 1.1)

        return round(spread, 1)

    def _simulate_slippage(
        self,
        symbol: str,
        size: float,
        order_type: str,
        side: str,
        volatility: str,
        regime: str,
        sym_spec: Dict
    ) -> Tuple[float, float]:
        """
        Simulate price slippage in pips and percentage.

        Returns:
            (slippage_pips, slippage_pct)
        """
        # Base slippage from broker
        base_slippage = np.random.normal(
            self.broker.avg_slippage_pips,
            self.broker.slippage_std_pips
        )

        # Size impact (larger orders = more slippage)
        size_factor = 1.0 + (size * 0.1)  # 10% more slippage per lot

        # Volatility impact
        vol_mult = {
            "LOW": 0.5,
            "MEDIUM": 1.0,
            "HIGH": 2.5
        }.get(volatility.upper(), 1.0)

        # Regime impact
        regime_mult = {
            "trending": 1.0,
            "ranging": 1.3,
            "volatile": 3.0,
            "news_event": 5.0
        }.get(regime, 1.0)

        # Order type impact
        order_mult = {
            "MARKET": 1.0,
            "LIMIT": 0.3,  # Less slippage for limits
            "STOP": 1.5    # More slippage for stops
        }.get(order_type.upper(), 1.0)

        slippage_pips = max(0, base_slippage * size_factor * vol_mult * regime_mult * order_mult)

        # Calculate percentage slippage
        if "JPY" in symbol:
            price = 150.0  # Approximate JPY rate
        elif "XAU" in symbol:
            price = 2000.0  # Approximate gold price
        elif "BTC" in symbol:
            price = 50000.0  # Approximate BTC price
        else:
            price = 1.1  # Approximate forex rate

        pip_value = sym_spec["pip_value"]
        slippage_pct = (slippage_pips * pip_value) / price * 100

        return round(slippage_pips, 2), round(slippage_pct, 4)

    def _calculate_fill_probability(
        self,
        order_type: str,
        volatility: str,
        regime: str,
        size: float
    ) -> float:
        """Calculate probability of order fill (0-100)."""
        base_prob = self.broker.fill_rate_pct

        # Order type adjustment
        if order_type.upper() == "LIMIT":
            base_prob *= 0.95  # Limits may not fill
        elif order_type.upper() == "STOP":
            base_prob *= 0.98  # Stops usually fill but not always

        # Volatility impact
        if volatility.upper() == "HIGH":
            base_prob *= 0.95

        # Regime impact
        if regime == "volatile":
            base_prob *= 0.90
        elif regime == "news_event":
            base_prob *= 0.85

        # Size impact (very large orders harder to fill)
        if size > 5.0:
            base_prob *= 0.95

        return min(100.0, base_prob)

    def _calculate_market_impact(
        self,
        size: float,
        sym_spec: Dict,
        regime: str
    ) -> float:
        """Calculate price impact from order size."""
        pip_value = sym_spec["pip_value"]
        liquidity = sym_spec["liquidity"]

        # Base impact per lot
        impact_per_lot = {
            "high": 0.01,     # 0.01 pips per lot
            "medium": 0.05,   # 0.05 pips per lot
            "low": 0.2        # 0.2 pips per lot
        }.get(liquidity, 0.05)

        # Regime multiplier
        regime_mult = {
            "trending": 1.0,
            "ranging": 0.8,
            "volatile": 1.5,
            "news_event": 2.5
        }.get(regime, 1.0)

        impact_pips = size * impact_per_lot * regime_mult
        impact_price = impact_pips * pip_value

        return round(impact_price, 6)

    def _get_symbol_spec(self, symbol: str) -> Dict:
        """Get symbol specifications."""
        for key in self.SYMBOL_SPECS:
            if key in symbol:
                return self.SYMBOL_SPECS[key]
        return self.SYMBOL_SPECS["default"]

    def _update_stats(self, result: ExecutionResult):
        """Update execution statistics."""
        if result.success and result.filled:
            n = self.stats["successful_fills"]
            if n > 0:
                self.stats["avg_slippage"] = (
                    (self.stats["avg_slippage"] * (n - 1) + result.slippage) / n
                )
            else:
                self.stats["avg_slippage"] = result.slippage

        self.stats["avg_latency_ms"] = np.mean([
            r.latency_ms for r in self.execution_history[-100:]
        ]) if self.execution_history else 0

    def get_stats(self) -> Dict:
        """Get execution statistics."""
        total = self.stats["total_orders"]
        if total == 0:
            return self.stats

        stats = self.stats.copy()
        stats["fill_rate_pct"] = (
            (self.stats["successful_fills"] / total) * 100
        )
        stats["partial_fill_rate_pct"] = (
            (self.stats["partial_fills"] / total) * 100
        )

        return stats

    def reset_stats(self):
        """Reset execution statistics."""
        self.execution_history.clear()
        self.stats = {
            "total_orders": 0,
            "successful_fills": 0,
            "partial_fills": 0,
            "rejected": 0,
            "avg_slippage": 0.0,
            "avg_latency_ms": 0.0,
        }

    def batch_simulate(
        self,
        orders: List[Dict],
        market_volatility: str = "MEDIUM",
        market_regime: str = "trending"
    ) -> List[ExecutionResult]:
        """
        Simulate multiple executions.

        Args:
            orders: List of order dicts with keys: symbol, order_type, side, size, requested_price
            market_volatility: LOW, MEDIUM, HIGH
            market_regime: trending, ranging, volatile, news_event

        Returns:
            List of ExecutionResult
        """
        results = []
        for order in orders:
            result = self.simulate_execution(
                symbol=order.get("symbol", "EURUSDm"),
                order_type=order.get("order_type", "MARKET"),
                side=order.get("side", "BUY"),
                size=order.get("size", 0.1),
                requested_price=order.get("requested_price", 1.0),
                market_volatility=market_volatility,
                market_regime=market_regime
            )
            results.append(result)

        return results

    def compare_profiles(
        self,
        symbol: str = "EURUSDm",
        n_simulations: int = 1000
    ) -> Dict[str, Dict]:
        """
        Compare execution quality across different network/broker profiles.

        Args:
            symbol: Symbol to test
            n_simulations: Number of simulations per profile

        Returns:
            Comparison statistics
        """
        results = {}

        test_order = {
            "symbol": symbol,
            "order_type": "MARKET",
            "side": "BUY",
            "size": 1.0,
            "requested_price": 1.0850
        }

        for net_name, net_profile in self.NETWORK_PROFILES.items():
            for brk_name, brk_profile in self.BROKER_PROFILES.items():
                # Create simulator with specific profiles
                sim = SpeedSimulator(
                    network_profile=net_name,
                    broker_profile=brk_name,
                    random_seed=42
                )

                # Run simulations
                exec_results = sim.batch_simulate(
                    [test_order] * n_simulations,
                    market_volatility="MEDIUM",
                    market_regime="trending"
                )

                # Calculate metrics
                filled = [r for r in exec_results if r.filled]
                avg_slip = np.mean([r.slippage for r in filled]) if filled else 0
                avg_latency = np.mean([r.latency_ms for r in exec_results])
                fill_rate = len(filled) / len(exec_results) * 100

                key = f"{net_name}_{brk_name}"
                results[key] = {
                    "fill_rate_pct": fill_rate,
                    "avg_slippage_pips": avg_slip,
                    "avg_latency_ms": avg_latency,
                    "network": net_name,
                    "broker": brk_name
                }

        return results


# Global instance
_speed_simulator = None


def get_speed_simulator() -> SpeedSimulator:
    """Get or create global speed simulator."""
    global _speed_simulator
    if _speed_simulator is None:
        _speed_simulator = SpeedSimulator()
    return _speed_simulator


def simulate_latency_only(
    network_profile: str = "good",
    duration_ms: int = None
) -> int:
    """
    Simulate just network latency (useful for simple delays).

    Args:
        network_profile: excellent/good/average/poor/mobile
        duration_ms: Optional fixed duration instead of random

    Returns:
        Simulated latency in milliseconds
    """
    profile = SpeedSimulator.NETWORK_PROFILES.get(
        network_profile, SpeedSimulator.NETWORK_PROFILES["good"]
    )

    if duration_ms is not None:
        return duration_ms

    jitter = random.randint(0, profile.jitter_ms)
    return profile.base_latency_ms + jitter


if __name__ == "__main__":
    # Demo usage
    print("=" * 60)
    print("SPEED SIMULATOR DEMO")
    print("=" * 60)

    sim = SpeedSimulator(network_profile="good", broker_profile="mm_premium")

    # Simulate different market conditions
    conditions = [
        ("LOW", "trending"),
        ("MEDIUM", "ranging"),
        ("HIGH", "volatile"),
        ("HIGH", "news_event"),
    ]

    for vol, regime in conditions:
        print(f"\n--- Testing: {vol} volatility, {regime} regime ---")

        result = sim.simulate_execution(
            symbol="EURUSDm",
            order_type="MARKET",
            side="BUY",
            size=1.0,
            requested_price=1.0850,
            market_volatility=vol,
            market_regime=regime
        )

        print(f"  Success: {result.success}")
        print(f"  Filled: {result.filled}")
        print(f"  Fill Price: {result.fill_price}")
        print(f"  Slippage: {result.slippage} pips ({result.slippage_pct:.4f}%)")
        print(f"  Latency: {result.latency_ms}ms")
        print(f"  Total Time: {result.execution_time_ms}ms")
        print(f"  Partial Fill: {result.partial_fill}")

    # Batch simulation
    print("\n" + "=" * 60)
    print("BATCH SIMULATION (100 orders)")
    print("=" * 60)

    orders = [
        {"symbol": "EURUSDm", "order_type": "MARKET", "side": "BUY",
         "size": 0.5 + (i % 5) * 0.5, "requested_price": 1.0850}
        for i in range(100)
    ]

    results = sim.batch_simulate(orders, market_volatility="MEDIUM", market_regime="trending")

    filled = [r for r in results if r.filled]
    avg_slip = np.mean([r.slippage for r in filled]) if filled else 0
    avg_latency = np.mean([r.latency_ms for r in results])

    print(f"Total Orders: {len(results)}")
    print(f"Filled: {len(filled)} ({len(filled)/len(results)*100:.1f}%)")
    print(f"Partial Fills: {sum(1 for r in results if r.partial_fill)}")
    print(f"Avg Slippage: {avg_slip:.2f} pips")
    print(f"Avg Latency: {avg_latency:.1f}ms")

    # Stats
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    print(sim.get_stats())
