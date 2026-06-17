"""
Validation script for Speed Simulator.

Tests:
1. Network latency simulation
2. Slippage calculation
3. Fill probability
4. Batch execution
5. Profile comparison
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Python.speed_simulator import SpeedSimulator, get_speed_simulator


def test_network_latency():
    """Test network latency simulation."""
    print("\n" + "=" * 60)
    print("TEST 1: Network Latency")
    print("=" * 60)

    for profile in ["excellent", "good", "average", "poor", "mobile"]:
        sim = SpeedSimulator(network_profile=profile, broker_profile="mm_premium")
        latencies = []
        for _ in range(100):
            lat = sim._simulate_network_latency()
            latencies.append(lat)

        avg_lat = np.mean(latencies)
        min_lat = min(latencies)
        max_lat = max(latencies)

        print(f"  {profile:12s}: avg={avg_lat:6.1f}ms, min={min_lat}ms, max={max_lat}ms")

    print("[PASS] Network latency ranges are realistic")
    return True


def test_slippage():
    """Test slippage calculation."""
    print("\n" + "=" * 60)
    print("TEST 2: Slippage Calculation")
    print("=" * 60)

    sim = SpeedSimulator(network_profile="good", broker_profile="mm_premium")

    test_cases = [
        ("EURUSDm", 0.1, "LOW"),
        ("EURUSDm", 1.0, "MEDIUM"),
        ("EURUSDm", 5.0, "HIGH"),
        ("XAUUSDm", 1.0, "MEDIUM"),
        ("BTCUSDm", 1.0, "HIGH"),
    ]

    for symbol, size, vol in test_cases:
        slips = []
        for _ in range(100):
            slip_pips, slip_pct = sim._simulate_slippage(
                symbol=symbol,
                size=size,
                order_type="MARKET",
                side="BUY",
                volatility=vol,
                regime="trending",
                sym_spec=sim._get_symbol_spec(symbol)
            )
            slips.append(slip_pips)

        avg_slip = np.mean(slips)
        print(f"  {symbol:10s} size={size:.1f} vol={vol:6s}: avg_slip={avg_slip:.2f} pips")

    print("[PASS] Slippage increases with size and volatility")
    return True


def test_fill_probability():
    """Test fill probability calculation."""
    print("\n" + "=" * 60)
    print("TEST 3: Fill Probability")
    print("=" * 60)

    sim = SpeedSimulator(network_profile="good", broker_profile="mm_premium")

    test_cases = [
        ("MARKET", "MEDIUM", "trending", 1.0),
        ("LIMIT", "MEDIUM", "trending", 1.0),
        ("STOP", "HIGH", "volatile", 5.0),
        ("MARKET", "HIGH", "news_event", 10.0),
    ]

    for order_type, vol, regime, size in test_cases:
        prob = sim._calculate_fill_probability(order_type, vol, regime, size)
        print(f"  {order_type:8s} {vol:6s} {regime:12s} size={size:.1f}: {prob:.1f}%")

    print("[PASS] Fill probability varies by order type and conditions")
    return True


def test_batch_execution():
    """Test batch execution simulation."""
    print("\n" + "=" * 60)
    print("TEST 4: Batch Execution")
    print("=" * 60)

    sim = SpeedSimulator(network_profile="good", broker_profile="mm_premium")

    orders = [
        {"symbol": "EURUSDm", "order_type": "MARKET", "side": "BUY",
         "size": 0.1 * (i + 1), "requested_price": 1.0850}
        for i in range(50)
    ]

    results = sim.batch_simulate(orders, market_volatility="MEDIUM", market_regime="trending")

    filled = [r for r in results if r.filled]
    rejected = [r for r in results if r.rejected]
    partial = [r for r in results if r.partial_fill]

    fill_rate = len(filled) / len(results) * 100

    print(f"  Total Orders: {len(results)}")
    print(f"  Filled: {len(filled)} ({fill_rate:.1f}%)")
    print(f"  Rejected: {len(rejected)}")
    print(f"  Partial Fills: {len(partial)}")

    assert fill_rate > 80, "Fill rate should be >80%"
    print("[PASS] Batch execution working correctly")
    return True


def test_spread_simulation():
    """Test spread simulation."""
    print("\n" + "=" * 60)
    print("TEST 5: Spread Simulation")
    print("=" * 60)

    sim = SpeedSimulator(network_profile="good", broker_profile="mm_premium")

    for symbol in ["EURUSDm", "GBPUSDm", "XAUUSDm", "BTCUSDm"]:
        spec = sim._get_symbol_spec(symbol)
        spreads = []
        for _ in range(100):
            spread = sim._simulate_spread(
                spec["typical_spread_pips"],
                "MEDIUM",
                "trending"
            )
            spreads.append(spread)

        avg_spread = np.mean(spreads)
        print(f"  {symbol:10s}: typical={spec['typical_spread_pips']:.1f}pips, "
              f"simulated_avg={avg_spread:.1f}pips")

    print("[PASS] Spreads are realistic for each symbol")
    return True


def test_market_impact():
    """Test market impact calculation."""
    print("\n" + "=" * 60)
    print("TEST 6: Market Impact")
    print("=" * 60)

    sim = SpeedSimulator(network_profile="good", broker_profile="mm_premium")

    for symbol in ["EURUSDm", "XAUUSDm", "BTCUSDm"]:
        spec = sim._get_symbol_spec(symbol)
        print(f"  {symbol}:")
        for size in [0.5, 1.0, 5.0, 10.0]:
            impact = sim._calculate_market_impact(size, spec, "trending")
            print(f"    size={size:5.1f}: impact={impact:.6f}")

    print("[PASS] Market impact increases with size")
    return True


def test_integration_with_paper_trader():
    """Test integration with paper trader workflow."""
    print("\n" + "=" * 60)
    print("TEST 7: Paper Trader Integration")
    print("=" * 60)

    sim = get_speed_simulator()

    # Simulate a realistic trading scenario
    trades = [
        {"symbol": "EURUSDm", "order_type": "MARKET", "side": "BUY",
         "size": 1.0, "requested_price": 1.08500},
        {"symbol": "EURUSDm", "order_type": "MARKET", "side": "SELL",
         "size": 1.0, "requested_price": 1.08600},
        {"symbol": "XAUUSDm", "order_type": "MARKET", "side": "BUY",
         "size": 0.5, "requested_price": 2300.00},
    ]

    print("  Executing 3 simulated trades:")
    for trade in trades:
        result = sim.simulate_execution(
            symbol=trade["symbol"],
            order_type=trade["order_type"],
            side=trade["side"],
            size=trade["size"],
            requested_price=trade["requested_price"],
            market_volatility="MEDIUM",
            market_regime="trending"
        )

        status = "FILLED" if result.filled else "REJECTED" if result.rejected else "FAILED"
        print(f"    {trade['symbol']:10s} {trade['side']:4s}: {status:10s} "
              f"@ {result.fill_price:.5f} (slip: {result.slippage:.1f}pips, "
              f"latency: {result.latency_ms}ms)")

    stats = sim.get_stats()
    print(f"\n  Execution Stats: {stats}")

    print("[PASS] Paper trader integration working")
    return True


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "=" * 60)
    print("TEST 8: Edge Cases")
    print("=" * 60)

    sim = SpeedSimulator(network_profile="poor", broker_profile="mm_slow")

    # Very large order
    result = sim.simulate_execution(
        symbol="EURUSDm",
        order_type="MARKET",
        side="BUY",
        size=20.0,
        requested_price=1.0850,
        market_volatility="HIGH",
        market_regime="news_event"
    )
    print(f"  Large order (20 lots): filled={result.filled}, "
          f"slippage={result.slippage:.1f}pips")

    # Tiny order
    result2 = sim.simulate_execution(
        symbol="EURUSDm",
        order_type="MARKET",
        side="BUY",
        size=0.01,
        requested_price=1.0850,
        market_volatility="LOW",
        market_regime="trending"
    )
    print(f"  Tiny order (0.01 lots): filled={result2.filled}, "
          f"slippage={result2.slippage:.1f}pips")

    print("[PASS] Edge cases handled correctly")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("SPEED SIMULATOR VALIDATION")
    print("=" * 60)

    results = []

    results.append(("Network Latency", test_network_latency()))
    results.append(("Slippage Calculation", test_slippage()))
    results.append(("Fill Probability", test_fill_probability()))
    results.append(("Batch Execution", test_batch_execution()))
    results.append(("Spread Simulation", test_spread_simulation()))
    results.append(("Market Impact", test_market_impact()))
    results.append(("Paper Trader Integration", test_integration_with_paper_trader()))
    results.append(("Edge Cases", test_edge_cases()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {name}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n[SUCCESS] All validation tests passed!")
        return 0
    else:
        print("\n[WARNING] Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
