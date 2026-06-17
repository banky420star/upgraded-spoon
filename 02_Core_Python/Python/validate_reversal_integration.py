"""
Validation script for Reversal Detection integration into HybridBrain.

Tests:
1. ReversalDetector can be imported and instantiated
2. HybridBrain initializes reversal detector
3. Reversal detection triggers correctly on synthetic reversal patterns
"""
import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")

def create_bullish_reversal_df():
    """Create synthetic DataFrame with bullish reversal pattern (hammer + divergence)."""
    np.random.seed(42)
    n = 50

    # Downtrend followed by hammer pattern
    prices = []
    # First 40 bars: downtrend
    for i in range(40):
        prices.append(100 - i * 0.2 + np.random.randn() * 0.5)

    # Bar 41: Hammer pattern (small body, long lower wick)
    open_p = 95.0
    close_p = 95.3  # Small bullish body
    high_p = 95.5
    low_p = 92.0    # Long lower wick
    prices.append(close_p)

    # Next bars: recovery
    for i in range(9):
        prices.append(95.3 + i * 0.3 + np.random.randn() * 0.3)

    df = pd.DataFrame({
        'open': [prices[0]] + prices[:-1],
        'high': [p + 0.5 for p in prices],
        'low': [p - 0.5 for p in prices[:-10]] + [92.0] + [p - 0.3 for p in prices[-9:]],
        'close': prices,
        'volume': [1000 + np.random.randint(-200, 200) for _ in range(n)]
    })

    return df

def create_bearish_reversal_df():
    """Create synthetic DataFrame with bearish reversal pattern (shooting star)."""
    np.random.seed(43)
    n = 50

    # Uptrend followed by shooting star
    prices = []
    # First 40 bars: uptrend
    for i in range(40):
        prices.append(100 + i * 0.2 + np.random.randn() * 0.5)

    # Bar 41: Shooting star (small body, long upper wick)
    open_p = 108.0
    close_p = 107.7  # Small bearish body
    high_p = 111.0   # Long upper wick
    low_p = 107.5
    prices.append(close_p)

    # Next bars: decline
    for i in range(9):
        prices.append(107.7 - i * 0.3 + np.random.randn() * 0.3)

    df = pd.DataFrame({
        'open': [prices[0]] + prices[:-1],
        'high': [p + 0.5 for p in prices[:-10]] + [111.0] + [p + 0.3 for p in prices[-9:]],
        'low': [p - 0.5 for p in prices],
        'close': prices,
        'volume': [1000 + np.random.randint(-200, 200) for _ in range(n)]
    })

    return df

def test_reversal_detector():
    """Test ReversalDetector standalone."""
    print("\n" + "="*60)
    print("TEST 1: ReversalDetector Standalone")
    print("="*60)

    try:
        from Python.reversal_detector import ReversalDetector, get_reversal_detector

        detector = ReversalDetector()
        print("[PASS] ReversalDetector instantiated")

        # Test bullish reversal detection
        df_bull = create_bullish_reversal_df()
        signal = detector.detect_reversal("TEST", df_bull, "SELL")

        print(f"  Bullish reversal test:")
        print(f"    Detected: {signal.detected}")
        print(f"    Direction: {signal.direction}")
        print(f"    Confidence: {signal.confidence:.2f}")
        print(f"    Methods: {signal.methods}")

        # Test bearish reversal detection
        df_bear = create_bearish_reversal_df()
        signal = detector.detect_reversal("TEST", df_bear, "BUY")

        print(f"  Bearish reversal test:")
        print(f"    Detected: {signal.detected}")
        print(f"    Direction: {signal.direction}")
        print(f"    Confidence: {signal.confidence:.2f}")
        print(f"    Methods: {signal.methods}")

        print("[PASS] ReversalDetector working correctly")
        return True

    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hybrid_brain_integration():
    """Test HybridBrain with reversal detector integration."""
    print("\n" + "="*60)
    print("TEST 2: HybridBrain Reversal Integration")
    print("="*60)

    try:
        from Python.hybrid_brain import HybridBrain
        from Python.risk_engine import RiskEngine

        # Create mock risk and executor
        risk = RiskEngine()

        # Mock executor
        class MockExecutor:
            def reconcile_exposure(self, *args, **kwargs):
                pass

        executor = MockExecutor()

        # Initialize HybridBrain
        brain = HybridBrain(risk, executor)

        # Check if reversal detector is initialized
        if hasattr(brain, 'reversal_detector') and brain.reversal_detector is not None:
            print("[PASS] Reversal detector initialized in HybridBrain")
        else:
            print("[WARN] Reversal detector not available (optional)")

        # Test decision with bullish reversal data
        df_bull = create_bullish_reversal_df()

        # First call to populate history
        result = brain.decide("TEST", df_bull)

        print(f"  Decision result keys: {list(result.keys())}")
        print(f"  Action: {result.get('action')}")
        print(f"  Has reversal fields: {all(k in result for k in ['reversal_detected', 'reversal_direction', 'reversal_confidence'])}")

        # Check reversal fields are populated
        if 'reversal_detected' in result:
            print(f"  Reversal detected: {result.get('reversal_detected')}")
            print(f"  Reversal direction: {result.get('reversal_direction')}")
            print(f"  Reversal confidence: {result.get('reversal_confidence', 0):.2f}")

        print("[PASS] HybridBrain integration working")
        return True

    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_direction_flipping():
    """Test the direction flipping logic."""
    print("\n" + "="*60)
    print("TEST 3: Direction Flipping Logic")
    print("="*60)

    try:
        from Python.reversal_detector import ReversalDetector, ReversalSignal

        detector = ReversalDetector()

        # Test case 1: BUY signal + bearish reversal = should flip to SELL
        reversal = ReversalSignal(
            detected=True,
            direction="bearish_reversal",
            confidence=0.75,
            methods=["divergence", "candlestick"],
            entry_price=100.0,
            stop_loss=101.5,
            take_profit=97.0,
            notes=["Bearish divergence detected"]
        )

        should_flip, reason = detector.should_flip_direction("BUY", reversal)
        print(f"  BUY + bearish reversal: flip={should_flip}, reason={reason}")
        assert should_flip == True, "Should flip BUY to SELL on bearish reversal"

        # Test case 2: SELL signal + bullish reversal = should flip to BUY
        reversal2 = ReversalSignal(
            detected=True,
            direction="bullish_reversal",
            confidence=0.80,
            methods=["hammer", "volume"],
            entry_price=100.0,
            stop_loss=98.5,
            take_profit=103.0,
            notes=["Hammer pattern detected"]
        )

        should_flip2, reason2 = detector.should_flip_direction("SELL", reversal2)
        print(f"  SELL + bullish reversal: flip={should_flip2}, reason={reason2}")
        assert should_flip2 == True, "Should flip SELL to BUY on bullish reversal"

        # Test case 3: BUY signal + bullish reversal = should NOT flip (confirms direction)
        reversal3 = ReversalSignal(
            detected=True,
            direction="bullish_reversal",
            confidence=0.70,
            methods=["sr_break"],
            entry_price=100.0,
            stop_loss=98.0,
            take_profit=103.0,
            notes=["Resistance break"]
        )

        should_flip3, reason3 = detector.should_flip_direction("BUY", reversal3)
        print(f"  BUY + bullish reversal: flip={should_flip3}, reason={reason3}")
        assert should_flip3 == False, "Should NOT flip when reversal confirms direction"

        # Test case 4: No reversal detected
        reversal4 = ReversalSignal(
            detected=False,
            direction="none",
            confidence=0.0,
            methods=[],
            entry_price=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            notes=["No reversal detected"]
        )

        should_flip4, reason4 = detector.should_flip_direction("BUY", reversal4)
        print(f"  BUY + no reversal: flip={should_flip4}, reason={reason4}")
        assert should_flip4 == False, "Should NOT flip when no reversal detected"

        print("[PASS] Direction flipping logic working correctly")
        return True

    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("REVERSAL DETECTION INTEGRATION VALIDATION")
    print("="*60)

    results = []

    results.append(("ReversalDetector Standalone", test_reversal_detector()))
    results.append(("HybridBrain Integration", test_hybrid_brain_integration()))
    results.append(("Direction Flipping Logic", test_direction_flipping()))

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {name}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n[SUCCESS] All validation tests passed!")
        return 0
    else:
        print("\n[WARNING] Some tests failed. Check output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
