#!/usr/bin/env python3
"""Direct MT5 connection test."""
import sys
import os

# Add paths
sys.path.insert(0, '02_Core_Python')
sys.path.insert(0, '02_Core_Python/Python')

try:
    import MetaTrader5 as mt5
    print("[OK] MetaTrader5 module imported")

    # Try to initialize
    print("\nInitializing MT5...")
    if not mt5.initialize():
        error = mt5.last_error()
        print(f"[FAIL] MT5 initialize failed: {error}")
        sys.exit(1)
    print("[OK] MT5 initialized successfully")

    # Get account info
    print("\nGetting account info...")
    account = mt5.account_info()
    if account is None:
        error = mt5.last_error()
        print(f"[FAIL] Failed to get account info: {error}")
        mt5.shutdown()
        sys.exit(1)

    print("[OK] Account connected!")
    print(f"  Login: {account.login}")
    print(f"  Server: {account.server}")
    print(f"  Balance: ${account.balance:,.2f}")
    print(f"  Equity: ${account.equity:,.2f}")
    print(f"  Currency: {account.currency}")
    print(f"  Leverage: 1:{account.leverage}")

    # Get symbols
    print("\nGetting available symbols...")
    symbols = mt5.symbols_get()
    if symbols:
        print(f"[OK] {len(symbols)} symbols available")

    # Check for specific symbols
    target_symbols = ["BTCUSDm", "XAUUSDm", "EURUSDm", "GBPUSDm"]
    print("\nChecking trading symbols:")
    for sym in target_symbols:
        tick = mt5.symbol_info_tick(sym)
        if tick:
            print(f"  [OK] {sym}: Bid={tick.bid}, Ask={tick.ask}")
        else:
            print(f"  [FAIL] {sym}: Not available")

    mt5.shutdown()
    print("\n[OK] MT5 connection test complete!")

except ImportError as e:
    print(f"[FAIL] Failed to import MetaTrader5: {e}")
    print("  Run: .venv312/Scripts/pip install MetaTrader5")
    sys.exit(1)
except Exception as e:
    print(f"[FAIL] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
