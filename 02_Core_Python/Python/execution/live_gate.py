"""Live and demo trading gates.

Implements the full safety checklist before ANY order can be sent to a broker.
"""

import os
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bool_env(name: str, default: bool = False) -> bool:
    return os.environ.get(name, "1" if default else "0").strip() == "1"


def _check_account_verified_real(account_state: dict) -> tuple[bool, str]:
    acct_type = str(account_state.get("account_type", ""))
    verified = bool(account_state.get("account_type_verified", False))
    if not verified:
        return False, "account_not_verified"
    if acct_type != "real":
        return False, f"account_type={acct_type}, expected real"
    return True, "ok"


def _check_account_verified_demo(account_state: dict) -> tuple[bool, str]:
    acct_type = str(account_state.get("account_type", ""))
    verified = bool(account_state.get("account_type_verified", False))
    if not verified:
        return False, "account_not_verified"
    if acct_type != "demo":
        return False, f"account_type={acct_type}, expected demo"
    return True, "ok"


# ---------------------------------------------------------------------------
# Live gate
# ---------------------------------------------------------------------------

def live_trading_allowed(
    config: dict,
    validation_state: dict,
    account_state: dict,
    test_state: dict,
) -> tuple[bool, str]:
    """Return (allowed, reason) for real-money live trading.

    Required gates (from spec):
      1. config.allow_real_live == True
      2. mode == real_live
      3. account verified real
      4. telemetry valid (balance > 0, equity > 0)
      5. tests clean
      6. champion passed
      7. MT5 source
      8. timesteps met
      9. OOS return > 0
      10. demo_canary passed
    """
    cfg_exec = config.get("execution", {}) if isinstance(config, dict) else {}
    if not cfg_exec.get("allow_real_live", False):
        return False, "config.allow_real_live=False"

    mode = os.environ.get("CHAIN_GAMBLER_EXECUTION_MODE", "paper").strip().lower()
    if mode != "live":
        return False, f"mode={mode}, expected live"

    if not _bool_env("CHAIN_GAMBLER_ALLOW_LIVE", False):
        return False, "CHAIN_GAMBLER_ALLOW_LIVE!=1"

    ok, reason = _check_account_verified_real(account_state)
    if not ok:
        return False, reason

    if not account_state.get("telemetry_valid", False):
        return False, "telemetry_invalid"
    if account_state.get("balance", 0.0) <= 0 or account_state.get("equity", 0.0) <= 0:
        return False, "balance_or_equity_non_positive"

    if not test_state.get("tests_clean", False):
        return False, "tests_failing"

    # Champion gates
    champion = validation_state.get("champion", {})
    if not champion.get("passed", False):
        return False, "champion_not_passed"
    if str(champion.get("data_source", "")).lower() != "mt5":
        return False, "champion_data_source_not_mt5"
    timesteps = int(champion.get("timesteps", 0))
    min_ts = int(os.environ.get("CHAIN_GAMBLER_MIN_TIMESTEPS", "10000"))
    if timesteps < min_ts:
        return False, f"timesteps {timesteps} < {min_ts}"
    oos_return = float(champion.get("oos_return", 0.0))
    if oos_return <= 0:
        return False, f"oos_return {oos_return} <= 0"

    # Demo canary passed
    demo_canary = validation_state.get("demo_canary", {})
    if not demo_canary.get("passed", False):
        return False, "demo_canary_not_passed"

    return True, "all_gates_passed"


# ---------------------------------------------------------------------------
# Demo gate
# ---------------------------------------------------------------------------

def demo_trading_allowed(
    config: dict,
    account_state: dict,
    risk_state: dict,
) -> tuple[bool, str]:
    """Return (allowed, reason) for demo-account live trading.

    Required gates:
      1. mode == demo_live  (or env CHAIN_GAMBLER_EXECUTION_MODE == "demo")
      2. account verified demo
      3. telemetry valid
      4. daily loss limit not reached
      5. max open positions not reached
    """
    mode = os.environ.get("CHAIN_GAMBLER_EXECUTION_MODE", "paper").strip().lower()
    # Accept either "demo" or "live" when the *account* is demo.
    # The tighter interpretation used here is: mode must be "demo" or we must
    # explicitly be in a demo-account context.
    if mode not in ("demo", "live"):
        return False, f"mode={mode}, expected demo or live"

    ok, reason = _check_account_verified_demo(account_state)
    if not ok:
        return False, reason

    if not account_state.get("telemetry_valid", False):
        return False, "telemetry_invalid"
    if account_state.get("balance", 0.0) <= 0 or account_state.get("equity", 0.0) <= 0:
        return False, "balance_or_equity_non_positive"

    # Risk-state gates
    if risk_state.get("halt", False):
        return False, "risk_halt"

    daily_pnl = float(risk_state.get("daily_pnl", 0.0))
    max_daily_loss = float(
        risk_state.get("max_daily_loss", config.get("risk", {}).get("max_daily_loss", 1000.0))
    )
    if daily_pnl <= -abs(max_daily_loss):
        return False, "daily_loss_limit_reached"

    open_positions = int(risk_state.get("open_positions", 0))
    max_open = int(
        risk_state.get("max_open_positions", config.get("risk", {}).get("max_open_positions", 6))
    )
    if open_positions >= max_open:
        return False, f"max_open_positions {open_positions} >= {max_open}"

    return True, "demo_gates_passed"
