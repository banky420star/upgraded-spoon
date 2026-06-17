"""MT5 account telemetry verifier.

Extracts and validates account metadata from MT5 account_info() output.
Rejects accounts with non-positive balance or equity.
"""

import os
import re
from typing import Any


_DEMO_HINTS = re.compile(
    r"demo|test|practice|training|virtual|simulat|fake|play",
    re.IGNORECASE,
)


def _is_demo_account(info: dict) -> bool:
    """Heuristic: look for demo keywords in company / name / server / login comment."""
    for key in ("company", "name", "server", "login"):
        val = str(info.get(key, ""))
        if _DEMO_HINTS.search(val):
            return True
    return False


def verify_account(mt5_account_info: dict[str, Any] | None) -> dict[str, Any]:
    """Validate MT5 account snapshot and return a normalized status dict.

    Returns:
        {
            "account_type":       "demo" | "real" | "unknown",
            "account_type_verified": bool,
            "telemetry_valid":      bool,
            "balance":              float,
            "equity":               float,
            "currency":             str,
            "server":               str,
            "login_masked":         str,
        }
    """
    if not isinstance(mt5_account_info, dict):
        mt5_account_info = {}

    balance = float(mt5_account_info.get("balance", 0.0) or 0.0)
    equity = float(mt5_account_info.get("equity", 0.0) or 0.0)
    currency = str(mt5_account_info.get("currency", "USD") or "USD")
    server = str(mt5_account_info.get("server", "") or "")
    login_raw = str(mt5_account_info.get("login", "") or "")

    # Mask login: keep first 2 and last 2 chars, replace middle with ***
    login_masked = login_raw
    if len(login_raw) > 4:
        login_masked = f"{login_raw[:2]}***{login_raw[-2:]}"
    elif len(login_raw) > 0:
        login_masked = "***"

    # Telemetry validity: must have positive balance and equity
    telemetry_valid = balance > 0.0 and equity > 0.0

    # Determine account type
    env_type = os.environ.get("CHAIN_GAMBLER_ACCOUNT_TYPE", "").strip().lower()
    if env_type in ("demo", "real"):
        account_type = env_type
    elif _is_demo_account(mt5_account_info):
        account_type = "demo"
    else:
        account_type = "real" if login_raw and server else "unknown"

    account_type_verified = telemetry_valid and account_type in ("demo", "real")

    return {
        "account_type": account_type,
        "account_type_verified": account_type_verified,
        "telemetry_valid": telemetry_valid,
        "balance": balance,
        "equity": equity,
        "currency": currency,
        "server": server,
        "login_masked": login_masked,
    }
