"""Execution mode resolver.

Reads environment variables and config to decide which of the four canonical
modes the system should run in:
  paper_sim        — simulation only, no real broker connection
  demo_live        — connected to a demo MT5 account, real orders but play money
  real_live_locked — real mode requested but safety lock is engaged
  real_live        — full live trading on a real account (all gates required)
"""

import os
from typing import Literal

ExecutionMode = Literal["paper_sim", "demo_live", "real_live_locked", "real_live"]


def resolve_mode(config: dict | None = None) -> ExecutionMode:
    """Return the canonical execution mode.

    Env vars consulted (in order of precedence):
      CHAIN_GAMBLER_EXECUTION_MODE   — "paper" | "demo" | "live"
      CHAIN_GAMBLER_ALLOW_LIVE       — "1" to unlock real_live
      CHAIN_GAMBLER_ACCOUNT_TYPE     — "demo" | "real" (fallback when mode==live)
    """
    config = config or {}

    env_mode = os.environ.get("CHAIN_GAMBLER_EXECUTION_MODE", "paper").strip().lower()
    allow_live = os.environ.get("CHAIN_GAMBLER_ALLOW_LIVE", "0").strip() == "1"
    env_account_type = os.environ.get("CHAIN_GAMBLER_ACCOUNT_TYPE", "").strip().lower()

    # Config-level overrides (lower precedence than env)
    cfg_exec = config.get("execution", {}) if isinstance(config, dict) else {}
    cfg_mode = str(cfg_exec.get("mode", "")).strip().lower()
    cfg_allow = bool(cfg_exec.get("allow_real_live", False))
    cfg_account = str(cfg_exec.get("account_type", "")).strip().lower()

    # Merge: env wins over config
    mode = env_mode if env_mode in ("paper", "demo", "live") else cfg_mode if cfg_mode in ("paper", "demo", "live") else "paper"
    allow = allow_live or cfg_allow
    account_type = env_account_type or cfg_account

    if mode == "paper":
        return "paper_sim"

    if mode == "demo":
        return "demo_live"

    if mode == "live":
        if not allow:
            return "real_live_locked"
        # When allow_live is true, the mode is real_live regardless of account_type.
        # The account_verifier / live_gate will block if the *actual* MT5 account
        # turns out to be demo.  resolve_mode only tells us the *intent*.
        return "real_live"

    # Fallback
    return "paper_sim"
