from __future__ import annotations

import json
import os
import subprocess
import sys
import time
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ACTIVE_JSON = os.path.join(_PROJECT_ROOT, "models", "registry", "active.json")
_TESTS_DIR = os.path.join(_PROJECT_ROOT, "tests")
_PYTEST_CACHE = {"passed": None, "checked_at": 0.0}


def get_execution_mode() -> str:
    """Return execution mode from env var, defaulting to paper."""
    mode = os.environ.get("CHAIN_GAMBLER_EXECUTION_MODE", "paper").strip().lower()
    if mode not in ("paper", "live", "demo"):
        return "paper"
    return mode


def _check_account_telemetry() -> dict:
    """Validate MT5 account telemetry: balance>0, equity>0, connected=True."""
    try:
        from Python.mt5_compat import mt5 as _mt5
    except Exception as exc:
        return {"ok": False, "reason": f"mt5_unavailable:{exc}"}
    try:
        if not _mt5.initialize():
            return {"ok": False, "reason": "mt5_not_initialized"}
        info = _mt5.account_info()
        if info is None:
            return {"ok": False, "reason": "mt5_account_info_none"}
        balance = float(getattr(info, "balance", 0.0) or 0.0)
        equity = float(getattr(info, "equity", 0.0) or 0.0)
        if balance <= 0:
            return {"ok": False, "reason": f"balance_non_positive:{balance}"}
        if equity <= 0:
            return {"ok": False, "reason": f"equity_non_positive:{equity}"}
        tinfo = _mt5.terminal_info()
        connected = bool(getattr(tinfo, "connected", False)) if tinfo else False
        if not connected:
            return {"ok": False, "reason": "mt5_not_connected"}
        return {
            "ok": True,
            "balance": balance,
            "equity": equity,
            "connected": connected,
        }
    except Exception as exc:
        return {"ok": False, "reason": f"telemetry_exception:{exc}"}


def _check_pytest_passes() -> dict:
    """Run pytest on tests/ and return whether it passes. Cached for 5 minutes."""
    global _PYTEST_CACHE
    now = time.time()
    if _PYTEST_CACHE["passed"] is not None and (now - _PYTEST_CACHE["checked_at"]) < 300:
        return {
            "ok": bool(_PYTEST_CACHE["passed"]),
            "reason": None if _PYTEST_CACHE["passed"] else "pytest_cached_failure",
            "cached": True,
        }
    if not os.path.isdir(_TESTS_DIR):
        _PYTEST_CACHE = {"passed": False, "checked_at": now}
        return {"ok": False, "reason": "tests_directory_missing", "cached": False}
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", _TESTS_DIR, "-q", "--tb=no"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        passed = result.returncode == 0
        _PYTEST_CACHE = {"passed": passed, "checked_at": now}
        if not passed:
            return {
                "ok": False,
                "reason": f"pytest_failure:exit_code={result.returncode}",
                "stdout": result.stdout[-500:] if result.stdout else "",
                "cached": False,
            }
        return {"ok": True, "reason": None, "cached": False}
    except Exception as exc:
        _PYTEST_CACHE = {"passed": False, "checked_at": now}
        return {"ok": False, "reason": f"pytest_exception:{exc}", "cached": False}


def _check_champion_validated() -> dict:
    """Validate champion: positive backtest return, MT5 data source, sufficient timesteps."""
    if not os.path.exists(_ACTIVE_JSON):
        return {"ok": False, "reason": "active_json_missing"}
    try:
        with open(_ACTIVE_JSON, "r", encoding="utf-8") as f:
            active = json.load(f)
    except Exception as exc:
        return {"ok": False, "reason": f"active_json_read_error:{exc}"}

    # Collect all champions (global + per-symbol)
    champions = []
    global_champ = active.get("champion")
    global_meta = active.get("champion_metadata") or active.get("registry_metadata", {}).get("champion_metadata", {})
    if global_champ:
        champions.append(("global", global_champ, global_meta))
    symbols = active.get("symbols", {})
    for sym, cfg in (symbols or {}).items():
        champ = cfg.get("champion")
        if champ:
            meta = cfg.get("metadata", {})
            champions.append((sym, champ, meta))

    if not champions:
        return {"ok": False, "reason": "no_champion_registered"}

    min_timesteps = int(os.environ.get("CHAIN_GAMBLER_MIN_TIMESTEPS", "10000"))
    ok_champions = []
    for scope, path, meta in champions:
        if not isinstance(meta, dict):
            meta = {}
        scorecard = meta.get("scorecard") or meta
        evaluation = meta.get("evaluation") if isinstance(meta, dict) else None
        data_source = meta.get("data_source") or scorecard.get("data_source") or "unknown"
        timesteps = int(meta.get("timesteps") or scorecard.get("timesteps") or 0)

        total_return = 0.0
        if evaluation and isinstance(evaluation, dict):
            per_symbol = evaluation.get("per_symbol", [])
            if per_symbol and isinstance(per_symbol, list):
                total_return = float(per_symbol[0].get("total_return", 0.0))
            else:
                total_return = float(evaluation.get("total_return", 0.0))
        else:
            total_return = float(scorecard.get("total_return", 0.0))

        gates = {
            "positive_backtest_return": total_return >= 0.0,
            "mt5_data_source": str(data_source).lower() == "mt5",
            "sufficient_timesteps": timesteps >= min_timesteps,
        }
        if all(gates.values()):
            ok_champions.append({"scope": scope, "path": path, "gates": gates})

    if ok_champions:
        return {"ok": True, "reason": None, "champions": ok_champions}
    return {
        "ok": False,
        "reason": "no_champion_passes_gates",
        "details": f"requires positive backtest return, MT5 data source, and >= {min_timesteps} timesteps",
    }


def _check_paper_canary_passed() -> dict:
    """Check if any paper canary has passed validation."""
    if not os.path.exists(_ACTIVE_JSON):
        return {"ok": False, "reason": "active_json_missing"}
    try:
        with open(_ACTIVE_JSON, "r", encoding="utf-8") as f:
            active = json.load(f)
    except Exception as exc:
        return {"ok": False, "reason": f"active_json_read_error:{exc}"}

    global_canary_state = active.get("canary_state") or {}
    if isinstance(global_canary_state, dict) and global_canary_state.get("passed"):
        return {"ok": True, "reason": None, "scope": "global"}

    symbols = active.get("symbols", {})
    for sym, cfg in (symbols or {}).items():
        state = cfg.get("canary_state") or {}
        if isinstance(state, dict) and state.get("passed"):
            return {"ok": True, "reason": None, "scope": sym}

    return {"ok": False, "reason": "no_canary_passed"}


def live_trading_allowed(force_refresh: bool = False) -> dict:
    """
    Central live-trading safety gate.

    Returns a dict with:
      - allowed: bool
      - mode: str (paper or live)
      - gates: list of individual gate results
    """
    if force_refresh:
        global _PYTEST_CACHE
        _PYTEST_CACHE = {"passed": None, "checked_at": 0.0}

    mode = get_execution_mode()
    if mode != "live":
        return {
            "allowed": False,
            "mode": "paper",
            "gates": [{"name": "execution_mode", "ok": True, "reason": "paper_mode_forced"}],
        }

    env_opt_in = os.environ.get("CHAIN_GAMBLER_ALLOW_LIVE", "0").strip()
    if env_opt_in != "1":
        return {
            "allowed": False,
            "mode": "paper",
            "gates": [{"name": "allow_live_env", "ok": False, "reason": "CHAIN_GAMBLER_ALLOW_LIVE!=1"}],
        }

    gates = [
        {"name": "allow_live_env", "ok": True, "reason": None},
    ]

    tel = _check_account_telemetry()
    gates.append({"name": "account_telemetry", "ok": tel["ok"], "reason": tel.get("reason")})

    tests = _check_pytest_passes()
    gates.append({"name": "pytest_passes", "ok": tests["ok"], "reason": tests.get("reason")})

    champ = _check_champion_validated()
    gates.append({"name": "champion_validated", "ok": champ["ok"], "reason": champ.get("reason")})

    canary = _check_paper_canary_passed()
    gates.append({"name": "paper_canary_passed", "ok": canary["ok"], "reason": canary.get("reason")})

    all_ok = all(g["ok"] for g in gates)
    force_live = os.environ.get("CHAIN_GAMBLER_FORCE_LIVE", "0").strip() == "1"
    
    if not all_ok and not force_live:
        mode = "paper"
        
    return {
        "allowed": all_ok or force_live,
        "mode": mode,
        "gates": gates,
    }
