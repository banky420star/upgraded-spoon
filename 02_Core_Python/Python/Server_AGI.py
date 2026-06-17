import atexit
import datetime
import json
import os
import signal
import subprocess
import threading
import time

try:
    from Python.mt5_compat import mt5
except Exception as exc:
    _MT5_IMPORT_ERROR = exc

    class _MissingMetaTrader5:
        def __getattr__(self, name):
            raise RuntimeError(
                "MetaTrader5 is required for live runtime operations and is unavailable in this environment."
            ) from _MT5_IMPORT_ERROR

    mt5 = _MissingMetaTrader5()

import pandas as pd
from loguru import logger

try:
    from Python import paper_trading as _srv_paper
except Exception:
    _srv_paper = None

from Python.config_utils import DEFAULT_TRADING_SYMBOLS, load_project_config, resolve_trading_symbols
try:
    from Python import live_safety
except Exception:
    live_safety = None
try:
    from Python.reversal_detector import get_reversal_detector
except Exception:
    get_reversal_detector = None

# Temporarily disabled — reversal detector was flattening every signal in ranging markets
get_reversal_detector = None


def _is_paper_mode() -> bool:
    return _srv_paper is not None and _srv_paper.get_mode() == "paper"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK_DIR = os.path.join(BASE_DIR, ".tmp")
LOCK_PATH = os.path.join(LOCK_DIR, "server_agi.lock")
LOG_DIR = os.path.join(BASE_DIR, "logs")
SERVER_LOG = os.path.join(LOG_DIR, "server.log")
AUDIT_LOG = os.path.join(LOG_DIR, "audit_events.jsonl")
TRADE_EVENTS_LOG = os.path.join(LOG_DIR, "trade_events.jsonl")
ACTIVE_MODELS_PATH = os.path.join(BASE_DIR, "models", "registry", "active.json")

os.makedirs(LOG_DIR, exist_ok=True)
logger.add(SERVER_LOG, rotation="10 MB", level="INFO")

_shutdown_flag = threading.Event()


def _shutdown_handler(signum, frame):
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    _shutdown_flag.set()


SYMBOL_EXECUTION_PROFILES = {
    "BTCUSDm": {
        "ppo_weight": 0.65,
        "dreamer_weight": 0.25,
        "agi_weight": 0.10,
        "min_trade_threshold": 0.18,
        "max_abs_target": 1.00,
        "cooldown_sec": 30,
    },
    "XAUUSDm": {
        "ppo_weight": 0.50,
        "dreamer_weight": 0.20,
        "agi_weight": 0.30,
        "min_trade_threshold": 0.12,
        "max_abs_target": 0.75,
        "cooldown_sec": 45,
    },
}


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _json_default(v):
    if isinstance(v, (datetime.datetime, datetime.date)):
        return v.isoformat()
    return str(v)


_JSONL_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per JSONL file before rotation


def _rotate_jsonl_if_needed(path: str) -> None:
    """Rename path -> path.1 (keeping one backup) when the file exceeds _JSONL_MAX_BYTES."""
    try:
        if os.path.exists(path) and os.path.getsize(path) >= _JSONL_MAX_BYTES:
            backup = path + ".1"
            if os.path.exists(backup):
                os.remove(backup)
            os.rename(path, backup)
    except Exception as exc:
        logger.warning(f"JSONL rotation failed for {path} (non-fatal): {exc}")


def _append_jsonl(path: str, row: dict):
    _rotate_jsonl_if_needed(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True, default=_json_default) + "\n")


def _append_audit(event: str, payload: dict):
    _append_jsonl(
        AUDIT_LOG,
        {
            "ts": _utc_now().isoformat(timespec="microseconds"),
            "event": event,
            "payload": payload,
        },
    )


def _append_trade_event(event: str, payload: dict):
    row = {
        "ts": _utc_now().isoformat(timespec="microseconds"),
        "event": event,
        "payload": payload,
    }
    _append_jsonl(TRADE_EVENTS_LOG, row)
    _append_audit(event, payload)


def _load_runtime_components():
    from Python.agi_brain import SmartAGI
    from Python.event_intel import EventIntel
    from Python.hybrid_brain import HybridBrain
    from Python.mt5_executor import MT5Executor
    from Python.risk_engine import RiskEngine
    from Python.risk_supervisor import RiskSupervisor
    from Python.trade_learning import build_trade_learning
    from alerts.telegram_alerts import TelegramAlerter

    return {
        "SmartAGI": SmartAGI,
        "EventIntel": EventIntel,
        "HybridBrain": HybridBrain,
        "MT5Executor": MT5Executor,
        "RiskEngine": RiskEngine,
        "RiskSupervisor": RiskSupervisor,
        "build_trade_learning": build_trade_learning,
        "TelegramAlerter": TelegramAlerter,
    }



def _read_active_models():
    if not os.path.exists(ACTIVE_MODELS_PATH):
        return {"champion": None, "canary": None}
    try:
        with open(ACTIVE_MODELS_PATH, "r", encoding="utf-8") as f:
            d = json.load(f)
        if not isinstance(d, dict):
            return {"champion": None, "canary": None}
        return {"champion": d.get("champion"), "canary": d.get("canary")}
    except Exception as exc:
        logger.warning(f"Failed to read active models registry (using defaults): {exc}")
        return {"champion": None, "canary": None}


def _write_live_state(risk, symbols, last_symbol_state, models, training_state=None):
    """Persist lightweight runtime snapshot for the standalone API server."""
    snap = {
        "timestamp": time.time(),
        "registry": models,
        "symbols": {},
        "trading": {
            "account": {
                "balance": float(getattr(risk, "_mt5_balance", 0.0) or 0.0),
                "equity": float(getattr(risk, "_current_equity", 0.0) or 0.0),
                "floatingPnl": float(getattr(risk, "_mt5_profit", 0.0) or 0.0),
                "realizedToday": float(getattr(risk, "realized_pnl_today", 0.0) or 0.0),
            },
            "risk": {
                "drawdownPct": float(getattr(risk, "current_dd", 0.0) or 0.0),
                "canTrade": bool(getattr(risk, "halt", False)) is False,
                "haltReason": str(getattr(risk, "_halt_reason", "")) or "",
            },
        },
        "training": training_state
        if isinstance(training_state, dict)
        else {"active_canary": False, "cycles_completed": 0},
    }
    for sym in symbols:
        sstate = last_symbol_state.get(str(sym), {})
        snap["symbols"][str(sym)] = {
            "champion": models.get("champion"),
            "canary": models.get("canary"),
            "canary_pnl": 0.0,
            "canary_set_time": None,
            "canary_trades": 0,
            "signal": sstate.get("signal", "UNKNOWN"),
            "regime": sstate.get("regime", "--"),
            "confidence": sstate.get("confidence", 0.0),
            "rainforest_regime": sstate.get("rainforest_regime", "ranging"),
            "rainforest_confidence": sstate.get("rainforest_confidence", 0.0),
            "ppo_exposure": sstate.get("ppo_exposure", 0.0),
            "dreamer_exposure": sstate.get("dreamer_exposure", 0.0),
            "blend_exposure": sstate.get("blend_exposure", 0.0),
            "floating_pnl": sstate.get("floating_pnl", 0.0),
            "open_positions": sstate.get("open_positions", 0),
        }
    path = os.path.join(BASE_DIR, "live_state.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snap, f, default=_json_default)
    except Exception as exc:
        logger.warning(f"Failed to persist live_state.json (non-fatal): {exc}")


def _training_state():
    out = {
        "lstm_running": False,
        "drl_running": False,
        "cycle_running": False,
        "lstm_symbol": None,
        "lstm_epoch": None,
        "lstm_epochs_total": None,
        "lstm_score": None,
        "drl_symbol": None,
        "drl_score": None,
    }

    # ── Detect training processes (Windows PowerShell + macOS/Linux ps) ──
    try:
        lines = []
        if sys.platform == "win32":
            cmd = (
                "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                "Select-Object CommandLine | ConvertTo-Json -Depth 3"
            )
            raw = subprocess.check_output(["powershell", "-NoProfile", "-Command", cmd], text=True, timeout=6)
            rows = json.loads(raw)
            if isinstance(rows, dict):
                rows = [rows]
            lines = [str((r or {}).get("CommandLine") or "").lower().replace("\\", "/") for r in (rows or [])]
        else:
            raw = subprocess.check_output(["ps", "-eo", "command"], text=True, timeout=6)
            lines = [x.lower().replace("\\", "/") for x in raw.splitlines()]
        out["lstm_running"] = any("training/train_lstm.py" in x for x in lines)
        out["drl_running"] = any("training/train_drl.py" in x for x in lines)
        out["cycle_running"] = any(("tools/champion_cycle.py" in x or "tools/champion_cycle_loop.py" in x) for x in lines)
    except Exception as exc:
        logger.debug(f"Training process scan failed (non-fatal, using fallbacks): {exc}")

    # ── Fallback: infer running from recent log activity ──
    now = time.time()
    try:
        lstm_log = os.path.join(LOG_DIR, "lstm_training.log")
        if os.path.exists(lstm_log) and (now - os.path.getmtime(lstm_log)) < 180:
            out["lstm_running"] = True
    except Exception as exc:
        logger.debug(f"lstm log mtime check failed: {exc}")
    try:
        ppo_log = os.path.join(LOG_DIR, "ppo_training.log")
        if os.path.exists(ppo_log) and (now - os.path.getmtime(ppo_log)) < 180:
            out["drl_running"] = True
    except Exception as exc:
        logger.debug(f"ppo log mtime check failed: {exc}")
    try:
        cycle_log = os.path.join(LOG_DIR, "champion_cycle.log")
        if os.path.exists(cycle_log) and (now - os.path.getmtime(cycle_log)) < 180:
            out["cycle_running"] = True
    except Exception as exc:
        logger.debug(f"cycle log mtime check failed: {exc}")

    # ── Parse LSTM log for progress metadata ──
    try:
        lstm_lines = []
        lstm_log = os.path.join(LOG_DIR, "lstm_training.log")
        if os.path.exists(lstm_log):
            with open(lstm_log, "r", encoding="utf-8", errors="replace") as f:
                lstm_lines = [x.rstrip("\n") for x in f.readlines()[-40:]]
        for line in reversed(lstm_lines):
            if " | epoch " in line and " | loss " in line:
                parts = line.split("|")
                if len(parts) >= 4:
                    left = parts[1].strip()
                    ep = parts[2].strip().replace("epoch", "").strip()
                    score = parts[3].strip()
                    sym = left.split()[-1]
                    out["lstm_symbol"] = sym
                    if "/" in ep:
                        a, b = ep.split("/", 1)
                        out["lstm_epoch"] = a.strip()
                        out["lstm_epochs_total"] = b.strip()
                    if "acc" in score.lower():
                        out["lstm_score"] = score
                    break
    except Exception:
        pass

    # ── Parse PPO log for progress metadata ──
    try:
        ppo_lines = []
        ppo_log = os.path.join(LOG_DIR, "ppo_training.log")
        if os.path.exists(ppo_log):
            with open(ppo_log, "r", encoding="utf-8", errors="replace") as f:
                ppo_lines = [x.rstrip("\n") for x in f.readlines()[-80:]]
        for line in reversed(ppo_lines):
            if "DRL Training | symbols=" in line:
                idx = line.find("symbols=")
                if idx >= 0:
                    chunk = line[idx + len("symbols=") :]
                    if "[" in chunk and "]" in chunk:
                        s = chunk[chunk.find("[") + 1 : chunk.find("]")]
                        first = s.split(",")[0].strip().strip("'\"")
                        out["drl_symbol"] = first or None
            if "best_score=" in line:
                out["drl_score"] = line.split("best_score=", 1)[1].strip()
                if out["drl_symbol"] is not None:
                    break
    except Exception:
        pass

    return out


def _runtime_owner_health():
    out = {"ok": True, "issues": []}
    try:
        cmd = (
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
            "Select-Object ProcessId,ParentProcessId,ExecutablePath,CommandLine | ConvertTo-Json -Depth 4"
        )
        raw = subprocess.check_output(["powershell", "-NoProfile", "-Command", cmd], text=True, timeout=6)
        rows = json.loads(raw)
        if isinstance(rows, dict):
            rows = [rows]
    except Exception:
        return out

    tokens = [
        ("server", "python.server_agi"),
        ("ui", "tools/project_status_ui.py"),
        ("cycle", "tools/champion_cycle.py"),
        ("train_lstm", "training/train_lstm.py"),
        ("train_drl", "training/train_drl.py"),
    ]
    try:
        cfg = _load_cfg(live=True)
        max_parallel_roots = max(1, len(resolve_trading_symbols(cfg, env_keys=("AGI_RUNTIME_SYMBOLS",), fallback=DEFAULT_TRADING_SYMBOLS)))
    except Exception:
        max_parallel_roots = max(1, len(DEFAULT_TRADING_SYMBOLS))
    parallel_roles = {"train_lstm", "train_drl"}

    for role, token in tokens:
        matches = []
        for r in rows or []:
            cmdline = str((r or {}).get("CommandLine") or "").lower().replace("\\", "/")
            if token in cmdline:
                matches.append(
                    {
                        "pid": int((r or {}).get("ProcessId") or 0),
                        "ppid": int((r or {}).get("ParentProcessId") or 0),
                        "exe": str((r or {}).get("ExecutablePath") or ""),
                    }
                )
        if not matches:
            continue

        pid_set = {m["pid"] for m in matches}
        roots = [m for m in matches if m["ppid"] not in pid_set]
        exe_paths = sorted({m["exe"].lower() for m in matches if m["exe"]})

        # Windows venv redirector chain: venv launcher roots the tree and the base
        # interpreter appears only as a child process for the same role token.
        allowed_paths = {
            "users\\administrator\\desktop\\python.exe",
            ".venv312\\scripts\\python.exe",
            ".venv\\scripts\\python.exe",
        }
        if len(roots) == 1 and exe_paths and all(any(token in p for token in allowed_paths) for p in exe_paths):
            non_root_children_ok = True
            for m in matches:
                if m["pid"] != (roots[0]["pid"] if roots else 0) and m["ppid"] not in pid_set:
                    non_root_children_ok = False
                    break
            if non_root_children_ok:
                continue

        if len(roots) > 1 and role in parallel_roles and len(roots) <= max_parallel_roots:
            continue

        if len(roots) > 1:
            out["ok"] = False
            out["issues"].append(
                {
                    "role": role,
                    "type": "multiple_root_owners",
                    "root_pids": [m["pid"] for m in roots],
                    "exe_paths": exe_paths,
                }
            )
        elif len(exe_paths) > 1:
            out["ok"] = False
            out["issues"].append(
                {
                    "role": role,
                    "type": "mixed_executables",
                    "root_pids": [m["pid"] for m in roots] or [matches[0]["pid"]],
                    "exe_paths": exe_paths,
                }
            )
    return out


def _acquire_single_instance_lock():
    """Improved single-instance lock using filelock (when available) + stale PID cleanup.
    Reduces classic TOCTOU window compared to pure check-then-create.
    """
    os.makedirs(LOCK_DIR, exist_ok=True)

    try:
        from filelock import FileLock
        # Use a .lock file for the advisory lock (more robust across restarts)
        lock = FileLock(LOCK_PATH + ".lock", timeout=0)
        lock.acquire()
        # Write our PID into the human-readable lock file for diagnostics
        try:
            with open(LOCK_PATH, "w") as f:
                f.write(str(os.getpid()))
        except Exception:
            pass

        def _cleanup_lock():
            try:
                lock.release()
            except Exception:
                pass
            try:
                if os.path.exists(LOCK_PATH):
                    os.remove(LOCK_PATH)
            except Exception:
                pass
        return True, _cleanup_lock
    except Exception:
        # Fallback to previous O_EXCL logic if filelock unavailable or fails
        pass

    # Fallback path (original improved logic)
    if os.path.exists(LOCK_PATH):
        try:
            with open(LOCK_PATH, "r") as f:
                old_pid = int(f.read().strip())
            try:
                os.kill(old_pid, 0)
                return False, None
            except ProcessLookupError:
                try:
                    os.remove(LOCK_PATH)
                except Exception:
                    pass
            except PermissionError:
                return False, None
        except Exception:
            try:
                os.remove(LOCK_PATH)
            except Exception:
                pass

    try:
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode("utf-8"))
        os.close(fd)
    except FileExistsError:
        return False, None

    def _cleanup_lock():
        try:
            if os.path.exists(LOCK_PATH):
                os.remove(LOCK_PATH)
        except Exception:
            pass

    return True, _cleanup_lock

    atexit.register(_cleanup_lock)
    return True


def _load_cfg(live: bool = False):
    return load_project_config(BASE_DIR, live_mode=bool(live))


def _resolve_env_ref(v):
    if isinstance(v, str) and v.startswith("ENV:"):
        return os.environ.get(v.split(":", 1)[1])
    return v


def _load_telegram_cfg(cfg):
    tcfg = cfg.get("telegram", {}) or {}
    token = os.environ.get("TELEGRAM_TOKEN") or _resolve_env_ref(tcfg.get("token"))
    chat_id = os.environ.get("TELEGRAM_CHAT_ID") or _resolve_env_ref(tcfg.get("chat_id"))

    if token in ("", "YOUR_BOT_TOKEN_HERE"):
        token = None
    if chat_id in ("", "YOUR_CHAT_ID_HERE"):
        chat_id = None

    return token, chat_id


def _init_mt5(cfg):
    mt5_cfg = cfg.get("mt5", {})
    login = int(os.environ.get("MT5_LOGIN", _resolve_env_ref(mt5_cfg.get("login", 0))) or 0)
    password = os.environ.get("MT5_PASSWORD") or _resolve_env_ref(mt5_cfg.get("password", ""))
    server = os.environ.get("MT5_SERVER") or _resolve_env_ref(mt5_cfg.get("server", ""))

    for attempt in range(1, 6):
        try:
            if login and password and server:
                ok = mt5.initialize(login=login, password=password, server=server)
            else:
                ok = mt5.initialize()
            if ok:
                return True
        except Exception as e:
            logger.warning(f"MT5 init attempt {attempt}/5 failed: {e}")
        if attempt < 5:
            time.sleep(2 ** attempt)  # exponential backoff: 2, 4, 8, 16s
    return False


def _to_mt5_timeframe(tf: str):
    mapping = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
    }
    return mapping.get((tf or "M5").upper(), mt5.TIMEFRAME_M5)


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _symbol_profile(symbol: str, cfg: dict | None = None) -> dict:
    base = dict(SYMBOL_EXECUTION_PROFILES.get(str(symbol), SYMBOL_EXECUTION_PROFILES["BTCUSDm"]))
    trading_cfg = (cfg or {}).get("trading", {}) if isinstance(cfg, dict) else {}
    symbol_profiles = trading_cfg.get("symbol_profiles", {}) or {}
    raw = symbol_profiles.get(str(symbol), {}) if isinstance(symbol_profiles, dict) else {}
    if isinstance(raw, dict):
        if "ppo_weight" in raw:
            base["ppo_weight"] = float(raw["ppo_weight"])
        if "dreamer_weight" in raw:
            base["dreamer_weight"] = float(raw["dreamer_weight"])
        if "agi_context_weight" in raw:
            base["agi_weight"] = float(raw["agi_context_weight"])
        if "min_actionable_exposure" in raw:
            base["min_trade_threshold"] = float(raw["min_actionable_exposure"])
        if "max_policy_exposure" in raw:
            base["max_abs_target"] = float(raw["max_policy_exposure"])
        if "min_trade_interval_sec" in raw:
            base["cooldown_sec"] = int(raw["min_trade_interval_sec"])
    return base


def _blend_symbol_decision(
    symbol: str,
    agi_meta: dict | None,
    ppo_meta: dict | None,
    dreamer_meta: dict | None,
    cfg: dict | None = None,
) -> dict:
    profile = _symbol_profile(symbol, cfg=cfg)

    ppo_target = float((ppo_meta or {}).get("target", 0.0) or 0.0)
    dreamer_target = float((dreamer_meta or {}).get("target", 0.0) or 0.0)

    agi_conf = float((agi_meta or {}).get("confidence", 0.0) or 0.0)
    agi_risk = float((agi_meta or {}).get("risk_scalar", 1.0) or 1.0)
    agi_bias = float((agi_meta or {}).get("trend_bias", 0.0) or 0.0)

    # Redistribute weights from absent models to AGI so the 150-feature
    # LSTM signal can drive real decisions before a PPO champion is promoted.
    # When both PPO and Dreamer are present, weights are used as configured.
    ppo_w = float(profile["ppo_weight"]) if ppo_meta is not None else 0.0
    dreamer_w = float(profile["dreamer_weight"]) if dreamer_meta is not None else 0.0
    agi_w = max(float(profile["agi_weight"]), 1.0 - ppo_w - dreamer_w)

    raw = ppo_w * ppo_target + dreamer_w * dreamer_target + agi_w * agi_bias

    adjusted = raw * agi_risk
    adjusted = _clip(adjusted, -float(profile["max_abs_target"]), float(profile["max_abs_target"]))

    if abs(adjusted) < float(profile["min_trade_threshold"]):
        adjusted = 0.0

    return {
        "target": adjusted,
        "raw_target": raw,
        "agi_confidence": agi_conf,
        "agi_risk_scalar": agi_risk,
        "ppo_target": ppo_target,
        "dreamer_target": dreamer_target,
        "agi_bias": agi_bias,
        "agi_direction": str((agi_meta or {}).get("direction", (agi_meta or {}).get("signal", "HOLD")) or "HOLD"),
        "agi_feature_version": str((agi_meta or {}).get("feature_version", "unknown") or "unknown"),
        "ppo_weight_used": round(ppo_w, 3),
        "dreamer_weight_used": round(dreamer_w, 3),
        "agi_weight_used": round(agi_w, 3),
        "profile": profile,
    }


def _low_volatility_memory_base(trade_memory: dict | None) -> float:
    memory = trade_memory or {}
    min_trades = int(os.environ.get("AGI_LOW_VOL_MIN_TRADES", "20") or 20)
    min_profit_factor = float(os.environ.get("AGI_LOW_VOL_MIN_PROFIT_FACTOR", "1.15") or 1.15)
    min_expectancy = float(os.environ.get("AGI_LOW_VOL_MIN_EXPECTANCY", "0.0") or 0.0)
    max_recent_loss_streak = int(os.environ.get("AGI_LOW_VOL_MAX_RECENT_LOSS_STREAK", "3") or 3)

    trades = int(memory.get("trades", 0) or 0)
    profit_factor = float(memory.get("profit_factor", 0.0) or 0.0)
    expectancy = float(memory.get("expectancy", 0.0) or 0.0)
    recent_loss_streak = int(memory.get("recent_loss_streak", 0) or 0)

    if trades < min_trades:
        return 0.0
    if profit_factor < min_profit_factor:
        return 0.0
    if expectancy < min_expectancy:
        return 0.0
    if recent_loss_streak > max_recent_loss_streak:
        return 0.0
    return 1.0


def _fetch_symbol_df(symbol: str, timeframe, bars=220):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) < 80:
        return None

    df = pd.DataFrame(rates)
    if df.empty:
        return None

    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={"tick_volume": "volume"})
    keep = ["time", "open", "high", "low", "close", "volume"]
    for k in keep:
        if k not in df.columns:
            return None

    out = df[keep].copy()
    out["symbol"] = symbol
    return out


def _account_snapshot():
    if _is_paper_mode():
        pacc = _srv_paper.get_paper_account()
        positions = _srv_paper.paper_positions_get()
        floating = sum(float(p.get("profit", 0.0)) for p in positions)
        return {
            "balance": float(pacc["balance"]),
            "equity": float(pacc["equity"]),
            "free_margin": float(pacc["free_margin"]),
            "pnl_today": float(pacc.get("realized_today", 0.0)),
            "floating": float(floating),
            "open_positions": len(positions),
        }

    info = mt5.account_info()
    positions = mt5.positions_get() or []
    floating = sum(float(getattr(p, "profit", 0.0)) for p in positions)

    pnl_today = 0.0
    try:
        now_utc = _utc_now()
        day_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        deals = mt5.history_deals_get(day_start, now_utc)
        for d in deals or []:
            if int(getattr(d, "entry", -1)) == int(mt5.DEAL_ENTRY_OUT):
                pnl_today += float(getattr(d, "profit", 0.0) + getattr(d, "commission", 0.0) + getattr(d, "swap", 0.0))
    except Exception as exc:
        logger.warning(f"Failed to compute today's realized PnL from history (using 0): {exc}")

    balance = None
    equity = None
    free_margin = None
    if info is not None:
        if getattr(info, "_valid", True):
            balance = float(info.balance)
            equity = float(info.equity)
            free_margin = float(info.margin_free)
        else:
            logger.warning(
                "[MT5 TELEMETRY] Invalid account snapshot: balance=%s, equity=%s, server=%s, login=%s",
                info.balance, info.equity, info.server, info.login,
            )

    return {
        "balance": balance,
        "equity": equity,
        "free_margin": free_margin,
        "pnl_today": float(pnl_today),
        "floating": float(floating),
        "open_positions": len(positions),
    }


def _expected_usd(symbol: str, side: str, entry: float, tp: float, sl: float, lots: float):
    try:
        info = mt5.symbol_info(symbol)
        tick_size = float(getattr(info, "trade_tick_size", 0.0) or 0.0)
        tick_value = float(getattr(info, "trade_tick_value", 0.0) or 0.0)
        if tick_size <= 0 or tick_value <= 0:
            return None, None
        usd_per_price = tick_value / tick_size
        if str(side).upper() == "BUY":
            tp_outcome = (float(tp) - float(entry)) * usd_per_price * float(lots)
            sl_outcome = (float(sl) - float(entry)) * usd_per_price * float(lots)
        else:
            tp_outcome = (float(entry) - float(tp)) * usd_per_price * float(lots)
            sl_outcome = (float(entry) - float(sl)) * usd_per_price * float(lots)
        return tp_outcome, sl_outcome
    except Exception as exc:
        logger.debug(f"_expected_usd calc failed for {symbol}: {exc}")
        return None, None


def _tick_spread_bps(symbol: str) -> float | None:
    try:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        bid = float(getattr(tick, "bid", 0.0) or 0.0)
        ask = float(getattr(tick, "ask", 0.0) or 0.0)
        mid = (bid + ask) * 0.5
        if bid <= 0.0 or ask <= 0.0 or mid <= 0.0:
            return None
        return float(((ask - bid) / mid) * 10000.0)
    except Exception as exc:
        logger.debug(f"_tick_spread_bps failed for {symbol}: {exc}")
        return None


def _position_exposure_state(symbol: str, max_lots: float) -> tuple[float, float, int, int]:
    if _is_paper_mode():
        positions = _srv_paper.paper_positions_get()
    else:
        positions = mt5.positions_get() or []
    symbol_positions = [p for p in positions if str(getattr(p, "symbol", "") if not isinstance(p, dict) else p.get("symbol", "")) == str(symbol)]
    current_symbol_exposure = 0.0
    total_exposure = 0.0
    for pos in positions:
        volume = float(getattr(pos, "volume", 0.0) or 0.0) if not isinstance(pos, dict) else float(pos.get("volume", 0.0))
        side = -1.0 if int(getattr(pos, "type", 0) if not isinstance(pos, dict) else (0 if pos.get("type") == "BUY" else 1)) == int(mt5.ORDER_TYPE_SELL) else 1.0
        exp = side * (volume / max(max_lots, 1e-8))
        total_exposure += abs(exp)
        if str(getattr(pos, "symbol", "") if not isinstance(pos, dict) else pos.get("symbol", "")) == str(symbol):
            current_symbol_exposure += exp
    return float(current_symbol_exposure), float(total_exposure), len(symbol_positions), len(positions)


def _scan_trade_events(alerter, risk, known_open_tickets, seen_closed_deals, last_deal_check):
    now_utc = _utc_now()
    closed_events = []

    if _is_paper_mode():
        positions = _srv_paper.paper_positions_get()
        current_open = {int(p.get("ticket")): p for p in positions}
    else:
        positions = mt5.positions_get() or []
        current_open = {int(p.ticket): p for p in positions}
    current_tickets = set(current_open.keys())

    new_tickets = sorted(current_tickets - known_open_tickets)
    for ticket in new_tickets:
        p = current_open[ticket]
        is_dict = isinstance(p, dict)
        pos_type = p.get("type") if is_dict else getattr(p, "type", 0)
        side = "BUY" if str(pos_type).upper() == "BUY" or int(pos_type) == int(mt5.ORDER_TYPE_BUY) else "SELL"
        payload = {
            "ticket": ticket,
            "symbol": str(p.get("symbol", "?") if is_dict else getattr(p, "symbol", "?")),
            "side": side,
            "volume": float(p.get("volume", 0.0) if is_dict else getattr(p, "volume", 0.0)),
            "open_price": float(p.get("open_price", 0.0) if is_dict else getattr(p, "price_open", 0.0)),
            "sl": float((p.get("sl", 0.0) or 0.0) if is_dict else (getattr(p, "sl", 0.0) or 0.0)),
            "tp": float((p.get("tp", 0.0) or 0.0) if is_dict else (getattr(p, "tp", 0.0) or 0.0)),
        }
        _append_trade_event("trade_open", payload)

        snap = _account_snapshot()
        alerter.trade(
            symbol=payload["symbol"],
            action=side,
            exposure=payload["volume"],
            confidence=1.0,
            balance=0.0 if snap["balance"] is None else snap["balance"],
            equity=0.0 if snap["equity"] is None else snap["equity"],
            free_margin=0.0 if snap["free_margin"] is None else snap["free_margin"],
        )

    removed_tickets = sorted(known_open_tickets - current_tickets)
    for ticket in removed_tickets:
        _append_trade_event("position_removed", {"ticket": ticket})

    try:
        deals = mt5.history_deals_get(last_deal_check, now_utc) or []
    except Exception as exc:
        logger.warning(f"MT5 history_deals_get failed (deals=[]): {exc}")
        deals = []

    for d in deals:
        try:
            if int(getattr(d, "entry", -1)) != int(mt5.DEAL_ENTRY_OUT):
                continue
            deal_id = int(getattr(d, "deal", 0))
            if deal_id <= 0 or deal_id in seen_closed_deals:
                continue
            seen_closed_deals.add(deal_id)

            pnl = float(getattr(d, "profit", 0.0) + getattr(d, "commission", 0.0) + getattr(d, "swap", 0.0))
            payload = {
                "deal_id": deal_id,
                "ticket": int(getattr(d, "position_id", 0) or 0),
                "symbol": str(getattr(d, "symbol", "?")),
                "volume": float(getattr(d, "volume", 0.0)),
                "price": float(getattr(d, "price", 0.0)),
                "profit": pnl,
                "comment": str(getattr(d, "comment", "")),
            }
            _append_trade_event("trade_closed", payload)
            closed_events.append(payload)
            try:
                risk.record_trade_result(payload["symbol"], pnl)
            except Exception as exc:
                logger.warning(f"risk.record_trade_result failed for {payload.get('symbol')}: {exc}")
            alerter.trade_closed(
                symbol=payload["symbol"],
                ticket=payload["ticket"],
                pnl=pnl,
                volume=payload["volume"],
                price=payload["price"],
                reason=payload.get("comment"),
                deal_id=deal_id,
            )
        except Exception as exc:
            logger.debug(f"Deal processing error (skipped): {exc}")
            continue

    if len(seen_closed_deals) > 20000:
        seen_closed_deals = set(sorted(seen_closed_deals)[-10000:])

    return current_tickets, seen_closed_deals, now_utc - datetime.timedelta(seconds=3), closed_events


def main(live=False):
    acquired, cleanup_fn = _acquire_single_instance_lock()
    if not acquired:
        raise RuntimeError("Server_AGI is already running (lock file exists)")

    if cleanup_fn:
        atexit.register(cleanup_fn)

    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)

    # Determine execution mode: env var is primary, --live flag is secondary
    env_mode = live_safety.get_execution_mode() if live_safety else "paper"
    requested_live = env_mode in ("live", "demo")
    if live and env_mode == "paper":
        logger.warning("--live flag ignored because CHAIN_GAMBLER_EXECUTION_MODE=paper")

    # Gate live mode with comprehensive safety checks
    if requested_live:
        if env_mode == "demo":
            logger.info("Demo mode enabled — real orders to demo account.")
            live = True
            os.environ["AGI_IS_LIVE"] = "1"
        elif live_safety is not None:
            gate = live_safety.live_trading_allowed()
            if not gate["allowed"]:
                logger.warning(f"Live trading requested but safety gates failed: {gate['gates']}")
                logger.warning("Forcing PAPER mode. Set CHAIN_GAMBLER_ALLOW_LIVE=1 and ensure all gates pass for live.")
                os.environ["CHAIN_GAMBLER_EXECUTION_MODE"] = "paper"
                live = False
            else:
                live = True
                os.environ["AGI_IS_LIVE"] = "1"
        else:
            logger.warning("live_safety module unavailable — forcing PAPER mode")
            live = False
    else:
        live = False

    if not live:
        os.environ.setdefault("CHAIN_GAMBLER_EXECUTION_MODE", "paper")

    cfg = _load_cfg(live=live)
    ok = _init_mt5(cfg)
    if not ok:
        if _is_paper_mode() or env_mode == "demo":
            try:
                err = mt5.last_error()
            except Exception as e:
                err = f"unknown error ({e})"
            logger.warning(f"MT5 unavailable ({err}), continuing in {env_mode}-only mode")
        else:
            try:
                err = mt5.last_error()
            except Exception as e:
                err = f"unknown error ({e})"
            raise RuntimeError(f"MT5 init failed: {err}")

    runtime = _load_runtime_components()
    risk = runtime["RiskEngine"]()
    supervisor = runtime["RiskSupervisor"](cfg)
    executor = runtime["MT5Executor"](risk)

    # Non-security fix for reviewer finding: protect shared mutable risk state
    # from the background training thread and main trading loop.
    risk_lock = threading.Lock()

    # Prevent HybridBrain from starting its own AutonomyLoop thread;
    # Server_AGI orchestrates training cycles directly in the main loop.
    os.environ["AGI_AUTONOMY_ENABLED"] = "false"
    brain = runtime["HybridBrain"](risk, executor)
    agi = runtime["SmartAGI"]()

    from Python.autonomy_loop import AutonomyLoop

    autonomy = AutonomyLoop(brain)
    autonomy._last_train_ts = time.time()

    trading_cfg = cfg.get("trading", {})
    symbols = resolve_trading_symbols(cfg, env_keys=("AGI_RUNTIME_SYMBOLS",), fallback=DEFAULT_TRADING_SYMBOLS)
    timeframe = _to_mt5_timeframe(trading_cfg.get("timeframe", "M5"))
    max_lots = float(cfg.get("risk", {}).get("max_lots", 1.0))

    token, chat_id = _load_telegram_cfg(cfg)
    alerter = runtime["TelegramAlerter"](token, chat_id)
    event_intel = runtime["EventIntel"](cfg, LOG_DIR)

    alerter.online("Trading engine initialized")

    def _notify_offline():
        try:
            snap = _account_snapshot()
            alerter.offline(
                f"Balance={0.0 if snap['balance'] is None else snap['balance']:.2f} | "
                f"Equity={0.0 if snap['equity'] is None else snap['equity']:.2f} | "
                f"Open={int(snap['open_positions'])}"
            )
        except Exception as exc:
            logger.warning(f"Failed to build detailed offline snapshot: {exc}")
            alerter.offline("Runtime exited")

    atexit.register(_notify_offline)

    known_open_tickets = set()
    seen_closed_deals = set()
    last_deal_check = _utc_now() - datetime.timedelta(minutes=30)

    start_time = time.time()
    heartbeat_sec = int(os.environ.get("AGI_HEARTBEAT_SEC", "600"))
    symbol_card_sec = int(os.environ.get("AGI_SYMBOL_CARD_SEC", "90"))
    learning_sec = int(os.environ.get("AGI_TRADE_LEARN_SEC", "600"))
    loop_sleep_sec = int(os.environ.get("AGI_LOOP_SEC", "20"))
    last_heartbeat = 0.0
    last_symbol_cards = 0.0
    last_learning = 0.0
    last_models = {"champion": None, "canary": None}
    last_owner_issue_key = None
    last_owner_issue_time = 0.0
    last_daily_profit_date = None
    last_symbol_state = {str(s): {} for s in symbols}
    last_closed_by_symbol = {}
    trade_learning_by_symbol = {}

    _loop_counter = 0
    _last_training_cycle = 0.0
    _training_cycle_interval = max(60, int(os.environ.get("AGI_TRAINING_CYCLE_SEC", "1800")))
    _training_cycle_every_n = max(1, int(os.environ.get("AGI_TRAINING_CYCLE_EVERY_N", "50")))
    _training_thread = None

    while not _shutdown_flag.is_set():
        now = time.time()
        _loop_counter += 1

        # Trigger autonomous training cycle every N loops or every 30 minutes.
        if (
            autonomy.enable_train
            and (_loop_counter % _training_cycle_every_n == 0 or now - _last_training_cycle >= _training_cycle_interval)
            and (_training_thread is None or not _training_thread.is_alive())
        ):
            _last_training_cycle = now
            _training_thread = threading.Thread(
                target=autonomy.training_cycle,
                args=(symbols, _shutdown_flag),
                name="training-cycle",
                daemon=False,
            )
            _training_thread.start()
            logger.info(f"Training cycle thread started (loop={_loop_counter})")

        if now - last_heartbeat >= max(15, heartbeat_sec):
            uptime = int(now - start_time)
            acc = mt5.account_info()
            if acc and getattr(acc, "_valid", True):
                with risk_lock:
                    risk.update_equity(float(acc.equity))
            elif acc and not getattr(acc, "_valid", True):
                logger.warning(
                    "[MT5 TELEMETRY] Invalid heartbeat telemetry: balance=%s, equity=%s",
                    acc.balance, acc.equity,
                )

            snap = _account_snapshot()
            tr_state = _training_state()
            models = _read_active_models()
            event_state = {}
            try:
                p = os.path.join(LOG_DIR, "event_intel_state.json")
                if os.path.exists(p):
                    with open(p, "r", encoding="utf-8") as f:
                        event_state = json.load(f)
            except Exception:
                event_state = {}
            alerter.heartbeat_full(
                uptime=str(uptime) + " sec",
                mt5_connected=(mt5.terminal_info() is not None),
                trading_enabled=not risk.halt,
                snapshot=snap,
                training=tr_state,
                models=models,
                event_intel=event_state,
            )
            alerter.snapshot(
                balance=0.0 if snap["balance"] is None else snap["balance"],
                equity=0.0 if snap["equity"] is None else snap["equity"],
                pnl_today=snap["pnl_today"],
                floating=snap["floating"],
                open_positions=snap["open_positions"],
            )
            _append_audit("snapshot", snap)
            if models.get("champion") != last_models.get("champion"):
                alerter.model(f"Champion changed: {models.get('champion') or 'none'}")
            if models.get("canary") != last_models.get("canary"):
                alerter.model(f"Canary changed: {models.get('canary') or 'none'}")
            last_models = models
            owner_health = _runtime_owner_health()
            _append_audit("runtime_owner_health", owner_health)
            if not owner_health.get("ok", True):
                issue_key = json.dumps(owner_health.get("issues", []), sort_keys=True, ensure_ascii=True)
                if issue_key != last_owner_issue_key or (now - last_owner_issue_time) > 1800:
                    lines = []
                    for it in owner_health.get("issues", []):
                        lines.append(
                            f"{it.get('role')}: {it.get('type')} | roots={it.get('root_pids')} | exe={it.get('exe_paths')}"
                        )
                    alerter.alert("RUNTIME OWNERSHIP WARNING\n" + "\n".join(lines))
                    last_owner_issue_key = issue_key
                    last_owner_issue_time = now
            last_heartbeat = now

        # Observe-only event intelligence (calendar/news/websocket).
        try:
            event_state = event_intel.tick(symbols)
            _append_audit("event_intel", event_state.get("summary", {}))
            for msg in event_intel.pop_alerts():
                alerter.alert(msg)
        except Exception as e:
            logger.warning(f"event_intel tick failed: {e}")

        if now - last_learning >= max(120, learning_sec):
            try:
                learn = runtime["build_trade_learning"](
                    log_dir=LOG_DIR,
                    out_dir=os.path.join(LOG_DIR, "learning"),
                    lookback_days=int(os.environ.get("AGI_TRADE_LEARN_DAYS", "30")),
                )
                _append_audit(
                    "trade_learning",
                    {
                        "trades": int(learn.get("trades", 0)),
                        "win_rate": float(learn.get("win_rate", 0.0)),
                        "expectancy": float(learn.get("expectancy", 0.0)),
                        "profit_factor": float(learn.get("profit_factor", 0.0)),
                        "total_pnl": float(learn.get("total_pnl", 0.0)),
                    },
                )
                try:
                    day = _utc_now().date().isoformat()
                    daily_hour = int(os.environ.get("AGI_DAILY_PROFIT_HOUR_UTC", "0"))
                    if _utc_now().hour >= max(0, min(23, daily_hour)) and day != last_daily_profit_date:
                        alerter.profitability_daily(learn)
                        last_daily_profit_date = day
                except Exception:
                    pass
                trade_learning_by_symbol = {
                    str((row or {}).get("symbol", "")): dict(row or {})
                    for row in (learn.get("by_symbol", []) if isinstance(learn, dict) else [])
                    if isinstance(row, dict) and row.get("symbol")
                }
            except Exception as e:
                logger.warning(f"trade learning update failed: {e}")
            last_learning = now

        for symbol in symbols:
            try:
                df = _fetch_symbol_df(symbol, timeframe)
                if df is None or df.empty:
                    continue

                agi_meta = agi.predict(df, production=True)
                conf = float((agi_meta or {}).get("confidence", 0.0) or 0.0)
                regime = str((agi_meta or {}).get("regime", (agi_meta or {}).get("signal", "UNKNOWN")))
                trade_memory = trade_learning_by_symbol.get(str(symbol), {})

                ppo_meta = brain.predict_ppo_action(symbol, df)
                dreamer_meta = brain.predict_dreamer_action(symbol, df)
                decision = _blend_symbol_decision(symbol, agi_meta, ppo_meta, dreamer_meta, cfg=cfg)
                exposure = float(decision["target"])

                # ── Reversal detection ──
                if get_reversal_detector is not None and abs(exposure) > 0.01:
                    try:
                        rev = get_reversal_detector().detect_reversal(symbol, df, "BUY" if exposure > 0 else "SELL")
                        if rev.detected and rev.confidence >= 0.65:
                            # Reversal against our direction — flatten or reduce
                            if (exposure > 0 and "bearish" in rev.direction) or (exposure < 0 and "bullish" in rev.direction):
                                logger.warning(
                                    f"REVERSAL {symbol} | {rev.direction} conf={rev.confidence:.2f} methods={rev.methods} — flattening exposure"
                                )
                                exposure = 0.0
                                decision["target"] = 0.0
                                decision["reversal_override"] = True
                                decision["reversal_signal"] = rev.direction
                                decision["reversal_confidence"] = rev.confidence
                                decision["reversal_methods"] = rev.methods
                    except Exception as rev_err:
                        logger.debug(f"Reversal detection failed for {symbol}: {rev_err}")

                logger.info(
                    "DECISION %s | regime=%s conf=%.4f risk=%.4f agi_bias=%.4f ppo=%.4f dreamer=%.4f raw=%.4f final=%.4f"
                    % (
                        symbol,
                        regime,
                        float((agi_meta or {}).get("confidence", 0.0) or 0.0),
                        float((agi_meta or {}).get("risk_scalar", 1.0) or 1.0),
                        float((agi_meta or {}).get("trend_bias", 0.0) or 0.0),
                        float((ppo_meta or {}).get("target", 0.0) or 0.0),
                        float((dreamer_meta or {}).get("target", 0.0) or 0.0),
                        float(decision["raw_target"]),
                        float(decision["target"]),
                    )
                )
                _append_audit(
                    "signal",
                    {
                        "symbol": symbol,
                        "regime": regime,
                        "confidence": conf,
                        "risk_scalar": float((agi_meta or {}).get("risk_scalar", 1.0) or 1.0),
                        "agi_bias": float((agi_meta or {}).get("trend_bias", 0.0) or 0.0),
                        "ppo_exposure": float((ppo_meta or {}).get("target", 0.0) or 0.0),
                        "dreamer_exposure": float((dreamer_meta or {}).get("target", 0.0) or 0.0),
                        "raw_target": float(decision["raw_target"]),
                        "exposure": float(exposure),
                        "decision_profile": dict(decision["profile"]),
                        "trade_memory": {
                            "trades": int(trade_memory.get("trades", 0) or 0),
                            "expectancy": float(trade_memory.get("expectancy", 0.0) or 0.0),
                            "profit_factor": float(trade_memory.get("profit_factor", 0.0) or 0.0),
                            "recent_loss_streak": int(trade_memory.get("recent_loss_streak", 0) or 0),
                        },
                    },
                )
                sym_state = last_symbol_state.setdefault(str(symbol), {})
                sym_state["signal"] = regime
                sym_state["regime"] = regime
                sym_state["rainforest_regime"] = str((agi_meta or {}).get("rainforest_regime", "ranging") or "ranging")
                sym_state["rainforest_confidence"] = float((agi_meta or {}).get("rainforest_confidence", 0.0) or 0.0)
                sym_state["confidence"] = conf
                sym_state["risk_scalar"] = float((agi_meta or {}).get("risk_scalar", 1.0) or 1.0)
                sym_state["trend_bias"] = float((agi_meta or {}).get("trend_bias", 0.0) or 0.0)
                sym_state["ppo_exposure"] = float((ppo_meta or {}).get("target", 0.0) or 0.0)
                sym_state["dreamer_exposure"] = float((dreamer_meta or {}).get("target", 0.0) or 0.0)
                sym_state["raw_target"] = float(decision["raw_target"])
                sym_state["blend_exposure"] = float(exposure)

                action_meta = ppo_meta or brain.get_last_action_meta(symbol=symbol)
                current_symbol_exposure, total_exposure, symbol_positions, total_positions = _position_exposure_state(
                    symbol, max_lots
                )
                supervisor_decision = supervisor.allow_trade(
                    symbol=symbol,
                    target_exposure=float(exposure),
                    confidence=conf,
                    spread_bps=_tick_spread_bps(symbol),
                    snapshot=snap,
                    symbol_positions=symbol_positions,
                    total_positions=total_positions,
                    current_symbol_exposure=current_symbol_exposure,
                    total_exposure=total_exposure,
                    drawdown_pct=float(risk.current_dd),
                )
                sym_state["risk_supervisor"] = {
                    "allowed": bool(supervisor_decision.allowed),
                    "reason": supervisor_decision.reason,
                    "current_symbol_exposure": float(current_symbol_exposure),
                    "total_exposure": float(total_exposure),
                }
                if supervisor_decision.allowed:
                    order_meta = brain.live_trade(
                        symbol,
                        exposure,
                        max_lots,
                        action_meta=action_meta,
                        execution_context={
                            "regime": regime,
                            "confidence": conf,
                            "target_exposure": float(exposure),
                            "raw_target": float(decision["raw_target"]),
                            "ppo_target": float((ppo_meta or {}).get("target", 0.0) or 0.0),
                            "dreamer_target": float((dreamer_meta or {}).get("target", 0.0) or 0.0),
                            "agi_bias": float((agi_meta or {}).get("trend_bias", 0.0) or 0.0),
                            "agi_risk_scalar": float((agi_meta or {}).get("risk_scalar", 1.0) or 1.0),
                        },
                    )
                    if order_meta and order_meta.get("executed"):
                        supervisor.mark_trade(symbol)
                else:
                    order_meta = None
                    _append_audit(
                        "risk_supervisor_block",
                        {
                            "symbol": symbol,
                            "target_exposure": float(exposure),
                            "reason": supervisor_decision.reason,
                            "confidence": conf,
                            "signal": regime,
                        },
                    )
                executor.manage_open_positions(symbol)
                if order_meta:
                    tp_outcome_usd, sl_outcome_usd = _expected_usd(
                        symbol=symbol,
                        side=str(order_meta.get("order_type", "BUY")),
                        entry=float(order_meta.get("entry_price", 0.0) or 0.0),
                        tp=float(order_meta.get("tp_price", 0.0) or 0.0),
                        sl=float(order_meta.get("sl_price", 0.0) or 0.0),
                        lots=float(order_meta.get("volume_lots", 0.0) or 0.0),
                    )
                    if tp_outcome_usd is not None:
                        order_meta["tp_outcome_usd"] = float(tp_outcome_usd)
                        order_meta["expected_profit_usd"] = float(tp_outcome_usd)
                    if sl_outcome_usd is not None:
                        order_meta["sl_outcome_usd"] = float(sl_outcome_usd)
                        order_meta["expected_loss_usd"] = float(sl_outcome_usd)
                    _append_audit("trade_action", dict(order_meta))
                    logger.info(
                        "ACTION %s | req=%s side=%s volume=%s target=%.4f ppo=%.4f dreamer=%.4f agi=%.4f magic=%s comment=%s ticket=%s retcode=%s TP=%s SL=%s"
                        % (
                            symbol,
                            order_meta.get("request_action"),
                            order_meta.get("order_type"),
                            order_meta.get("executed_lots", order_meta.get("volume_lots")),
                            float(order_meta.get("target_exposure", order_meta.get("exposure", 0.0)) or 0.0),
                            float(order_meta.get("ppo_target", 0.0) or 0.0),
                            float(order_meta.get("dreamer_target", 0.0) or 0.0),
                            float(order_meta.get("agi_bias", 0.0) or 0.0),
                            order_meta.get("magic"),
                            order_meta.get("comment"),
                            order_meta.get("ticket"),
                            order_meta.get("retcode"),
                            order_meta.get("tp_price"),
                            order_meta.get("sl_price"),
                        )
                    )
                    alerter.trade_action(symbol, order_meta)

                acc = mt5.account_info()
                if acc and getattr(acc, "_valid", True):
                    with risk_lock:
                        risk.update_equity(float(acc.equity))
                elif acc and not getattr(acc, "_valid", True):
                    logger.warning(
                        "[MT5 TELEMETRY] Invalid execution-loop telemetry: balance=%s, equity=%s",
                        acc.balance, acc.equity,
                    )
            except Exception as exc:
                with risk_lock:
                    risk.record_error()
                alerter.alert(f"Execution loop error on {symbol}: {exc}")
                logger.exception(f"Execution loop error on {symbol}: {exc}")

        known_open_tickets, seen_closed_deals, last_deal_check, closed_events = _scan_trade_events(
            alerter,
            risk,
            known_open_tickets,
            seen_closed_deals,
            last_deal_check,
        )
        for c in closed_events:
            try:
                last_closed_by_symbol[str(c.get("symbol", "?"))] = c
            except Exception:
                pass

        if now - last_symbol_cards >= max(15, symbol_card_sec):
            for symbol in symbols:
                sym = str(symbol)
                sstate = dict(last_symbol_state.get(sym, {}))
                pos_rows = mt5.positions_get(symbol=sym) or []
                sstate["open_positions"] = len(pos_rows)
                sstate["floating_pnl"] = sum(float(getattr(p, "profit", 0.0)) for p in pos_rows)
                if pos_rows:
                    p0 = pos_rows[0]
                    p_side = "BUY" if int(getattr(p0, "type", 0)) == int(mt5.ORDER_TYPE_BUY) else "SELL"
                    p_vol = float(getattr(p0, "volume", 0.0) or 0.0)
                    p_entry = float(getattr(p0, "price_open", 0.0) or 0.0)
                    p_tp = float(getattr(p0, "tp", 0.0) or 0.0)
                    p_sl = float(getattr(p0, "sl", 0.0) or 0.0)
                    sstate["position_side"] = p_side
                    sstate["position_volume"] = p_vol
                    sstate["position_entry"] = p_entry
                    sstate["position_tp"] = p_tp
                    sstate["position_sl"] = p_sl
                    tpv, slv = _expected_usd(sym, p_side, p_entry, p_tp, p_sl, p_vol)
                    sstate["position_tp_value_usd"] = None if tpv is None else float(tpv)
                    sstate["position_sl_value_usd"] = None if slv is None else float(slv)
                else:
                    sstate["position_side"] = None
                    sstate["position_volume"] = None
                    sstate["position_entry"] = None
                    sstate["position_tp"] = None
                    sstate["position_sl"] = None
                    sstate["position_tp_value_usd"] = None
                    sstate["position_sl_value_usd"] = None
                sstate["last_closed"] = last_closed_by_symbol.get(sym)
                alerter.symbol_status(sym, sstate)
            last_symbol_cards = now

        # Write live_state.json for the standalone API server / dashboard
        try:
            _write_live_state(
                risk=risk,
                symbols=symbols,
                last_symbol_state=last_symbol_state,
                models=_read_active_models(),
                training_state=getattr(autonomy, "training_state", None),
            )
        except Exception:
            pass

        time.sleep(max(5, loop_sleep_sec))

    logger.info("Shutdown flag set; exiting main loop gracefully.")

    if _training_thread is not None and _training_thread.is_alive():
        logger.info("Waiting for training thread to finish (max 30s)...")
        _training_thread.join(timeout=30)
        if _training_thread.is_alive():
            logger.warning("Training thread did not finish in time; proceeding with shutdown.")

    try:
        snap = _account_snapshot()
        logger.info(
            f"Shutdown snapshot — balance={snap['balance']}, equity={snap['equity']}, "
            f"floating={snap['floating']}, open_positions={snap['open_positions']}"
        )
        _append_audit("shutdown", snap)
    except Exception as exc:
        logger.warning(f"Failed to record shutdown snapshot: {exc}")

    try:
        _write_live_state(
            risk=risk,
            symbols=symbols,
            last_symbol_state=last_symbol_state,
            models=_read_active_models(),
            training_state=getattr(autonomy, "training_state", None),
        )
    except Exception:
        pass

    logger.info("Server_AGI stopped gracefully.")


if __name__ == "__main__":
    import sys

    live_flag = "--live" in sys.argv
    main(live=live_flag)
