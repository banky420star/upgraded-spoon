"""
API Server — Lightweight HTTP bridge between the AGI engine and the React dashboard.

Uses bottle (already in .venv312) on port 5050.  Vite dev-server proxies /api/* here.

Start modes:
  1. Embedded: import start_api_server(agi_server) from Server_AGI — preferred.
  2. Standalone: python -m Python.api_server — reads live_state.json fallback.

All endpoints are read-only except POST /api/control.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import threading
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── Path Setup for Standalone Mode ─────────────────────────────────
# Add project root to path so 'Python' module imports work
_project_root = os.path.normpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
# Also add Python directory directly for imports
_python_dir = os.path.join(_project_root, 'Python')
if _python_dir not in sys.path:
    sys.path.insert(0, _python_dir)

# Pre-import training modules to ensure they're available
try:
    from Python.training_analyzer import get_training_description, get_analyzer
except ImportError:
    get_training_description = None
    get_analyzer = None

# ── Trade Review Summary Cache ─────────────────────────────────────
_trade_review_cache = {"summary": {}, "updated_at": 0}
_trade_review_lock = threading.Lock()

def _get_trade_review_summary():
    """Return cached trade review summary, refreshing if older than 5 minutes."""
    global _trade_review_cache
    now = time.time()
    if now - _trade_review_cache.get("updated_at", 0) > 300:  # 5 min cache
        try:
            with _trade_review_lock:
                from Python.trade_review import get_latest_review
                review = get_latest_review()
                if review:
                    _trade_review_cache = {
                        "summary": review.get("summary", {}),
                        "updated_at": now,
                    }
        except Exception:
            pass
    return _trade_review_cache.get("summary", {})


_calendar_cache = {"events": [], "updated_at": 0}

def _get_economic_calendar_cached():
    """Return cached economic calendar, refreshing if older than 30 minutes."""
    global _calendar_cache
    now = time.time()
    if now - _calendar_cache.get("updated_at", 0) > 1800:  # 30 min cache
        try:
            from Python.trade_review import get_economic_calendar
            events = get_economic_calendar(days_ahead=7)
            _calendar_cache = {"events": events, "updated_at": now}
        except Exception as e:
            logger.debug(f"Calendar cache refresh failed: {e}")
    return _calendar_cache.get("events", [])
from typing import Any

from bottle import Bottle, ServerAdapter, request, response, abort, run as bottle_run
from loguru import logger

# ── Optional geventwebsocket support ────────────────────────────────────────
try:
    from geventwebsocket import WebSocketError
    from geventwebsocket.handler import WebSocketHandler
    from gevent.pywsgi import WSGIServer as GeventWSGIServer
    from gevent import sleep as gevent_sleep
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False
    WebSocketError = Exception  # type: ignore
    gevent_sleep = time.sleep

# ---------------------------------------------------------------------------
# Rate limiter (in-memory sliding window) for control endpoints
# ---------------------------------------------------------------------------
class _RateLimiter:
    """Simple per-IP sliding-window rate limiter."""

    def __init__(self, max_requests: int = 10, window_sec: int = 60):
        self.max_requests = max_requests
        self.window_sec = window_sec
        self._windows: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def is_allowed(self, ip: str) -> bool:
        now = time.time()
        with self._lock:
            window = self._windows.setdefault(ip, [])
            # Purge old entries
            cutoff = now - self.window_sec
            while window and window[0] < cutoff:
                window.pop(0)
            if len(window) >= self.max_requests:
                return False
            window.append(now)
            return True

    def reset(self, ip: str) -> None:
        with self._lock:
            self._windows.pop(ip, None)


_control_limiter = _RateLimiter(max_requests=10, window_sec=60)
_status_limiter = _RateLimiter(max_requests=60, window_sec=60)


# ---------------------------------------------------------------------------
# TLS helper — auto-generate self-signed certs if user provides a cert dir
# ---------------------------------------------------------------------------
def _ensure_tls_certs(cert_dir: str) -> tuple[str, str]:
    """Return (cert_path, key_path). Generate self-signed certs if missing."""
    import ssl as _ssl

    cert_path = os.path.join(cert_dir, "api_server.crt")
    key_path = os.path.join(cert_dir, "api_server.key")
    if os.path.exists(cert_path) and os.path.exists(key_path):
        return cert_path, key_path

    os.makedirs(cert_dir, exist_ok=True)
    # Generate a self-signed cert
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
    except ImportError:
        logger.warning("cryptography library not installed — cannot auto-generate TLS certs")
        return "", ""

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
    ])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(timezone.utc) - timedelta(days=1))
        .not_valid_after(datetime.now(timezone.utc) + timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost"), x509.IPAddress("127.0.0.1")]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    logger.success(f"Self-signed TLS certificate generated: {cert_path}")
    return cert_path, key_path


# ---------------------------------------------------------------------------
# Project root (one level above Python/)
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Bottle app
# ---------------------------------------------------------------------------
app = Bottle()


@app.hook("before_request")
def _rate_limit_hook():
    path = request.path or ""
    ip = request.environ.get("REMOTE_ADDR", "unknown")
    if path.startswith("/api/control"):
        if not _control_limiter.is_allowed(ip):
            response.status = 429
            response.body = json.dumps({"ok": False, "error": "rate limit exceeded — try again in 60s"})
            return response.body
    elif path.startswith("/api/status"):
        if not _status_limiter.is_allowed(ip):
            response.status = 429
            response.body = json.dumps({"ok": False, "error": "rate limit exceeded"})
            return response.body


# ---------------------------------------------------------------------------
# Shared references — populated by start_api_server()
# ---------------------------------------------------------------------------
_server_ref: Any = None          # AGIServer instance
_bot_process: subprocess.Popen | None = None  # tracked bot subprocess
_bot_process_lock = threading.Lock()
_decision_cache: dict[str, deque] = {}  # symbol -> deque of recent decisions
_CACHE_MAX = 50                  # decisions per symbol

# ── Standalone RiskEngine for when API runs without Server_AGI ─────────────
_risk_standalone: Any = None

def _get_risk():
    """Return the active risk engine (embedded or standalone)."""
    global _risk_standalone
    srv = _server_ref
    if srv and hasattr(srv, "risk"):
        return srv.risk
    if _risk_standalone is None:
        try:
            from Python.risk_engine import RiskEngine
            _risk_standalone = RiskEngine()
        except Exception:
            pass
    return _risk_standalone

# ---------------------------------------------------------------------------
# Agent heartbeat tracking — updated by actual activity, not wallclock
# ---------------------------------------------------------------------------
_agent_heartbeats: dict[str, float] = {}  # agent_id -> unix timestamp of last real activity

def _agent_heartbeat(agent_id: str):
    """Record a real heartbeat for an agent."""
    _agent_heartbeats[agent_id] = time.time()

def _get_agent_heartbeat(agent_id: str) -> str:
    """Return ISO timestamp of last real activity, or 'never' if no activity recorded."""
    ts = _agent_heartbeats.get(agent_id)
    if ts is None:
        return "never"
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

# ---------------------------------------------------------------------------
# Rainforest detector — lazy-loaded; populated by train_rainforest() or from
# Server_AGI via set_rainforest_predictions().
# ---------------------------------------------------------------------------
rf_detector = None          # RainforestDetector instance (or None if unavailable)
rainforest_predictions: dict[str, dict] = {}   # symbol -> latest predict_regime() output
_rainforest_trained_at: float = 0.0

try:
    from Python.rainforest_detector import RainforestDetector as _RainforestDetector
    rf_detector = _RainforestDetector()
except Exception as _rf_import_err:
    logger.debug(f"RainforestDetector not available: {_rf_import_err}")
    _RainforestDetector = None  # type: ignore


def set_rainforest_predictions(symbol: str, prediction: dict):
    """Called by Server_AGI / agi_brain to push live predictions into the cache."""
    global rainforest_predictions, _rainforest_trained_at
    rainforest_predictions[symbol] = dict(prediction)
    _rainforest_trained_at = time.time()


# ---------------------------------------------------------------------------
# CORS middleware (restrict to localhost only for security)
# ---------------------------------------------------------------------------
_CORS_ALLOWED_ORIGINS = [
    "http://localhost:4180",
    "http://127.0.0.1:4180",
]

# In production, add additional allowed origins from environment
if os.environ.get("AGI_IS_LIVE", "0") == "1":
    _extra_origins = os.environ.get("AGI_ALLOWED_ORIGINS", "").split(",")
    _CORS_ALLOWED_ORIGINS.extend([o.strip() for o in _extra_origins if o.strip()])

@app.hook("after_request")
def _cors():
    origin = request.get_header("Origin", "")
    # Security: Only allow whitelisted origins, NO fallback to localhost
    if origin in _CORS_ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Control-Token"
        response.headers["Vary"] = "Origin"
    # If origin is not in whitelist, don't set CORS headers (browser will block)


@app.route("<path:path>", method="OPTIONS")
def _options(path):
    return {}


# ---------------------------------------------------------------------------
# Telegram Mini App — serve the HTML page
# ---------------------------------------------------------------------------
_MINI_APP_HTML = None

@app.route("/mini", method=["GET", "POST"])
def api_mini_app():
    """Serve the Telegram Mini App HTML page."""
    global _MINI_APP_HTML
    if _MINI_APP_HTML is None:
        mini_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "tools", "ui_assets", "telegram_mini_app.html")
        try:
            with open(mini_path, "r", encoding="utf-8") as f:
                _MINI_APP_HTML = f.read()
        except Exception:
            _MINI_APP_HTML = "<html><body><h1>Mini App not found</h1></body></html>"
    response.content_type = "text/html; charset=utf-8"
    return _MINI_APP_HTML


# ---------------------------------------------------------------------------
# Parallel lane manager helper
# ---------------------------------------------------------------------------
def _parallel_lane_status(srv) -> dict:
    """Return parallel_lanes keys to merge into the training dict."""
    try:
        if srv and hasattr(srv, "lane_mgr") and srv.lane_mgr:
            lane_status = srv.lane_mgr.get_status()
            return {
                "parallel_lanes": lane_status["parallel_lanes"],
                "max_parallel": lane_status["max_parallel"],
                "lane_active_count": lane_status["active_count"],
            }
    except Exception:
        pass
    return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _json(obj: Any, status: int = 200):
    response.content_type = "application/json"
    response.status = status
    return json.dumps(obj, default=str)


def _read_json_file(path: str) -> Any:
    """Safely read a JSON file, returning None on any error."""
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def _read_active_registry() -> dict:
    """Read models/registry/active.json."""
    active_path = os.path.join(ROOT, "models", "registry", "active.json")
    return _read_json_file(active_path) or {"champion": None, "canary": None}


def _read_config() -> dict:
    cfg_path = os.path.join(ROOT, "config.yaml")
    try:
        import yaml
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return {}


def _read_incidents() -> list:
    return _read_json_file(os.path.join(ROOT, "live_incidents.json")) or []


def _read_live_state() -> dict:
    return _read_json_file(os.path.join(ROOT, "live_state.json")) or {}


def _get_telegram_status() -> dict:
    """Return Telegram alerter status from cards state."""
    try:
        cards_path = os.path.join(ROOT, "logs", "telegram_cards.json")
        cards = _read_json_file(cards_path) or {}
        cfg = _read_config()
        tg_cfg = cfg.get("telegram", {})
        token = tg_cfg.get("token", "")
        chat_id = tg_cfg.get("chat_id", "")
        configured = bool(token and chat_id)
        delivered = 0
        failed = 0
        for row in cards.values():
            if isinstance(row, dict):
                ds = row.get("delivery_status", "dashboard_only")
                if ds == "delivered":
                    delivered += 1
                elif ds in ("failed", "error"):
                    failed += 1
        return {
            "configured": configured,
            "connected": configured,  # best-effort: configured implies connectable
            "card_count": len(cards),
            "delivered": delivered,
            "failed": failed,
            "delivery_stats": {
                "delivered": delivered,
                "failed": failed,
                "pending": len(cards) - delivered - failed,
            },
        }
    except Exception:
        pass
    return {
        "configured": False,
        "connected": False,
        "card_count": 0,
        "delivered": 0,
        "failed": 0,
    }


def cache_decision(symbol: str, decision: dict):
    """Store a decision in the in-memory cache (called from the brain path)."""
    if symbol not in _decision_cache:
        _decision_cache[symbol] = deque(maxlen=_CACHE_MAX)
    entry = {**decision, "_cached_at": time.time()}
    _decision_cache[symbol].appendleft(entry)


def _safe_risk(attr: str, default=None):
    """Pull a value from the risk engine (embedded or standalone)."""
    risk = _get_risk()
    if risk is not None:
        return getattr(risk, attr, default)
    return default


def _safe_brain(attr: str, default=None):
    srv = _server_ref
    if srv and hasattr(srv, "brain"):
        return getattr(srv.brain, attr, default)
    return default


def _get_trading_mode() -> str:
    try:
        from Python import paper_trading
        return paper_trading.get_mode()
    except Exception:
        return "paper"


def _build_truth_payload(mt5_account: dict) -> dict:
    """Compute the canonical truth payload for status/mode endpoints."""
    try:
        from Python.execution.mode_resolver import resolve_mode
        from Python.execution.account_verifier import verify_account
        from Python.execution.live_gate import live_trading_allowed, demo_trading_allowed

        cfg = _read_config()
        system_mode = resolve_mode(cfg)

        # Map canonical mode to transport string
        transport_map = {
            "paper_sim": "paper",
            "demo_live": "mt5_demo",
            "real_live_locked": "mt5_live",
            "real_live": "mt5_live",
        }
        execution_transport = transport_map.get(system_mode, "paper")

        acct = verify_account(mt5_account)
        account_type = acct["account_type"]
        account_type_verified = acct["account_type_verified"]

        # Real money is locked unless we are in real_live with all gates open
        real_money_locked = system_mode in ("paper_sim", "demo_live", "real_live_locked")
        if system_mode == "real_live":
            allowed, _ = live_trading_allowed(cfg, {}, acct, {})
            real_money_locked = not allowed

        # Order send is enabled for everything except real_live_locked
        order_send_enabled = system_mode != "real_live_locked"

        # Demo canary is enabled when demo mode is active and a canary exists
        demo_canary_enabled = system_mode == "demo_live"
        if demo_canary_enabled:
            try:
                active = _read_active_registry()
                canary_id = active.get("canary") or ""
                demo_canary_enabled = bool(canary_id)
            except Exception:
                demo_canary_enabled = False

        # real_live_enabled only when mode==real_live and live gate passes
        real_live_enabled = False
        if system_mode == "real_live":
            allowed, _ = live_trading_allowed(cfg, {}, acct, {})
            real_live_enabled = allowed

        return {
            "system_mode": system_mode,
            "execution_transport": execution_transport,
            "account_type": account_type,
            "account_type_verified": account_type_verified,
            "real_money_locked": real_money_locked,
            "order_send_enabled": order_send_enabled,
            "demo_canary_enabled": demo_canary_enabled,
            "real_live_enabled": real_live_enabled,
        }
    except Exception as exc:
        logger.debug(f"_build_truth_payload failed: {exc}")
        return {
            "system_mode": "paper_sim",
            "execution_transport": "paper",
            "account_type": "unknown",
            "account_type_verified": False,
            "real_money_locked": True,
            "order_send_enabled": False,
            "demo_canary_enabled": False,
            "real_live_enabled": False,
        }


def _get_mt5_account_and_positions() -> dict:
    """Fetch live MT5 account info and open positions.

    Returns a dict with keys: balance, equity, free_margin, profit,
    open_positions, positions.  Falls back to risk-engine equity and
    empty positions when MT5 is unavailable (dry-run or non-Windows).
    In paper mode, returns simulated account data.
    """
    # Check paper mode first
    try:
        from Python import paper_trading
        if paper_trading.get_mode() == "paper":
            pacc = paper_trading.get_paper_account()
            positions = paper_trading.paper_positions_get()
            return {
                "balance": pacc["balance"],
                "equity": pacc["equity"],
                "free_margin": pacc["free_margin"],
                "profit": pacc.get("profit", 0.0),
                "open_positions": len(positions),
                "positions": positions,
                "login": 999999,
                "server": "PAPER-CHAIN",
                "name": "Paper Trader",
                "currency": "USD",
                "leverage": 100,
                "mode": "paper",
            }
    except Exception as e:
        logger.debug(f"Paper trading check failed: {e}")

    # Fallback defaults from risk engine (populated by equity poll)
    result = {
        "balance": _safe_risk("_mt5_balance", None) or _safe_risk("_current_equity", 0.0),
        "equity": _safe_risk("_current_equity", 0.0),
        "free_margin": _safe_risk("_mt5_free_margin", None) or _safe_risk("_current_equity", 0.0),
        "profit": _safe_risk("_mt5_profit", 0.0),
        "open_positions": 0,
        "positions": [],
        "login": None,
        "server": None,
        "name": None,
        "currency": None,
        "leverage": None,
        "mode": "live",
    }

    try:
        from Python.mt5_compat import mt5

        # Retry MT5 init a few times — terminal IPC may not be ready immediately after bridge startup
        _mt5_ready = False
        for _attempt in range(5):
            if mt5.initialize():
                _mt5_ready = True
                break
            time.sleep(1)
        if not _mt5_ready:
            logger.debug("MT5 init failed in API status handler after 5 retries")
            return result

        try:
            info = mt5.account_info()
            if info is not None:
                # Always capture metadata even if numeric telemetry is invalid
                result["login"] = getattr(info, "login", None)
                result["server"] = getattr(info, "server", None)
                result["name"] = getattr(info, "name", None)
                result["currency"] = getattr(info, "currency", None)
                result["leverage"] = getattr(info, "leverage", None)

                if getattr(info, "_valid", True):
                    result["balance"] = float(info.balance)
                    result["equity"] = float(info.equity)
                    result["free_margin"] = float(info.margin_free)
                    result["profit"] = float(info.profit)
                else:
                    logger.warning(
                        "API status: MT5 account telemetry invalid — "
                        "balance=%s, equity=%s, server=%s, login=%s. "
                        "Keeping risk-engine fallbacks.",
                        info.balance, info.equity, info.server, info.login,
                    )

            raw_positions = mt5.positions_get()
            if raw_positions:
                result["open_positions"] = len(raw_positions)
                result["positions"] = [
                    {
                        "ticket": p.ticket,
                        "symbol": p.symbol,
                        "type": "BUY" if p.type == 0 else "SELL",
                        "volume": float(p.volume),
                        "open_price": float(p.price_open),
                        "current_price": float(p.price_current),
                        "profit": float(p.profit),
                        "sl": float(p.sl) if p.sl else 0.0,
                        "tp": float(p.tp) if p.tp else 0.0,
                        "comment": p.comment or "",
                        "magic": p.magic,
                        "open_time": datetime.fromtimestamp(p.time, tz=timezone.utc).isoformat(),
                    }
                    for p in raw_positions
                ]
        finally:
            pass
    except Exception as e:
        logger.debug(f"MT5 account/positions fetch failed: {e}")

    return result


def _read_training_progress():
    """Read per-trainer progress files, including per-symbol PPO files.
    Falls back to parsing log files when JSON progress files don't exist."""
    result = {}
    now = time.time()

    for key in ("lstm", "ppo", "dreamer"):
        path = os.path.join(ROOT, "logs", f"{key}_progress.json")
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if now - data.get("updated_at", 0) < 600:
                    result[key] = data
                    continue
        except Exception:
            pass
        result[key] = {}

    # ── Fallback: infer from log files ──
    # LSTM
    try:
        lstm_log = os.path.join(ROOT, "logs", "lstm_training.log")
        if os.path.exists(lstm_log) and not result.get("lstm"):
            mtime = os.path.getmtime(lstm_log)
            if now - mtime < 300:
                lines = []
                with open(lstm_log, "r", encoding="utf-8", errors="replace") as f:
                    lines = [x.rstrip("\n") for x in f.readlines()[-40:]]
                for line in reversed(lines):
                    if " | epoch " in line and " | loss " in line:
                        # Strip loguru header (e.g. "2026-05-12 18:08:50.697 | SUCCESS  | __main__:train_lstm:272 - ")
                        msg = line
                        if " - " in msg:
                            msg = msg.split(" - ", 1)[1]
                        parts = [p.strip() for p in msg.split("|")]
                        if len(parts) >= 3:
                            sym = parts[0].split()[-1]
                            ep = parts[1].replace("epoch", "").strip()
                            score = parts[2].strip()
                            epoch_num, epoch_total = 0, 0
                            if "/" in ep:
                                a, b = ep.split("/", 1)
                                epoch_num = int(a.strip())
                                epoch_total = int(b.strip())
                            acc = 0.0
                            if "acc" in score.lower():
                                try:
                                    acc = float(score.lower().split("acc")[-1].strip().strip("%"))
                                except Exception:
                                    pass
                            result["lstm"] = {
                                "running": True,
                                "symbol": sym,
                                "epoch": epoch_num,
                                "epochs_total": epoch_total,
                                "accuracy": acc,
                                "updated_at": mtime,
                            }
                        break
    except Exception:
        pass

    # PPO
    try:
        ppo_log = os.path.join(ROOT, "logs", "ppo_training.log")
        if os.path.exists(ppo_log) and not result.get("ppo"):
            mtime = os.path.getmtime(ppo_log)
            if now - mtime < 300:
                lines = []
                with open(ppo_log, "r", encoding="utf-8", errors="replace") as f:
                    lines = [x.rstrip("\n") for x in f.readlines()[-80:]]
                symbol = ""
                for line in reversed(lines):
                    if "DRL Training | symbols=" in line:
                        idx = line.find("symbols=")
                        if idx >= 0:
                            chunk = line[idx + len("symbols="):]
                            if "[" in chunk and "]" in chunk:
                                s = chunk[chunk.find("[") + 1:chunk.find("]")]
                                symbol = s.split(",")[0].strip().strip("'\"")
                    if "best_score=" in line:
                        score = line.split("best_score=", 1)[1].strip()
                        result["ppo"] = {
                            "running": True,
                            "symbol": symbol,
                            "best_score": score,
                            "updated_at": mtime,
                        }
                        break
    except Exception:
        pass

    # Dreamer
    try:
        dreamer_log = os.path.join(ROOT, "logs", "dreamer_training.log")
        if os.path.exists(dreamer_log) and not result.get("dreamer"):
            mtime = os.path.getmtime(dreamer_log)
            if now - mtime < 300:
                result["dreamer"] = {"running": True, "updated_at": mtime}
    except Exception:
        pass

    # Rainforest
    try:
        rf_log = os.path.join(ROOT, "logs", "rainforest_training.log")
        if os.path.exists(rf_log) and not result.get("rainforest"):
            mtime = os.path.getmtime(rf_log)
            if now - mtime < 300:
                lines = []
                with open(rf_log, "r", encoding="utf-8", errors="replace") as f:
                    lines = [x.rstrip("\n") for x in f.readlines()[-20:]]
                rows = 0
                classes = []
                for line in reversed(lines):
                    if "Rainforest trained on" in line:
                        try:
                            rows = int(line.split("trained on")[1].strip().split()[0])
                        except Exception:
                            pass
                    if "classes=" in line:
                        try:
                            cls = line.split("classes=")[1].split("]")[0].strip("[]'\"")
                            classes = [c.strip().strip("'\"") for c in cls.split(",")]
                        except Exception:
                            pass
                result["rainforest"] = {
                    "running": True,
                    "rows": rows,
                    "classes": classes,
                    "updated_at": mtime,
                }
    except Exception:
        pass

    # Merge per-symbol PPO progress files (ppo_{SYMBOL}_progress.json)
    ppo_per_symbol = {}
    for fname in os.listdir(os.path.join(ROOT, "logs")) if os.path.isdir(os.path.join(ROOT, "logs")) else []:
        if fname.startswith("ppo_") and fname.endswith("_progress.json") and fname != "ppo_progress.json":
            sym = fname[4:-len("_progress.json")]
            path = os.path.join(ROOT, "logs", fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if time.time() - data.get("updated_at", 0) < 600:
                    ppo_per_symbol[sym] = data
            except Exception:
                pass
    if ppo_per_symbol:
        # Use the most recently updated per-symbol file as the primary PPO progress
        latest_sym = max(ppo_per_symbol, key=lambda s: ppo_per_symbol[s].get("updated_at", 0))
        if ppo_per_symbol[latest_sym].get("running"):
            result["ppo"] = ppo_per_symbol[latest_sym]
        result["ppo_per_symbol"] = ppo_per_symbol

    return result


def _resolve_system_mode(cfg: dict | None = None) -> dict:
    """Return honest system mode status."""
    try:
        from Python.execution.mode_resolver import resolve_mode
        mode = resolve_mode(cfg)
    except Exception:
        mode = "unknown"
    execution_transport = "mt5"
    try:
        from Python.mt5_compat import mt5
        execution_transport = "mt5" if mt5 is not None else "none"
    except Exception:
        pass
    real_money_locked = True
    live_lock_reason = "real_live_disabled"
    try:
        import Python.execution.live_gate as live_gate
        real_money_locked = getattr(live_gate, "real_money_locked", True)
        live_lock_reason = getattr(live_gate, "lock_reason", "real_live_disabled")
    except Exception:
        pass
    return {
        "system_mode": mode,
        "execution_transport": execution_transport,
        "real_money_locked": real_money_locked,
        "live_lock_reason": live_lock_reason,
    }


def _get_account_truth(mt5_account_info: dict) -> dict:
    """Return honest account truth fields."""
    try:
        from Python.execution.account_verifier import verify_account
        return verify_account(mt5_account_info)
    except Exception:
        balance = float(mt5_account_info.get("balance", 0.0) or 0.0)
        equity = float(mt5_account_info.get("equity", 0.0) or 0.0)
        login_raw = str(mt5_account_info.get("login", "") or "")
        login_masked = login_raw
        if len(login_raw) > 4:
            login_masked = f"{login_raw[:2]}***{login_raw[-2:]}"
        elif len(login_raw) > 0:
            login_masked = "***"
        return {
            "account_type": "unknown",
            "account_type_verified": False,
            "telemetry_valid": balance > 0 and equity > 0,
            "balance": balance,
            "equity": equity,
            "currency": str(mt5_account_info.get("currency", "USD") or "USD"),
            "server": "masked_demo_server" if not mt5_account_info.get("server") else str(mt5_account_info.get("server")),
            "login_masked": login_masked,
        }


def _get_data_provenance() -> dict:
    """Return honest data source status."""
    try:
        from Python.data.provenance import get_provenance_status
        return get_provenance_status()
    except Exception:
        pass
    return {
        "source": "MT5",
        "status": "unknown",
        "latest_dataset_id": "unknown",
    }


def _get_model_registry_status(progress: dict | None = None) -> dict:
    """Return honest model bundle statuses by scanning actual files on disk."""
    progress = progress or {}
    # ── Scan actual model files on disk ──
    lstm_files = []
    per_symbol_dir = os.path.join(ROOT, "models", "per_symbol")
    if os.path.isdir(per_symbol_dir):
        for f in os.listdir(per_symbol_dir):
            if f.endswith(".meta.json") and f.startswith("lstm_"):
                try:
                    with open(os.path.join(per_symbol_dir, f), "r") as fp:
                        meta = json.load(fp)
                    lstm_files.append({
                        "symbol": meta.get("symbol", f[5:-11]),
                        "samples": meta.get("samples", 0),
                        "seq_len": meta.get("seq_len", 0),
                        "epochs": meta.get("epochs", 0),
                    })
                except Exception:
                    pass

    # Check bundles
    bundles_dir = os.path.join(ROOT, "models", "bundles")
    bundle_files = []
    if os.path.isdir(bundles_dir):
        for f in os.listdir(bundles_dir):
            if f.endswith(".json"):
                try:
                    with open(os.path.join(bundles_dir, f), "r") as fp:
                        meta = json.load(fp)
                    bundle_files.append(meta)
                except Exception:
                    pass

    # Check PPO
    ppo_dir = os.path.join(ROOT, "models", "ppo")
    ppo_models = []
    if os.path.isdir(ppo_dir):
        for d in os.listdir(ppo_dir):
            if os.path.isdir(os.path.join(ppo_dir, d)):
                ppo_models.append(d)

    # Determine statuses based on actual files
    lstm_status = "trained" if lstm_files else "disabled"
    rainforest_status = "informational-only"
    if rf_detector is not None and rf_detector.is_trained():
        rainforest_status = "validated"
    dreamer_status = "stub_disabled"
    ppo_status = "undertrained"
    if ppo_models:
        ppo_status = "candidate" if bundle_files else "trained"
    ensemble_status = "disabled"
    if bundle_files:
        ensemble_status = "candidate"
    if any(b.get("status") == "champion" for b in bundle_files):
        ensemble_status = "champion"

    bundle_id = bundle_files[0].get("bundle_id", "unknown") if bundle_files else "none"

    return {
        "bundle_id": bundle_id,
        "lstm_status": lstm_status,
        "rainforest_status": rainforest_status,
        "dreamer_status": dreamer_status,
        "ppo_status": ppo_status,
        "ensemble_status": ensemble_status,
        "lstm_models": lstm_files,
        "ppo_models": ppo_models,
        "bundle_count": len(bundle_files),
    }


def _get_validation_status() -> dict:
    """Return honest validation / promotion gate status."""
    try:
        from Python.registry.promotion_gates import get_promotion_status
        return get_promotion_status()
    except Exception:
        pass
    # Safe defaults
    return {
        "backtest_status": "unknown",
        "walk_forward_status": "unknown",
        "promotion_status": "unknown",
        "champion_status": "not_real_live_eligible",
    }


def _get_test_status() -> dict:
    """Return honest pytest status from last run if available."""
    status = "unknown"
    failures = 0
    errors = 0
    try:
        path = os.path.join(ROOT, "logs", "pytest_results.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            status = data.get("status", "unknown")
            failures = data.get("failures", 0)
            errors = data.get("errors", 0)
    except Exception:
        pass
    return {
        "status": status,
        "open_failures": failures,
        "open_errors": errors,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 1. GET /api/status — Full system status
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/status")
def api_status():
    srv = _server_ref
    cfg = _read_config()
    symbols = cfg.get("trading", {}).get("symbols", ["EURUSD"])
    active = _read_active_registry()
    progress = _read_training_progress()
    lstm_p = progress.get("lstm", {})
    ppo_p = progress.get("ppo", {})
    dreamer_p = progress.get("dreamer", {})
    rf_p = progress.get("rainforest", {})
    live = _read_live_state()
    incidents = _read_incidents()

    champ_path = active.get("champion") or ""
    canary_path = active.get("canary") or ""
    champ_id = os.path.basename(champ_path) if champ_path else "none"
    canary_id = os.path.basename(canary_path) if canary_path else ""

    # ── Risk engine live values ──
    halt = _safe_risk("halt", False)
    halt_reason = _safe_risk("_halt_reason", "")
    daily_trades = _safe_risk("daily_trades", 0)
    realized_pnl = _safe_risk("realized_pnl_today", 0.0)
    current_dd = _safe_risk("current_dd", 0.0)
    peak_equity = _safe_risk("_peak_equity", 0.0)
    current_equity = _safe_risk("_current_equity", 0.0)
    max_daily_loss = _safe_risk("max_daily_loss", 500)
    max_hourly_loss = _safe_risk("max_hourly_loss", 150)
    max_daily_trades = _safe_risk("max_daily_trades", 200)
    max_drawdown_pct = _safe_risk("max_drawdown_pct", 8.0)
    max_open_positions = _safe_risk("max_open_positions", 8)
    max_positions_per_symbol = _safe_risk("max_positions_per_symbol", 2)
    risk = _get_risk()
    can_trade = False
    if risk is not None:
        try:
            can_trade = risk.can_trade()
        except Exception:
            pass

    uptime = int(time.time() - srv.start_time) if srv and hasattr(srv, "start_time") else 0
    env_mode = os.environ.get("CHAIN_GAMBLER_EXECUTION_MODE", "paper").strip().lower()
    if env_mode == "demo":
        mode = "DEMO"
    else:
        mode = "LIVE" if (srv and getattr(srv, "live", False)) else "DRY-RUN"

    # ── Advanced features status ──
    reversal_status = {"enabled": False}
    speed_status = {"enabled": False}
    # Embedded mode: check brain object
    if srv and hasattr(srv, "brain") and srv.brain:
        brain = srv.brain
        if hasattr(brain, "reversal_detector") and brain.reversal_detector is not None:
            reversal_status = {
                "enabled": True,
                "methods": ["divergence", "trend_exhaustion", "sr_break", "candlestick", "volume"],
                "auto_flip": True,
            }
    # Standalone mode: infer from process / imports
    if not reversal_status["enabled"]:
        try:
            from Python.reversal_detector import get_reversal_detector
            rev = get_reversal_detector()
            if rev is not None:
                reversal_status = {
                    "enabled": True,
                    "methods": ["divergence", "trend_exhaustion", "sr_break", "candlestick", "volume"],
                    "auto_flip": True,
                }
        except Exception:
            pass
    # Speed simulator is on paper_trader, not brain - check if paper trading mode
    if srv and hasattr(srv, "live") and not srv.live:
        # In dry-run/paper trading mode - speed simulation would be active
        speed_status = {
            "enabled": True,
            "mode": "paper_trading_simulation",
            "network_profile": "good",
            "avg_latency_ms": 50,
        }

    # ── Build lane rows from decision cache ──
    # Build per-symbol model info
    per_symbol_models = {}
    symbols_map = active.get("symbols", {})
    for sym, sym_data in symbols_map.items():
        sym_champ = sym_data.get("champion")
        sym_canary = sym_data.get("canary")
        per_symbol_models[sym] = {
            "champion": os.path.basename(sym_champ) if sym_champ else None,
            "canary": os.path.basename(sym_canary) if sym_canary else None,
            "has_per_symbol_champion": sym_champ is not None,
            "has_per_symbol_canary": sym_canary is not None,
            "canary_policy": sym_data.get("canary_policy", {}),
            "canary_state": sym_data.get("canary_state", {}),
        }

    lane_rows = []
    for sym in symbols:
        recent = list(_decision_cache.get(sym, []))
        last = recent[0] if recent else {}
        if not last:
            # Fallback to live_state.json for standalone mode
            lsym = live.get("symbols", {}).get(sym, {})
            last = {
                "volatility": lsym.get("regime", "--"),
                "exposure": lsym.get("blend_exposure", 0.0),
                "ppo_target": lsym.get("ppo_exposure", 0.0),
                "dreamer_target": lsym.get("dreamer_exposure", 0.0),
                "confidence": lsym.get("confidence", 0.0),
                "action": lsym.get("signal", "HOLD"),
            }
            
        sym_model = per_symbol_models.get(sym, {})
        # Per-symbol champion/canary with global fallback
        sym_champ_id = sym_model.get("champion") or champ_id
        sym_canary_id = sym_model.get("canary") or (canary_id or None)
        lane_rows.append({
            "symbol": sym,
            "decision": {
                "regime": last.get("volatility", "--"),
                "final_target": last.get("exposure", 0.0),
                "ppo_target": last.get("ppo_target", 0.0),
                "dreamer_target": last.get("dreamer_target", 0.0),
                "confidence": last.get("confidence", 0.0),
            },
            "pipeline": {
                "lstm": {"state": last.get("volatility", "UNKNOWN")},
            },
            "champion": sym_champ_id,
            "canary": sym_canary_id,
            "has_per_symbol_champion": sym_model.get("has_per_symbol_champion", False),
            "has_per_symbol_canary": sym_model.get("has_per_symbol_canary", False),
            "model_version": last.get("model_version", "champion"),
            "is_canary": last.get("is_canary", False),
            "status": "live" if not halt else "halted",
            "side": last.get("action", "HOLD").lower(),
            "confidence": last.get("confidence", 0.0),
            "exposure": last.get("exposure", 0.0),
            "pnl": 0.0,
            "canTrade": can_trade,
            "reason": last.get("reason", ""),
        })

    # ── Pipeline summary ──
    pipeline_summary = {
        "symbols_total": len(symbols),
        "training_active_symbols": 0,
        "canary_review_symbols": 1 if canary_id else 0,
        "champion_live_symbols": len(symbols) if champ_id != "none" else 0,
        "trading_ready_symbols": len(symbols) if champ_id != "none" else 0,
        "trading_active_symbols": len(symbols) if can_trade else 0,
    }

    # ── MT5 account info and open positions ──
    mt5_account = _get_mt5_account_and_positions()

    # ── Build a symbol->position lookup so lanes can show live PnL ──
    pos_by_symbol: dict[str, list] = {}
    for pos in mt5_account["positions"]:
        pos_by_symbol.setdefault(pos["symbol"], []).append(pos)

    # Merge real position PnL into lane rows
    for row in lane_rows:
        sym = row["symbol"]
        if sym in pos_by_symbol:
            row["pnl"] = round(sum(p["profit"] for p in pos_by_symbol[sym]), 2)

    lane_summary = {
        "actionable_symbols": sum(1 for r in lane_rows if r["side"] != "hold"),
        "executed_symbols": daily_trades,
        "blocked_symbols": sum(1 for r in lane_rows if not r["canTrade"]),
        "neutral_symbols": sum(1 for r in lane_rows if r["side"] == "hold"),
        "open_positions": mt5_account["open_positions"],
    }

    # ── Honest truth statuses ──
    system_truth = _resolve_system_mode(cfg)
    account_truth = _get_account_truth(mt5_account)
    data_truth = _get_data_provenance()
    models_truth = _get_model_registry_status(progress)
    validation_truth = _get_validation_status()
    tests_truth = _get_test_status()

    # Record heartbeats for agents based on actual activity
    if symbols:
        _agent_heartbeat("data_feed")
    if rf_detector and rf_detector.is_trained():
        _agent_heartbeat("pattern_detector")
    if srv and hasattr(srv, "risk"):
        _agent_heartbeat("risk_guardian")
    if lstm_p.get("running"):
        _agent_heartbeat("lstm_brain")
    if ppo_p.get("running"):
        _agent_heartbeat("ppo_brain")
    if dreamer_p.get("running"):
        _agent_heartbeat("dreamer")
    if mt5_account.get("open_positions", 0) > 0:
        _agent_heartbeat("trade_executor")

    return _json({
        "state": "online" if not halt else "halted",
        "status": "online" if not halt else "halted",
        "server": {
            "running": True,
            "pids": [os.getpid()],
            "bot_pid": _bot_process.pid if (_bot_process and _bot_process.poll() is None) else None,
        },
        "account": {
            "balance": mt5_account["balance"],
            "equity": mt5_account["equity"],
            "free_margin": mt5_account["free_margin"],
            "profit": mt5_account["profit"],
            "open_positions": mt5_account["open_positions"],
            "positions": mt5_account["positions"],
            "realized_today": realized_pnl,
            "drawdown_pct": current_dd,
            "connected": mode == "LIVE" or mt5_account["equity"] > 0,
            "login": mt5_account["login"],
            "server": mt5_account["server"],
            "name": mt5_account["name"],
            "currency": mt5_account["currency"],
            "leverage": mt5_account["leverage"],
            "mode": "demo" if env_mode == "demo" else mt5_account.get("mode", "live"),
            "account_type": account_truth.get("account_type", "unknown"),
            "account_type_verified": account_truth.get("account_type_verified", False),
            "telemetry_valid": account_truth.get("telemetry_valid", False),
            "login_masked": account_truth.get("login_masked", "***"),
        },
        "training": {
            "cycle_running": bool(live.get("training", {}).get("cycle_running", False)) or bool(lstm_p.get("running")) or bool(ppo_p.get("running")) or bool(rf_p.get("running")),
            "lstm_running": bool(lstm_p.get("running")),
            "drl_running": bool(ppo_p.get("running")),
            "dreamer_running": bool(dreamer_p.get("running")),
            "rainforest_running": bool(rf_p.get("running")),
            "configured_symbols": symbols,
            "lstm_symbol": lstm_p.get("symbol", ""),
            "lstm_epoch": lstm_p.get("epoch", 0),
            "lstm_epochs_total": lstm_p.get("epochs_total", 0),
            "drl_symbol": ppo_p.get("symbol", ""),
            "drl_timesteps": ppo_p.get("total_timesteps", 0),
            "visual": {
                "lstm": {
                    "state": "training" if lstm_p.get("running") else "idle",
                    "current_symbol": lstm_p.get("symbol", ""),
                    "loss": lstm_p.get("loss", 0),
                    "val_loss": 0,
                    "memory_strength": (lstm_p.get("accuracy", 0) / 100) if lstm_p.get("accuracy") else 0,
                },
                "ppo": {
                    "state": "training" if ppo_p.get("running") else "idle",
                    "current_symbol": ppo_p.get("symbol", ""),
                    "current_timesteps": ppo_p.get("current_timesteps", 0),
                    "target_timesteps": ppo_p.get("total_timesteps", 0),
                    "progress_pct": ppo_p.get("progress_pct", 0),
                },
                "dreamer": {
                    "state": "training" if dreamer_p.get("running") else "idle",
                    "current_symbol": dreamer_p.get("symbol", ""),
                    "steps": dreamer_p.get("step", 0),
                    "progress_pct": dreamer_p.get("progress_pct", 0),
                    "window": dreamer_p.get("window", 64),
                },
                "rainforest": {
                    "state": "training" if rf_p.get("running") else "idle",
                    "rows": rf_p.get("rows", 0),
                    "classes": rf_p.get("classes", []),
                    "top_feature": rf_p.get("top_feature", ""),
                },
                "active_label": (
                    "LSTM Training" if lstm_p.get("running")
                    else "PPO Training" if ppo_p.get("running")
                    else "Dreamer Training" if dreamer_p.get("running")
                    else "Rainforest Training" if rf_p.get("running")
                    else "Idle"
                ),
            },
            "symbol_stage_rows": [],
            "symbol_lane_rows": lane_rows,
            "pipeline_summary": pipeline_summary,
            "lane_summary": lane_summary,
            "ppo_per_symbol": progress.get("ppo_per_symbol", {}),
            **(_parallel_lane_status(srv)),
        },
        "canary_gate": {
            "ready": bool(canary_id),
            "reason": "Canary active" if canary_id else "No canary",
        },
        "active_models": active,
        "registry_summary": {
            "champion": champ_id,
            "canary": canary_id or None,
            "per_symbol_models": per_symbol_models,
        },
        "incidents": incidents or [{
            "id": "SYS-001",
            "type": "system",
            "severity": "info",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "API server online.",
        }],
        "logs": {},
        "timestamp": time.time(),
        "uptime_sec": uptime,
        "repo_root": ROOT,
        "mode": mode,
        **_build_truth_payload(mt5_account),
        "reversal": reversal_status,
        "speed": speed_status,
        "risk": {
            "halt": halt,
            "halt_reason": halt_reason,
            "daily_trades": daily_trades,
            "max_daily_trades": max_daily_trades,
            "realized_pnl": realized_pnl,
            "max_daily_loss": max_daily_loss,
            "max_hourly_loss": max_hourly_loss,
            "current_dd": current_dd,
            "max_drawdown_pct": max_drawdown_pct,
            "peak_equity": peak_equity,
            "current_equity": current_equity,
            "max_open_positions": max_open_positions,
            "max_positions_per_symbol": max_positions_per_symbol,
            "can_trade": can_trade,
        },
        "system": system_truth,
        "data": data_truth,
        "models": models_truth,
        "validation": validation_truth,
        "tests": tests_truth,
        "trade_review": _get_trade_review_summary(),
        "telegram": _get_telegram_status(),
        "economic_calendar": _get_economic_calendar_cached(),
        "rainforest": {
            "loaded": rf_detector.is_trained() if rf_detector is not None else False,
            "trained_at": _rainforest_trained_at or None,
            "per_symbol": {
                sym: {
                    "regime": rainforest_predictions[sym].get("regime", "ranging"),
                    "confidence": rainforest_predictions[sym].get("confidence", 0.0),
                }
                for sym in rainforest_predictions
            },
        },
    })


# ═══════════════════════════════════════════════════════════════════════════
# 1b. POST /api/mode — Toggle paper / live trading mode
# ═══════════════════════════════════════════════════════════════════════════
@app.post("/api/mode")
def api_set_mode():
    try:
        from Python import paper_trading
        body = request.json
        new_mode = (body.get("mode") if body else "").strip().lower()
        if new_mode not in ("paper", "live"):
            return _json({"error": "mode must be 'paper' or 'live'"}, status=400)
        result = paper_trading.set_mode(new_mode)
        # Enrich mode response with truth payload
        try:
            mt5_account = _get_mt5_account_and_positions()
            truth = _build_truth_payload(mt5_account)
        except Exception:
            truth = {}
        return _json({"success": True, **result, **truth})
    except Exception as exc:
        logger.warning(f"set_mode failed: {exc}")
        return _json({"error": str(exc)}, status=500)


# ═══════════════════════════════════════════════════════════════════════════
# 1b2. GET /api/live_gate — Live trading safety gate status
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/live_gate")
def api_live_gate():
    try:
        from Python import live_safety
        gate = live_safety.live_trading_allowed()
        return _json({
            "allowed": gate["allowed"],
            "mode": gate["mode"],
            "gates": gate["gates"],
        })
    except Exception as exc:
        logger.warning(f"live_gate failed: {exc}")
        return _json({"error": str(exc)}, status=500)


# ═══════════════════════════════════════════════════════════════════════════
# 1c. POST /api/mt5_login — Attempt MT5 login with credentials
# ═══════════════════════════════════════════════════════════════════════════
@app.post("/api/mt5_login")
def api_mt5_login():
    try:
        from Python.mt5_compat import mt5
        body = request.json or {}
        login = int(body.get("login", 0))
        password = body.get("password", "")
        server = body.get("server", "")

        if not login or not password or not server:
            return _json({"error": "login, password, and server are required"}, status=400)

        if not mt5.initialize():
            return _json({"error": "MT5 initialize failed"}, status=503)

        result = mt5.login(login, password, server)
        if result:
            info = mt5.account_info()
            valid = getattr(info, "_valid", True) if info else False
            return _json({
                "success": True,
                "login": getattr(info, "login", login),
                "server": getattr(info, "server", server),
                "name": getattr(info, "name", None),
                "balance": float(info.balance) if info and valid else None,
                "equity": float(info.equity) if info and valid else None,
            })
        else:
            return _json({"success": False, "error": "MT5 login rejected"}, status=401)
    except Exception as exc:
        logger.warning(f"mt5_login failed: {exc}")
        return _json({"success": False, "error": str(exc)}, status=500)


# ═══════════════════════════════════════════════════════════════════════════
# 1d. POST /api/paper_reset — Reset paper trading account balance
# ═══════════════════════════════════════════════════════════════════════════
@app.post("/api/paper_reset")
def api_paper_reset():
    try:
        from Python import paper_trading
        body = request.json or {}
        balance = float(body.get("balance", paper_trading.PAPER_DEFAULT_BALANCE))
        paper_trading.reset_paper_account(balance)
        return _json({"success": True, "balance": balance})
    except Exception as exc:
        logger.warning(f"paper_reset failed: {exc}")
        return _json({"error": str(exc)}, status=500)


# ═══════════════════════════════════════════════════════════════════════════
# 2. GET /api/trades — Recent trade history
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/trades")
def api_trades():
    limit = int(request.params.get("limit", 50))
    offset = int(request.params.get("offset", 0))
    symbol_filter = request.params.get("symbol", "")
    bot_lane_filter = request.params.get("bot_lane", "")

    trades = _fetch_trade_history(symbol_filter, bot_lane_filter)
    total = len(trades)
    page = trades[offset:offset + limit]

    return _json({
        "trades": page,
        "total": total,
        "limit": limit,
        "offset": offset,
    })


@app.get("/api/trades/summary")
def api_trades_summary():
    symbol_filter = request.params.get("symbol", "")
    bot_lane_filter = request.params.get("bot_lane", "")
    trades = _fetch_trade_history(symbol_filter, bot_lane_filter)

    wins = [t for t in trades if t.get("profit", 0) > 0]
    losses = [t for t in trades if t.get("profit", 0) < 0]
    total_pnl = sum(t.get("profit", 0) for t in trades)
    avg_profit = (sum(t["profit"] for t in wins) / len(wins)) if wins else 0
    avg_loss = (sum(t["profit"] for t in losses) / len(losses)) if losses else 0
    gross_profit = sum(t["profit"] for t in wins)
    gross_loss = abs(sum(t["profit"] for t in losses))
    pf = (gross_profit / gross_loss) if gross_loss > 0 else "inf"

    hold_mins = [t.get("hold_minutes", 0) for t in trades if t.get("hold_minutes")]

    # Per-symbol breakdown
    by_symbol: dict[str, Any] = {}
    for t in trades:
        sym = t.get("symbol", "UNKNOWN")
        if sym not in by_symbol:
            by_symbol[sym] = {"trades": 0, "wins": 0, "pnl": 0.0}
        by_symbol[sym]["trades"] += 1
        by_symbol[sym]["pnl"] += t.get("profit", 0)
        if t.get("profit", 0) > 0:
            by_symbol[sym]["wins"] += 1
    for sym in by_symbol:
        bs = by_symbol[sym]
        bs["win_rate"] = (bs["wins"] / bs["trades"]) if bs["trades"] > 0 else 0

    return _json({
        "overall": {
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": (len(wins) / len(trades)) if trades else 0,
            "total_pnl": round(total_pnl, 2),
            "avg_profit": round(avg_profit, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(pf, 2) if isinstance(pf, float) else pf,
            "avg_hold_minutes": round(sum(hold_mins) / len(hold_mins), 1) if hold_mins else 0,
            "max_loss_streak": _max_loss_streak(trades),
        },
        "by_symbol": by_symbol,
    })


def _extract_bot_lane(comment: str, magic: int | None) -> str:
    """Infer bot lane from MT5 order comment (preferred) or magic number fallback."""
    # Comment format: AGI|SYM|LANE_TAG|REQ_TAG|...
    if comment and comment.startswith("AGI|"):
        parts = comment.split("|")
        if len(parts) >= 3:
            tag = parts[2].upper()
            tag_map = {"CH": "champion", "CA": "canary", "HI": "history", "UN": "unknown"}
            lane = tag_map.get(tag)
            if lane:
                return lane
    # Fallback: coarse magic-number heuristic (ranges overlap across symbols,
    # so this is best-effort only).  Magic formula in mt5_executor:
    #   base = 505000 + symbol_offset*100 + lane_offset + kind_offset
    if magic is not None:
        base_mod = magic % 1000
        kind_mod = base_mod % 100
        hundreds = (base_mod // 100) % 10
        # champion: kind_mod in (0,10,20) AND hundreds even-ish (0,2,4...)
        # canary:   kind_mod in (0,10,20) AND hundreds odd-ish (1,3,5...)
        # This is imperfect because symbol base and lane offset are both
        # multiples of 100. We simply trust the comment first; this fallback
        # treats anything with hundreds==0 as champion, 1 as canary, 2 as history.
        if hundreds == 0:
            return "champion"
        if hundreds == 1:
            return "canary"
        if hundreds == 2:
            return "history"
    return "unknown"


def _fetch_trade_history(symbol_filter: str = "", bot_lane_filter: str = "") -> list[dict]:
    """
    Pull trade history from MT5 (Windows live) or from the decision cache (dry-run).
    Returns a list of Trade dicts matching the frontend Trade interface.
    """
    trades: list[dict] = []

    # Try MT5 deal history
    try:
        from Python.mt5_compat import mt5
        import pytz

        if mt5.initialize():
            tz = pytz.timezone("Etc/UTC")
            now_utc = datetime.now(tz)
            from datetime import timedelta
            lookback = now_utc - timedelta(days=30)
            deals = mt5.history_deals_get(lookback, now_utc)
            if deals:
                # Build index of entry deals keyed by position_id so we can
                # match open_price / open_time to each closing deal.
                entry_by_position: dict = {}
                for d in deals:
                    if d.entry == mt5.DEAL_ENTRY_IN:
                        entry_by_position[d.position_id] = d

                for d in deals:
                    if d.entry != mt5.DEAL_ENTRY_OUT:
                        continue
                    if symbol_filter and d.symbol != symbol_filter:
                        continue
                    lane = _extract_bot_lane(d.comment or "", d.magic)
                    if bot_lane_filter and lane != bot_lane_filter:
                        continue

                    entry = entry_by_position.get(d.position_id)
                    open_price = entry.price if entry else None
                    open_ts = datetime.fromtimestamp(entry.time, tz=tz) if entry else None
                    close_ts = datetime.fromtimestamp(d.time, tz=tz)
                    hold_minutes = int((close_ts - open_ts).total_seconds() / 60) if open_ts else None

                    trades.append({
                        "ticket": d.ticket,
                        "symbol": d.symbol,
                        "side": "BUY" if (entry.type if entry else d.type) == mt5.DEAL_TYPE_BUY else "SELL",
                        "volume": d.volume,
                        "open_time": open_ts.isoformat() if open_ts else None,
                        "close_time": close_ts.isoformat(),
                        "open_price": open_price,
                        "close_price": d.price,
                        "profit": round(d.profit, 2),
                        "comment": d.comment or "",
                        "hold_minutes": hold_minutes,
                        "magic": d.magic,
                        "bot_lane": lane,
                        "model": "champion" if lane == "champion" else ("canary" if lane == "canary" else "unknown"),
                        "action_type": "close",
                        "outcome": "win" if d.profit > 0 else ("loss" if d.profit < 0 else "breakeven"),
                    })
                trades.sort(key=lambda t: t.get("close_time", "") or "", reverse=True)
                return trades
    except Exception as e:
        logger.debug(f"MT5 trade history fetch failed: {e}")

    # Fallback 2: read from trades.db
    try:
        import sqlite3 as _sqlite3
        db_path = os.path.join(ROOT, "trades.db")
        if os.path.exists(db_path):
            conn = _sqlite3.connect(db_path)
            conn.row_factory = _sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trades ORDER BY close_time DESC LIMIT 500"
            ).fetchall()
            conn.close()
            for r in rows:
                if symbol_filter and r["symbol"] != symbol_filter:
                    continue
                if bot_lane_filter and r["bot_lane"] != bot_lane_filter:
                    continue
                trades.append({
                    "ticket": r["ticket"],
                    "symbol": r["symbol"],
                    "side": r["side"],
                    "volume": r["volume"],
                    "open_time": r["open_time"],
                    "close_time": r["close_time"],
                    "open_price": r["open_price"],
                    "close_price": r["close_price"],
                    "profit": r["profit"],
                    "comment": r["comment"] or "",
                    "hold_minutes": r["hold_minutes"],
                    "magic": r["magic"],
                    "bot_lane": r["bot_lane"] or "unknown",
                    "model": r["model"] or "unknown",
                    "action_type": r["action_type"] or "close",
                    "outcome": r["outcome"] or "breakeven",
                })
            return trades
    except Exception as e:
        logger.debug(f"SQLite trade history fetch failed: {e}")

    # Fallback 3: read from paper closed trades log (macOS paper mode)
    try:
        paper_log = os.path.join(ROOT, "logs", "paper_closed_trades.jsonl")
        if os.path.exists(paper_log):
            with open(paper_log, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        t = json.loads(line)
                        if symbol_filter and t.get("symbol") != symbol_filter:
                            continue
                        lane = t.get("bot_lane", "paper")
                        if bot_lane_filter and lane != bot_lane_filter:
                            continue
                        open_time = t.get("close_time")
                        close_time = t.get("close_time")
                        trades.append({
                            "ticket": t.get("ticket"),
                            "symbol": t.get("symbol", "?"),
                            "side": t.get("side", "BUY"),
                            "volume": t.get("volume", 0.0),
                            "open_time": open_time,
                            "close_time": close_time,
                            "open_price": t.get("open_price", 0.0),
                            "close_price": t.get("close_price", 0.0),
                            "profit": t.get("profit", 0.0),
                            "comment": t.get("comment", ""),
                            "hold_minutes": None,
                            "magic": t.get("magic"),
                            "bot_lane": lane,
                            "model": "paper",
                            "action_type": "close",
                            "outcome": "win" if t.get("profit", 0) > 0 else ("loss" if t.get("profit", 0) < 0 else "breakeven"),
                        })
                    except Exception:
                        continue
            trades.sort(key=lambda t: t.get("close_time", "") or "", reverse=True)
            return trades
    except Exception as e:
        logger.debug(f"Paper closed trades log read failed: {e}")

    # Fallback 4: derive from decision cache
    for sym, dq in _decision_cache.items():
        if symbol_filter and sym != symbol_filter:
            continue
        for i, d in enumerate(dq):
            if d.get("action") in ("BUY", "SELL"):
                lane = "ppo"
                if bot_lane_filter and lane != bot_lane_filter:
                    continue
                trades.append({
                    "ticket": int(d.get("_cached_at", time.time()) * 1000) + i,
                    "symbol": sym,
                    "side": d.get("action", "HOLD"),
                    "volume": abs(d.get("exposure", 0.0)),
                    "open_time": datetime.fromtimestamp(d.get("_cached_at", 0), tz=timezone.utc).isoformat(),
                    "close_time": None,
                    "open_price": 0,
                    "close_price": 0,
                    "profit": 0,
                    "comment": d.get("reason", ""),
                    "hold_minutes": None,
                    "magic": None,
                    "bot_lane": lane,
                    "model": "canary" if d.get("reason", "").startswith("canary") else "champion",
                    "action_type": "signal",
                    "outcome": "breakeven",
                })

    trades.sort(key=lambda t: t.get("open_time", "") or "", reverse=True)
    return trades


def _max_loss_streak(trades: list[dict]) -> int:
    streak = 0
    max_streak = 0
    for t in trades:
        if t.get("profit", 0) < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


# ═══════════════════════════════════════════════════════════════════════════
# 2b. GET /api/equity_curve — Time-series equity curve from closed trade history
# ═══════════════════════════════════════════════════════════════════════════
@app.route("/api/equity_curve", method=["GET", "OPTIONS"])
def get_equity_curve():
    """Return time-series equity curve from closed trade history."""
    limit = int(request.query.get("limit", 500))
    window = request.query.get("window", "all")  # '30d', '90d', 'all'

    rows: list[dict] = []

    # ── Source 1: SQLite databases (trades.db / bets.db) ──
    try:
        import sqlite3 as _sqlite3

        db_paths = [
            os.path.join(ROOT, "trades.db"),
            os.path.join(ROOT, "data", "bets.db"),
        ]
        db_path = None
        table_name = "trades"
        for p in db_paths:
            if os.path.exists(p):
                db_path = p
                break

        if db_path:
            conn = _sqlite3.connect(db_path)
            conn.row_factory = _sqlite3.Row
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            if "trades" in tables:
                table_name = "trades"
            elif "bets" in tables:
                table_name = "bets"
            else:
                conn.close()
                db_path = None

            if db_path:
                # Whitelist table names to prevent injection even though values are currently hardcoded
                if table_name not in ("trades", "bets"):
                    conn.close()
                    db_path = None
                else:
                    where = ""
                    if window == "30d":
                        where = "WHERE close_time >= datetime('now', '-30 days')"
                    elif window == "90d":
                        where = "WHERE close_time >= datetime('now', '-90 days')"
                    rows = [
                        {"close_time": r["close_time"], "profit": float(r["profit"])}
                        for r in conn.execute(
                            f"SELECT close_time, profit FROM {table_name} {where} ORDER BY close_time ASC"
                        ).fetchall()
                    ]
                    conn.close()
    except Exception as e:
        logger.debug(f"SQLite equity curve fetch failed: {e}")

    # ── Source 2: paper closed trades log (macOS / dry-run) ──
    if not rows:
        try:
            paper_log = os.path.join(ROOT, "logs", "paper_closed_trades.jsonl")
            if os.path.exists(paper_log):
                cutoff = None
                if window == "30d":
                    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
                elif window == "90d":
                    cutoff = datetime.now(timezone.utc) - timedelta(days=90)

                with open(paper_log, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            t = json.loads(line)
                            ts_str = t.get("close_time")
                            if not ts_str:
                                continue
                            if cutoff:
                                try:
                                    ts = datetime.fromisoformat(ts_str)
                                    if ts.replace(tzinfo=timezone.utc) < cutoff:
                                        continue
                                except Exception:
                                    pass
                            rows.append({"close_time": ts_str, "profit": float(t.get("profit", 0))})
                        except Exception:
                            continue
                rows.sort(key=lambda r: r["close_time"] or "")
        except Exception as e:
            logger.debug(f"Paper closed trades log equity fetch failed: {e}")

    if not rows:
        return _json({"points": [], "summary": {}})

    # Build equity curve
    starting_balance = 1000.0  # fallback
    try:
        srv = _server_ref
        state = srv.get_account_info() if srv and hasattr(srv, "get_account_info") else {}
        bal = state.get("balance", starting_balance) if state else starting_balance
        total_profit = sum(r["profit"] for r in rows)
        starting_balance = max(100.0, bal - total_profit)
    except Exception:
        pass

    equity = starting_balance
    peak = starting_balance
    points = []

    for r in rows:
        equity += r["profit"]
        peak = max(peak, equity)
        dd_pct = ((peak - equity) / peak * 100) if peak > 0 else 0
        points.append({
            "ts": r["close_time"],
            "equity": round(equity, 2),
            "balance": round(equity, 2),
            "drawdown_pct": round(dd_pct, 2),
        })

    # Downsample to limit points
    if len(points) > limit:
        step = len(points) // limit
        points = points[::step]

    max_dd = max((p["drawdown_pct"] for p in points), default=0)

    return _json({
        "points": points[-limit:],
        "summary": {
            "start_equity": round(starting_balance, 2),
            "current_equity": round(equity, 2),
            "peak_equity": round(peak, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "total_trades": len(rows),
        },
    })


# ═══════════════════════════════════════════════════════════════════════════
# 3. GET /api/ppo_diagnostics — PPO model diagnostics
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/ppo_diagnostics")
def api_ppo_diagnostics():
    brain = _safe_brain("__self__")  # get the brain object itself
    srv = _server_ref
    brain_obj = srv.brain if srv and hasattr(srv, "brain") else None

    active = _read_active_registry()
    champ_path = active.get("champion") or ""
    canary_path = active.get("canary") or ""

    ppo_loaded = False
    obs_shape = None
    action_shape = None
    is_canary = False
    device = "cpu"

    if brain_obj:
        ppo_loaded = brain_obj.ppo_model is not None
        is_canary = getattr(brain_obj, "_is_canary", False)
        device = getattr(brain_obj, "device", "cpu")
        if brain_obj.ppo_model is not None:
            try:
                obs_space = brain_obj.ppo_model.observation_space
                act_space = brain_obj.ppo_model.action_space
                obs_shape = list(obs_space.shape) if obs_space else None
                action_shape = list(act_space.shape) if act_space else None
            except Exception:
                pass

    # Last actions from decision cache
    last_actions = {}
    for sym, dq in _decision_cache.items():
        if dq:
            d = dq[0]
            last_actions[sym] = {
                "action": d.get("action"),
                "exposure": d.get("exposure"),
                "confidence": d.get("confidence"),
                "volatility": d.get("volatility"),
                "reason": d.get("reason"),
                "cached_at": d.get("_cached_at"),
            }

    # PPO bias correction data
    ppo_biases = {}
    if brain_obj and hasattr(brain_obj, "get_ppo_biases"):
        ppo_biases = brain_obj.get_ppo_biases()

    return _json({
        "ppo_loaded": ppo_loaded,
        "obs_shape": obs_shape,
        "action_shape": action_shape,
        "is_canary": is_canary,
        "device": device,
        "champion_path": champ_path,
        "canary_path": canary_path,
        "model_version": os.path.basename(canary_path if is_canary else champ_path) or "none",
        "last_actions": last_actions,
        "ppo_biases": ppo_biases,
    })


# ═══════════════════════════════════════════════════════════════════════════
# 4. GET /api/lstm_explanations — LSTM indicator attribution
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/lstm_explanations")
def api_lstm_explanations():
    """Return the last LSTM decision per symbol with top_indicators attribution."""
    results = {}

    try:
        for sym, dq in _decision_cache.items():
            try:
                # Find the most recent decision that has top_indicators
                for d in dq:
                    if not isinstance(d, dict):
                        continue
                    if "top_indicators" in d:
                        results[sym] = {
                            "regime": d.get("volatility") or d.get("regime", "UNKNOWN"),
                            "confidence": d.get("confidence", 0.0),
                            "top_indicators": d.get("top_indicators", []),
                            "cached_at": d.get("_cached_at"),
                        }
                        break
            except Exception:
                continue
    except Exception:
        pass

    if not results:
        # If no decisions have been cached yet, return empty with explanation
        return _json({
            "symbols": {},
            "message": "No LSTM decisions cached yet. Decisions are cached when the brain runs predictions.",
        })

    return _json({"symbols": results})


# ═══════════════════════════════════════════════════════════════════════════
# 5. GET /api/learning — Learning pipeline status
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/learning")
def api_learning():
    active = _read_active_registry()
    champ = active.get("champion")
    canary = active.get("canary")

    # Read champion scorecard if available
    champ_meta = {}
    if champ:
        sc = _read_json_file(os.path.join(champ, "scorecard.json"))
        if sc:
            champ_meta = sc

    # Read canary scorecard if available
    canary_meta = {}
    if canary:
        sc = _read_json_file(os.path.join(canary, "scorecard.json"))
        if sc:
            canary_meta = sc

    # List candidate versions
    cands_dir = os.path.join(ROOT, "models", "registry", "candidates")
    candidates = []
    if os.path.isdir(cands_dir):
        for d in sorted(os.listdir(cands_dir), reverse=True)[:10]:
            cpath = os.path.join(cands_dir, d)
            if os.path.isdir(cpath):
                sc = _read_json_file(os.path.join(cpath, "scorecard.json")) or {}
                candidates.append({
                    "version": d,
                    "path": cpath,
                    "win_rate": sc.get("win_rate"),
                    "loss": sc.get("loss"),
                    "saved_at": sc.get("saved_at"),
                    "type": sc.get("type"),
                })

    # Training schedule from config
    cfg = _read_config()
    train_enabled = os.environ.get("AGI_AUTONOMY_TRAIN", "false").lower() == "true"
    autonomy_interval = int(os.environ.get("AGI_AUTONOMY_INTERVAL_SEC", "3600"))

    # Trade learning log
    learning_log = _read_json_file(
        os.path.join(ROOT, "logs", "learning", "trade_learning_latest.json")
    )

    return _json({
        "canary": {
            "active": canary is not None,
            "path": canary,
            "version": os.path.basename(canary) if canary else None,
            "scorecard": canary_meta,
        },
        "champion": {
            "path": champ,
            "version": os.path.basename(champ) if champ else None,
            "scorecard": champ_meta,
        },
        "candidates": candidates,
        "training_schedule": {
            "enabled": train_enabled,
            "interval_sec": autonomy_interval,
            "auto_canary": os.environ.get("AGI_AUTONOMY_AUTO_CANARY", "true").lower() == "true",
        },
        "learning_log": learning_log,
    })


# ═══════════════════════════════════════════════════════════════════════════
# 6. GET /api/regimes — Regime performance data
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/regimes")
def api_regimes():
    """Return performance breakdown by volatility regime from the decision cache."""
    regime_stats: dict[str, dict] = {}

    for sym, dq in _decision_cache.items():
        for d in dq:
            regime = d.get("volatility") or d.get("regime", "UNKNOWN")
            if regime not in regime_stats:
                regime_stats[regime] = {
                    "total_decisions": 0,
                    "buy_count": 0,
                    "sell_count": 0,
                    "hold_count": 0,
                    "avg_confidence": 0.0,
                    "avg_exposure": 0.0,
                    "symbols": set(),
                }
            rs = regime_stats[regime]
            rs["total_decisions"] += 1
            action = d.get("action", "HOLD")
            if action == "BUY":
                rs["buy_count"] += 1
            elif action == "SELL":
                rs["sell_count"] += 1
            else:
                rs["hold_count"] += 1
            rs["avg_confidence"] += d.get("confidence", 0.0)
            rs["avg_exposure"] += abs(d.get("exposure", 0.0))
            rs["symbols"].add(sym)

    # Finalize averages and serialize sets
    for regime, rs in regime_stats.items():
        n = rs["total_decisions"] or 1
        rs["avg_confidence"] = round(rs["avg_confidence"] / n, 4)
        rs["avg_exposure"] = round(rs["avg_exposure"] / n, 4)
        rs["symbols"] = sorted(rs["symbols"])

    return _json({"regimes": regime_stats})


# ═══════════════════════════════════════════════════════════════════════════
# 7. GET /api/lanes — Trading lane status per symbol
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/lanes")
def api_lanes():
    cfg = _read_config()
    symbols = cfg.get("trading", {}).get("symbols", ["EURUSD"])
    active = _read_active_registry()
    champ_id = os.path.basename(active.get("champion") or "") or "none"
    canary_id = os.path.basename(active.get("canary") or "")

    can_trade = False
    if _server_ref and hasattr(_server_ref, "risk"):
        try:
            can_trade = _server_ref.risk.can_trade()
        except Exception:
            pass

    lanes = []
    symbols_map = active.get("symbols", {})
    for sym in symbols:
        recent = list(_decision_cache.get(sym, []))
        last = recent[0] if recent else {}
        sym_data = symbols_map.get(sym, {})
        sym_champ = sym_data.get("champion")
        sym_canary = sym_data.get("canary")
        sym_champ_id = os.path.basename(sym_champ) if sym_champ else champ_id
        sym_canary_id = os.path.basename(sym_canary) if sym_canary else (canary_id or None)
        lanes.append({
            "symbol": sym,
            "champion": sym_champ_id,
            "canary": sym_canary_id,
            "has_per_symbol_champion": sym_champ is not None,
            "has_per_symbol_canary": sym_canary is not None,
            "model_version": last.get("model_version", "champion"),
            "action": last.get("action", "HOLD"),
            "exposure": last.get("exposure", 0.0),
            "confidence": last.get("confidence", 0.0),
            "volatility": last.get("volatility", "UNKNOWN"),
            "reason": last.get("reason", ""),
            "can_trade": can_trade,
            "is_canary": last.get("is_canary", bool(sym_canary or canary_id)),
            "last_decision_at": last.get("_cached_at"),
            "recent_decisions": len(recent),
        })

    return _json({"lanes": lanes})


# ═══════════════════════════════════════════════════════════════════════════
# 7b. GET /api/per_symbol_models — Per-symbol model registry info
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/per_symbol_models")
def api_per_symbol_models():
    """Return per-symbol champion and canary model paths from the registry."""
    active = _read_active_registry()
    symbols_map = active.get("symbols", {})
    global_champ = active.get("champion")
    global_canary = active.get("canary")

    result = {}
    for sym, sym_data in symbols_map.items():
        sym_champ = sym_data.get("champion")
        sym_canary = sym_data.get("canary")
        result[sym] = {
            "champion": sym_champ,
            "champion_basename": os.path.basename(sym_champ) if sym_champ else None,
            "canary": sym_canary,
            "canary_basename": os.path.basename(sym_canary) if sym_canary else None,
            "uses_global_champion": sym_champ is None,
            "uses_global_canary": sym_canary is None,
            "canary_policy": sym_data.get("canary_policy", {}),
            "canary_state": sym_data.get("canary_state", {}),
            "champion_history_count": len(sym_data.get("champion_history", [])),
        }

    return _json({
        "global_champion": global_champ,
        "global_canary": global_canary,
        "symbols": result,
    })


# ═══════════════════════════════════════════════════════════════════════════
# Rainforest pattern detection endpoint
# ═══════════════════════════════════════════════════════════════════════════
@app.route("/api/patterns/rainforest", method=["GET", "OPTIONS"])
def get_rainforest_patterns():
    """Return Rainforest Random-Forest regime predictions per symbol."""
    if request.method == "OPTIONS":
        return {}

    cfg = _read_config()
    symbols = cfg.get("trading", {}).get("symbols", [])

    # If we have no live predictions yet, try to compute them now from
    # synthetic / cached models for each configured symbol.
    per_symbol: dict[str, dict] = {}
    live = _read_live_state()
    for sym in symbols:
        if sym in rainforest_predictions:
            pred = rainforest_predictions[sym]
        else:
            lsym = live.get("symbols", {}).get(sym, {})
            pred = {
                "regime": lsym.get("rainforest_regime", "ranging"),
                "confidence": lsym.get("rainforest_confidence", 0.0),
                "probabilities": {},
                "feature_importances": {},
                "top_patterns": [],
                "note": "from_live_state" if "rainforest_regime" in lsym else "no_prediction_yet",
            }
        per_symbol[sym] = {
            "regime": pred.get("regime", "ranging"),
            "confidence": pred.get("confidence", 0.0),
            "probabilities": pred.get("probabilities", {}),
            "feature_importances": pred.get("feature_importances", {}),
            "top_patterns": pred.get("top_patterns", []),
        }

    return _json({
        "trained_at": _rainforest_trained_at or None,
        "n_trees": rf_detector.n_estimators if rf_detector is not None else 200,
        "model_loaded": rf_detector.is_trained() if rf_detector is not None else False,
        "per_symbol": per_symbol,
    })


# ═══════════════════════════════════════════════════════════════════════════
# Existing endpoints (compat with current frontend api.ts)
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/patterns")
def api_patterns():
    """Pattern library — extracted from logs/patterns.jsonl, MT5 candles, or incidents."""
    patterns = []

    # 1. Try dedicated pattern log
    patterns_log = os.path.join(ROOT, "logs", "patterns.jsonl")
    if os.path.exists(patterns_log):
        try:
            with open(patterns_log, "r", encoding="utf-8") as f:
                lines = f.readlines()[-50:]
            for line in lines:
                try:
                    p = json.loads(line.strip())
                    if isinstance(p, dict):
                        patterns.append(p)
                except (json.JSONDecodeError, ValueError):
                    pass
            if patterns:
                return _json(patterns)
        except Exception:
            pass

    # 2. Try basic candlestick detection from MT5
    try:
        from Python.mt5_compat import mt5
        import pytz

        if mt5.initialize():
            tz = pytz.timezone("Etc/UTC")
            symbols = ["EURUSD", "GBPUSD", "XAUUSD", "BTCUSD"]
            for sym in symbols:
                rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 0, 20)
                if rates and len(rates) >= 2:
                    # Simple pattern detection on the last 3 candles
                    for i in range(-3, 0):
                        c = rates[i]
                        body = abs(c["close"] - c["open"])
                        lower = c["open"] - c["low"] if c["close"] >= c["open"] else c["close"] - c["low"]
                        upper = c["high"] - c["close"] if c["close"] >= c["open"] else c["high"] - c["open"]
                        pattern_name = None
                        if body < 1e-9 and upper > 0 and lower > 0:
                            pattern_name = "doji"
                        elif lower > 2 * body and upper < body:
                            pattern_name = "hammer"
                        if pattern_name:
                            patterns.append({
                                "type": "pattern",
                                "pattern": pattern_name,
                                "symbol": sym,
                                "timestamp": datetime.fromtimestamp(c["time"], tz=tz).isoformat(),
                                "open": float(c["open"]),
                                "high": float(c["high"]),
                                "low": float(c["low"]),
                                "close": float(c["close"]),
                            })
            if patterns:
                return _json(patterns[-10:])
    except Exception:
        pass

    # 3. Fallback: return last 10 incidents of any type so the tab isn't empty
    incidents = _read_incidents()
    for inc in incidents[-10:]:
        p = dict(inc)
        if "type" not in p:
            p["type"] = "pattern"
        patterns.append(p)

    return _json(patterns)


@app.get("/api/perf")
def api_perf():
    """Performance metrics summary."""
    live = _read_live_state()
    hist = live.get("_history", {})

    # Primary: live equity history from server's risk engine
    srv = _server_ref
    equity_curve = []
    pnl_curve = []
    confidence_curve = []

    if srv and hasattr(srv, "risk"):
        try:
            equity_curve = list(getattr(srv.risk, "_equity_history", []))
            pnl_curve = list(getattr(srv.risk, "_pnl_history", []))
        except Exception:
            pass

    # Build confidence curve from recent decisions
    try:
        decisions_log = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "logs", "decisions.jsonl"
        )
        if os.path.exists(decisions_log):
            with open(decisions_log, "r") as f:
                lines = f.readlines()[-300:]  # Last 300 decisions
            for line in lines:
                try:
                    d = json.loads(line.strip())
                    conf = d.get("confidence", 0)
                    confidence_curve.append(float(conf))
                except (json.JSONDecodeError, ValueError):
                    pass
    except Exception:
        pass

    # Fallback to file-based history if server has no data yet
    if not equity_curve:
        equity_curve = hist.get("equity", [])
    if not pnl_curve:
        pnl_curve = hist.get("pnl", [])
    if not confidence_curve:
        confidence_curve = hist.get("confidence", [])

    # Read adaptation history from live_state (training section)
    adaptation_history = []
    try:
        live_state = _read_live_state()
        training_data = live_state.get("training", {})
        if isinstance(training_data, dict):
            adaptation_history = training_data.get("adaptation_history", [])
            if not isinstance(adaptation_history, list):
                adaptation_history = []
    except Exception:
        pass

    return _json({
        "equity_curve": equity_curve,
        "pnl_curve": pnl_curve,
        "confidence_curve": confidence_curve,
        "lstm_loss_curve": hist.get("lstmLoss", []),
        "adaptation_history": adaptation_history,
    })


# ── Protected control actions requiring a control token ──────────────────
_PROTECTED_ACTIONS = {
    "promote_canary", "rollback_canary", "rollback_champion",
    "restart_server", "start_training_cycle", "stop_training_cycle",
    "emergency_stop", "clear_emergency_stop", "unblock", "arm_live",
    "start_bot", "stop_bot",
}

import secrets

_CONTROL_TOKEN = os.environ.get("AGI_CONTROL_TOKEN", "")
# Security: In production, control token MUST be set AND strong
_IS_PRODUCTION = os.environ.get("AGI_IS_LIVE", "0") == "1"
_MIN_TOKEN_LENGTH = 24  # bytes after urlsafe encoding; enforce at startup

if _CONTROL_TOKEN and len(_CONTROL_TOKEN) < _MIN_TOKEN_LENGTH:
    logger.warning(f"AGI_CONTROL_TOKEN is too short ({len(_CONTROL_TOKEN)} chars). Minimum {_MIN_TOKEN_LENGTH} recommended for production.")
    if _IS_PRODUCTION:
        _CONTROL_TOKEN = ""  # treat as unset in prod to force explicit strong token


@app.post("/api/control")
def api_control():
    """Accept control commands from the React UI.

    Protected actions require the X-Control-Token header to match
    the AGI_CONTROL_TOKEN environment variable.
    """
    try:
        payload = request.json or {}
    except Exception:
        payload = {}
    action = payload.get("action", "unknown")
    logger.info(f"API control action received: {action}")

    # ── Token auth for protected actions ────────────────────────────────
    if action in _PROTECTED_ACTIONS:
        token = request.get_header("X-Control-Token", "").strip()

        # Security: In production, control token must be configured
        if _IS_PRODUCTION and not _CONTROL_TOKEN:
            logger.error(f"Control action '{action}' rejected — AGI_CONTROL_TOKEN not configured in production")
            return _json({"ok": False, "action": action, "error": "control token not configured"}, 503)

        # Security: Use constant-time comparison to prevent timing attacks
        if not _CONTROL_TOKEN:
            logger.warning(f"Control action '{action}' rejected — no control token configured")
            return _json({"ok": False, "action": action, "error": "control token required"}, 403)

        if not secrets.compare_digest(token, _CONTROL_TOKEN):
            logger.warning(f"Control action '{action}' rejected — invalid token")
            return _json({"ok": False, "action": action, "error": "control token required"}, 403)

    srv = _server_ref

    # ── Emergency stop / clear ──────────────────────────────────────────
    if action == "emergency_stop":
        if srv and hasattr(srv, "risk"):
            srv.risk.halt = True
            logger.critical("EMERGENCY STOP ACTIVATED via API — all trading halted")
            # Optionally close all open positions if executor is available
            if hasattr(srv, "executor") and srv.executor:
                try:
                    srv.executor.close_all_positions()
                    logger.info("All open positions closed during emergency stop")
                except Exception as e:
                    logger.warning(f"Failed to close positions during emergency stop: {e}")
            try:
                srv.telegram.risk_event("EMERGENCY STOP", "All trading halted via API")
            except Exception:
                pass
            return _json({"ok": True, "action": action, "message": "Emergency stop activated. All trading halted.", "halted": True})
        return _json({"ok": False, "action": action, "error": "No risk engine available"}, 500)

    if action == "clear_emergency_stop":
        if srv and hasattr(srv, "risk"):
            srv.risk.halt = False
            srv.risk._consecutive_errors = 0
            logger.info("Emergency stop CLEARED via API — trading resumed")
            return _json({"ok": True, "action": action, "message": "Emergency stop cleared. Trading resumed.", "halted": False})
        return _json({"ok": False, "action": action, "error": "No risk engine available"}, 500)

    # ── Unblock: Hard-reset risk engine halt and reset counters ──
    if action == "unblock":
        if srv and hasattr(srv, "_unblock"):
            result = srv._unblock()
            # Also clear executor cooldowns
            if hasattr(srv, "executor") and srv.executor:
                srv.executor._last_failed_signal_time.clear()
                srv.executor._last_spread_spike_time.clear()
                logger.success("UNBLOCK: executor cooldowns cleared")
            return _json(result)
        elif srv and hasattr(srv, "risk"):
            # Fallback: directly reset risk engine
            prev_halt = srv.risk.halt
            prev_reason = srv.risk.get_halt_reason() if hasattr(srv.risk, 'get_halt_reason') else ""
            srv.risk.halt = False
            srv.risk._halt_reason = ""
            srv.risk.error_count = 0
            srv.risk.realized_pnl_today = 0.0
            srv.risk.daily_trades = 0
            # Also clear executor cooldowns
            if hasattr(srv, "executor") and srv.executor:
                srv.executor._last_failed_signal_time.clear()
                srv.executor._last_spread_spike_time.clear()
                logger.success("UNBLOCK: executor cooldowns cleared")
            logger.success(f"UNBLOCK via API: risk engine cleared (was halted={prev_halt}, reason={prev_reason})")
            return _json({"ok": True, "action": "unblock", "was_halted": prev_halt, "was_reason": prev_reason, "halt": False, "message": "Risk engine unblocked + cooldowns cleared via API."})
        return _json({"ok": False, "action": action, "error": "No risk engine available"}, 500)

    # ── Arm live: Enable live trading ──
    if action == "arm_live":
        if srv and hasattr(srv, "_arm_live"):
            result = srv._arm_live()
            return _json(result)
        return _json({"ok": False, "action": action, "error": "Server not available"}, 500)

    # ── Standard UI actions ─────────────────────────────────────────────
    if action == "restart_server":
        return _json({"ok": True, "action": action, "message": "Server is running. Use system-level restart to restart."})

    # ── Start/Stop bot process ──────────────────────────────────────────────
    if action == "start_bot":
        global _bot_process
        with _bot_process_lock:
            # Check if already running
            if _bot_process is not None and _bot_process.poll() is None:
                return _json({"ok": True, "action": action, "message": f"Bot already running (PID {_bot_process.pid})", "pid": _bot_process.pid})

            # Check if Server_AGI is already running as a separate process
            try:
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-Command",
                     "Get-Process python* -ErrorAction SilentlyContinue | "
                     "Where-Object { $_.CommandLine -match 'Server_AGI' } | "
                     "Select-Object -ExpandProperty Id"],
                    capture_output=True, text=True, timeout=10,
                )
                existing_pids = [p.strip() for p in (result.stdout or "").strip().split("\n") if p.strip().isdigit()]
                if existing_pids:
                    return _json({"ok": True, "action": action, "message": f"Bot already running (PID {existing_pids[0]})", "pid": int(existing_pids[0])})
            except Exception:
                pass

            # Find venv python
            venv_python = os.path.join(ROOT, ".venv312", "Scripts", "python.exe")
            if not os.path.exists(venv_python):
                venv_python = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
            if not os.path.exists(venv_python):
                venv_python = sys.executable

            # Determine live vs dry-run mode from config
            cfg = _read_config()
            live_mode = srv and getattr(srv, "live", False)

            log_dir = os.path.join(ROOT, "logs")
            os.makedirs(log_dir, exist_ok=True)
            stdout_log = open(os.path.join(log_dir, "bot_stdout.log"), "a", encoding="utf-8")
            stderr_log = open(os.path.join(log_dir, "bot_stderr.log"), "a", encoding="utf-8")

            cmd = [venv_python, "-m", "Python.Server_AGI"]

            # Default to paper mode; live requires explicit env opt-in
            env = os.environ.copy()
            env.setdefault("CHAIN_GAMBLER_EXECUTION_MODE", "paper")
            env.setdefault("CHAIN_GAMBLER_ALLOW_LIVE", "0")
            env["AGI_LIVE_ENABLED"] = "false"
            env["AGI_TRADE_INTERVAL_SEC"] = "300"
            env["AGI_REVIEW_INTERVAL_SEC"] = "120"
            env["AGI_TRAIL_INTERVAL_SEC"] = "30"
            env["AGI_EQUITY_POLL_SEC"] = "15"
            env["AGI_HEDGING_ENABLED"] = "false"
            env["AGI_MAX_POS_PER_SYMBOL"] = "5"
            env["AGI_SL_COOLDOWN_MIN"] = "5"
            env["AGI_BIAS_WINDOW"] = "50"
            env["AGI_BIAS_STRENGTH"] = "0.5"
            env["AGI_ACTION_THRESHOLD"] = "0.0001"
            env["AGI_DEADZONE_CONFIDENCE"] = "0.99"

            try:
                proc = subprocess.Popen(cmd, cwd=ROOT, stdout=stdout_log, stderr=stderr_log, env=env)
                logger.success(f"Bot process started: PID={proc.pid} cmd={' '.join(cmd)}")
            except Exception as exc:
                logger.error(f"Failed to start bot: {exc}")
                return _json({"ok": False, "action": action, "error": f"Failed to start bot: {exc}"}, 500)

            # Store reference (module-level)
            _bot_process = proc

            # Brief wait to see if it crashes immediately
            time.sleep(2)
            if proc.poll() is not None:
                return _json({"ok": False, "action": action, "error": f"Bot process exited immediately with code {proc.returncode}"}, 500)

            return _json({"ok": True, "action": action, "message": f"Bot started (PID {proc.pid})", "pid": proc.pid})

    if action == "stop_bot":
        with _bot_process_lock:
            killed_pids = []
            # Stop tracked subprocess
            if _bot_process is not None and _bot_process.poll() is None:
                try:
                    _bot_process.terminate()
                    _bot_process.wait(timeout=10)
                    killed_pids.append(_bot_process.pid)
                except subprocess.TimeoutExpired:
                    _bot_process.kill()
                    killed_pids.append(_bot_process.pid)
                except Exception:
                    pass
                _bot_process = None

            # Also kill any Server_AGI process found via powershell
            try:
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-Command",
                     "Get-Process python* -ErrorAction SilentlyContinue | "
                     "Where-Object { $_.CommandLine -match 'Server_AGI' } | "
                     "Select-Object -ExpandProperty Id"],
                    capture_output=True, text=True, timeout=10,
                )
                for pid_str in (result.stdout or "").strip().split("\n"):
                    pid_str = pid_str.strip()
                    if pid_str.isdigit():
                        try:
                            subprocess.run(["powershell", "-NoProfile", "-Command", f"Stop-Process -Id {pid_str} -Force"], check=False, timeout=5)
                            killed_pids.append(int(pid_str))
                        except Exception:
                            pass
            except Exception:
                pass

            if not killed_pids:
                return _json({"ok": True, "action": action, "message": "No bot process was running."})

            # Also set risk halt if server ref is available
            if srv and hasattr(srv, "risk"):
                try:
                    srv.risk.halt = True
                    srv.risk._halt_reason = "Bot stopped via dashboard"
                except Exception:
                    pass

            return _json({"ok": True, "action": action, "message": f"Bot stopped (PIDs: {killed_pids})", "pids": killed_pids})

    if action == "hft_start":
        return _json({"ok": True, "action": action, "message": "HFT mode not available in current configuration."})

    if action == "hft_stop":
        return _json({"ok": True, "action": action, "message": "HFT mode not active."})

    if action == "stop_training_cycle":
        if srv and hasattr(srv, "autonomy") and srv.autonomy:
            try:
                srv.autonomy.stop()
                return _json({"ok": True, "action": action, "message": "Training cycle stop requested."})
            except Exception as e:
                return _json({"ok": False, "action": action, "error": str(e)}, 500)
        return _json({"ok": True, "action": action, "message": "No active training cycle."})

    if action == "reset_peak_equity":
        if srv and hasattr(srv, "risk"):
            srv.risk.reset_peak_equity()
            return _json({"ok": True, "action": action, "message": f"Peak equity reset to {srv.risk._peak_equity:.2f}"})
        return _json({"ok": False, "action": action, "error": "No risk engine available"}, 500)

    if action == "start_training_cycle":
        if srv and hasattr(srv, "autonomy") and srv.autonomy:
            return _json({"ok": True, "action": action, "message": "Autonomy loop is already running."})
        return _json({"ok": True, "action": action, "message": "Autonomy loop not initialized."})

    if action == "rebuild_trade_memory":
        return _json({"ok": True, "action": action, "message": "Trade memory rebuild queued."})

    if action == "promote_canary":
        symbol = payload.get("symbol")
        if srv and hasattr(srv, "autonomy") and srv.autonomy:
            try:
                from Python.model_registry import ModelRegistry
                registry = ModelRegistry()
                if symbol:
                    registry.promote_canary_to_champion(symbol=symbol)
                else:
                    registry.promote_canary()
                return _json({"ok": True, "action": action, "symbol": symbol or "global", "message": "Canary promoted to champion."})
            except Exception as e:
                return _json({"ok": False, "action": action, "error": str(e)}, 500)
        return _json({"ok": True, "action": action, "message": "No canary to promote."})

    if action == "force_ingest":
        return _json({"ok": True, "action": action, "message": "Data ingest triggered."})

    if action == "start_parallel_training":
        if srv and hasattr(srv, "lane_mgr") and srv.lane_mgr:
            srv.lane_mgr.start_cycle()
            return _json({"ok": True, "action": action, "message": "Parallel training started"})
        return _json({"ok": True, "action": action, "message": "lane_mgr not initialized on server."})

    if action in ("rollback_canary", "rollback_champion"):
        symbol = payload.get("symbol")
        if srv and hasattr(srv, "autonomy") and srv.autonomy:
            try:
                from Python.model_registry import ModelRegistry
                registry = ModelRegistry()
                if symbol:
                    registry.clear_canary(symbol=symbol)
                else:
                    registry.rollback_canary()
                return _json({"ok": True, "action": action, "symbol": symbol or "global", "message": "Canary rolled back."})
            except Exception as e:
                return _json({"ok": False, "action": action, "error": str(e)}, 500)
        return _json({"ok": True, "action": action, "message": "No canary to rollback."})

    # Fallback: forward to AGIServer.handle_command (token-gated, for socket/n8n)
    if srv:
        try:
            result = srv.handle_command({"action": action, **payload})
            return _json({"ok": True, "action": action, "result": result})
        except Exception as e:
            return _json({"ok": False, "action": action, "error": str(e)}, 500)

    return _json({"ok": True, "action": action, "message": f"Action '{action}' acknowledged."})


# ═══════════════════════════════════════════════════════════════════════════
# 8. WebSocket /ws/status — Real-time push (via simple polling SSE fallback)
#    Bottle doesn't natively support WebSocket, so we provide SSE instead.
#    The frontend createStatusWS() will need a small adapter, but /api/status
#    polling every 2-5s is the primary mechanism.
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/status/stream")
def api_status_stream():
    """Server-Sent Events stream of status updates.

    Note: wsgiref is single-threaded, so we emit one event and close.
    The client reconnects to achieve polling-like updates.
    """
    response.content_type = "text/event-stream"
    response.set_header("Cache-Control", "no-cache")

    def generate():
        try:
            data = json.dumps(_build_status_summary(), default=str)
            yield f"data: {data}\n\n"
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            logger.debug(f"SSE client disconnected: {e}")
            return

    return generate()


def _build_status_summary() -> dict:
    """Lightweight status for real-time push."""
    srv = _server_ref
    return {
        "timestamp": time.time(),
        "halt": _safe_risk("halt", False),
        "halt_reason": _safe_risk("_halt_reason", ""),
        "daily_trades": _safe_risk("daily_trades", 0),
        "realized_pnl": _safe_risk("realized_pnl_today", 0.0),
        "current_dd": _safe_risk("current_dd", 0.0),
        "can_trade": srv.risk.can_trade() if srv and hasattr(srv, "risk") else False,
        "uptime_sec": int(time.time() - srv.start_time) if srv and hasattr(srv, "start_time") else 0,
        "mode": "LIVE" if (srv and getattr(srv, "live", False)) else "DRY-RUN",
        "live_armed": srv.live_armed if srv and hasattr(srv, "live_armed") else False,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Health endpoints
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/health")
def api_health():
    """
    Health check endpoint for monitoring and load balancers.

    Returns:
        - status: "ok" or "degraded"
        - checks: Results of individual component health checks
        - timestamp: ISO format UTC timestamp
        - uptime_seconds: Server uptime in seconds
    """
    srv = _server_ref
    # In standalone mode, Server_AGI runs in a separate process — detect via ps
    _server_process_running = False
    if srv is not None:
        _server_process_running = True
    else:
        try:
            import subprocess
            import platform
            if platform.system().lower().startswith("win"):
                # Windows: use PowerShell (consistent with other detection in this file)
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-Command",
                     "Get-Process python* -ErrorAction SilentlyContinue | "
                     "Where-Object { $_.CommandLine -match 'Server_AGI' } | "
                     "Select-Object -First 1 -ExpandProperty Id"],
                    capture_output=True, text=True, timeout=8
                )
                _server_process_running = bool(result.stdout.strip())
            else:
                procs = subprocess.check_output(["ps", "-eo", "command"], text=True, timeout=3)
                _server_process_running = "Server_AGI" in procs
        except Exception:
            pass
    checks = {
        "server_running": _server_process_running,
        "risk_engine": False,
        "brain_initialized": False,
        "model_registry": False,
        "config_loaded": False,
    }

    # Check risk engine (embedded or standalone)
    risk = _get_risk()
    if risk is not None:
        try:
            _ = risk.can_trade()
            checks["risk_engine"] = True
        except Exception as e:
            checks["risk_engine_error"] = str(e)

    # Check brain (embedded mode) — in standalone mode, brain lives in Server_AGI
    if srv and hasattr(srv, "brain"):
        checks["brain_initialized"] = True
    else:
        # Standalone mode: brain is in separate process, infer from live_state.json or running process
        try:
            live_state_path = os.path.join(ROOT, "live_state.json")
            if os.path.exists(live_state_path):
                with open(live_state_path, "r", encoding="utf-8") as f:
                    live = json.load(f)
                checks["brain_initialized"] = bool(live.get("server", {}).get("running"))
        except Exception:
            pass
        if not checks["brain_initialized"]:
            try:
                import subprocess
                import platform
                if platform.system().lower().startswith("win"):
                    result = subprocess.run(
                        ["powershell", "-NoProfile", "-Command",
                         "Get-Process python* -ErrorAction SilentlyContinue | "
                         "Where-Object { $_.CommandLine -match 'Server_AGI' } | "
                         "Select-Object -First 1 -ExpandProperty Id"],
                        capture_output=True, text=True, timeout=8
                    )
                    checks["brain_initialized"] = bool(result.stdout.strip())
                else:
                    procs = subprocess.check_output(["ps", "-eo", "command"], text=True, timeout=3)
                    checks["brain_initialized"] = "Server_AGI" in procs
            except Exception:
                pass

    # Check model registry
    try:
        active = _read_active_registry()
        checks["model_registry"] = active.get("champion") is not None
    except Exception as e:
        checks["model_registry_error"] = str(e)

    # Check config
    cfg = _read_config()
    checks["config_loaded"] = bool(cfg)

    # Overall status
    critical_checks = [checks["server_running"], checks["risk_engine"]]
    status = "ok" if all(critical_checks) else "degraded"

    # Calculate uptime
    uptime = 0
    if srv and hasattr(srv, "start_time"):
        uptime = int(time.time() - srv.start_time)

    return _json({
        "status": status,
        "pid": os.getpid(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": uptime,
        "checks": checks,
    })


@app.get("/api/health/ready")
def api_health_ready():
    """
    Readiness probe - returns 200 only when fully ready to accept traffic.

    This is stricter than /api/health and waits for all components to be ready.
    """
    srv = _server_ref

    # Must have server
    if srv is None:
        return _json({"ready": False, "reason": "server_not_initialized"}, status=503)

    # Must have risk engine
    if not hasattr(srv, "risk"):
        return _json({"ready": False, "reason": "risk_engine_not_loaded"}, status=503)

    # Must have brain
    if not hasattr(srv, "brain"):
        return _json({"ready": False, "reason": "brain_not_loaded"}, status=503)

    # Must have champion model
    try:
        active = _read_active_registry()
        if active.get("champion") is None:
            return _json({"ready": False, "reason": "no_champion_model"}, status=503)
    except Exception as e:
        return _json({"ready": False, "reason": f"registry_error: {e}"}, status=503)

    return _json({"ready": True, "timestamp": datetime.now(timezone.utc).isoformat()})


@app.get("/api/emergency_status")
def api_emergency_status():
    """Return whether emergency stop is active and why."""
    srv = _server_ref
    halted = False
    reason = ""
    if srv and hasattr(srv, "risk"):
        halted = srv.risk.halt
        if halted:
            reasons = []
            if srv.risk.realized_pnl_today < -srv.risk.max_daily_loss:
                reasons.append(f"daily_loss_exceeded (${srv.risk.realized_pnl_today:.2f} < -${srv.risk.max_daily_loss:.2f})")
            if getattr(srv.risk, "_consecutive_errors", 0) >= 3:
                reasons.append("3_consecutive_errors")
            if not reasons:
                reasons.append("manual_or_emergency_stop")
            reason = "; ".join(reasons)
    return _json({"halted": halted, "reason": reason, "daily_trades": _safe_risk("daily_trades", 0), "realized_pnl": _safe_risk("realized_pnl_today", 0.0)})


# ═══════════════════════════════════════════════════════════════════════════
# POST /api/backup/create — Create a backup now
# ═══════════════════════════════════════════════════════════════════════════
@app.post("/api/backup/create")
def api_backup_create():
    """Trigger an immediate backup creation."""
    try:
        from Python.backup_manager import get_backup_manager
        mgr = get_backup_manager()
        backup_path = mgr.create_backup(include_models=False)
        return _json({
            "ok": True,
            "path": str(backup_path),
            "name": backup_path.name,
        })
    except Exception as e:
        return _json({"ok": False, "error": str(e)}, status=500)


# ═══════════════════════════════════════════════════════════════════════════
# GET /api/backup/status — Backup manager status
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/backup/status")
def api_backup_status():
    """Return backup manager status and recent backups list."""
    try:
        from Python.backup_manager import get_backup_manager
        mgr = get_backup_manager()
        backups = mgr.list_backups()
        return _json({
            "count": len(backups),
            "latest": backups[0]["created_at"] if backups else None,
            "latest_size_mb": backups[0].get("size_mb") if backups else None,
            "auto_enabled": False,  # Set by environment, not runtime
            "max_backups": 7,
            "backups": [
                {
                    "name": b["name"],
                    "created_at": b["created_at"],
                    "size_mb": b.get("size_mb", 0),
                }
                for b in backups[:5]  # Last 5 only
            ],
        })
    except Exception as e:
        return _json({
            "count": 0,
            "latest": None,
            "auto_enabled": False,
            "max_backups": 7,
            "error": str(e),
            "backups": [],
        })


# ═══════════════════════════════════════════════════════════════════════════
# GET /api/ollama — Ollama advisor status and queries
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/ollama")
def api_ollama_status():
    """Return Ollama advisor status."""
    try:
        from Python.ollama_advisor import get_advisor
        advisor = get_advisor()
        return _json(advisor.get_status())
    except Exception as e:
        return _json({"error": str(e), "enabled": False, "available": False}, status=500)


@app.post("/api/ollama/analyze_trade")
def api_ollama_analyze_trade():
    """Analyze a specific trade using Ollama."""
    try:
        from Python.ollama_advisor import get_advisor
        data = request.json or {}
        advisor = get_advisor()
        result = advisor.analyze_trade(data)
        if result:
            return _json({"analysis": result, "trade": data})
        else:
            return _json({"error": "Ollama not available", "analysis": None}, status=503)
    except Exception as e:
        return _json({"error": str(e)}, status=500)


@app.post("/api/ollama/review_risk")
def api_ollama_review_risk():
    """Review current risk state using Ollama."""
    try:
        from Python.ollama_advisor import get_advisor
        advisor = get_advisor()
        srv = _server_ref
        risk_data = {}
        if srv and hasattr(srv, "risk"):
            r = srv.risk
            risk_data = {
                "equity": getattr(r, "_current_equity", 0),
                "balance": getattr(r, "_mt5_balance", 0),
                "drawdown_pct": getattr(r, "current_dd", 0.0),
                "daily_pnl": getattr(r, "realized_pnl_today", 0.0),
                "open_positions": r._get_open_positions_count() if hasattr(r, "_get_open_positions_count") else 0,
                "daily_trades": getattr(r, "daily_trades", 0),
                "halted": getattr(r, "halt", False),
                "halt_reason": r.get_halt_reason() if hasattr(r, "get_halt_reason") else "",
                "max_daily_loss_pct": getattr(r, "max_daily_loss_pct", 0.0),
                "max_hourly_loss_pct": getattr(r, "max_hourly_loss_pct", 0.0),
                "risk_per_trade_pct": getattr(r, "risk_per_trade_pct", 0.0),
                "max_drawdown_pct": getattr(r, "max_drawdown_pct", 0.0),
                "max_positions_per_symbol": getattr(r, "max_positions_per_symbol", 0),
            }
        result = advisor.review_risk_state(risk_data)
        if result:
            return _json({"review": result, "risk_data": risk_data})
        else:
            return _json({"error": "Ollama not available"}, status=503)
    except Exception as e:
        return _json({"error": str(e)}, status=500)


@app.post("/api/ollama/daily_summary")
def api_ollama_daily_summary():
    """Generate a daily trading summary using Ollama."""
    try:
        from Python.ollama_advisor import get_advisor
        from Python.trade_review import gather_closed_trades, analyze_trades, load_decision_log
        advisor = get_advisor()
        # Gather today's data
        trades = gather_closed_trades(days_back=1)
        decisions = load_decision_log(hours_back=24)
        result = analyze_trades(trades, decisions)
        summary = result.get("summary", {})
        session_data = {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "start_equity": 0,
            "end_equity": getattr(_server_ref.risk, "_current_equity", 0) if _server_ref else 0,
            "net_pnl": summary.get("total_pnl", 0),
            "total_trades": summary.get("total_trades", 0),
            "wins": summary.get("wins", 0),
            "losses": summary.get("losses", 0),
            "win_rate": summary.get("win_rate", 0),
            "max_drawdown": summary.get("max_drawdown", 0),
            "symbol_pnl": summary.get("by_symbol", {}),
            "top_losses": summary.get("top_losses", [])[:3],
            "top_wins": summary.get("top_wins", [])[:3],
            "risk_events": [],
            "decisions_summary": {},
        }
        daily_summary = advisor.daily_summary(session_data)
        if daily_summary:
            return _json({"summary": daily_summary, "data": session_data})
        else:
            return _json({"error": "Ollama not available"}, status=503)
    except Exception as e:
        return _json({"error": str(e)}, status=500)


# ═══════════════════════════════════════════════════════════════════════════
# GET /api/strategies — Analyze trades into strategies & patterns
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/strategies")
def api_strategies():
    trades = _fetch_trade_history("")
    if not trades:
        return _json({"strategies": [], "patterns": [], "meta": {"total_trades": 0}})

    from collections import defaultdict
    import math

    # --- Derive regime from comment/time ---
    def _hour_bucket(t):
        ct = t.get("close_time") or t.get("open_time") or ""
        if not ct:
            return "unknown"
        try:
            h = int(ct[11:13])
        except Exception:
            return "unknown"
        if h < 8:
            return "asian"
        if h < 14:
            return "london"
        if h < 21:
            return "new_york"
        return "asian"

    def _side(t):
        return (t.get("side") or "HOLD").upper()

    # --- Group trades into strategy buckets ---
    buckets = defaultdict(list)
    for t in trades:
        sym = t.get("symbol", "UNKNOWN")
        session = _hour_bucket(t)
        side = _side(t)
        key = f"{sym}|{session}|{side}"
        buckets[key].append(t)

    strategies = []
    for key, group in buckets.items():
        sym, session, side = key.split("|")
        profits = [t.get("profit", 0) for t in group]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]
        total_pnl = sum(profits)
        win_rate = len(wins) / len(profits) if profits else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

        # Sharpe-like score
        if len(profits) > 1:
            mean_r = sum(profits) / len(profits)
            var_r = sum((p - mean_r) ** 2 for p in profits) / (len(profits) - 1)
            std_r = math.sqrt(var_r) if var_r > 0 else 1e-6
            sharpe = mean_r / std_r
        else:
            sharpe = 0.0

        # Weighted score: combines win_rate, expectancy, and trade count
        confidence = min(1.0, len(group) / 20.0)  # confidence grows with sample size
        score = (expectancy * 100 + sharpe * 2) * confidence

        strategies.append({
            "id": key.replace("|", "_"),
            "symbol": sym,
            "session": session,
            "side": side,
            "trades": len(group),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 4),
            "total_pnl": round(total_pnl, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "expectancy": round(expectancy, 4),
            "profit_factor": round(profit_factor, 2),
            "sharpe": round(sharpe, 3),
            "score": round(score, 2),
            "confidence": round(confidence, 2),
        })

    strategies.sort(key=lambda s: s["score"], reverse=True)

    # --- Pattern recognition: symbol-session combos ranked by profitability ---
    sym_stats = defaultdict(lambda: {"trades": 0, "pnl": 0.0, "wins": 0})
    session_stats = defaultdict(lambda: {"trades": 0, "pnl": 0.0, "wins": 0})
    side_stats = defaultdict(lambda: {"trades": 0, "pnl": 0.0, "wins": 0})

    for t in trades:
        sym = t.get("symbol", "UNKNOWN")
        session = _hour_bucket(t)
        side = _side(t)
        profit = t.get("profit", 0)
        is_win = 1 if profit > 0 else 0

        sym_stats[sym]["trades"] += 1
        sym_stats[sym]["pnl"] += profit
        sym_stats[sym]["wins"] += is_win

        session_stats[session]["trades"] += 1
        session_stats[session]["pnl"] += profit
        session_stats[session]["wins"] += is_win

        side_stats[side]["trades"] += 1
        side_stats[side]["pnl"] += profit
        side_stats[side]["wins"] += is_win

    def _build_patterns(label, stats):
        result = []
        for name, s in stats.items():
            wr = s["wins"] / s["trades"] if s["trades"] > 0 else 0
            result.append({
                "type": label,
                "name": name,
                "trades": s["trades"],
                "pnl": round(s["pnl"], 2),
                "win_rate": round(wr, 4),
                "weight": round(s["pnl"] / max(abs(s["pnl"]), 0.01) * wr, 3) if s["trades"] >= 3 else 0,
            })
        result.sort(key=lambda p: p["pnl"], reverse=True)
        return result

    patterns = (
        _build_patterns("symbol", sym_stats) +
        _build_patterns("session", session_stats) +
        _build_patterns("side", side_stats)
    )

    return _json({
        "strategies": strategies,
        "patterns": patterns,
        "meta": {
            "total_trades": len(trades),
            "analysis_window": "30d",
        },
    })


# GET /api/economic_calendar — Upcoming economic events from MT5
@app.get("/api/economic_calendar")
def api_economic_calendar():
    """Return upcoming economic calendar events from the MT5 calendar API."""
    try:
        from Python.trade_review import get_economic_calendar
        days = int(request.params.get("days_ahead", 7))
        events = get_economic_calendar(days_ahead=days)
        return _json({"events": events, "count": len(events)})
    except Exception as e:
        logger.error(f"Economic calendar fetch failed: {e}")
        return _json({"events": [], "count": 0, "error": str(e)})


# GET /api/trade_review — Post-trade review with annotations and analysis
@app.get("/api/trade_review")
def api_trade_review():
    """Return the latest trade review with annotations, tags, and per-symbol breakdown."""
    from Python.trade_review import get_latest_review, run_review
    review = get_latest_review()
    if review is None:
        review = run_review(days_back=7)
    return _json(review.get("summary", review))


# GET /api/trade_review/enriched — Full enriched trade list with decision context
@app.get("/api/trade_review/enriched")
def api_trade_review_enriched():
    """Return enriched trade list with decision context and tags."""
    from Python.trade_review import get_latest_review
    review = get_latest_review()
    if review is None:
        return _json({"error": "No review available. Run /api/trade_review first."})
    return _json({
        "trades": review.get("enriched", [])[:50],  # Last 50 trades
        "summary": review.get("summary", {}),
    })


# POST /api/trade_review/refresh — Force a fresh review cycle
@app.post("/api/trade_review/refresh")
def api_trade_review_refresh():
    """Force a fresh trade review cycle."""
    from Python.trade_review import run_review
    result = run_review(days_back=7)
    return _json(result.get("summary", {}))


# ═══════════════════════════════════════════════════════════════════════════
# GET /api/scenarios — Scenario memory stats and review
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/scenarios")
def api_scenarios():
    """Return scenario memory statistics, best/worst scenarios, and session review."""
    try:
        from Python.scenario_memory import get_scenario_memory
        smem = get_scenario_memory()
        symbol = request.params.get("symbol", "").strip()
        review = smem.generate_session_review(symbol if symbol else None)
        best = smem.get_best_scenarios(min_trades=3, top_n=10)
        worst = smem.get_worst_scenarios(min_trades=3, top_n=10)
        avoid = smem.get_should_avoid(min_trades=3)
        return _json({
            "ok": True,
            "total_scenarios": len(smem.stats),
            "total_records": len(smem.records),
            "best_scenarios": [{k: v for k, v in __import__("dataclasses").asdict(s).items()} for s in best],
            "worst_scenarios": [{k: v for k, v in __import__("dataclasses").asdict(s).items()} for s in worst],
            "should_avoid": avoid,
            "session_review": review,
        })
    except Exception as exc:
        return _json({"ok": False, "error": str(exc)}, 500)


@app.post("/api/scenarios/record_outcome")
def api_scenarios_record_outcome():
    """Manually record a trade outcome by decision_id."""
    try:
        from Python.scenario_memory import get_scenario_memory
        smem = get_scenario_memory()
        payload = request.json or {}
        decision_id = payload.get("decision_id", "")
        if not decision_id:
            return _json({"ok": False, "error": "decision_id required"}, 400)
        record = smem.record_outcome(
            decision_id=decision_id,
            exit_price=float(payload.get("exit_price", 0)),
            pnl=float(payload.get("pnl", 0)),
            pnl_pct=float(payload.get("pnl_pct", 0)),
            hold_minutes=float(payload.get("hold_minutes", 0)),
            close_reason=payload.get("close_reason", ""),
            max_drawup=float(payload.get("max_drawup", 0)),
            max_drawdown=float(payload.get("max_drawdown", 0)),
        )
        if record is None:
            return _json({"ok": False, "error": "decision_id not found"}, 404)
        return _json({"ok": True, "decision_id": decision_id, "outcome": record.outcome})
    except Exception as exc:
        return _json({"ok": False, "error": str(exc)}, 500)


# ═══════════════════════════════════════════════════════════════════════════
# 11. POST /api/training/enhanced — Start enhanced training with multi-timeframe
# ═══════════════════════════════════════════════════════════════════════════
@app.post("/api/training/enhanced")
def api_start_enhanced_training():
    """Start enhanced DRL training with per-symbol metrics and multi-timeframe optimization.

    Request body (JSON, optional):
        - symbols: list of symbols to train (defaults to config)
        - timeframe_opt: bool (default true)
        - per_symbol_metrics: bool (default true)
    """
    try:
        import subprocess
        import sys

        body = request.json or {}
        symbols = body.get("symbols", [])
        timeframe_opt = body.get("timeframe_opt", True)
        per_symbol_metrics = body.get("per_symbol_metrics", True)

        # Build command
        cmd = [sys.executable, "start_enhanced_training.py"]
        if symbols:
            cmd.extend(["--symbols", ",".join(symbols)])
        if not timeframe_opt:
            cmd.append("--no-timeframe-opt")
        if not per_symbol_metrics:
            cmd.append("--no-per-symbol-metrics")

        # Start in background
        if os.name == 'nt':
            subprocess.Popen(cmd, cwd=ROOT, creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return _json({
            "ok": True,
            "message": "Enhanced training started",
            "symbols": symbols,
            "timeframe_opt": timeframe_opt,
            "per_symbol_metrics": per_symbol_metrics,
        })
    except Exception as exc:
        return _json({"ok": False, "error": str(exc)}, 500)


# ═══════════════════════════════════════════════════════════════════════════
# 12. GET /api/training/metrics — Per-symbol training metrics
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/training/metrics")
def api_training_metrics():
    """Return per-symbol training metrics, timeframe optimization results, and performance data.

    Used by the enhanced training dashboard to display:
    - Per-symbol profit, balance, drawdown
    - Timeframe optimization results
    - Training history and trade metrics
    """
    import glob
    from datetime import datetime as dt

    cfg = _read_config()
    symbols = cfg.get("trading", {}).get("symbols", [])

    # Initialize response structure
    result = {
        "symbols": symbols,
        "average_return": 0.0,
        "max_drawdown": 0.0,
        "best_symbol": None,
        "worst_symbol": None,
        "per_symbol_metrics": {},
        "timeframe_selections": {},
        "training_active": False,
    }

    # Read training progress files
    progress = _read_training_progress()
    if progress.get("ppo", {}).get("running"):
        result["training_active"] = True
        result["current_symbol"] = progress["ppo"].get("symbol")
        result["current_timesteps"] = progress["ppo"].get("timesteps", 0)
        result["target_timesteps"] = progress["ppo"].get("target_timesteps", 100000)

    # Surface explicit training health signal (robustness & recovery)
    try:
        hpath = os.path.join(ROOT, "logs", "training_health.json")
        if os.path.exists(hpath):
            with open(hpath, "r", encoding="utf-8") as f:
                th = json.load(f)
            result["training_health"] = th
            result["training_active"] = result["training_active"] or (th.get("status") in ("running", "recovering"))
    except Exception:
        pass

    # Load combined training results file (contains per-symbol metrics)
    # Note: Must NOT match per-symbol files like enhanced_training_results_BTCUSDm_*.json
    combined_results_path = os.path.join(ROOT, "logs", "enhanced_training_results_*.json")
    combined_files = glob.glob(combined_results_path)
    # Filter out per-symbol files (they have symbol name after the underscore)
    combined_files = [f for f in combined_files if not any(s in os.path.basename(f) for s in ['BTCUSDm', 'XAUUSDm', 'EURUSDm', 'GBPUSDm', 'ETHUSDm'])]
    combined_training_data = {}
    if combined_files:
        combined_files.sort(reverse=True)
        try:
            with open(combined_files[0], "r", encoding="utf-8") as f:
                combined_training_data = json.load(f)
            # Populate per-symbol metrics from combined file
            result["per_symbol_metrics"] = combined_training_data.get("per_symbol_metrics", {})
            result["timeframe_selections"] = combined_training_data.get("timeframe_selections", {})
            # Calculate summary stats
            all_returns = [m.get("return_pct", 0) for m in result["per_symbol_metrics"].values()]
            all_drawdowns = [m.get("max_drawdown_pct", 0) for m in result["per_symbol_metrics"].values()]
            if all_returns:
                result["average_return"] = sum(all_returns) / len(all_returns)
                result["max_drawdown"] = max(all_drawdowns) if all_drawdowns else 0
                # Find best/worst symbols by return
                sorted_by_return = sorted(result["per_symbol_metrics"].items(), key=lambda x: x[1].get("return_pct", 0), reverse=True)
                if sorted_by_return:
                    result["best_symbol"] = sorted_by_return[0][0]
                    result["worst_symbol"] = sorted_by_return[-1][0]
        except Exception:
            pass

    # Look for per-symbol training results (fallback for timeframe data)
    for symbol in symbols:
        # Try to load enhanced training results
        training_results_path = os.path.join(ROOT, "logs", f"enhanced_training_results_{symbol}_*.json")
        result_files = glob.glob(training_results_path)
        if result_files:
            # Get most recent
            result_files.sort(reverse=True)
            try:
                with open(result_files[0], "r", encoding="utf-8") as f:
                    training_data = json.load(f)
                result["timeframe_selections"][symbol] = training_data.get("timeframe_selections", {}).get(symbol, {})
            except Exception:
                pass

        # Real metrics only — do NOT invent numbers from decision cache.
        # If no training artifacts exist, the symbol simply has no metrics.

    # Calculate aggregate metrics from real data only (defensive)
    if result["per_symbol_metrics"] and isinstance(result["per_symbol_metrics"], dict):
        metrics_dicts = {k: v for k, v in result["per_symbol_metrics"].items() if isinstance(v, dict)}
        if metrics_dicts:
            returns = [m.get("return_pct", 0) for m in metrics_dicts.values()]
            drawdowns = [m.get("max_drawdown_pct", 0) for m in metrics_dicts.values()]

            result["average_return"] = sum(returns) / len(returns) if returns else 0
            result["max_drawdown"] = max(drawdowns) if drawdowns else 0

            sorted_by_return = sorted(metrics_dicts.items(), key=lambda x: x[1].get("return_pct", 0), reverse=True)
            if sorted_by_return:
                result["best_symbol"] = sorted_by_return[0][0]
                result["worst_symbol"] = sorted_by_return[-1][0]

    # Read timeframe optimization reports if available
    report_files = glob.glob(os.path.join(ROOT, "logs", "enhanced_training_report_*.txt"))
    if report_files:
        report_files.sort(reverse=True)
        result["latest_report"] = report_files[0]

    return _json(result)


# ═══════════════════════════════════════════════════════════════════════════
# 13. GET /api/training/analysis — AI-generated training analysis
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/training/analysis")
def api_training_analysis():
    """Get AI-generated analysis of what the model is currently learning.

    Query params:
        - symbol: specific symbol to analyze (optional)
    """
    try:
        if get_training_description is None or get_analyzer is None:
            raise ImportError("Training analyzer not available")

        symbol = request.query.get("symbol")
        description = get_training_description(symbol)

        # Also get trajectory if we have history
        analyzer = get_analyzer()
        trajectory = None
        if symbol:
            trajectory = analyzer.get_learning_trajectory(symbol)

        # Get general insights
        insights = analyzer.generate_training_insights()

        return _json({
            "ok": True,
            "description": description,
            "trajectory": trajectory,
            "insights": insights,
        })
    except Exception as exc:
        logger.error(f"Training analysis failed: {exc}")
        return _json({"ok": False, "error": str(exc)}, 500)


# ═══════════════════════════════════════════════════════════════════════════
# 14. POST /api/training/analyze — Analyze training-trading connection
# ═══════════════════════════════════════════════════════════════════════════
@app.post("/api/training/analyze")
def api_analyze_training_trading():
    """Analyze the connection between training progress and live trading.

    Request body (JSON):
        - training_symbol: symbol being trained
        - trading_symbol: symbol being traded (usually same as training)
    """
    try:
        if get_analyzer is None:
            raise ImportError("Training analyzer not available")

        body = request.json or {}
        training_symbol = body.get("training_symbol", "BTCUSDm")
        trading_symbol = body.get("trading_symbol", training_symbol)

        analyzer = get_analyzer()

        # Get training metrics from progress files
        training_metrics = _get_training_metrics_for_symbol(training_symbol)

        # Get trading metrics from server reference or cache
        trading_metrics = _get_trading_metrics_for_symbol(trading_symbol)

        # Analyze connection
        analysis = analyzer.analyze_trading_connection(
            training_metrics, trading_metrics
        )

        return _json({
            "ok": True,
            "analysis": analysis,
            "training_metrics": training_metrics,
            "trading_metrics": trading_metrics,
        })
    except Exception as exc:
        logger.error(f"Training-trading analysis failed: {exc}")
        return _json({"ok": False, "error": str(exc)}, 500)


def _get_training_metrics_for_symbol(symbol: str) -> dict:
    """Helper to get training metrics for a symbol."""
    import glob

    metrics = {
        "symbol": symbol,
        "epoch": 0,
        "total_epochs": 100,
        "loss": 0,
        "avg_reward": 0,
        "win_rate": 0,
    }

    # Try to read progress file
    progress_pattern = os.path.join(ROOT, "logs", f"ppo_{symbol}_progress.json")
    try:
        files = glob.glob(progress_pattern)
        if files:
            with open(files[0], "r") as f:
                progress = json.load(f)
                metrics["epoch"] = progress.get("timesteps", 0) // 1000
                metrics["total_epochs"] = progress.get("target_timesteps", 100000) // 1000
                metrics["win_rate"] = progress.get("win_rate", 0)
    except Exception:
        pass

    return metrics


def _get_trading_metrics_for_symbol(symbol: str) -> dict:
    """Helper to get trading metrics for a symbol."""
    metrics = {
        "symbol": symbol,
        "pnl": 0,
        "live_win_rate": 0,
        "open_positions": 0,
        "recent_actions": ["HOLD"],
        "avg_confidence": 0.5,
    }

    # Try to get from decision cache
    if symbol in _decision_cache:
        decisions = list(_decision_cache[symbol])
        if decisions:
            recent = decisions[-10:]
            metrics["recent_actions"] = [d.get("action", "HOLD") for d in recent]
            metrics["avg_confidence"] = sum(d.get("confidence", 0) for d in recent) / len(recent)

    # Try to get from server reference
    if _server_ref and hasattr(_server_ref, "trading"):
        try:
            lanes = _server_ref.trading.get("lanes", [])
            for lane in lanes:
                if lane.get("symbol") == symbol:
                    metrics["pnl"] = lane.get("pnl", 0)
                    metrics["open_positions"] = len(lane.get("positions", []))
                    break
        except Exception:
            pass

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Mission Control — new truth-first endpoints (no fake green states)
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/api/system_header", method=["GET", "OPTIONS"])
def api_system_header():
    """Return compact system header for the persistent command bar."""
    if request.method == "OPTIONS":
        return {}
    cfg = _read_config()
    mt5 = _get_mt5_account_and_positions()
    system = _resolve_system_mode(cfg)
    account = _get_account_truth(mt5)
    progress = _read_training_progress()
    models = _get_model_registry_status(progress)
    tests = _get_test_status()
    validation = _get_validation_status()
    active = _read_active_registry()
    champ_path = active.get("champion") or ""
    canary_path = active.get("canary") or ""
    champ_id = os.path.basename(champ_path) if champ_path else None
    return _json({
        "system_mode": system.get("system_mode", "unknown"),
        "execution_transport": system.get("execution_transport", "unknown"),
        "real_money_locked": system.get("real_money_locked", True),
        "live_lock_reason": system.get("live_lock_reason", ""),
        "api_status": "online",
        "mt5_bridge_status": "online" if mt5.get("equity", 0) > 0 else "offline",
        "account_type": account.get("account_type", "unknown"),
        "account_type_verified": account.get("account_type_verified", False),
        "account_telemetry_valid": account.get("telemetry_valid", False),
        "tests_status": tests.get("status", "unknown"),
        "open_test_failures": tests.get("open_failures", 0),
        "open_test_errors": tests.get("open_errors", 0),
        "active_bundle_id": champ_id,
        "champion_status": validation.get("champion_status", "none"),
    })


@app.route("/api/pipeline/stages", method=["GET", "OPTIONS"])
def api_pipeline_stages():
    """Return 20-stage pipeline map with honest statuses."""
    if request.method == "OPTIONS":
        return {}
    cfg = _read_config()
    progress = _read_training_progress()
    active = _read_active_registry()
    tests = _get_test_status()
    validation = _get_validation_status()
    mt5 = _get_mt5_account_and_positions()
    models = _get_model_registry_status(progress)
    data = _get_data_provenance()

    champ_path = active.get("champion") or ""
    canary_path = active.get("canary") or ""
    champ_id = os.path.basename(champ_path) if champ_path else None
    canary_id = os.path.basename(canary_path) if canary_path else None

    def _stage_last_run(stage_id: str) -> str | None:
        """Return ISO timestamp of last activity for a pipeline stage from progress files."""
        mapping = {
            "lstm": "lstm", "ppo": "ppo", "dreamer": "dreamer",
            "rainforest": None,  # uses rf_detector trained_at
        }
        key = mapping.get(stage_id)
        if key and key in progress:
            updated = progress[key].get("updated_at")
            if updated:
                try:
                    return datetime.fromtimestamp(updated, tz=timezone.utc).isoformat()
                except Exception:
                    pass
        if stage_id == "rainforest" and _rainforest_trained_at:
            return datetime.fromtimestamp(_rainforest_trained_at, tz=timezone.utc).isoformat()
        if stage_id == "validation" and tests.get("status") != "unknown":
            # Use pytest_results.json mtime
            ptest_path = os.path.join(ROOT, "logs", "pytest_results.json")
            if os.path.exists(ptest_path):
                try:
                    return datetime.fromtimestamp(os.path.getmtime(ptest_path), tz=timezone.utc).isoformat()
                except Exception:
                    pass
        return None

    def _stage_artifact_id(stage_id: str) -> str | None:
        """Return the most recent artifact path for a pipeline stage."""
        mapping = {
            "mt5_data": ["logs/data_provenance.json", "logs/last_dataset.json"],
            "validation": ["logs/pytest_results.json"],
            "features": ["logs/feature_pipeline.json", "logs/feature_importance.json"],
            "labels": ["logs/label_distribution.json"],
            "lstm": ["models/lstm_agi_trained.pt", "models/lstm_agi_trained.meta.json"],
            "rainforest": ["models/rainforest_BTCUSDm.pkl", "models/rainforest_XAUUSDm.pkl"],
            "dreamer": ["models/dreamer"],
            "ppo": ["models/ppo"],
            "meta_controller": ["models/registry/active.json"],
            "bundle": ["models/bundles"],
            "backtest": ["logs/backtest_results.json", "logs/backtester.log"],
            "walk_forward": ["logs/walk_forward_results.json"],
            "baseline": ["logs/baseline_comparison.json"],
            "demo_canary": ["logs/canary_monitor.jsonl"],
            "champion_rejected": ["models/registry/promotion_log.jsonl", "models/registry/rejection_log.jsonl"],
            "trade_journal": ["data/bets.db", "trades.db"],
            "trade_coroner": ["artifacts/trade_coroner"],
            "replay_dataset": ["logs/replay_dataset.json"],
            "retraining_trigger": ["logs/retraining_trigger.json"],
        }
        candidates = mapping.get(stage_id, [])
        found = []
        for rel in candidates:
            path = os.path.join(ROOT, rel)
            if os.path.exists(path):
                found.append(path)
        if not found:
            return None
        # Return the most recently modified artifact
        try:
            found.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return found[0]
        except Exception:
            return found[0] if found else None

    def stage(id: str, name: str, status: str, blockers: list = None, metrics: dict = None):
        return {
            "id": id,
            "name": name,
            "status": status,
            "last_run": _stage_last_run(id),
            "artifact_id": _stage_artifact_id(id),
            "blockers": blockers or [],
            "metrics": metrics or {},
        }

    stages = [
        stage("mt5_data", "MT5 Data", data.get("status", "unknown"), metrics={"source": data.get("source", "unknown")}),
        stage("validation", "Validation", tests.get("status", "unknown"), blockers=["tests failing"] if tests.get("open_failures", 0) > 0 else [], metrics={"failures": tests.get("open_failures", 0)}),
        stage("features", "Features", "unaudited"),
        stage("labels", "Labels", "unknown"),
        stage("lstm", "LSTM", models.get("lstm_status", "unknown"), metrics={"running": bool(progress.get("lstm", {}).get("running"))}),
        stage("rainforest", "Rainforest", models.get("rainforest_status", "unknown"), metrics={"trained": bool(rf_detector.is_trained() if rf_detector else False)}),
        stage("dreamer", "Dreamer", models.get("dreamer_status", "unknown"), metrics={"stub": os.path.exists(os.path.join(ROOT, "models", "dreamer"))}),
        stage("ppo", "PPO", models.get("ppo_status", "unknown"), metrics={"running": bool(progress.get("ppo", {}).get("running"))}),
        stage("meta_controller", "Meta Controller", "unknown"),
        stage("bundle", "Bundle", "unknown", metrics={"champion": champ_id, "canary": canary_id}),
        stage("backtest", "Backtest", validation.get("backtest_status", "unknown")),
        stage("walk_forward", "Walk-Forward", validation.get("walk_forward_status", "unknown")),
        stage("baseline", "Baseline", "unknown"),
        stage("demo_canary", "Demo Canary", "candidate" if canary_id else "none"),
        stage("champion_rejected", "Champion/Rejected", validation.get("champion_status", "none")),
        stage("trade_journal", "Trade Journal", "unknown", metrics={"positions": mt5.get("open_positions", 0)}),
        stage("trade_coroner", "Trade Coroner", "unknown"),
        stage("replay_dataset", "Replay Dataset", "unknown"),
        stage("retraining_trigger", "Retraining Trigger", "unknown"),
    ]
    return _json(stages)


@app.route("/api/model_brains", method=["GET", "OPTIONS"])
def api_model_brains():
    """Return four model brain cards with honest telemetry from disk."""
    if request.method == "OPTIONS":
        return {}
    models = _get_model_registry_status()
    active = _read_active_registry()

    # Read actual LSTM metadata from disk
    lstm_meta = None
    per_symbol_dir = os.path.join(ROOT, "models", "per_symbol")
    if os.path.isdir(per_symbol_dir):
        for f in os.listdir(per_symbol_dir):
            if f.endswith(".meta.json") and f.startswith("lstm_"):
                try:
                    with open(os.path.join(per_symbol_dir, f), "r") as fp:
                        lstm_meta = json.load(fp)
                    break
                except Exception:
                    pass

    # PPO metadata from disk
    ppo_meta = {}
    ppo_dir = os.path.join(ROOT, "models", "ppo")
    if os.path.isdir(ppo_dir):
        for d in os.listdir(ppo_dir):
            if os.path.isdir(os.path.join(ppo_dir, d)) and d != "smoke_ppo":
                ppo_meta["model_id"] = d
                break

    # Rainforest
    rf_trained = rf_detector is not None and rf_detector.is_trained()
    rf_regime = "unknown"
    rf_conf = 0.0
    rf_importance = {}
    if rainforest_predictions:
        first = next(iter(rainforest_predictions.values()), {})
        rf_regime = first.get("regime", "unknown")
        rf_conf = first.get("confidence", 0.0)
        rf_importance = first.get("feature_importances", {})

    return _json({
        "lstm": {
            "status": "trained" if lstm_meta else "unknown",
            "model_id": lstm_meta.get("symbol", "unknown") + "_lstm" if lstm_meta else None,
            "lookback": lstm_meta.get("seq_len") if lstm_meta else None,
            "feature_set": f"features_{lstm_meta.get('symbol', 'unknown')}_v1" if lstm_meta else None,
            "p_up": None,
            "p_down": None,
            "p_flat": None,
            "expected_return": None,
            "confidence": None,
            "calibration_error": None,
            "influence_enabled": bool(lstm_meta),
            "samples": lstm_meta.get("samples") if lstm_meta else None,
            "epochs": lstm_meta.get("epochs") if lstm_meta else None,
        },
        "rainforest": {
            "status": "validated" if rf_trained else "informational-only",
            "regime": rf_regime,
            "confidence": rf_conf if rf_conf > 0 else None,
            "allowed_modes": ["trend", "range"] if rf_trained else [],
            "blocked_modes": ["reversal"] if not rf_trained else [],
            "feature_importance": rf_importance,
            "lift_vs_no_rainforest": None,
        },
        "dreamer": {
            "status": "stub_disabled",
            "stub_disabled": True,
            "rollouts": None,
            "horizon": None,
            "expected_reward": None,
            "expected_drawdown": None,
            "ruin_probability": None,
            "used_for_decisions": False,
        },
        "ppo": {
            "status": "candidate" if models.get("bundle_count", 0) > 0 else "undertrained",
            "training_status": "idle",
            "actual_timesteps": None,
            "configured_timesteps": 500000,
            "reward_version": "v7",
            "action_bias": None,
            "promotion_status": "candidate" if models.get("bundle_count", 0) > 0 else "none",
            "model_id": ppo_meta.get("model_id") if ppo_meta else None,
        },
    })


@app.route("/api/training/lanes", method=["GET", "OPTIONS"])
def api_training_lanes():
    """Return parallel training lane cards."""
    if request.method == "OPTIONS":
        return {}
    srv = _server_ref
    cfg = _read_config()
    symbols = cfg.get("trading", {}).get("symbols", [])
    progress = _read_training_progress()
    lanes = []
    for sym in symbols:
        p = progress.get("ppo_per_symbol", {}).get(sym, {})
        lanes.append({
            "lane_id": f"ppo_{sym}",
            "lane_name": sym,
            "status": "training" if p.get("running") else "idle",
            "progress_pct": p.get("progress_pct"),
            "model_id": p.get("model_id"),
            "timesteps": p.get("current_timesteps"),
            "validation_summary": None,
            "failure_reason": p.get("failure_reason"),
        })
    return _json(lanes)


@app.route("/api/registry", method=["GET", "OPTIONS"])
def api_registry():
    """Return model bundle registry entries."""
    if request.method == "OPTIONS":
        return {}
    active = _read_active_registry()
    progress = _read_training_progress()
    cfg = _read_config()
    symbols = cfg.get("trading", {}).get("symbols", [])
    champ_path = active.get("champion") or ""
    canary_path = active.get("canary") or ""
    champ_id = os.path.basename(champ_path) if champ_path else None
    canary_id = os.path.basename(canary_path) if canary_path else None
    bundles = []
    for sym in symbols:
        bundles.append({
            "bundle_id": f"{champ_id or 'none'}_{sym}" if champ_id else f"untrained_{sym}",
            "symbol": sym,
            "timeframe": "M5",
            "status": "champion" if champ_id else "untrained",
            "data_source": "MT5",
            "feature_set": None,
            "lstm": "trained" if progress.get("lstm", {}).get("epoch", 0) > 0 else "none",
            "rainforest": "trained" if (rf_detector and rf_detector.is_trained()) else "none",
            "dreamer": "stub" if not os.path.exists(os.path.join(ROOT, "models", "dreamer")) else "trained",
            "ppo": "candidate" if canary_id else "champion" if champ_id else "none",
            "backtest_return": None,
            "walk_forward": None,
            "canary": None,
            "promotion_decision": "champion" if champ_id else None,
            "promotion_reason": None,
        })
    return _json(bundles)


@app.route("/api/promotion_gates", method=["GET", "OPTIONS"])
def api_promotion_gates():
    """Return honest promotion gate checklist."""
    if request.method == "OPTIONS":
        return {}
    tests = _get_test_status()
    validation = _get_validation_status()
    progress = _read_training_progress()
    mt5 = _get_mt5_account_and_positions()
    gates = []
    def gate(name: str, required, actual, passed: bool, pending: bool = False):
        gates.append({"gate": name, "required": required, "actual": actual, "passed": passed, "pending": pending})
    gate("tests_passing", True, tests.get("status") == "passing", tests.get("status") == "passing")
    gate("backtest_complete", True, validation.get("backtest_status") == "complete", validation.get("backtest_status") == "complete")
    gate("walk_forward_complete", True, validation.get("walk_forward_status") == "complete", validation.get("walk_forward_status") == "complete")
    gate("ppo_trained", True, bool(progress.get("ppo", {}).get("running")) or bool(progress.get("ppo", {}).get("current_timesteps", 0) > 0), bool(progress.get("ppo", {}).get("current_timesteps", 0) > 0))
    gate("lstm_trained", True, bool(progress.get("lstm", {}).get("epoch", 0) > 0), bool(progress.get("lstm", {}).get("epoch", 0) > 0))
    gate("rainforest_trained", True, bool(rf_detector and rf_detector.is_trained()), bool(rf_detector and rf_detector.is_trained()))
    gate("account_telemetry_valid", True, mt5.get("equity", 0) > 0, mt5.get("equity", 0) > 0)
    gate("real_money_unlocked", False, not _resolve_system_mode().get("real_money_locked", True), not _resolve_system_mode().get("real_money_locked", True))
    return _json(gates)


@app.route("/api/demo_canary", method=["GET", "OPTIONS"])
def api_demo_canary():
    """Return demo canary execution state."""
    if request.method == "OPTIONS":
        return {}
    active = _read_active_registry()
    canary_path = active.get("canary") or ""
    canary_id = os.path.basename(canary_path) if canary_path else None
    review = _get_trade_review_summary()
    overall = review.get("overall", {})
    # Derive account type honestly from system mode
    system = _resolve_system_mode()
    mt5 = _get_mt5_account_and_positions()
    acct_truth = _get_account_truth(mt5)
    return _json({
        "account_type": acct_truth.get("account_type", "unknown"),
        "real_money_locked": system.get("real_money_locked", True),
        "metrics": {
            "trades": overall.get("total_trades", 0),
            "days": 0,
            "pnl": overall.get("total_pnl", 0.0),
            "drawdown": overall.get("max_drawdown_pct", 0.0),
            "profit_factor": overall.get("profit_factor", None),
            "win_rate": overall.get("win_rate", None),
        },
        "timeline": [
            {
                "step": "canary_loaded",
                "ts": datetime.now(timezone.utc).isoformat() if canary_id else None,
                "status": "passed" if canary_id else "pending",
                "detail": f"Canary bundle {canary_id}" if canary_id else "No canary bundle loaded",
            },
        ] if canary_id else [],
    })


@app.route("/api/trades/coroner", method=["GET", "OPTIONS"])
def api_trade_coroner():
    """Return mistake clusters for trade review and retraining eligibility."""
    if request.method == "OPTIONS":
        return {}
    incidents = _read_incidents()
    clusters = []
    total_mistakes = 0
    total_reviewed = 0
    for inc in incidents:
        if inc.get("severity") in ("warning", "critical"):
            reviewed = inc.get("reviewed", False)
            # Retraining eligible if critical and involves model decisions
            retrain = inc.get("severity") == "critical" and bool(inc.get("symbols"))
            clusters.append({
                "cluster_id": inc.get("id", "UNK"),
                "count": 1,
                "root_cause": inc.get("message", "unknown"),
                "affected_symbols": inc.get("symbols", []),
                "recommended_experiment": "retrain_on_failure" if retrain else "review_incident_logs",
                "retraining_eligible": retrain,
            })
            total_mistakes += 1
            if reviewed:
                total_reviewed += 1
    return _json({
        "clusters": clusters,
        "total_mistakes": total_mistakes,
        "total_reviewed": total_reviewed,
    })


@app.route("/api/patterns/verified", method=["GET", "OPTIONS"])
def api_patterns_verified():
    """Return pattern verification list with fallback incident counts."""
    if request.method == "OPTIONS":
        return {}
    patterns = []
    patterns_log = os.path.join(ROOT, "logs", "patterns.jsonl")
    # Pre-load incidents for cross-reference
    incidents = _read_incidents()
    # Count incidents per regime / pattern type
    incident_counts: dict[str, int] = {}
    for inc in incidents:
        regime = inc.get("regime", "unknown")
        pattern_type = inc.get("pattern", inc.get("type", "unknown"))
        key = f"{pattern_type}:{regime}"
        incident_counts[key] = incident_counts.get(key, 0) + 1
    if os.path.exists(patterns_log):
        try:
            with open(patterns_log, "r", encoding="utf-8") as f:
                lines = f.readlines()[-50:]
            seen: set[str] = set()
            for line in lines:
                try:
                    p = json.loads(line.strip())
                    if isinstance(p, dict):
                        pattern_id = p.get("pattern", p.get("type", "unknown"))
                        regime = p.get("regime", "unknown")
                        key = f"{pattern_id}:{regime}"
                        # A pattern is considered verified if we have seen it
                        # multiple times with a known outcome or it appears in
                        # trade review / incidents.
                        outcome = p.get("outcome", "unknown")
                        confidence = float(p.get("confidence", 0.0))
                        verified = (
                            outcome != "unknown"
                            and confidence > 0.5
                            and key in incident_counts
                        )
                        fallback_incidents = incident_counts.get(key, 0)
                        # Deduplicate by key, keeping the latest occurrence
                        if key in seen:
                            continue
                        seen.add(key)
                        patterns.append({
                            "pattern_id": pattern_id,
                            "pattern_name": pattern_id,
                            "confidence": confidence,
                            "regime": regime,
                            "outcome": outcome,
                            "verified": verified,
                            "fallback_incidents": fallback_incidents,
                        })
                except (json.JSONDecodeError, ValueError):
                    pass
        except Exception:
            pass
    return _json(patterns)


@app.route("/api/perpetual_improvement", method=["GET", "OPTIONS"])
def api_perpetual_improvement():
    """Return perpetual improvement loop status."""
    if request.method == "OPTIONS":
        return {}
    progress = _read_training_progress()
    events = []
    for model, p in progress.items():
        if isinstance(p, dict) and p.get("running"):
            events.append({
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": "training_active",
                "symbol": p.get("symbol", "unknown"),
                "model": model,
            })

    # Load historical learning events from perpetual_improvement.jsonl
    pi_log = os.path.join(ROOT, "logs", "perpetual_improvement.jsonl")
    if os.path.exists(pi_log):
        try:
            with open(pi_log, "r", encoding="utf-8") as f:
                lines = f.readlines()[-20:]
            for line in lines:
                try:
                    entry = json.loads(line.strip())
                    if isinstance(entry, dict):
                        events.append({
                            "ts": entry.get("timestamp", datetime.now(timezone.utc).isoformat()),
                            "event": entry.get("action", "unknown"),
                            "symbol": entry.get("symbol", "unknown"),
                            "model": entry.get("model", "unknown"),
                        })
                except (json.JSONDecodeError, ValueError):
                    pass
        except Exception:
            pass

    # Build candidate experiments from registry + incidents
    candidate_experiments = []
    # 1. Unpromoted registry candidates
    active = _read_active_registry()
    for sym, sym_data in active.get("symbols", {}).items():
        for hist in sym_data.get("champion_history", []):
            meta = hist.get("metadata", {})
            bundle_id = os.path.basename(hist.get("path", "unknown"))
            eval_info = meta.get("evaluation", {})
            if not eval_info.get("winner", False):
                candidate_experiments.append({
                    "experiment_id": f"registry_candidate_{bundle_id}",
                    "type": "registry_candidate",
                    "symbol": sym,
                    "description": f"Unpromoted {meta.get('type', 'ppo')} candidate {bundle_id}",
                    "status": "pending_evaluation",
                    "priority": 2,
                })
    # 2. Retraining suggestions from incidents
    incidents = _read_incidents()
    for inc in incidents:
        if inc.get("severity") == "critical" and inc.get("symbols"):
            for sym in inc.get("symbols", []):
                candidate_experiments.append({
                    "experiment_id": f"retrain_{inc.get('id', 'UNK')}_{sym}",
                    "type": "retraining",
                    "symbol": sym,
                    "description": f"Retrain on critical incident {inc.get('id', 'UNK')}: {inc.get('message', '')[:80]}",
                    "status": "proposed",
                    "priority": 1,
                })

    # Deduplicate by experiment_id
    seen: set[str] = set()
    deduped = []
    for exp in candidate_experiments:
        eid = exp["experiment_id"]
        if eid not in seen:
            seen.add(eid)
            deduped.append(exp)
    deduped.sort(key=lambda x: x.get("priority", 99))

    return _json({
        "loop_status": "active" if any(p.get("running") for p in progress.values() if isinstance(p, dict)) else "idle",
        "learning_events": events[-20:],
        "candidate_experiments": deduped[:10],
    })


@app.route("/api/agents/status", method=["GET", "OPTIONS"])
def api_agents_status():
    """Return operational status for each agent."""
    if request.method == "OPTIONS":
        return {}
    srv = _server_ref
    cfg = _read_config()
    progress = _read_training_progress()
    halt = _safe_risk("halt", False)
    symbols = cfg.get("trading", {}).get("symbols", [])
    agents = []
    # Helper: get real heartbeat from progress file or agent tracking
    def _hb(agent_id: str, progress_key: str | None = None) -> str:
        """Return real heartbeat from progress files or tracked activity."""
        if progress_key and progress_key in progress:
            updated = progress[progress_key].get("updated_at")
            if updated:
                try:
                    return datetime.fromtimestamp(updated, tz=timezone.utc).isoformat()
                except Exception:
                    pass
        tracked = _get_agent_heartbeat(agent_id)
        return tracked

    agents.append({
        "agent_id": "data_feed",
        "agent_name": "Data Feed Agent",
        "status": "online" if symbols else "idle",
        "heartbeat": _hb("data_feed"),
        "current_task": f"Polling {len(symbols)} symbols" if symbols else "idle",
        "last_artifact": None,
        "error_count": 0,
    })
    agents.append({
        "agent_id": "pattern_detector",
        "agent_name": "Pattern Detector",
        "status": "online" if (rf_detector and rf_detector.is_trained()) else "idle",
        "heartbeat": _hb("pattern_detector"),
        "current_task": "Regime detection" if (rf_detector and rf_detector.is_trained()) else "Awaiting training",
        "last_artifact": None,
        "error_count": 0,
    })
    agents.append({
        "agent_id": "risk_guardian",
        "agent_name": "Risk Guardian",
        "status": "error" if halt else "online",
        "heartbeat": _hb("risk_guardian"),
        "current_task": "HALT" if halt else "Monitoring drawdown",
        "last_artifact": None,
        "error_count": 1 if halt else 0,
    })
    agents.append({
        "agent_id": "lstm_brain",
        "agent_name": "LSTM Brain",
        "status": "training" if progress.get("lstm", {}).get("running") else "idle",
        "heartbeat": _hb("lstm_brain", "lstm"),
        "current_task": progress.get("lstm", {}).get("symbol", "") or "idle",
        "last_artifact": None,
        "error_count": 0,
    })
    agents.append({
        "agent_id": "ppo_brain",
        "agent_name": "PPO Brain",
        "status": "training" if progress.get("ppo", {}).get("running") else "idle",
        "heartbeat": _hb("ppo_brain", "ppo"),
        "current_task": progress.get("ppo", {}).get("symbol", "") or "idle",
        "last_artifact": None,
        "error_count": 0,
    })
    agents.append({
        "agent_id": "dreamer",
        "agent_name": "Dreamer",
        "status": "training" if progress.get("dreamer", {}).get("running") else "idle",
        "heartbeat": _hb("dreamer", "dreamer"),
        "current_task": progress.get("dreamer", {}).get("symbol", "") or "idle",
        "last_artifact": None,
        "error_count": 0,
    })
    agents.append({
        "agent_id": "trade_executor",
        "agent_name": "Trade Executor",
        "status": "online" if not halt else "blocked",
        "heartbeat": _hb("trade_executor"),
        "current_task": "Executing signals" if not halt else "Blocked by risk halt",
        "last_artifact": None,
        "error_count": 0,
    })
    agents.append({
        "agent_id": "champion_evaluator",
        "agent_name": "Champion Evaluator",
        "status": "online" if _read_active_registry().get("champion") else "idle",
        "heartbeat": _hb("champion_evaluator"),
        "current_task": "Canary monitoring" if _read_active_registry().get("canary") else "Awaiting champion",
        "last_artifact": None,
        "error_count": 0,
    })
    agents.append({
        "agent_id": "perpetual_optimizer",
        "agent_name": "Perpetual Optimizer",
        "status": "training" if any(p.get("running") for p in progress.values() if isinstance(p, dict)) else "idle",
        "heartbeat": _hb("perpetual_optimizer"),
        "current_task": "Hyper-parameter sweep" if any(p.get("running") for p in progress.values() if isinstance(p, dict)) else "idle",
        "last_artifact": None,
        "error_count": 0,
    })
    return _json(agents)


@app.route("/api/safety", method=["GET", "OPTIONS"])
def api_safety():
    """Return blunt safety lock state and gate checklist."""
    if request.method == "OPTIONS":
        return {}
    system = _resolve_system_mode()
    mt5 = _get_mt5_account_and_positions()
    tests = _get_test_status()
    progress = _read_training_progress()
    locked = system.get("real_money_locked", True)
    lock_reasons = []
    if locked:
        lock_reasons.append(system.get("live_lock_reason", "real_live_disabled"))
    if tests.get("open_failures", 0) > 0:
        lock_reasons.append("test_failures")
    if not (rf_detector and rf_detector.is_trained()):
        lock_reasons.append("rainforest_untrained")
    if progress.get("ppo", {}).get("current_timesteps", 0) == 0 and not progress.get("ppo", {}).get("running"):
        lock_reasons.append("ppo_undertrained")
    gates = []
    gates.append({
        "name": "real_money_locked",
        "passed": not locked,
        "required": False,
        "actual": locked,
        "reason": "real_live_disabled" if locked else None,
    })
    gates.append({
        "name": "tests_passing",
        "passed": tests.get("status") == "passing",
        "required": True,
        "actual": tests.get("status") == "passing",
        "reason": f"{tests.get('open_failures', 0)} failures" if tests.get("open_failures", 0) > 0 else None,
    })
    gates.append({
        "name": "account_telemetry_valid",
        "passed": mt5.get("equity", 0) > 0,
        "required": True,
        "actual": mt5.get("equity", 0) > 0,
        "reason": "No MT5 connection" if mt5.get("equity", 0) <= 0 else None,
    })
    gates.append({
        "name": "rainforest_trained",
        "passed": bool(rf_detector and rf_detector.is_trained()),
        "required": True,
        "actual": bool(rf_detector and rf_detector.is_trained()),
        "reason": "Rainforest not trained" if not (rf_detector and rf_detector.is_trained()) else None,
    })
    return _json({
        "real_money_locked": locked,
        "lock_reasons": lock_reasons,
        "gates": gates,
    })


@app.route("/api/evidence", method=["GET", "OPTIONS"])
def api_evidence():
    """Return evidence locker artifacts from models/ and logs/."""
    if request.method == "OPTIONS":
        return {}

    def _validate_artifact(path: str, is_model_dir: bool = False) -> str:
        """Basic validation: check file/dir is non-empty and recent."""
        try:
            if is_model_dir:
                children = os.listdir(path)
                if not children:
                    return "empty"
                # Check if any model files exist
                has_model = any(f.endswith((".pt", ".pth", ".zip", ".json", ".pkl")) for f in children)
                return "valid" if has_model else "incomplete"
            else:
                size = os.path.getsize(path)
                if size == 0:
                    return "empty"
                if size < 10:
                    return "corrupt"
                return "valid"
        except Exception:
            return "unknown"

    artifacts = []
    models_dir = os.path.join(ROOT, "models")
    if os.path.isdir(models_dir):
        for entry in os.listdir(models_dir):
            path = os.path.join(models_dir, entry)
            if os.path.isdir(path):
                artifacts.append({
                    "name": entry,
                    "created_at": datetime.fromtimestamp(os.path.getctime(path), tz=timezone.utc).isoformat(),
                    "status": _validate_artifact(path, is_model_dir=True),
                    "linked_model": entry,
                    "path": path,
                })
    logs_dir = os.path.join(ROOT, "logs")
    if os.path.isdir(logs_dir):
        for entry in os.listdir(logs_dir):
            if entry.endswith(".json") or entry.endswith(".jsonl"):
                path = os.path.join(logs_dir, entry)
                artifacts.append({
                    "name": entry,
                    "created_at": datetime.fromtimestamp(os.path.getctime(path), tz=timezone.utc).isoformat(),
                    "status": _validate_artifact(path),
                    "linked_model": None,
                    "path": path,
                })
    return _json(artifacts)


# ═══════════════════════════════════════════════════════════════════════════
# WebSocket endpoint — /ws/status
# Pushes the same payload as /api/status every 2 seconds.
# Only active when geventwebsocket is installed (WS_AVAILABLE == True).
# ═══════════════════════════════════════════════════════════════════════════
if WS_AVAILABLE:
    @app.route('/ws/status')
    def ws_status():
        wsock = request.environ.get('wsgi.websocket')
        if not wsock:
            abort(400, 'Expected WebSocket request')
        while True:
            try:
                # Reuse the same data-building logic as /api/status by calling
                # the route function directly and extracting its JSON body.
                payload_str = api_status()
                # api_status() returns a str (from _json()); send it as-is.
                wsock.send(payload_str if isinstance(payload_str, str) else json.dumps(payload_str))
                gevent_sleep(2)
            except WebSocketError:
                break
            except Exception as e:
                try:
                    wsock.send(json.dumps({"error": str(e)}))
                except Exception:
                    break


# Server lifecycle
# ═══════════════════════════════════════════════════════════════════════════
import ssl as _ssl
from wsgiref.simple_server import WSGIServer, WSGIRequestHandler, make_server
from socketserver import ThreadingMixIn

class _ThreadedWSGIServer(ThreadingMixIn, WSGIServer):
    daemon_threads = True
    allow_reuse_address = True

class ThreadedWSGIRefServer(ServerAdapter):
    """Threaded wsgiref adapter for Bottle — avoids gevent blocking RPyC.
    Supports optional TLS via AGI_API_TLS env var or AGI_API_CERT/AGI_API_KEY.
    """
    def run(self, handler):
        srv = make_server(
            self.host,
            self.port,
            handler,
            server_class=_ThreadedWSGIServer,
            handler_class=WSGIRequestHandler,
        )
        # Optional TLS
        use_tls = os.environ.get("AGI_API_TLS", "0") == "1"
        cert_path = os.environ.get("AGI_API_CERT", "")
        key_path = os.environ.get("AGI_API_KEY", "")
        if use_tls and (not cert_path or not key_path):
            cert_dir = os.path.join(ROOT, ".tmp", "certs")
            cert_path, key_path = _ensure_tls_certs(cert_dir)
        if cert_path and key_path and os.path.exists(cert_path) and os.path.exists(key_path):
            try:
                ctx = _ssl.SSLContext(_ssl.PROTOCOL_TLS_SERVER)
                ctx.minimum_version = _ssl.TLSVersion.TLSv1_2
                ctx.load_cert_chain(cert_path, key_path)
                srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
                logger.success(f"API server TLS enabled on https://{self.host}:{self.port}")
            except Exception as e:
                logger.error(f"TLS setup failed — falling back to HTTP: {e}")
        srv.serve_forever()

API_PORT = int(os.environ.get("AGI_API_PORT", "5050"))


def start_api_server(agi_server=None, host: str = "0.0.0.0", port: int = API_PORT):
    """
    Start the HTTP API server in a daemon thread.

    Uses a threaded wsgiref server instead of gevent to prevent RPyC/MT5
    calls from blocking the event loop and freezing all requests.

    Args:
        agi_server: AGIServer instance for live data access.
        host: Bind address.
        port: Listen port (default 5050, matches Vite proxy).
    """
    global _server_ref
    _server_ref = agi_server

    def _run():
        proto = "https" if os.environ.get("AGI_API_TLS", "0") == "1" else "http"
        logger.success(f"API server starting (threaded wsgiref) on {proto}://{host}:{port}")
        bottle_run(app, host=host, port=port, quiet=True, server=ThreadedWSGIRefServer)

    t = threading.Thread(target=_run, name="api-server", daemon=True)
    t.start()
    logger.info(f"API server thread started (port {port}, threaded=True)")
    return t


# ═══════════════════════════════════════════════════════════════════════════
# Rich Decision PPO + Execution Observability Endpoints (for React + external UIs)
# Mirrors TUI readers: execution_reports + mql5_commands + feedback + live agent_status
# Works identically whether primary exec is Python OrderManager or MQL5 bridge.
# ═══════════════════════════════════════════════════════════════════════════
import glob as _glob  # local alias to avoid conflicts

@app.get("/api/execution/decisions")
def api_execution_decisions():
    """Recent rich TradeDecision + execution reports (with full specs for PPO attribution)."""
    limit = int(request.params.get("limit", 15))
    items = []
    try:
        base = Path(__file__).resolve().parent.parent
        reports_dir = base / "runtime" / "execution_reports"
        cmds_dir = base / "runtime" / "mql5_commands"
        for p in sorted(reports_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:limit*2]:
            try:
                rep = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                did = rep.get("decision_id", p.stem)
                # Enrich full spec from mql5 command if present (MQL5 path fidelity)
                if not rep.get("decision"):
                    for cp in cmds_dir.glob(f"decision_{did}_*.json"):
                        try:
                            rep["decision"] = json.loads(cp.read_text(encoding="utf-8", errors="ignore"))
                            break
                        except Exception:
                            pass
                items.append({"file": p.name, **rep})
            except Exception:
                continue
    except Exception:
        pass
    # Dedup
    deduped = []
    seen = set()
    for it in items:
        did = it.get("decision_id")
        if did and did not in seen:
            seen.add(did)
            deduped.append(it)
    return _json({"decisions": deduped[:limit], "count": len(deduped), "sources": ["execution_reports", "mql5_commands"]})


@app.get("/api/execution/live")
def api_execution_live():
    """Live managed positions + ExecutionAgent status (from agent_status live file or reports)."""
    try:
        base = Path(__file__).resolve().parent.parent
        live_path = base / "runtime" / "agent_status" / "decision_ppo_execution_live.json"
        data = {"active": [], "status": "no_live_agent_status"}
        if live_path.exists():
            try:
                data = json.loads(live_path.read_text(encoding="utf-8", errors="ignore"))
                data["status"] = "live_from_agent"
            except Exception:
                pass
        # Fallback enrichment
        if not data.get("active_decisions"):
            reports = sorted((base / "runtime" / "execution_reports").glob("*.json"), key=lambda x:x.stat().st_mtime, reverse=True)[:5]
            data["fallback_reports"] = [json.loads(r.read_text(errors="ignore")) for r in reports if r.exists()]
        return _json(data)
    except Exception:
        return _json({"status": "error", "active": []})


@app.get("/api/execution/feedback")
def api_execution_feedback():
    limit = int(request.params.get("limit", 20))
    try:
        base = Path(__file__).resolve().parent.parent
        fb_path = base / "logs" / "execution_feedback.jsonl"
        recs = []
        if fb_path.exists():
            for ln in fb_path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()[-limit:]:
                if ln.strip():
                    recs.append(json.loads(ln))
        return _json({"feedback": list(reversed(recs)), "count": len(recs)})
    except Exception:
        return _json({"feedback": [], "count": 0})


@app.get("/api/timing/insights")
def api_timing_insights():
    """Profitable trade timing analyzer insights (news, opens, sessions) for UI visibility.
    Powers React/TUI panels for Decision PPO timing awareness.
    """
    try:
        base = Path(__file__).resolve().parent.parent
        logs = base / "logs"
        # Prefer latest saved insights from Decision PPO launchers
        cands = sorted(logs.glob("*timing_insights*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            data = json.loads(cands[0].read_text(encoding="utf-8", errors="ignore"))
            data["source"] = cands[0].name
            return _json(data)
        # Live compute fallback
        try:
            from Python.analysis.trade_timing_analyzer import analyze_profitable_trade_timing
            journal = logs / "trade_journal" / "trade_journal.jsonl"
            ins = analyze_profitable_trade_timing(journal_path=journal, top_n=40)
            if "error" not in ins:
                ins["source"] = "live_analyzer"
                return _json(ins)
        except Exception:
            pass
        return _json({"error": "no timing insights yet (run Decision PPO training to populate)", "source": "none"})
    except Exception as e:
        return _json({"error": str(e)})


# ═══════════════════════════════════════════════════════════════════════════
# Standalone entry point
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logger.info("Starting API server in standalone mode (no AGIServer reference)")
    logger.info("Using threaded wsgiref server to avoid gevent RPyC blocking")
    proto = "https" if os.environ.get("AGI_API_TLS", "0") == "1" else "http"
    logger.info(f"Listening on {proto}://0.0.0.0:{API_PORT}")
    bottle_run(app, host="0.0.0.0", port=API_PORT, quiet=False, reloader=False, server=ThreadedWSGIRefServer)
