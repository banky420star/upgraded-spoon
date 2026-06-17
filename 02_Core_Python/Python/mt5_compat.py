from __future__ import annotations

import logging
import os


class _AttrDict(dict):
    """Dict that supports attribute access for backward-compat with MT5 NamedTuples."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


def _sanitize_account_info(raw):
    """Convert raw MT5 account_info to a plain _AttrDict with guaranteed fields.

    Logs a loud warning and sets ``_valid = False`` when balance or equity
    are missing or non-positive so callers can fall back to risk-engine
    estimates instead of broadcasting zeros.
    """
    logger = logging.getLogger("mt5_compat")

    if raw is None:
        logger.warning("[MT5 TELEMETRY] account_info returned None — telemetry invalid")
        return None

    # Convert netref / NamedTuple / dict to plain dict
    if isinstance(raw, dict):
        data = dict(raw)
    else:
        keys = (
            "login", "server", "currency", "balance", "equity", "margin",
            "margin_free", "leverage", "name", "profit", "company",
            "trade_allowed", "trade_expert",
        )
        data = {}
        for k in keys:
            try:
                data[k] = getattr(raw, k, None)
            except Exception:
                data[k] = None

    # Ensure required fields exist
    required = ("login", "server", "currency", "balance", "equity",
                "margin", "margin_free", "leverage", "name")
    for k in required:
        if k not in data:
            data[k] = None

    # Coerce numeric fields
    for k in ("balance", "equity", "margin", "margin_free", "leverage", "profit"):
        try:
            data[k] = float(data[k]) if data[k] is not None else None
        except Exception:
            data[k] = None

    # Validate
    balance = data.get("balance")
    equity = data.get("equity")
    if (balance is not None and balance <= 0) or (equity is not None and equity <= 0):
        logger.warning(
            "[MT5 TELEMETRY] Invalid account telemetry: balance=%s, equity=%s, "
            "login=%s, server=%s — marking as invalid",
            balance, equity, data.get("login"), data.get("server"),
        )
        data["_valid"] = False
    else:
        data["_valid"] = True

    return _AttrDict(data)


# ── try native MetaTrader5 first (Windows / Linux) ──
_WINE_BRIDGE_ERROR = None
_MT5_IMPORT_ERROR = None

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
    MT5_IMPORT_ERROR = None

    # Wrap so Wine and native paths both return sanitized _AttrDicts
    _native_account_info = mt5.account_info

    def _wrapped_native_account_info(*args, **kwargs):
        return _sanitize_account_info(_native_account_info(*args, **kwargs))

    mt5.account_info = _wrapped_native_account_info
except Exception as exc:
    _MT5_IMPORT_ERROR = exc
    MT5_AVAILABLE = False

    # ── fallback 1: Wine RPyC bridge (macOS) ──
    _WINE_HOST = os.environ.get("MT5_WINE_HOST", "127.0.0.1")
    _WINE_PORT = int(os.environ.get("MT5_WINE_PORT", "18812"))

    try:
        import rpyc
        _conn = rpyc.classic.connect(_WINE_HOST, _WINE_PORT)
        _conn._config["sync_request_timeout"] = 300
        _conn.execute("import MetaTrader5 as mt5")
        _conn.execute("import datetime")

        class _WineMT5Bridge:
            def __init__(self, conn):
                self._conn = conn
                self._host = _WINE_HOST
                self._port = _WINE_PORT

            def _serialize(self, value):
                """Serialize a value for remote eval; convert datetimes to ISO strings."""
                import datetime as _dt
                if isinstance(value, _dt.datetime):
                    return f"datetime.datetime.fromisoformat({repr(value.astimezone().isoformat())})"
                return repr(value)

            def _reconnect(self):
                """Reconnect to the RPyC bridge after a restart."""
                import rpyc as _rpyc
                self._conn = _rpyc.classic.connect(self._host, self._port)
                self._conn._config["sync_request_timeout"] = 300
                self._conn.execute("import MetaTrader5 as mt5")
                self._conn.execute("import datetime")

            def _call(self, remote_expr):
                """Evaluate a remote expression, reconnecting once on stale connection."""
                try:
                    return self._conn.eval(remote_expr)
                except (EOFError, ConnectionError, BrokenPipeError):
                    self._reconnect()
                    return self._conn.eval(remote_expr)

            def __getattr__(self, name):
                return self._call(f"mt5.{name}")

            def initialize(self, *args, **kwargs):
                # On Wine, passing path= forces process creation which often fails;
                # prefer no-arg initialize so we connect to an already-running terminal.
                if kwargs:
                    return self._call(f"mt5.initialize(*{args}, **{kwargs})")
                return self._call(f"mt5.initialize(*{args})")

            def login(self, *args, **kwargs):
                return self._call(f"mt5.login(*{args}, **{kwargs})")

            def shutdown(self, *args, **kwargs):
                return self._call(f"mt5.shutdown(*{args}, **{kwargs})")

            def version(self, *args, **kwargs):
                return self._call(f"mt5.version(*{args}, **{kwargs})")

            def last_error(self, *args, **kwargs):
                return self._call(f"mt5.last_error(*{args}, **{kwargs})")

            def _obtain(self, remote_expr):
                """Eval + obtain with automatic reconnect on stale connection."""
                try:
                    return rpyc.utils.classic.obtain(self._conn.eval(remote_expr))
                except (EOFError, ConnectionError, BrokenPipeError):
                    self._reconnect()
                    return rpyc.utils.classic.obtain(self._conn.eval(remote_expr))

            def account_info(self, *args, **kwargs):
                args_str = ", ".join(self._serialize(a) for a in args)
                kwargs_str = ", ".join(f"{k}={self._serialize(v)}" for k, v in kwargs.items())
                all_args = ", ".join(p for p in [args_str, kwargs_str] if p)
                remote_expr = (
                    "(lambda ai: None if ai is None else {"
                    "'login': getattr(ai, 'login', None),"
                    "'server': getattr(ai, 'server', None),"
                    "'currency': getattr(ai, 'currency', None),"
                    "'balance': getattr(ai, 'balance', None),"
                    "'equity': getattr(ai, 'equity', None),"
                    "'margin': getattr(ai, 'margin', None),"
                    "'margin_free': getattr(ai, 'margin_free', None),"
                    "'leverage': getattr(ai, 'leverage', None),"
                    "'name': getattr(ai, 'name', None),"
                    "'profit': getattr(ai, 'profit', None),"
                    "'company': getattr(ai, 'company', None),"
                    "'trade_allowed': getattr(ai, 'trade_allowed', None),"
                    "'trade_expert': getattr(ai, 'trade_expert', None),"
                    "})(mt5.account_info(" + all_args + "))"
                )
                raw = self._obtain(remote_expr)
                return _sanitize_account_info(raw)

            def terminal_info(self, *args, **kwargs):
                return self._call(f"mt5.terminal_info(*{args}, **{kwargs})")

            def symbols_total(self, *args, **kwargs):
                return self._call(f"mt5.symbols_total(*{args}, **{kwargs})")

            def symbols_get(self, *args, **kwargs):
                return self._call(f"mt5.symbols_get(*{args}, **{kwargs})")

            def symbol_info(self, *args, **kwargs):
                return self._call(f"mt5.symbol_info(*{args}, **{kwargs})")

            def symbol_info_tick(self, *args, **kwargs):
                return self._call(f"mt5.symbol_info_tick(*{args}, **{kwargs})")

            def symbol_select(self, *args, **kwargs):
                return self._call(f"mt5.symbol_select(*{args}, **{kwargs})")

            def copy_rates_from(self, symbol, timeframe, date_from, count):
                return self._obtain(
                    f"mt5.copy_rates_from({repr(symbol)}, {timeframe}, {self._serialize(date_from.astimezone())}, {count})"
                )

            def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
                return self._obtain(
                    f"mt5.copy_rates_from_pos({repr(symbol)}, {timeframe}, {start_pos}, {count})"
                )

            def copy_rates_range(self, symbol, timeframe, date_from, date_to):
                return self._obtain(
                    f"mt5.copy_rates_range({repr(symbol)}, {timeframe}, {self._serialize(date_from.astimezone())}, {self._serialize(date_to.astimezone())})"
                )

            def copy_ticks_from(self, symbol, date_from, count, flags):
                return self._obtain(
                    f"mt5.copy_ticks_from({repr(symbol)}, {self._serialize(date_from.astimezone())}, {count}, {flags})"
                )

            def copy_ticks_range(self, symbol, date_from, date_to, flags):
                return self._obtain(
                    f"mt5.copy_ticks_range({repr(symbol)}, {self._serialize(date_from.astimezone())}, {self._serialize(date_to.astimezone())}, {flags})"
                )

            def orders_total(self, *args, **kwargs):
                return self._call(f"mt5.orders_total(*{args}, **{kwargs})")

            def orders_get(self, *args, **kwargs):
                return self._call(f"mt5.orders_get(*{args}, **{kwargs})")

            def order_calc_margin(self, *args, **kwargs):
                return self._call(f"mt5.order_calc_margin(*{args}, **{kwargs})")

            def order_calc_profit(self, *args, **kwargs):
                return self._call(f"mt5.order_calc_profit(*{args}, **{kwargs})")

            def order_check(self, *args, **kwargs):
                return self._call(f"mt5.order_check(*{args}, **{kwargs})")

            def order_send(self, request):
                # Serialize request dict safely via JSON instead of raw f-string interpolation
                import json
                payload = json.dumps(
                    request,
                    default=lambda o: o.item() if hasattr(o, "item") else str(o),
                )
                return self._call(
                    f"mt5.order_send(__import__('json').loads({repr(payload)}))"
                )

            def positions_total(self, *args, **kwargs):
                return self._call(f"mt5.positions_total(*{args}, **{kwargs})")

            def positions_get(self, *args, **kwargs):
                return self._call(f"mt5.positions_get(*{args}, **{kwargs})")

            def history_orders_total(self, date_from, date_to):
                return self._call(f"mt5.history_orders_total({self._serialize(date_from.astimezone())}, {self._serialize(date_to.astimezone())})")

            def history_orders_get(self, *args, **kwargs):
                args_str = ", ".join(self._serialize(a) for a in args)
                kwargs_str = ", ".join(f"{k}={self._serialize(v)}" for k, v in kwargs.items())
                all_args = ", ".join(p for p in [args_str, kwargs_str] if p)
                return self._call(f"mt5.history_orders_get({all_args})")

            def history_deals_total(self, date_from, date_to):
                return self._call(f"mt5.history_deals_total({self._serialize(date_from.astimezone())}, {self._serialize(date_to.astimezone())})")

            def history_deals_get(self, *args, **kwargs):
                args_str = ", ".join(self._serialize(a) for a in args)
                kwargs_str = ", ".join(f"{k}={self._serialize(v)}" for k, v in kwargs.items())
                all_args = ", ".join(p for p in [args_str, kwargs_str] if p)
                return self._call(f"mt5.history_deals_get({all_args})")

        mt5 = _WineMT5Bridge(_conn)
        MT5_AVAILABLE = True
        MT5_IMPORT_ERROR = None
    except Exception as bridge_exc:
        _WINE_BRIDGE_ERROR = bridge_exc

        # ── fallback 2: stub (no MT5 at all) ──
        class _MissingMetaTrader5:
            ORDER_TYPE_BUY = 0
            ORDER_TYPE_SELL = 1
            ORDER_FILLING_FOK = 0
            ORDER_FILLING_IOC = 1
            ORDER_FILLING_RETURN = 2
            ORDER_TIME_GTC = 0
            TRADE_ACTION_DEAL = 1
            TRADE_ACTION_SLTP = 6
            TRADE_RETCODE_DONE = 10009
            TIMEFRAME_M1 = 1
            TIMEFRAME_M5 = 5
            TIMEFRAME_M15 = 15
            TIMEFRAME_M30 = 30
            TIMEFRAME_H1 = 60
            TIMEFRAME_H4 = 240
            TIMEFRAME_D1 = 1440

            class Tick:
                pass

            def initialize(self, *args, **kwargs):
                return False

            def last_error(self):
                return str(_MT5_IMPORT_ERROR or _WINE_BRIDGE_ERROR)

            def symbol_info(self, *args, **kwargs):
                return None

            def symbol_info_tick(self, *args, **kwargs):
                return None

            def symbol_select(self, *args, **kwargs):
                return False

            def copy_rates_from_pos(self, *args, **kwargs):
                return None

            def copy_rates_from(self, *args, **kwargs):
                return None

            def copy_rates_range(self, *args, **kwargs):
                return None

            def positions_get(self, *args, **kwargs):
                return []

            def order_send(self, *args, **kwargs):
                raise RuntimeError(
                    "MetaTrader5 is required for live trading and is unavailable in this environment."
                ) from (_MT5_IMPORT_ERROR or _WINE_BRIDGE_ERROR)

            def account_info(self, *args, **kwargs):
                return None

            def terminal_info(self, *args, **kwargs):
                return None

            def __getattr__(self, name):
                raise AttributeError(
                    f"MetaTrader5 has no attribute {name!r} (unavailable in this environment)"
                ) from (_MT5_IMPORT_ERROR or _WINE_BRIDGE_ERROR)

        mt5 = _MissingMetaTrader5()
