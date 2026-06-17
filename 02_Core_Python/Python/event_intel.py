import datetime as dt
import json
import os
import threading
from collections import defaultdict

import requests
from loguru import logger

try:
    import aiohttp
except Exception:
    aiohttp = None


def _utc_now():
    return dt.datetime.now(dt.timezone.utc)


def _parse_iso(ts):
    try:
        v = dt.datetime.fromisoformat(str(ts))
        if v.tzinfo is None:
            v = v.replace(tzinfo=dt.timezone.utc)
        return v.astimezone(dt.timezone.utc)
    except Exception:
        return None


def _impact_rank(v):
    s = str(v or "").strip().lower()
    if s in {"high", "3", "h"}:
        return 3
    if s in {"medium", "med", "2", "m"}:
        return 2
    if s in {"low", "1", "l"}:
        return 1
    return 0


def _impact_name(rank):
    return {3: "high", 2: "medium", 1: "low"}.get(int(rank), "none")


class EventIntel:
    """Observe-only calendar/news intelligence. No trade behavior changes."""

    def __init__(self, cfg: dict, log_dir: str):
        ecfg = (cfg or {}).get("event_intel", {}) or {}
        self.enabled = bool(ecfg.get("enabled", True))
        self.calendar_url = str(ecfg.get("calendar_url", "") or "").strip()
        self.news_url = str(ecfg.get("news_url", "") or "").strip()
        self.websocket_url = str(ecfg.get("websocket_url", "") or "").strip()
        self.poll_sec = int(ecfg.get("poll_sec", 60))
        self.pre_event_min = int(ecfg.get("pre_event_min", 60))
        self.post_event_min = int(ecfg.get("post_event_min", 30))
        self.news_hold_min = int(ecfg.get("news_hold_min", 20))
        self.max_events = int(ecfg.get("max_events", 200))
        self.log_events = bool(ecfg.get("log_events", True))

        self._last_poll_ts = 0.0
        self._events = []
        self._last_regime = {}
        self._pending_alerts = []
        self._ws_queue = []
        self._lock = threading.Lock()

        self._state_path = os.path.join(log_dir, "event_intel_state.json")
        self._jsonl_path = os.path.join(log_dir, "event_intel_events.jsonl")
        self._log_dir = log_dir

        if self.enabled and self.websocket_url:
            self._start_ws_listener()

    def _start_ws_listener(self):
        if aiohttp is None:
            logger.warning("event_intel websocket configured but aiohttp missing; websocket feed disabled")
            return

        def _runner():
            import asyncio

            async def _loop():
                while True:
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.ws_connect(self.websocket_url, heartbeat=20, timeout=20) as ws:
                                logger.info(f"event_intel websocket connected: {self.websocket_url}")
                                async for msg in ws:
                                    if msg.type == aiohttp.WSMsgType.TEXT:
                                        try:
                                            payload = json.loads(msg.data)
                                            with self._lock:
                                                if isinstance(payload, list):
                                                    self._ws_queue.extend(payload)
                                                else:
                                                    self._ws_queue.append(payload)
                                        except Exception:
                                            continue
                                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                        break
                    except Exception as e:
                        logger.warning(f"event_intel websocket reconnect in 5s: {e}")
                        await asyncio.sleep(5)

            asyncio.run(_loop())

        t = threading.Thread(target=_runner, daemon=True)
        t.start()

    def _fetch_json(self, url: str):
        if not url:
            return []
        try:
            r = requests.get(url, timeout=8)
            if not r.ok:
                return []
            data = r.json()
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                for k in ("events", "data", "items", "news"):
                    v = data.get(k)
                    if isinstance(v, list):
                        return v
                return [data]
        except Exception:
            return []
        return []

    def _to_event(self, raw: dict, source: str):
        if not isinstance(raw, dict):
            return None
        time_raw = raw.get("time_utc") or raw.get("ts") or raw.get("time") or raw.get("published_at")
        ts = _parse_iso(time_raw)
        if ts is None:
            # News item without explicit timestamp: treat as now.
            ts = _utc_now()
        currency = str(raw.get("currency") or raw.get("ccy") or "").upper()
        symbol = str(raw.get("symbol") or "").upper() or None
        impact_raw = raw.get("impact") or raw.get("importance") or raw.get("priority") or "low"
        impact = _impact_name(_impact_rank(impact_raw))
        headline = str(raw.get("headline") or raw.get("title") or raw.get("event") or "event")
        sentiment = raw.get("sentiment")
        try:
            sentiment = float(sentiment) if sentiment is not None else None
        except Exception:
            sentiment = None
        return {
            "source": source,
            "time_utc": ts.isoformat(),
            "currency": currency or None,
            "symbol": symbol,
            "impact": impact,
            "headline": headline[:180],
            "sentiment": sentiment,
        }

    def _rebuild_events(self):
        events = []
        # REST calendar/news
        for item in self._fetch_json(self.calendar_url):
            ev = self._to_event(item, "calendar")
            if ev:
                events.append(ev)
        for item in self._fetch_json(self.news_url):
            ev = self._to_event(item, "news")
            if ev:
                events.append(ev)
        # websocket queue
        with self._lock:
            ws_items = self._ws_queue
            self._ws_queue = []
        for item in ws_items:
            ev = self._to_event(item, "websocket")
            if ev:
                events.append(ev)

        # merge with previous short-memory to avoid empty feed gaps
        now = _utc_now()
        merged = []
        for e in self._events + events:
            ts = _parse_iso(e.get("time_utc"))
            if ts is None:
                continue
            age_min = (now - ts).total_seconds() / 60.0
            if age_min <= max(self.post_event_min, self.news_hold_min, 120):
                merged.append(e)

        # de-dup
        dedup = {}
        for e in merged:
            k = (
                str(e.get("source")),
                str(e.get("time_utc")),
                str(e.get("currency") or ""),
                str(e.get("symbol") or ""),
                str(e.get("headline") or ""),
            )
            dedup[k] = e
        out = list(dedup.values())
        out.sort(key=lambda x: x.get("time_utc", ""))
        self._events = out[-self.max_events :]

    def _symbol_matches(self, symbol: str, event: dict):
        sym = str(symbol or "").upper()
        esym = str(event.get("symbol") or "").upper()
        if esym and esym == sym:
            return True
        cc = str(event.get("currency") or "").upper()
        return bool(cc and cc in sym)

    def _build_state(self, symbols):
        now = _utc_now()
        symbols = [str(s).upper() for s in (symbols or [])]
        by_symbol = {s: {"regime": "normal", "impact": "none", "minutes_to_event": None, "headline": None} for s in symbols}
        upcoming = []
        active = []

        for e in self._events:
            ts = _parse_iso(e.get("time_utc"))
            if ts is None:
                continue
            mins = (ts - now).total_seconds() / 60.0
            is_news = str(e.get("source")) in {"news", "websocket"}
            if is_news and mins < 0:
                # News is treated as active for short hold window.
                active_window = abs(mins) <= self.news_hold_min
            else:
                active_window = (-self.post_event_min) <= mins <= self.pre_event_min

            if mins >= 0 and mins <= 24 * 60:
                upcoming.append({**e, "minutes_to_event": round(mins, 1)})
            if active_window:
                active.append({**e, "minutes_to_event": round(mins, 1)})

            rank = _impact_rank(e.get("impact"))
            for s in symbols:
                if not self._symbol_matches(s, e):
                    continue
                cur = by_symbol[s]
                cur_rank = _impact_rank(cur.get("impact"))
                if rank < cur_rank:
                    continue
                if mins >= 0:
                    regime = "pre_event"
                elif mins >= -5:
                    regime = "event_live"
                else:
                    regime = "post_event"
                cur["regime"] = regime if active_window else cur["regime"]
                cur["impact"] = e.get("impact", "none")
                cur["minutes_to_event"] = round(mins, 1)
                cur["headline"] = e.get("headline")

        high_upcoming = [e for e in upcoming if _impact_rank(e.get("impact")) >= 3]
        high_active = [e for e in active if _impact_rank(e.get("impact")) >= 3]

        state = {
            "updated_utc": now.isoformat(),
            "enabled": self.enabled,
            "sources": {
                "calendar_url": bool(self.calendar_url),
                "news_url": bool(self.news_url),
                "websocket_url": bool(self.websocket_url),
            },
            "summary": {
                "upcoming_24h": len(upcoming),
                "active_window": len(active),
                "high_upcoming_24h": len(high_upcoming),
                "high_active": len(high_active),
            },
            "upcoming": sorted(upcoming, key=lambda x: x["minutes_to_event"])[:20],
            "active": sorted(active, key=lambda x: abs(x["minutes_to_event"]))[:20],
            "by_symbol": by_symbol,
        }
        return state

    def _emit_regime_alerts(self, state):
        by_symbol = state.get("by_symbol", {})
        for sym, cur in by_symbol.items():
            prev = self._last_regime.get(sym, {"regime": "normal", "impact": "none"})
            if (cur.get("regime"), cur.get("impact")) != (prev.get("regime"), prev.get("impact")):
                if cur.get("regime") != "normal" and _impact_rank(cur.get("impact")) >= 2:
                    msg = (
                        f"NEWS/CALENDAR WINDOW {sym}\n"
                        f"Regime={cur.get('regime')} Impact={cur.get('impact')} "
                        f"mins={cur.get('minutes_to_event')} "
                        f"title={cur.get('headline') or '-'}"
                    )
                    self._pending_alerts.append(msg)
            self._last_regime[sym] = cur

    def pop_alerts(self):
        out = self._pending_alerts[:]
        self._pending_alerts = []
        return out

    def _persist(self, state):
        try:
            with open(self._state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=True, indent=2)
            if self.log_events:
                with open(self._jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"ts": _utc_now().isoformat(), "state": state["summary"]}, ensure_ascii=True) + "\n")
        except Exception:
            pass

    def tick(self, symbols):
        if not self.enabled:
            return {
                "enabled": False,
                "updated_utc": _utc_now().isoformat(),
                "summary": {"upcoming_24h": 0, "active_window": 0, "high_upcoming_24h": 0, "high_active": 0},
                "upcoming": [],
                "active": [],
                "by_symbol": {},
            }
        now = _utc_now().timestamp()
        if now - self._last_poll_ts >= max(10, self.poll_sec):
            self._rebuild_events()
            self._last_poll_ts = now
        state = self._build_state(symbols)
        self._emit_regime_alerts(state)
        self._persist(state)
        return state

