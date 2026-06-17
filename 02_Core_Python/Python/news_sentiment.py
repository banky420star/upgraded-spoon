"""
NewsSentimentEngine — Free-tier news/sentiment aggregation with per-symbol scoring.

Data sources:
- Finnhub free tier (60 calls/min): company/news + sentiment
- alternative.me Crypto Fear & Greed Index (free, no auth)
- ForexFactory economic calendar (scrape, free)
- MT5 built-in calendar (fallback)

Design principles:
- Never blocks trading. Sentiment is a signal modifier, not a gate.
- Score of 0.0 = no data = neutral = no effect.
- All results cached with TTL to respect API rate limits.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import requests
from loguru import logger

# ── Symbol → currency mapping ───────────────────────────────────────────
_SYMBOL_CURRENCIES = {
    "EURUSDm": ("EUR", "USD"),
    "GBPUSDm": ("GBP", "USD"),
    "XAUUSDm": ("XAU", "USD"),
    "BTCUSDm": ("BTC", "USD"),
    "USDJPYm": ("USD", "JPY"),
    "AUDUSDm": ("AUD", "USD"),
    "NZDUSDm": ("NZD", "USD"),
    "USDCADm": ("USD", "CAD"),
    "USDCHFm": ("USD", "CHF"),
}

# ── Finnhub symbol mapping ─────────────────────────────────────────────
_FINNHUB_MAP = {
    "EURUSDm": "EURUSD",
    "GBPUSDm": "GBPUSD",
    "XAUUSDm": "OANDA:XAU_USD",
    "BTCUSDm": "BINANCE:BTCUSDT",
}


def _utc_now():
    return dt.datetime.now(dt.timezone.utc)


class NewsSentimentEngine:
    """Free-tier news/sentiment aggregation with per-symbol scoring."""

    def __init__(self, config: dict):
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.finnhub_key = os.environ.get("FINNHUB_API_KEY", cfg.get("finnhub_api_key", ""))
        self.cache_ttl = int(cfg.get("cache_ttl_seconds", 300))
        self.sentiment_weight = float(cfg.get("sentiment_weight", 0.3))
        self.extreme_fear_threshold = int(cfg.get("extreme_fear_threshold", 20))
        self.extreme_greed_threshold = int(cfg.get("extreme_greed_threshold", 80))
        self.fear_reduction = float(cfg.get("fear_reduction", 0.5))
        self.greed_reduction = float(cfg.get("greed_reduction", 0.7))

        # Caches: key -> (value, timestamp)
        self._sentiment_cache: dict[str, tuple[float, float]] = {}
        self._fgi_cache: tuple[int, float] | None = None
        self._impact_cache: dict[str, tuple[dict, float]] = {}

        # Cache MT5 calendar API availability (resolved once at init)
        self._mt5_has_calendar = False
        try:
            from Python.mt5_compat import mt5 as _mt5
            self._mt5_has_calendar = hasattr(_mt5, "calendar_country")
        except ImportError:
            pass

        # Log directory for sentiment events
        self._log_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "logs"
        self._log_dir.mkdir(exist_ok=True)
        self._sentiment_log = self._log_dir / "sentiment_events.jsonl"

        logger.info(
            f"NewsSentimentEngine initialized: enabled={self.enabled}, "
            f"finnhub={'configured' if self.finnhub_key else 'not configured'}, "
            f"weight={self.sentiment_weight}"
        )

    # ── Public API ──────────────────────────────────────────────────────

    def get_symbol_sentiment(self, symbol: str) -> float:
        """Return sentiment score [-1.0, +1.0] for a symbol.

        Aggregates multiple sources. Returns 0.0 if no data (neutral).
        """
        if not self.enabled:
            return 0.0

        # Check cache
        cached = self._sentiment_cache.get(symbol)
        if cached and (time.time() - cached[1]) < self.cache_ttl:
            return cached[0]

        score = self._compute_symbol_sentiment(symbol)
        self._sentiment_cache[symbol] = (score, time.time())
        return score

    def get_economic_impact(self, symbol: str) -> dict:
        """Return upcoming event impact for a symbol.

        Returns: {
            "impact_score": float [0-1],
            "minutes_to_event": int,
            "event_name": str,
            "currency": str
        }
        """
        if not self.enabled:
            return {"impact_score": 0.0, "minutes_to_event": 999, "event_name": "", "currency": ""}

        cached = self._impact_cache.get(symbol)
        if cached and (time.time() - cached[1]) < self.cache_ttl:
            return cached[0]

        impact = self._compute_economic_impact(symbol)
        self._impact_cache[symbol] = (impact, time.time())
        return impact

    def get_fear_greed_index(self) -> int:
        """Return Fear & Greed Index (0=extreme fear, 100=extreme greed).

        Falls back to 50 (neutral) if unavailable.
        """
        if not self.enabled:
            return 50

        if self._fgi_cache and (time.time() - self._fgi_cache[1]) < self.cache_ttl:
            return self._fgi_cache[0]

        fgi = self._fetch_fear_greed()
        self._fgi_cache = (fgi, time.time())
        return fgi

    def compute_exposure_modifier(self, symbol: str) -> dict:
        """Compute the full sentiment-based exposure modifier.

        Returns: {
            "sentiment": float,          # [-1, 1]
            "fgi": int,                  # [0, 100]
            "event_impact": dict,        # economic event info
            "exposure_mult": float,      # multiplier to apply to exposure
            "reason": str                # human-readable explanation
        }
        """
        sentiment = self.get_symbol_sentiment(symbol)
        fgi = self.get_fear_greed_index()
        impact = self.get_economic_impact(symbol)

        # Start with base multiplier
        mult = 1.0
        reasons = []

        # Sentiment bias
        if abs(sentiment) > 0.01:
            sentiment_adj = 1 + self.sentiment_weight * sentiment
            mult *= sentiment_adj
            reasons.append(f"sentiment={sentiment:.2f}x{self.sentiment_weight}")

        # Fear & Greed adjustments
        if fgi < self.extreme_fear_threshold:
            mult *= self.fear_reduction
            reasons.append(f"extreme_fear(FGI={fgi})")
        elif fgi > self.extreme_greed_threshold:
            mult *= self.greed_reduction
            reasons.append(f"extreme_greed(FGI={fgi})")

        # Event impact: reduce exposure if high-impact event is within 30 min
        impact_score = impact.get("impact_score", 0.0)
        minutes_to = impact.get("minutes_to_event", 999)
        if impact_score > 0.5 and minutes_to < 30:
            event_mult = max(0.3, 1.0 - impact_score * 0.5)
            mult *= event_mult
            reasons.append(f"pre_event({impact.get('event_name','')[:20]} in {minutes_to}m)")

        # Log significant sentiment events
        if abs(sentiment) > 0.3 or fgi < self.extreme_fear_threshold or fgi > self.extreme_greed_threshold:
            self._log_sentiment_event(symbol, sentiment, fgi, impact, mult)

        return {
            "sentiment": round(sentiment, 4),
            "fgi": fgi,
            "event_impact": impact,
            "exposure_mult": round(mult, 4),
            "reason": "; ".join(reasons) if reasons else "neutral",
        }

    # ── Internal: Sentiment computation ─────────────────────────────────

    def _compute_symbol_sentiment(self, symbol: str) -> float:
        """Aggregate sentiment from all available sources."""
        scores = []

        # Source 1: Finnhub
        finnhub_score = self._fetch_finnhub(symbol)
        if finnhub_score is not None:
            scores.append(("finnhub", finnhub_score, 1.0))

        # Source 2: ForexFactory (for FX pairs)
        currencies = _SYMBOL_CURRENCIES.get(symbol, ())
        if currencies:
            ff_score = self._fetch_forexfactory(currencies)
            if ff_score is not None:
                scores.append(("forexfactory", ff_score, 0.5))

        # Source 3: FGI as macro bias
        fgi = self.get_fear_greed_index()
        fgi_score = (fgi - 50) / 50.0  # normalize to [-1, 1]
        # FGI is more relevant for crypto/risk assets
        if "BTC" in symbol or "XAU" in symbol:
            scores.append(("fgi_macro", fgi_score, 0.4))

        if not scores:
            return 0.0

        # Weighted average
        total_weight = sum(w for _, _, w in scores)
        weighted_sum = sum(score * w for _, score, w in scores)
        return max(-1.0, min(1.0, weighted_sum / total_weight))

    def _compute_economic_impact(self, symbol: str) -> dict:
        """Compute economic event impact for a symbol using MT5 calendar."""
        default = {"impact_score": 0.0, "minutes_to_event": 999, "event_name": "", "currency": ""}

        try:
            from Python.mt5_compat import mt5
            if not mt5.initialize():
                return default

            currencies = _SYMBOL_CURRENCIES.get(symbol, ())
            if not currencies:
                mt5.shutdown()
                return default

            now = _utc_now()
            best_event = default

            if not self._mt5_has_calendar:
                mt5.shutdown()
                return default
            try:
                countries = mt5.calendar_country()
            except AttributeError:
                mt5.shutdown()
                return default

            if not countries:
                mt5.shutdown()
                return default

            for country in countries:
                currency = getattr(country, "currency", "") or getattr(country, "code", "")
                if currency not in currencies:
                    continue

                try:
                    events = mt5.calendar_value_last_by_country(
                        country.code,
                        now - dt.timedelta(minutes=5),
                        now + dt.timedelta(hours=4),
                    )
                except (AttributeError, TypeError):
                    continue

                if not events:
                    continue

                for ev in events:
                    importance = getattr(ev, "importance", 0)
                    if importance < 2:  # only high-impact
                        continue
                    ev_time = getattr(ev, "time", None)
                    if ev_time is None:
                        continue
                    if isinstance(ev_time, (int, float)):
                        ev_time = dt.datetime.fromtimestamp(ev_time, tz=dt.timezone.utc)
                    minutes_to = max(0, (ev_time - now).total_seconds() / 60)

                    # Higher impact + closer = higher score
                    proximity_score = max(0, 1.0 - minutes_to / 240)  # fade over 4h
                    impact_score = proximity_score * (importance / 3.0)

                    if impact_score > best_event.get("impact_score", 0):
                        best_event = {
                            "impact_score": round(impact_score, 3),
                            "minutes_to_event": int(minutes_to),
                            "event_name": getattr(ev, "name", "unknown")[:40],
                            "currency": currency,
                        }

            mt5.shutdown()
            return best_event

        except Exception as e:
            logger.debug(f"Economic impact check failed for {symbol}: {e}")
            return default

    # ── Internal: Data sources ──────────────────────────────────────────

    def _fetch_finnhub(self, symbol: str) -> float | None:
        """Fetch sentiment from Finnhub news API (free tier)."""
        if not self.finnhub_key:
            return None

        try:
            finnhub_sym = _FINNHUB_MAP.get(symbol, symbol.replace("m", ""))
            url = f"https://finnhub.io/api/v1/company-news"
            params = {
                "symbol": finnhub_sym,
                "from": (_utc_now() - dt.timedelta(days=3)).strftime("%Y-%m-%d"),
                "to": _utc_now().strftime("%Y-%m-%d"),
                "token": self.finnhub_key,
            }
            resp = requests.get(url, params=params, timeout=5)
            if resp.status_code != 200:
                return None

            articles = resp.json()
            if not isinstance(articles, list) or not articles:
                return None

            # Average sentiment from recent articles (last 20)
            # Finnhub doesn't provide sentiment per article on free tier,
            # so we use a simple heuristic: recent news volume + category
            recent = articles[:20]
            score = 0.0
            for article in recent:
                headline = (article.get("headline", "") or "").lower()
                # Simple keyword-based sentiment
                positive_words = ["bullish", "rally", "gain", "surge", "rise", "growth", "beat", "upgrade", "strong", "optimistic"]
                negative_words = ["bearish", "drop", "fall", "decline", "loss", "cut", "risk", "fear", "weak", "crash", "recession"]
                for w in positive_words:
                    if w in headline:
                        score += 0.15
                        break
                for w in negative_words:
                    if w in headline:
                        score -= 0.15
                        break

            # Normalize
            if recent:
                score = max(-1.0, min(1.0, score / len(recent) * 3))

            return score

        except Exception as e:
            logger.debug(f"Finnhub sentiment fetch failed for {symbol}: {e}")
            return None

    def _fetch_forexfactory(self, currencies: tuple[str, str]) -> float | None:
        """Fetch sentiment from ForexFactory weekly calendar (scrape)."""
        try:
            url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
            resp = requests.get(url, timeout=5)
            if resp.status_code != 200:
                return None

            events = resp.json()
            if not isinstance(events, list):
                return None

            now = _utc_now()
            relevant_score = 0.0
            count = 0

            for event in events:
                currency = event.get("currency", "")
                if currency not in currencies:
                    continue

                impact = event.get("impact", "").lower()
                if impact not in ("high", "medium"):
                    continue

                # Parse event time
                date_str = event.get("date", "")
                try:
                    ev_time = dt.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S%z")
                except (ValueError, TypeError):
                    continue

                # Only care about events in the next 4 hours
                minutes_to = (ev_time - now).total_seconds() / 60
                if minutes_to < -30 or minutes_to > 240:
                    continue

                # Forecast vs previous: positive surprise = positive sentiment
                forecast = event.get("forecast", "")
                previous = event.get("previous", "")
                try:
                    f_val = float(str(forecast).replace("%", "").replace("K", "").strip())
                    p_val = float(str(previous).replace("%", "").replace("K", "").strip())
                    surprise = (f_val - p_val) / max(abs(p_val), 0.01)
                    weight = 1.0 if impact == "high" else 0.5
                    relevance = max(0, 1.0 - minutes_to / 240) if minutes_to > 0 else 0.5
                    relevant_score += surprise * weight * relevance
                    count += 1
                except (ValueError, TypeError):
                    # Can't parse numbers, just note the event exists
                    count += 0.5

            if count == 0:
                return None

            return max(-1.0, min(1.0, relevant_score / max(count, 1)))

        except Exception as e:
            logger.debug(f"ForexFactory sentiment fetch failed: {e}")
            return None

    def _fetch_fear_greed(self) -> int:
        """Fetch Fear & Greed Index from alternative.me (free, no auth)."""
        try:
            url = "https://api.alternative.me/fng/?limit=1"
            resp = requests.get(url, timeout=5)
            if resp.status_code != 200:
                return 50

            data = resp.json()
            value = int(data.get("data", [{}])[0].get("value", 50))
            return max(0, min(100, value))

        except Exception as e:
            logger.debug(f"Fear & Greed fetch failed: {e}")
            return 50

    # ── Internal: Logging ────────────────────────────────────────────────

    def _log_sentiment_event(self, symbol, sentiment, fgi, impact, mult):
        """Log significant sentiment events for audit trail."""
        try:
            entry = {
                "timestamp": _utc_now().isoformat(),
                "symbol": symbol,
                "sentiment": sentiment,
                "fgi": fgi,
                "impact_score": impact.get("impact_score", 0),
                "exposure_mult": mult,
            }
            with open(self._sentiment_log, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.debug(f"Failed to write sentiment log: {e}")