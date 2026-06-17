"""
FinBERT financial sentiment feature extractor.

Analyzes crypto news headlines and returns a sentiment score (-1 to +1)
that can be used as an extra observation feature in TradingEnv.

Gated by env var: AGI_USE_SENTIMENT=1
"""

import os
import logging
from datetime import datetime

import numpy as np
import requests

logger = logging.getLogger(__name__)

# Default crypto headlines used as fallback when no live news is available
_DEFAULT_CRYPTO_HEADLINES = [
    "Bitcoin market shows mixed signals today",
    "Crypto trading volume remains steady",
    "Market participants await key economic data",
]


class SentimentExtractor:
    """
    Extracts a single sentiment score (-1 to +1) from crypto news headlines
    using FinBERT. Falls back to a price-movement heuristic if the model
    is unavailable or inference fails.

    The score is updated every N steps (configurable) to simulate periodic
    news arrival without paying inference cost on every environment step.
    """

    def __init__(
        self,
        model_id: str = "ProsusAI/finbert",
        device: str = "cpu",
        update_interval: int = 100,
    ):
        self.model_id = model_id
        self.device = device
        self.update_interval = max(1, int(update_interval))
        self._pipeline = None
        self._loaded = False
        self._current_score = 0.0
        self._step_counter = 0
        self._fallback_mode = False

    @property
    def sentiment_dim(self) -> int:
        """Returns 1 — a single scalar sentiment score."""
        return 1

    @property
    def current_score(self) -> float:
        return self._current_score

    def update(
        self,
        force: bool = False,
        close_prices: np.ndarray | None = None,
    ) -> float:
        """
        Update the sentiment score if enough steps have elapsed.

        Args:
            force: If True, bypass the update_interval check.
            close_prices: Optional close prices for price-movement heuristic fallback.

        Returns:
            Current sentiment score (-1 to +1).
        """
        self._step_counter += 1

        if not force and self._step_counter < self.update_interval:
            return self._current_score

        self._step_counter = 0

        try:
            if not self._loaded:
                self._load_pipeline()

            if self._pipeline is not None and not self._fallback_mode:
                headlines = self._fetch_headlines()
                score = self._analyze_sentiment(headlines)
            else:
                score = self._price_heuristic(close_prices)

            self._current_score = float(np.clip(score, -1.0, 1.0))
        except Exception as exc:
            logger.debug("SentimentExtractor: update failed: %s", exc)
            self._current_score = 0.0

        return self._current_score

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_pipeline(self) -> None:
        """Lazy-load the FinBERT pipeline."""
        self._loaded = True
        try:
            from transformers import pipeline

            logger.info(
                "SentimentExtractor: loading %s on %s", self.model_id, self.device
            )
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_id,
                device=self.device if self.device != "cpu" else -1,
                top_k=None,
            )
            logger.info("SentimentExtractor: loaded OK")
        except Exception as exc:
            logger.warning(
                "SentimentExtractor: failed to load %s: %s — using price-heuristic fallback",
                self.model_id,
                exc,
            )
            self._fallback_mode = True
            self._pipeline = None

    def _fetch_headlines(self) -> list[str]:
        """
        Fetch recent crypto news headlines from multiple sources.

        Tries in order:
        1. NewsAPI.org (requires ``AGI_NEWSAPI_KEY`` env var)
        2. Economic calendar events (from ``trade_review.get_economic_calendar``)
        3. Hardcoded placeholders (ultimate fallback)
        """
        # Tier 1: NewsAPI.org
        api_key = os.environ.get("AGI_NEWSAPI_KEY", "")
        if api_key:
            try:
                url = (
                    "https://newsapi.org/v2/everything"
                    "?q=cryptocurrency"
                    "&language=en"
                    "&sortBy=publishedAt"
                    "&pageSize=10"
                    f"&apiKey={api_key}"
                )
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    articles = data.get("articles", [])
                    if articles:
                        headlines = [
                            a["title"] for a in articles
                            if a.get("title") and a["title"] != "[Removed]"
                        ]
                        if headlines:
                            # Prepend a market summary headline
                            current_hour = datetime.utcnow().strftime("%H:%M UTC")
                            headlines.insert(0, f"Crypto market snapshot at {current_hour}")
                            logger.info(
                                "SentimentExtractor: fetched %d headlines from NewsAPI",
                                len(headlines),
                            )
                            return headlines
                    logger.warning(
                        "SentimentExtractor: NewsAPI returned no articles"
                    )
                else:
                    logger.warning(
                        "SentimentExtractor: NewsAPI error %d: %s",
                        resp.status_code,
                        resp.text[:200],
                    )
            except Exception as exc:
                logger.debug("SentimentExtractor: NewsAPI request failed: %s", exc)

        # Tier 2: Economic calendar events
        try:
            # Lazy import to avoid circular deps at module level
            from Python.trade_review import get_economic_calendar
            events = get_economic_calendar(days_ahead=3)
            if events:
                headlines = []
                for ev in events:
                    name = ev.get("name", "")
                    currency = ev.get("currency", "")
                    importance = ev.get("importance_label", "low")
                    if name:
                        prefix = "HIGH IMPACT: " if importance == "high" else ""
                        headlines.append(f"{prefix}{name} ({currency})")
                if headlines:
                    headlines.append(
                        "Traders monitoring {0} economic events".format(len(events))
                    )
                    logger.info(
                        "SentimentExtractor: %d economic calendar events",
                        len(headlines),
                    )
                    return headlines[:10]  # cap at 10
        except Exception as exc:
            logger.debug(
                "SentimentExtractor: economic calendar fallback failed: %s", exc
            )

        # Tier 3: Hardcoded placeholders
        logger.debug("SentimentExtractor: using hardcoded headlines")
        return _DEFAULT_CRYPTO_HEADLINES

    def _analyze_sentiment(self, headlines: list[str]) -> float:
        """
        Run FinBERT on the headlines and aggregate into a single score.

        Score = mean(P(positive) - P(negative)) across all headlines.
        Ranges from -1 (all strongly negative) to +1 (all strongly positive).
        """
        if not headlines:
            return 0.0

        results = self._pipeline(headlines)

        scores = []
        for result in results:
            # result is a list of dicts: [{'label': 'positive', 'score': 0.98}, ...]
            label_map = {d["label"].lower(): d["score"] for d in result}
            pos = label_map.get("positive", 0.0)
            neg = label_map.get("negative", 0.0)
            scores.append(pos - neg)

        return float(np.mean(scores)) if scores else 0.0

    @staticmethod
    def _price_heuristic(close_prices: np.ndarray | None) -> float:
        """
        Derive a sentiment score from recent price action.

        Used as fallback when FinBERT is unavailable. Uses the slope of
        the close-price window as a crude market sentiment proxy.
        """
        if close_prices is None or len(close_prices) < 10:
            return 0.0

        returns = (close_prices[-1] / close_prices[0]) - 1.0
        # Map return to [-1, +1] with soft saturation
        score = np.tanh(returns * 50.0)  # 2% move → ~0.76, 5% → ~0.99
        return float(score)
