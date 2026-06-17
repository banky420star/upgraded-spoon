"""
Chronos-bolt-base time-series foundation model feature extractor.

Transforms OHLCV windows into dense embeddings using Amazon's Chronos-Bolt model.
These embeddings supplement the existing ultimate_150 feature pipeline,
providing the PPO agent with learned time-series representations.

Gated by env var: AGI_USE_CHRONOS=1 (or model ID like chronos-bolt-base)

Usage:
    extractor = ChronosExtractor()
    embedding = extractor(close_prices)  # returns (512,) numpy array
"""

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Default embedding dimension for chronos-bolt-base
_CHRONOS_BASE_DIM = 512

# Mapping of known Chronos model IDs to their embedding dims
_CHRONOS_MODEL_DIMS = {
    "amazon/chronos-bolt-tiny": 256,
    "amazon/chronos-bolt-mini": 256,
    "amazon/chronos-bolt-small": 384,
    "amazon/chronos-bolt-base": 512,
    "amazon/chronos-bolt-large": 1024,
}


def _resolve_model_id(candidate: str | None) -> str:
    """Resolve a short name or env-var value to a full Hugging Face model ID."""
    if not candidate:
        candidate = os.environ.get("AGI_CHRONOS_MODEL", "amazon/chronos-bolt-base")
    candidate = candidate.strip()

    # Allow short names
    short_map = {
        "tiny": "amazon/chronos-bolt-tiny",
        "mini": "amazon/chronos-bolt-mini",
        "small": "amazon/chronos-bolt-small",
        "base": "amazon/chronos-bolt-base",
        "large": "amazon/chronos-bolt-large",
        "chronos-bolt-base": "amazon/chronos-bolt-base",
    }
    return short_map.get(candidate.lower(), candidate)


def chronos_embedding_dim(model_id: str | None = None) -> int:
    """Return the embedding dimension for a given (or default) Chronos model."""
    resolved = _resolve_model_id(model_id)
    return _CHRONOS_MODEL_DIMS.get(resolved, _CHRONOS_BASE_DIM)


class ChronosExtractor:
    """
    Extracts Chronos-bolt-base embeddings from OHLCV close-price windows.

    The embedding is mean-pooled across the time dimension to produce a
    fixed-size vector (default 512-dim for chronos-bolt-base).

    Model loading is deferred to the first __call__ so that environments
    can be created without paying the HF download cost upfront.
    """

    def __init__(self, model_id: str | None = None, device: str = "cpu"):
        self.model_id = _resolve_model_id(model_id)
        self.device = device
        self._model = None
        self._loaded = False

    @property
    def embedding_dim(self) -> int:
        return chronos_embedding_dim(self.model_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, close_prices: np.ndarray) -> np.ndarray:
        """
        Extract a Chronos embedding from a 1-D close-price window.

        Args:
            close_prices: 1-D float array of close prices, length >= 10.

        Returns:
            (embedding_dim,) float32 numpy array.
            Returns a zero vector if the model is unavailable or on error.
        """
        if len(close_prices) < 10:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        if not self._loaded:
            self._load_model()

        if self._model is None:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        try:
            values = self._preprocess(close_prices)       # (seq_len,)
            embedding = self._forward(values)             # (embedding_dim,)
            return embedding
        except Exception as exc:
            logger.warning("ChronosExtractor: inference failed: %s", exc)
            return np.zeros(self.embedding_dim, dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Lazy-load the model (called once on first __call__)."""
        self._loaded = True
        try:
            import torch
            from transformers import AutoModel

            logger.info("ChronosExtractor: loading %s on %s …", self.model_id, self.device)
            self._model = (
                AutoModel.from_pretrained(self.model_id, trust_remote_code=True)
                .to(self.device)
                .eval()
            )
            logger.info("ChronosExtractor: loaded %s OK", self.model_id)
        except Exception as exc:
            logger.warning("ChronosExtractor: failed to load %s: %s", self.model_id, exc)
            self._model = None

    @staticmethod
    def _preprocess(close_prices: np.ndarray) -> np.ndarray:
        """
        Convert close prices to stationary log-returns.

        Chronos tokenises continuous values internally, but benefits from
        roughly-normalised inputs.  Log-returns are preferred over raw prices
        because they are stationary and typically in a reasonable range.
        """
        log_p = np.log(np.maximum(close_prices, 1e-10))
        returns = np.diff(log_p, prepend=log_p[0:1])
        return returns.astype(np.float32)

    def _forward(self, values: np.ndarray) -> np.ndarray:
        """
        Run the Chronos backbone and return a mean-pooled embedding.

        values: (seq_len,) float32 numpy array of preprocessed log-returns.
        Returns: (embedding_dim,) float32 numpy array.
        """
        import torch

        # Chronos expects shape (1, seq_len)
        inp = torch.from_numpy(values).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self._model(inp, output_hidden_states=True)
            # hidden_states[-1] is the last decoder layer → (1, seq_len, hidden)
            hidden = outputs.hidden_states[-1]
            # Mean-pool across the time axis
            pooled = hidden.mean(dim=1)  # (1, hidden)

        return pooled.squeeze(0).cpu().numpy().astype(np.float32)
