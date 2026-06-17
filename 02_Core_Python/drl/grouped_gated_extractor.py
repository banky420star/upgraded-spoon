"""
Grouped Gated Feature Extractor for Stable Baselines3.

Splits the flat observation vector into semantic groups, encodes each group
independently, and learns to gate/weight each group adaptively before
concatenation. Designed for the Supreme Chainsaw trading system.

Reference architecture from the plan:
    - Per-group Linear -> LayerNorm -> ReLU -> Linear -> ReLU encoders
    - Learned gate: softmax over groups, applied as per-group attention weights
    - Compatible with SB3 MultiInputPolicy
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ---------------------------------------------------------------------------
# Feature group definitions
# ---------------------------------------------------------------------------
# These map column index ranges in the per-bar feature vector to semantic groups.
# The full per-bar feature vector has 59 columns:
#   [0..3]   price      (open_rel, high_rel, low_rel, close_rel)
#   [4..7]   vol_ret    (log_vol, log_ret1, log_ret5, log_ret20)
#   [8..11]  candle     (body_ratio, upper_wick, lower_wick, range_ratio)
#   [12..14] volatility (rv_20, rel_volume, spread_est_bps)
#   [15..18] time       (hour_sin, hour_cos, dow_sin, dow_cos)
#   [19..20] trend      (htf_trend, vol_bucket)
#   [21..23] session    (session_london, session_ny, major_open)
#   [24..29] news       (5 news-avoidance placeholders)
#   [29..40] patterns   (11 classical pattern detectors)
#   [40..58] cross_asset (6 features x 3 symbols = 18)
#   [58..59] ml_signal  (XGBoost directional probability)
#
# Plus trailing features (not windowed):
#   portfolio (9): balance, position, drawdown, ...
#   regime (6): 5 one-hot + 1 confidence (conditional)

FEATURE_GROUPS: dict[str, tuple[int, int]] = {
    "price": (0, 4),
    "vol_ret": (4, 8),
    "candle": (8, 12),
    "volatility": (12, 15),
    "time": (15, 19),
    "trend": (19, 21),
    "session": (21, 24),
    "news": (24, 29),
    "patterns": (29, 40),
    "cross_asset": (40, 58),
    "ml_signal": (58, 59),
}

GROUP_NAMES = list(FEATURE_GROUPS.keys())

# Trailing (non-windowed) feature group names
TRAILING_GROUPS = ["portfolio", "regime"]

# Default hidden size for per-group encoders
DEFAULT_HIDDEN = 64


class GroupedGatedExtractor(BaseFeaturesExtractor):
    """
    SB3-compatible feature extractor for Dict observations.

    Each observation dict contains:
      - Windowed feature groups: {name: (window * group_size,) flattened tensor}
      - Trailing groups: {name: (group_size,) tensor}

    Architecture:
      1. Per-group encoder: Linear(group_dim, hidden) -> LayerNorm -> ReLU
         -> Linear(hidden, hidden) -> ReLU
      2. Learned gate: Linear(hidden * n_groups, n_groups) -> softmax
      3. Gated concatenation: each group embedding weighted by its gate value
      4. Output: features_dim = hidden * (n_windowed + n_trailing)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        hidden: int = DEFAULT_HIDDEN,
    ):
        n_windowed = sum(
            1 for k in observation_space.spaces if k in FEATURE_GROUPS
        )
        n_trailing = sum(
            1 for k in observation_space.spaces if k in TRAILING_GROUPS
        )
        features_dim = hidden * (n_windowed + n_trailing)

        super().__init__(observation_space, features_dim=features_dim)

        self.hidden = hidden
        self._n_windowed = n_windowed
        self._n_trailing = n_trailing

        # --- Per-group encoders ---
        self.encoders = nn.ModuleDict()
        for name, space in observation_space.spaces.items():
            if name in FEATURE_GROUPS or name in TRAILING_GROUPS:
                in_dim = int(np.prod(space.shape))
                self.encoders[name] = nn.Sequential(
                    nn.Linear(in_dim, hidden),
                    nn.LayerNorm(hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                )

        # --- Learned gate over windowed groups ---
        if n_windowed > 0:
            self.gate = nn.Sequential(
                nn.Linear(hidden * n_windowed, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_windowed),
            )
        else:
            self.gate = None
        # Diagnostic storage (populated during forward for TensorBoard logging)
        self._last_gate_weights = None
        self._last_gate_logits = None
        self._last_encoded = None
        self._last_windowed_names = None

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        # Encode all groups
        encoded: dict[str, torch.Tensor] = {}
        for name, x in observations.items():
            if name in self.encoders:
                encoded[name] = self.encoders[name](x)

        # Gate windowed groups
        windowed_embeddings = []
        windowed_names = []
        for name in GROUP_NAMES:
            if name in encoded:
                windowed_embeddings.append(encoded[name])
                windowed_names.append(name)

        if self.gate is not None and len(windowed_embeddings) > 0:
            stacked = torch.stack(windowed_embeddings, dim=1)
            batch_size = stacked.shape[0]
            gate_input = stacked.reshape(batch_size, -1)
            gate_logits = self.gate(gate_input)
            gate_weights = torch.softmax(gate_logits, dim=1)

            # Store for diagnostics
            self._last_gate_weights = gate_weights.detach()
            self._last_gate_logits = gate_logits.detach()
            self._last_encoded = {k: v.detach() for k, v in encoded.items()}
            self._last_windowed_names = windowed_names

            gated = []
            for i, emb in enumerate(windowed_embeddings):
                w = gate_weights[:, i:i+1]
                gated.append(emb * w)
        else:
            gated = windowed_embeddings
            self._last_gate_weights = None
            self._last_gate_logits = None
            self._last_encoded = None
            self._last_windowed_names = None

        # Add trailing group embeddings (ungated)
        for name in TRAILING_GROUPS:
            if name in encoded:
                gated.append(encoded[name])

        return torch.cat(gated, dim=1)


def build_grouped_obs_space(
    window_size: int = 100,
    portfolio_feature_count: int = 9,
    regime_dim: int = 0,
) -> gym.spaces.Dict:
    """Build a Dict observation space for grouped extractor."""
    spaces_dict: dict[str, gym.Space] = {}

    for name, (start, end) in FEATURE_GROUPS.items():
        group_size = end - start
        spaces_dict[name] = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size * group_size,),
            dtype=np.float32,
        )

    spaces_dict["portfolio"] = gym.spaces.Box(
        low=-np.inf, high=np.inf,
        shape=(portfolio_feature_count,),
        dtype=np.float32,
    )

    if regime_dim > 0:
        spaces_dict["regime"] = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(regime_dim,),
            dtype=np.float32,
        )

    return gym.spaces.Dict(spaces_dict)


def split_obs_into_groups(
    feature_window: np.ndarray,
    portfolio_state: np.ndarray,
    regime_emb: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Split a flat feature window into grouped observation dict.

    Handles partial feature matrices gracefully: if a group's column range
    exceeds the available columns, the group is zero-padded to the expected size.
    """
    obs: dict[str, np.ndarray] = {}
    n_available = feature_window.shape[1]

    for name, (start, end) in FEATURE_GROUPS.items():
        expected_cols = end - start
        if end <= n_available:
            group_slice = feature_window[:, start:end]
        elif start < n_available:
            # Partial group available - pad with zeros
            available = feature_window[:, start:n_available]
            pad_width = expected_cols - available.shape[1]
            group_slice = np.pad(available, ((0, 0), (0, pad_width)), mode='constant')
        else:
            # Entire group missing - fill with zeros
            group_slice = np.zeros((feature_window.shape[0], expected_cols), dtype=np.float32)
        obs[name] = group_slice.flatten().astype(np.float32)

    obs["portfolio"] = portfolio_state.astype(np.float32)

    if regime_emb is not None:
        obs["regime"] = regime_emb.astype(np.float32)

    return obs
