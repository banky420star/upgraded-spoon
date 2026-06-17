"""
Trend + Momentum Bias Layer

Computes a directional market-bias signal from LSTM-extracted features,
providing the policy with a soft directional prior. Instead of treating
all features equally, the policy can condition its decisions on whether
the market is trending up/down, ranging, or unstable.

Architecture placement:
    AdaptiveLSTMFeatureExtractor
        -> TrendMomentumBiasLayer (NEW)
        -> FeatureGroupGate
        -> RegimeRoutedPolicy
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class TrendMomentumBiasLayer(nn.Module):
    """
    Computes directional market bias from LSTM-extracted features.

    Adds a soft directional prior so the policy can distinguish between
    bullish, bearish, ranging, and unstable market states rather than
    treating all feature combinations equally.

    Produces 6 bias features appended to the input:
        col 0: trend_score        [-1 .. +1]  direction of trend
        col 1: momentum_score     [-1 .. +1]  force behind movement
        col 2: direction_bias     [-1 .. +1]  combined (trend+momentum)/2
        col 3: confidence         [ 0 ..  1]  how reliable the bias is
        col 4: agreement          [ 0 ..  1]  trend-momentum alignment
        col 5: persistent_bias    [-1 .. +1]  hysteresis-smoothed bias

    Uses temperature-scaled activations with small-weight init to prevent
    saturation — the Tanh outputs start near 0 and grow only as the PPO
    policy finds directional signal worth amplifying.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, fixed_temperature: float | None = None):
        super().__init__()
        self.input_dim = input_dim
        self.num_bias_features = 6
        self.fixed_temperature = fixed_temperature  # if set, overrides learnable temp

        # Trend score: shared encoder + temperature-scaled Tanh head
        self.trend_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.trend_head = nn.Linear(hidden_dim, 1)
        self.trend_temp = nn.Parameter(th.tensor(-0.5))  # softplus(-0.5) ≈ 0.47

        # Momentum score: shared encoder + temperature-scaled Tanh head
        self.momentum_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.momentum_head = nn.Linear(hidden_dim, 1)
        self.momentum_temp = nn.Parameter(th.tensor(-0.5))

        # Confidence: shared encoder + temperature-scaled Sigmoid head
        self.confidence_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.confidence_head = nn.Linear(hidden_dim, 1)
        self.confidence_temp = nn.Parameter(th.tensor(-0.5))

        # ── Initialise final linear heads with tiny weights so bias
        #    outputs start near zero rather than saturating Tanh/Sigmoid.
        for head in [self.trend_head, self.momentum_head, self.confidence_head]:
            nn.init.normal_(head.weight, std=0.01)

        # Persistent bias state for hysteresis smoothing during eval
        self.register_buffer('_persistent_bias', th.zeros(1))

        # Stored scores from the last forward pass for diagnostics
        self.last_scores: dict[str, th.Tensor] = {}

    def forward(self, features: th.Tensor, use_persistence: bool = True) -> th.Tensor:
        """
        Args:
            features: (batch, input_dim) -- projected features from LSTM extractor.
            use_persistence: apply hysteresis smoothing during single-step eval.

        Returns:
            (batch, input_dim + 6) -- original features with 6 bias signals appended.
        """
        # ── Trend: encode → head → temperature-scaled Tanh ──
        t_enc = self.trend_encoder(features)
        t_logits = self.trend_head(t_enc)
        t_temp = self.fixed_temperature if self.fixed_temperature is not None else F.softplus(self.trend_temp)
        trend = th.tanh(t_logits * t_temp)             # (batch, 1)

        # ── Momentum: encode → head → temperature-scaled Tanh ──
        m_enc = self.momentum_encoder(features)
        m_logits = self.momentum_head(m_enc)
        m_temp = self.fixed_temperature if self.fixed_temperature is not None else F.softplus(self.momentum_temp)
        momentum = th.tanh(m_logits * m_temp)          # (batch, 1)

        # ── Confidence: encode → head → temperature-scaled Sigmoid ──
        c_enc = self.confidence_encoder(features)
        c_logits = self.confidence_head(c_enc)
        c_temp = self.fixed_temperature if self.fixed_temperature is not None else F.softplus(self.confidence_temp)
        confidence = th.sigmoid(c_logits * c_temp)     # (batch, 1)

        direction_bias = (trend + momentum) / 2.0      # (batch, 1)
        agreement = 1.0 - th.abs(trend - momentum) / 2.0  # (batch, 1)

        batch_size = features.shape[0]
        if use_persistence and not self.training:
            if batch_size == 1:
                # Single-step eval: hysteresis to prevent flip-flopping
                diff = direction_bias - self._persistent_bias
                d = diff[0, 0].item()
                c = confidence[0, 0].item()
                if c > 0.35 and abs(d) > 0.15:
                    # Smooth transition: blend 40% toward new bias
                    persistent_bias = self._persistent_bias + diff * 0.4
                else:
                    # Stay with current bias
                    persistent_bias = self._persistent_bias.clone().view(1, 1)
                self._persistent_bias = persistent_bias.detach()
            else:
                # Batched eval: no per-sample state, use raw bias
                persistent_bias = direction_bias
        else:
            persistent_bias = direction_bias

        bias_features = th.cat(
            [trend, momentum, direction_bias, confidence, agreement, persistent_bias],
            dim=1,
        )

        # Store raw scores for diagnostics (detached, float32)
        self.last_scores = {
            "trend": trend.detach(),
            "momentum": momentum.detach(),
            "direction_bias": direction_bias.detach(),
            "confidence": confidence.detach(),
            "agreement": agreement.detach(),
            "persistent_bias": persistent_bias.detach(),
        }

        return th.cat([features, bias_features], dim=1)

    def reset_persistent_bias(self):
        """Reset hysteresis state -- call on environment reset during live trading."""
        self._persistent_bias.fill_(0.0)
