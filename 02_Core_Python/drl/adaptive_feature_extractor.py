import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from drl.trend_momentum_bias import TrendMomentumBiasLayer

# Number of regime classes (one-hot dimension, excluding confidence)
try:
    from drl.regime_detector import NUM_REGIMES
except Exception:
    NUM_REGIMES = 5


class FeatureGroupGate(torch.nn.Module):
    """
    Regime-conditional feature group gating.

    Splits the LSTM hidden dimension into N groups and applies group-specific
    linear projections whose weights are modulated by regime probabilities.
    This allows the network to learn "which features matter in which regime".

    In trending regimes, trend-related projections get higher weight.
    In ranging regimes, mean-reversion projections dominate.
    The gate learns this mapping from the regime classifier's output.

    Args:
        hidden_dim: LSTM hidden dimension (e.g., 320 for bidirectional 160*2).
        num_groups: Number of feature groups (default 4 = trend/momentum/vol/other).
        num_regimes: Number of regime classes (default 5).
    """

    def __init__(self, hidden_dim: int, num_groups: int = 4, num_regimes: int = 5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_groups = num_groups
        group_size = hidden_dim // num_groups

        # Each group gets its own learned linear projection
        self.group_projections = torch.nn.ModuleList([
            torch.nn.Linear(group_size, group_size, bias=False)
            for _ in range(num_groups)
        ])

        # Gating network: regime probs -> group weights
        # A small MLP that learns which groups matter in each regime
        self.gate_net = torch.nn.Sequential(
            torch.nn.Linear(num_regimes, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, num_groups),
        )

        # Residual projection to preserve dimensionality if group split uneven
        self.residual_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)             if hidden_dim % num_groups != 0 else None

    def forward(self, x: torch.Tensor, regime_probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, hidden_dim] - LSTM/attention output
            regime_probs: [batch, num_regimes] - regime probabilities
        Returns:
            [batch, hidden_dim] - gated feature representation
        """
        batch_size = x.shape[0]
        group_size = self.hidden_dim // self.num_groups

        # Compute group weights from regime probs
        gate_logits = self.gate_net(regime_probs)  # [batch, num_groups]
        gate_weights = torch.softmax(gate_logits, dim=1)  # [batch, num_groups]
        self._last_gate_weights = gate_weights.detach()

        # Split hidden dim into groups and apply group-specific projections
        chunks = torch.split(x, group_size, dim=1)
        gated = []
        for i in range(self.num_groups):
            projected = self.group_projections[i](chunks[i])  # [batch, group_size]
            weight = gate_weights[:, i:i+1]  # [batch, 1]
            gated.append(projected * weight)

        out = torch.cat(gated, dim=1)

        # Residual connection
        if self.residual_proj is not None:
            out = out + self.residual_proj(x)
        else:
            out = out + x

        return out

    def get_gate_weights(self) -> torch.Tensor | None:
        """Return last gate weights for diagnostics."""
        if hasattr(self, '_last_gate_weights'):
            return self._last_gate_weights
        return None


class MultiHeadAttentionPooling(torch.nn.Module):
    """
    Multi-head self-attention pooling over the LSTM sequence dimension.

    Uses 4 independent learned query vectors (heads) to attend to different
    temporal patterns simultaneously. Each head can focus on a different set
    of bars (e.g., head 1 = recent price action, head 2 = volatility regime,
    head 3 = support/resistance levels, head 4 = volume patterns).

    The 4 context vectors are concatenated and projected back to hidden_dim
    via a learned linear layer.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.scale = (hidden_dim // num_heads) ** -0.5
        self.queries = torch.nn.Parameter(torch.randn(num_heads, hidden_dim) * 0.01)
        self.out_proj = torch.nn.Linear(num_heads * hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        scores = torch.matmul(x, self.queries.T) * self.scale
        scores = scores.permute(0, 2, 1)
        weights = torch.softmax(scores, dim=-1)
        self.last_weights = weights.mean(dim=1).detach()
        context = torch.matmul(weights, x)
        context = context.reshape(batch_size, -1)
        return self.out_proj(context)


class AdaptiveLSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    Trainable extractor for newer feature contracts where the live AGI LSTM
    bundle should not dictate PPO observation handling.

    v2: Bidirectional LSTM - processes the time-window forward AND backward
    so the representation at each step benefits from both past AND future
    context. Output dimension is doubled (hidden_size * 2) before projection.

    v5 (Regime Injection Fix): When regime_dim > 0, the regime features
    (one-hot + confidence, at the END of the observation tail) are extracted,
    repeated across the sequence dimension, and concatenated to the per-bar
    features BEFORE the LSTM. This lets the LSTM learn regime-conditional
    temporal patterns. Portfolio state (non-regime tail) is still concatenated
    after the LSTM as before.
    """

    def __init__(self, observation_space, features_dim=256, window_size=100, num_heads=4, regime_dim=0, use_feature_gate=False, use_trend_momentum_bias=False, bias_fixed_temperature=None):
        total_obs = int(observation_space.shape[0])
        self.seq_window = int(window_size)
        self.regime_dim = int(regime_dim)
        self.portfolio_dim = total_obs % self.seq_window
        seq_flat = total_obs - self.portfolio_dim
        if seq_flat <= 0 or seq_flat % self.seq_window != 0:
            raise ValueError(
                f"Invalid observation shape: total={total_obs}, window={self.seq_window}, port={self.portfolio_dim}"
            )
        actual_portfolio_dim = max(0, self.portfolio_dim - self.regime_dim)

        # Trend + Momentum bias layer: soft directional prior
        # Create temporary instance first to get num_bias_features for _features_dim
        _tmp_bias = None
        bias_extra = 0
        if use_trend_momentum_bias:
            _tmp_bias = TrendMomentumBiasLayer(input_dim=features_dim, fixed_temperature=bias_fixed_temperature)
            bias_extra = _tmp_bias.num_bias_features

        super().__init__(observation_space, features_dim=features_dim + bias_extra + actual_portfolio_dim)

        # Now assign the bias layer to self (Module.__init__ has been called)
        self.trend_momentum_bias = _tmp_bias
        self.seq_feature_dim = seq_flat // self.seq_window
        self.num_regime_classes = min(NUM_REGIMES, max(0, self.regime_dim))
        lstm_input_dim = self.seq_feature_dim + self.regime_dim
        self.encoder = torch.nn.LSTM(
            input_size=lstm_input_dim, hidden_size=160, num_layers=2,
            dropout=0.2, batch_first=True, bidirectional=True,
        )
        self.lstm_hidden = 320
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(self.lstm_hidden, features_dim),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Linear(features_dim, features_dim),
        )
        self.attention = MultiHeadAttentionPooling(self.lstm_hidden, num_heads=num_heads)
        self.lstm_norm = torch.nn.LayerNorm(self.lstm_hidden)

        # Regime-conditional feature group gating
        # num_regime_classes = number of one-hot regime classes (5), not
        # regime_dim (6, which includes confidence).  Using regime_dim here
        # would create a dimension mismatch when the gate receives only the
        # one-hot slice of the tail.
        if use_feature_gate and self.regime_dim > 0:
            self.feature_gate = FeatureGroupGate(
                hidden_dim=self.lstm_hidden,
                num_groups=4,
                num_regimes=self.num_regime_classes,
            )
        else:
            self.feature_gate = None

    def forward(self, observations):
        batch_size = observations.shape[0]
        seq_features = observations[:, :-self.portfolio_dim] if self.portfolio_dim else observations
        tail = observations[:, -self.portfolio_dim:] if self.portfolio_dim else observations.new_zeros((batch_size, 0))
        if self.regime_dim > 0 and tail.shape[-1] >= self.regime_dim:
            regime = tail[:, -self.regime_dim:]
            portfolio_state = tail[:, :-self.regime_dim]
        else:
            regime = None
            portfolio_state = tail
        seq = seq_features.view(batch_size, self.seq_window, self.seq_feature_dim)
        if regime is not None:
            regime_expanded = regime.unsqueeze(1).expand(-1, self.seq_window, -1)
            seq = torch.cat([seq, regime_expanded], dim=-1)
        encoded, _ = self.encoder(seq)
        lstm_out = self.attention(encoded)
        lstm_out = self.lstm_norm(lstm_out)

        # v6: Regime-conditional feature group gating
        if self.feature_gate is not None and self.regime_dim > 0 and regime is not None:
            # Extract the one-hot slice from the regime tail.
            # regime: [batch, regime_dim] where first num_regime_classes dims
            # are one-hot, and the remaining is confidence.
            regime_onehot = regime[:, :self.num_regime_classes]
            if regime_onehot.shape[-1] >= self.num_regime_classes:
                lstm_out = self.feature_gate(lstm_out, regime_onehot)

        projected = self.projection(lstm_out)
        if self.trend_momentum_bias is not None:
            projected = self.trend_momentum_bias(projected)
        if portfolio_state.shape[-1] > 0:
            return torch.cat([projected, portfolio_state], dim=1)
        return projected

    def get_attention_weights(self):
        if hasattr(self.attention, "last_weights"):
            return self.attention.last_weights
        return None
