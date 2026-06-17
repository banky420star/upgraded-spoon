"""
DecisionHead — Policy head for Decision PPO (Rich Trade Specification).

This is the high-level "brain" component that outputs the complete
structured trade decision vector (lot sizing, TP/SL in multiple units,
trailing, partials, breakeven, max hold, etc.).

Designed to sit on top of multi-timeframe (1m+5m+15m+1h) feature
extractors (LSTM / Transformer / Dreamer backbone) and per-symbol
best feature params.

The output of forward() is a raw continuous vector in [-1,1]^DECISION_ACTION_DIM
which TradingEnv.decode_action() turns into a full DecisionSpec.

Usage in custom training (beyond default SB3 PPO):
    from drl.decision_head import DecisionHead
    head = DecisionHead(obs_dim=..., hidden=256, action_dim=DECISION_ACTION_DIM)
    action_logits = head(features)

For SB3 integration, subclass stable_baselines3.common.policies.ActorCriticPolicy
and replace the action net with this head (or use as feature extractor head).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

try:
    from drl.trading_env import DECISION_ACTION_DIM
except Exception:
    DECISION_ACTION_DIM = 18


class DecisionHead(nn.Module):
    """
    Multi-head rich decision policy head.

    Outputs:
      - mean vector for the full Decision PPO action (continuous)
      - Optional: log_std for stochastic policy (used in PPO)
      - Can be extended with discrete heads (e.g. for entry_type logits) in future.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        action_dim: int = DECISION_ACTION_DIM,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)

        layers = []
        prev = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(prev, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden_dim
        self.backbone = nn.Sequential(*layers)

        # Mean head for the full rich decision vector
        self.mean_head = nn.Linear(hidden_dim, action_dim)

        # Learnable log_std (state-independent for simplicity; can be state-dep)
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)

        # Optional auxiliary heads (future: discrete choice for TP type etc. can be added here)
        # self.tp_type_head = nn.Linear(hidden_dim, 4)  # e.g. 4 TP modes

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            features: [B, input_dim] or [B, T, F] flattened upstream
        Returns:
            dict with 'action_mean', 'action_log_std', 'action_dist' (Normal)
        """
        if features.dim() > 2:
            # Assume last timestep or mean pool if sequence provided
            features = features[:, -1, :] if features.dim() == 3 else features.reshape(features.shape[0], -1)

        h = self.backbone(features)
        mean = torch.tanh(self.mean_head(h))  # bound to [-1, 1] exactly as env expects
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std).clamp(1e-4, 2.0)

        dist = Normal(mean, std)

        return {
            "action_mean": mean,
            "action_log_std": log_std,
            "action_std": std,
            "action_dist": dist,
            "raw_action": mean,  # deterministic sample for inference
        }

    def sample_action(self, features: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        out = self.forward(features)
        if deterministic:
            return out["action_mean"]
        action = out["action_dist"].sample()
        # Enforce bounds (tanh already applied on mean; clip samples)
        return torch.clamp(action, -1.0, 1.0)

    def log_prob(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        out = self.forward(features)
        return out["action_dist"].log_prob(actions).sum(dim=-1)

    def entropy(self, features: torch.Tensor) -> torch.Tensor:
        out = self.forward(features)
        return out["action_dist"].entropy().sum(dim=-1)


class DecisionPPOActorCritic(nn.Module):
    """
    Minimal standalone actor-critic using DecisionHead (for custom loops or distillation).
    For full SB3 PPO use, prefer wrapping DecisionHead inside a custom ActorCriticPolicy.
    """

    def __init__(self, obs_dim: int, action_dim: int = DECISION_ACTION_DIM, hidden: int = 256):
        super().__init__()
        self.actor = DecisionHead(obs_dim, hidden_dim=hidden, action_dim=action_dim)
        # Simple value head (critic)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward_actor(self, obs: torch.Tensor, deterministic: bool = False):
        return self.actor.sample_action(obs, deterministic=deterministic)

    def forward_critic(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() > 2:
            obs = obs[:, -1, :] if obs.dim() == 3 else obs.reshape(obs.shape[0], -1)
        return self.critic(obs).squeeze(-1)

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        actor_out = self.actor.forward(obs)
        logp = actor_out["action_dist"].log_prob(actions).sum(-1)
        entropy = actor_out["action_dist"].entropy().sum(-1)
        value = self.forward_critic(obs)
        return logp, entropy, value


def make_decision_head_for_sb3_policy(obs_dim: int, action_dim: int = DECISION_ACTION_DIM) -> DecisionHead:
    """Factory for use when building custom SB3 policies."""
    return DecisionHead(input_dim=obs_dim, action_dim=action_dim)


# Example integration note (not executed):
# from stable_baselines3.common.policies import ActorCriticPolicy
# class DecisionPPO_Policy(ActorCriticPolicy):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         # Replace self.action_net with DecisionHead(...)
#         ...
