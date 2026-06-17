"""
Regime-Routed Policy Head.

Architecture:
  latent_pi  -> actor_regime_classifier -> actor_regime_probs -> weight action_nets
  latent_vf  -> value_classifier -> value_regime_probs -> weight value_nets

Independent regime classifiers for actor and critic give each branch
its own regime decomposition without conflicting gradients.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3 import PPO as _PPO
from stable_baselines3.common.policies import ActorCriticPolicy

logger = logging.getLogger("regime_routed")

try:
    from drl.regime_detector import NUM_REGIMES
except Exception:
    NUM_REGIMES = 5


class _RegimeWeightedValue(nn.Module):
    """
    Value head with its own regime classifier.

    Operates on latent_vf independently from the actor's regime_classifier,
    learning a separate regime decomposition for the value function.
    """

    def __init__(
        self,
        regime_value_nets: nn.ModuleList,
        vf_latent_dim: int,
        num_regimes: int,
    ):
        super().__init__()
        self.regime_value_nets = regime_value_nets
        self.value_classifier = nn.Linear(vf_latent_dim, num_regimes)
        self._last_value_probs = None

    def forward(self, latent_vf: torch.Tensor) -> torch.Tensor:
        # Independent regime classification on latent_vf
        regime_logits = self.value_classifier(latent_vf)
        regime_probs = torch.softmax(regime_logits, dim=1)
        self._last_value_probs = regime_probs.detach()

        # Weight each regime's value network by its regime probability
        values = torch.cat(
            [net(latent_vf) for net in self.regime_value_nets], dim=1
        )
        return (values * regime_probs).sum(dim=1, keepdim=True)


class RegimeRoutedActorCriticPolicy(ActorCriticPolicy):
    """
    ActorCriticPolicy with regime-routed action heads and an independent
    regime-conditional value function.

    Actor: latent_pi -> actor_regime_classifier -> regime_probs -> weight action_nets
    Critic: latent_vf -> value_classifier -> value_probs -> weight value_nets
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        *args,
        num_regimes: int = NUM_REGIMES,
        regime_dim: int = 0,
        action_head_lr_mult: float | None = None,
        **kwargs,
    ):
        self.num_regimes = num_regimes
        self.regime_dim = int(regime_dim)
        self.action_head_lr_mult = action_head_lr_mult or float(
            os.environ.get("AGI_PPO_ACTION_HEAD_LR_MULT", "2.0")
        )
        self._last_regime_logits = None
        self._last_regime_probs = None
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        pi_latent_dim = self.mlp_extractor.latent_dim_pi
        vf_latent_dim = self.mlp_extractor.latent_dim_vf

        # Actor: regime-specific action networks
        self.regime_action_nets = nn.ModuleList([
            nn.Linear(pi_latent_dim, self.action_space.shape[0])
            for _ in range(self.num_regimes)
        ])

        # Actor: regime classifier (on latent_pi)
        self.regime_classifier = nn.Linear(pi_latent_dim, self.num_regimes)

        # Critic: regime-specific value networks
        self.regime_value_nets = nn.ModuleList([
            nn.Linear(vf_latent_dim, 1) for _ in range(self.num_regimes)
        ])

        # Critic: independent regime classifier (on latent_vf)
        # Replaces the single value_net with regime-weighted ensemble
        self.value_net = _RegimeWeightedValue(
            self.regime_value_nets,
            vf_latent_dim,
            self.num_regimes,
        )

        # Initialise — match SB3's standard action_net gain (0.01) so actions
        # start near zero and the policy explores before saturating at ±1.
        # Previous gain=5.0 caused instant hard-clipping to ±1 with zero gradient
        # signal to recover, since the Gaussian mean → ±5 → clipped to ±1.
        for net in self.regime_action_nets:
            nn.init.orthogonal_(net.weight, gain=0.01)
            nn.init.zeros_(net.bias)
        nn.init.xavier_uniform_(self.regime_classifier.weight)
        nn.init.zeros_(self.regime_classifier.bias)
        for net in self.regime_value_nets:
            nn.init.orthogonal_(net.weight, gain=1.0)
            nn.init.zeros_(net.bias)
        nn.init.xavier_uniform_(self.value_net.value_classifier.weight)
        nn.init.zeros_(self.value_net.value_classifier.bias)

        self._rebuild_optimizer(lr_schedule)

    def _rebuild_optimizer(self, lr_schedule) -> None:
        base_lr = lr_schedule(1.0)
        action_lr = base_lr * self.action_head_lr_mult

        action_head_names = [
            "regime_action_nets", "regime_value_nets",
            "action_scale", "log_std", "regime_classifier",
        ]
        body_params, action_head_params = [], []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(h in name for h in action_head_names):
                action_head_params.append(param)
            else:
                body_params.append(param)

        optim_class = self.optimizer_class
        optim_kwargs = self.optimizer_kwargs.copy() if self.optimizer_kwargs else {}
        clean_kwargs = {k: v for k, v in optim_kwargs.items() if k != "params"}
        clean_kwargs.setdefault("eps", 1e-5)

        self.optimizer = optim_class([
            {"params": body_params, **clean_kwargs, "lr": base_lr},
            {"params": action_head_params, **clean_kwargs, "lr": action_lr},
        ], lr=base_lr, **clean_kwargs)

    def _get_action_dist_from_latent(self, latent_pi):
        regime_logits = self.regime_classifier(latent_pi)
        regime_probs = torch.softmax(regime_logits, dim=1)

        self._last_regime_logits = regime_logits
        self._last_regime_probs = regime_probs.detach()

        weighted_mean = None
        for i in range(self.num_regimes):
            mean_i = self.regime_action_nets[i](latent_pi)
            w = regime_probs[:, i:i+1]
            if weighted_mean is None:
                weighted_mean = mean_i * w
            else:
                weighted_mean = weighted_mean + mean_i * w

        if hasattr(self, "action_scale"):
            weighted_mean = weighted_mean * self.action_scale

        return self.action_dist.proba_distribution(weighted_mean, self.log_std)

    def regime_supervised_loss(self, observations):
        """Cross-entropy between actor regime classifier and heuristic labels.

        Computes regime_logits internally from the observations, removing the
        need for the caller to do a separate forward pass.
        """
        if self.regime_dim <= 0 or observations.shape[-1] < self.regime_dim:
            return observations.new_zeros(())

        # Extract heuristic labels from observation tail
        regime_tail = observations[:, -self.regime_dim :]
        regime_onehot = regime_tail[:, : self.num_regimes]
        targets = regime_onehot.argmax(dim=1).detach()

        # Compute regime_logits from a forward pass
        features = self.extract_features(observations)
        latent_pi, _ = self.mlp_extractor(features)
        regime_logits = self.regime_classifier(latent_pi)

        return F.cross_entropy(regime_logits, targets)

    def get_regime_probs(self, obs):
        """Actor's regime probabilities (from latent_pi)."""
        features = self.extract_features(obs)
        latent_pi, _ = self.mlp_extractor(features)
        regime_logits = self.regime_classifier(latent_pi)
        return torch.softmax(regime_logits, dim=1)

    def get_value_regime_probs(self, obs):
        """Critic's regime probabilities (from latent_vf)."""
        features = self.extract_features(obs)
        _, latent_vf = self.mlp_extractor(features)
        _ = self.value_net(latent_vf)  # forward populates _last_value_probs
        return self.value_net._last_value_probs

    def get_regime_label(self, obs):
        """Actor's most likely regime label."""
        probs = self.get_regime_probs(obs)
        idx = probs.argmax(dim=1).item()
        try:
            from drl.regime_detector import REGIME_LABELS
            return REGIME_LABELS[idx] if idx < len(REGIME_LABELS) else f"regime_{idx}"
        except Exception:
            return f"regime_{idx}"

    def get_value_regime_label(self, obs):
        """Critic's most likely regime label."""
        probs = self.get_value_regime_probs(obs)
        if probs is None:
            return 'unknown'
        idx = probs.argmax(dim=1).item()
        try:
            from drl.regime_detector import REGIME_LABELS
            return REGIME_LABELS[idx] if idx < len(REGIME_LABELS) else f'regime_{idx}'
        except Exception:
            return f'regime_{idx}' 


# ---- RegimeRoutedPPO ---------------------------------------------------
class RegimeRoutedPPO(_PPO):
    """
    PPO variant that adds regime-supervised loss during training.
    Wraps standard PPO with the regime-routed policy.
    """

    def __init__(self, policy, env, regime_loss_coef=0.05, **kwargs):
        self.regime_loss_coef = regime_loss_coef
        super().__init__(policy, env, **kwargs)

    def train(self):
        """
        Override PPO.train() to incorporate regime-supervised loss.

        After the standard PPO training step, computes the regime supervised
        loss from rollout buffer observations and does an additional
        backward+step. The regime_loss_coef controls the contribution.
        """
        super().train()

        if self.regime_loss_coef <= 0:
            return
        if not hasattr(self.policy, 'regime_supervised_loss'):
            return
        if not hasattr(self, 'rollout_buffer') or self.rollout_buffer is None:
            return

        try:
            obs = self.rollout_buffer.observations
            # Flatten: [n_steps, n_envs, ...] -> [n_steps * n_envs, ...]
            obs_flat = obs.reshape(-1, *obs.shape[2:])

            sup_loss = self.policy.regime_supervised_loss(obs_flat)
            if sup_loss is None or sup_loss.numel() == 0:
                return
            if sup_loss.item() == 0.0:
                return

            total = sup_loss * self.regime_loss_coef
            self.policy.optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.max_grad_norm
            )
            self.policy.optimizer.step()

            self.logger.record('train/regime_supervised_loss', sup_loss.item())
        except Exception as e:
            if self.verbose >= 1:
                print(f'[RegimeRoutedPPO] supervised loss: {e}')
            pass
