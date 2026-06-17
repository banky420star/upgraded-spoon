import os
import sys

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Python.agi_brain import SmartAGI


class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    Trainable LSTM feature extractor for joint LSTM + PPO training.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        total_obs = int(observation_space.shape[0])
        self.seq_window = 100
        self.portfolio_dim = total_obs % self.seq_window
        seq_flat = total_obs - self.portfolio_dim
        if seq_flat <= 0 or seq_flat % self.seq_window != 0:
            raise ValueError(
                f"Invalid observation shape for LSTM extractor: total={total_obs}, "
                f"window={self.seq_window}, portfolio_dim={self.portfolio_dim}"
            )

        super().__init__(observation_space, features_dim=features_dim + self.portfolio_dim)

        logger.info("initializing LSTMFeatureExtractor")
        self.lstm_brain = SmartAGI()

        self.projection = torch.nn.Linear(128, features_dim)

        # Keep all extractor params trainable.
        for param in self.lstm_brain.model.parameters():
            param.requires_grad = True

        self.seq_feature_dim = seq_flat // self.seq_window

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        device = observations.device
        self.lstm_brain.model.to(device)
        self.projection = self.projection.to(device)

        batch_size = observations.shape[0]

        seq_features = observations[:, :-self.portfolio_dim]
        portfolio_state = observations[:, -self.portfolio_dim :]

        seq = seq_features.view(batch_size, self.seq_window, self.seq_feature_dim)
        lstm_embedding = self.lstm_brain.extract_features(seq)

        projected = self.projection(lstm_embedding)
        return torch.cat([projected, portfolio_state], dim=1)
