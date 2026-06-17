import json
import os

import numpy as np
import pandas as pd
import torch

from Python.feature_pipeline import ULTIMATE_150, build_env_feature_matrix
from drl.dreamer_agent import DreamerV3Agent


class DreamerPolicy:
    def __init__(self, checkpoint_path: str, metadata_path: str):
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f) or {}
        self.checkpoint_path = checkpoint_path
        self.window_size = int(self.meta.get("window_size", 64) or 64)
        self.feature_version = str(self.meta.get("feature_version", ULTIMATE_150) or ULTIMATE_150)
        self.obs_dim = int(self.meta.get("obs_dim", 0) or 0)
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        self.agent = DreamerV3Agent(obs_dim=self.obs_dim, action_dim=3, device=self.device)
        self.agent.load(self.checkpoint_path)

    def _build_obs(self, df: pd.DataFrame) -> np.ndarray | None:
        features = build_env_feature_matrix(df.copy(), feature_version=self.feature_version)
        if len(features) < self.window_size:
            return None
        obs = np.concatenate([features[-self.window_size :].reshape(-1), np.array([0.0], dtype=np.float32)]).astype(np.float32)
        if self.obs_dim and obs.shape[0] != self.obs_dim:
            return None
        return obs

    def predict_exposure(self, df: pd.DataFrame) -> float | None:
        obs = self._build_obs(df)
        if obs is None:
            return None
        action_onehot, _ = self.agent.act(obs, deterministic=True)
        action_idx = int(np.argmax(action_onehot))
        if action_idx == 1:
            return 1.0
        if action_idx == 2:
            return -1.0
        return 0.0

    @staticmethod
    def load_symbol(symbol: str):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ckpt = os.path.join(root, "models", "dreamer", f"dreamer_{symbol}.pt")
        meta = os.path.join(root, "models", "dreamer", f"dreamer_{symbol}.json")
        if os.path.exists(ckpt) and os.path.exists(meta):
            return DreamerPolicy(ckpt, meta)
        return None
