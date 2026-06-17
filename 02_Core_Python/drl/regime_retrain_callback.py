"""
RegimeRetrainCallback — periodic RegimeDetector retraining during PPO training.

Initialises the RF with heuristic labels before training and periodically
retrains on accumulated training data.
"""

from __future__ import annotations

import logging

from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger("regime_retrain_callback")


class RegimeRetrainCallback(BaseCallback):
    """
    Periodic RegimeDetector retraining callback.

    - On training start: calls fit_heuristic() on the initial DataFrame to
      initialise the RF with proper heuristic labels.
    - Every retrain_freq steps: calls fit_online() to retrain on accumulated
      training data (which now stores heuristic labels).
    """

    def __init__(self, retrain_freq: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.retrain_freq = retrain_freq
        self._last_retrain = 0

    def _on_training_start(self) -> None:
        """Initialise RF with heuristic labels before training begins."""
        try:
            env = self.training_env
            if hasattr(env, "envs"):
                env = env.envs[0]
            if hasattr(env, "unwrapped"):
                env = env.unwrapped

            regime_detector = getattr(env, "_regime_detector", None)
            df = getattr(env, "df", None)

            if regime_detector is not None and df is not None and len(df) > 100:
                regime_detector.fit_heuristic(df)
                if self.verbose > 0:
                    print(f"[RegimeRetrain] Initialised RF on {len(df)} bars")
        except Exception as e:
            if self.verbose > 0:
                print(f"[RegimeRetrain] Init skipped: {e}")

    def _on_step(self) -> bool:
        """Periodically retrain RF on accumulated heuristic data."""
        if self.num_timesteps - self._last_retrain < self.retrain_freq:
            return True

        try:
            env = self.training_env
            if hasattr(env, "envs"):
                env = env.envs[0]
            if hasattr(env, "unwrapped"):
                env = env.unwrapped

            regime_detector = getattr(env, "_regime_detector", None)
            if regime_detector is not None:
                regime_detector.fit_online()
                self._last_retrain = self.num_timesteps
                if self.verbose > 0:
                    print(f"[RegimeRetrain] Retrained RF at step {self.num_timesteps}")
        except Exception as e:
            if self.verbose > 0:
                print(f"[RegimeRetrain] Retrain skipped: {e}")

        return True
