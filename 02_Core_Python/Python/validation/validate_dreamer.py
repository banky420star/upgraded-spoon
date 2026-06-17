"""
DreamerValidator — Validate Dreamer world-model predictions.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


class DreamerValidator:
    """
    Validates Dreamer world model on replay data.
    """

    def __init__(
        self,
        max_latent_loss: float = 0.15,
        max_reward_error: float = 0.20,
        max_drawdown_error: float = 0.25,
        min_regime_accuracy: float = 0.60,
        max_distribution_distance: float = 0.30,
        min_rollout_stability: float = 0.70,
        max_ruin_calibration_error: float = 0.20,
    ):
        self.max_latent_loss = float(max_latent_loss)
        self.max_reward_error = float(max_reward_error)
        self.max_drawdown_error = float(max_drawdown_error)
        self.min_regime_accuracy = float(min_regime_accuracy)
        self.max_distribution_distance = float(max_distribution_distance)
        self.min_rollout_stability = float(min_rollout_stability)
        self.max_ruin_calibration_error = float(max_ruin_calibration_error)

    def validate(
        self,
        real_states: pd.DataFrame,
        imagined_states: pd.DataFrame,
        real_rewards: pd.Series,
        imagined_rewards: pd.Series,
        real_drawdowns: pd.Series,
        imagined_drawdowns: pd.Series,
        real_regimes: pd.Series,
        imagined_regimes: pd.Series,
        rollout_stability_scores: Optional[List[float]] = None,
        ruin_probabilities: Optional[pd.Series] = None,
        actual_ruins: Optional[pd.Series] = None,
        dreamer_id: Optional[str] = None,
        bundle_id: str = "",
    ) -> Dict[str, Any]:
        """
        Run Dreamer validation suite.

        Returns:
            JSON-serializable artifact with validation metrics.
        """
        dreamer_id = dreamer_id or str(uuid.uuid4())

        # 1. Latent prediction loss (MSE between latent state representations)
        latent_loss = self._latent_prediction_loss(real_states, imagined_states)

        # 2. Reward prediction error (MAE normalized by std)
        reward_error = self._reward_prediction_error(real_rewards, imagined_rewards)

        # 3. Drawdown prediction error
        drawdown_error = self._drawdown_prediction_error(real_drawdowns, imagined_drawdowns)

        # 4. Regime transition accuracy
        regime_accuracy = self._regime_transition_accuracy(real_regimes, imagined_regimes)

        # 5. Distribution distance (Wasserstein-1 on returns)
        distribution_distance = self._distribution_distance(real_rewards, imagined_rewards)

        # 6. Rollout stability
        rollout_stability = self._rollout_stability(rollout_stability_scores)

        # 7. Ruin probability calibration
        ruin_calibration = self._ruin_calibration(ruin_probabilities, actual_ruins)

        checks = {
            "latent_prediction_loss": latent_loss <= self.max_latent_loss,
            "reward_prediction_error": reward_error <= self.max_reward_error,
            "drawdown_prediction_error": drawdown_error <= self.max_drawdown_error,
            "regime_transition_accuracy": regime_accuracy >= self.min_regime_accuracy,
            "imagined_vs_real_distribution_distance": distribution_distance <= self.max_distribution_distance,
            "rollout_stability": rollout_stability >= self.min_rollout_stability,
            "ruin_probability_calibration": ruin_calibration <= self.max_ruin_calibration_error,
        }

        passed = all(checks.values())
        status = "passed" if passed else "stub_disabled"

        return {
            "dreamer_id": dreamer_id,
            "bundle_id": bundle_id,
            "latent_prediction_loss": float(latent_loss),
            "reward_prediction_error": float(reward_error),
            "drawdown_prediction_error": float(drawdown_error),
            "regime_transition_accuracy": float(regime_accuracy),
            "imagined_vs_real_distribution_distance": float(distribution_distance),
            "rollout_stability": float(rollout_stability),
            "ruin_probability_calibration": float(ruin_calibration),
            "passed": bool(passed),
            "status": status,
            "checks": checks,
        }

    def _latent_prediction_loss(self, real: pd.DataFrame, imagined: pd.DataFrame) -> float:
        try:
            cols = [c for c in real.columns if c in imagined.columns]
            if not cols:
                return float("inf")
            diff = (real[cols].values - imagined[cols].values) ** 2
            return float(np.mean(diff))
        except Exception:
            return float("inf")

    def _reward_prediction_error(self, real: pd.Series, imagined: pd.Series) -> float:
        try:
            real_vals = real.dropna().values
            imagined_vals = imagined.dropna().values
            min_len = min(len(real_vals), len(imagined_vals))
            if min_len == 0:
                return float("inf")
            mae = np.mean(np.abs(real_vals[:min_len] - imagined_vals[:min_len]))
            std = np.std(real_vals) + 1e-12
            return float(mae / std)
        except Exception:
            return float("inf")

    def _drawdown_prediction_error(self, real: pd.Series, imagined: pd.Series) -> float:
        try:
            real_vals = real.dropna().values
            imagined_vals = imagined.dropna().values
            min_len = min(len(real_vals), len(imagined_vals))
            if min_len == 0:
                return float("inf")
            mae = np.mean(np.abs(real_vals[:min_len] - imagined_vals[:min_len]))
            std = np.std(real_vals) + 1e-12
            return float(mae / std)
        except Exception:
            return float("inf")

    def _regime_transition_accuracy(self, real: pd.Series, imagined: pd.Series) -> float:
        try:
            real_vals = real.dropna().astype(str).values
            imagined_vals = imagined.dropna().astype(str).values
            min_len = min(len(real_vals), len(imagined_vals))
            if min_len == 0:
                return 0.0
            return float(np.mean(real_vals[:min_len] == imagined_vals[:min_len]))
        except Exception:
            return 0.0

    def _distribution_distance(self, real: pd.Series, imagined: pd.Series) -> float:
        try:
            r = real.dropna().values
            i = imagined.dropna().values
            min_len = min(len(r), len(i))
            if min_len < 2:
                return float("inf")
            r = r[:min_len]
            i = i[:min_len]
            # Empirical Wasserstein-1 via scipy
            return float(stats.wasserstein_distance(r, i))
        except Exception:
            return float("inf")

    def _rollout_stability(self, scores: Optional[List[float]]) -> float:
        if not scores:
            return 0.0
        return float(np.mean(scores))

    def _ruin_calibration(self, probs: Optional[pd.Series], actual: Optional[pd.Series]) -> float:
        try:
            if probs is None or actual is None:
                return float("inf")
            p = probs.dropna().values
            a = actual.dropna().values
            min_len = min(len(p), len(a))
            if min_len == 0:
                return float("inf")
            # Calibration error = mean absolute difference between predicted ruin prob and actual ruin rate
            # We bin by predicted probability deciles
            df = pd.DataFrame({"prob": p[:min_len], "actual": a[:min_len]})
            df["bin"] = pd.qcut(df["prob"], q=min(10, len(df)), duplicates="drop")
            cal_errors = []
            for _, group in df.groupby("bin"):
                if len(group) > 0:
                    pred = group["prob"].mean()
                    obs = group["actual"].mean()
                    cal_errors.append(abs(pred - obs))
            return float(np.mean(cal_errors)) if cal_errors else float("inf")
        except Exception:
            return float("inf")
