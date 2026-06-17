"""
Target-KL annealing callback for PPO.

Addresses the "dead-Gaussian" pathology where:
  - PPO's internal target_kl=0.08 is too conservative for a 1:3 RR env with
    heavily-penalized HOLD steps
  - clip_range=0.2 + 10 epochs drives the policy into a stable local minimum
    (mean ≈ 0, std = 1.0) within the first few updates
  - Once the policy is locally optimal at "tiny actions, never trade",
    approx_kl stays <0.005 and gradient signal vanishes (policy_gradient_loss ≈ 0)

Fix: linearly anneal target_kl from AGI_TARGET_KL_START (loose) to
AGI_TARGET_KL_END (tight) over the first AGI_TARGET_KL_ANNEAL_FRAC of training.
This forces PPO to make larger, more varied updates in early training, escaping
the HOLD-collapse basin, then settle into a tight local optimum near the end.

Usage:
    from Python.training.target_kl_anneal_callback import TargetKLAnnealCallback
    cb = TargetKLAnnealCallback(total_timesteps=50_000)
    model.learn(50_000, callback=cb)

Env vars (all optional, defaults match current "_default_ppo_params"):
    AGI_TARGET_KL_START       default 0.30  (loose; allows large early updates)
    AGI_TARGET_KL_END         default 0.05  (tight; stable final policy)
    AGI_TARGET_KL_ANNEAL_FRAC default 0.5   (anneal over first 50% of steps)

Why these defaults:
    - AGI_TARGET_KL_START=0.30 is well above SB3's default 0.05 and the
      "v6reward" run's 0.08. At 0.30 the policy can move ~30% per update
      without triggering early stop, which is the only way to escape a
      flat-Gaussian local minimum where actions are all near zero.
    - AGI_TARGET_KL_END=0.05 matches the proven conservative profile that
      produced stable exp_var=0.42-0.62 results in earlier runs.
    - 0.5 anneal fraction means the first 25k of a 50k run gets the loose
      target (escape HOLD basin), the last 25k gets the tight target
      (settle into a real policy).

SB3 internals note: PPO's target_kl attribute is read every update inside
train(). We assign model.target_kl = new_value at each _on_step; the change
takes effect on the *next* PPO update (this is fine — we only need gradual
annealing, not instant reaction).
"""
import os
from typing import Optional


try:
    from stable_baselines3.common.callbacks import BaseCallback
except Exception:  # pragma: no cover — soft import (matches style of other modules)
    BaseCallback = object  # type: ignore


class TargetKLAnnealCallback(BaseCallback):
    """
    Linearly anneal PPO's target_kl from a loose start value to a tight end value.

    The callback reads AGI_TARGET_KL_START / AGI_TARGET_KL_END / AGI_TARGET_KL_ANNEAL_FRAC
    at construction time (env vars are evaluated at launch, not per-step).
    """

    def __init__(
        self,
        total_timesteps: int,
        target_kl_start: Optional[float] = None,
        target_kl_end: Optional[float] = None,
        anneal_frac: Optional[float] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.total_timesteps = max(1, int(total_timesteps))
        self.target_kl_start = float(
            os.environ.get("AGI_TARGET_KL_START", "0.30")
            if target_kl_start is None
            else target_kl_start
        )
        self.target_kl_end = float(
            os.environ.get("AGI_TARGET_KL_END", "0.05")
            if target_kl_end is None
            else target_kl_end
        )
        self.anneal_frac = float(
            os.environ.get("AGI_TARGET_KL_ANNEAL_FRAC", "0.5")
            if anneal_frac is None
            else anneal_frac
        )
        # Clamp anneal_frac to (0, 1] — a non-positive fraction would never anneal,
        # a fraction >1 would anneal past the end of training.
        if self.anneal_frac <= 0.0:
            self.anneal_frac = 0.5
        elif self.anneal_frac > 1.0:
            self.anneal_frac = 1.0
        # Initial assignment: set the model to the start value on training start
        # so the very first PPO update uses the loose target.
        self._initial_value: Optional[float] = None
        self._last_value: Optional[float] = None

    def _on_training_start(self) -> None:
        # Apply the start value immediately so the first update uses the loose target.
        if self.model is not None:
            try:
                self.model.target_kl = float(self.target_kl_start)
                self._initial_value = float(self.target_kl_start)
                self._last_value = float(self.target_kl_start)
                if self.verbose > 0:
                    print(
                        f"[TargetKLAnneal] start: target_kl={self.target_kl_start:.3f} "
                        f"(will anneal to {self.target_kl_end:.3f} over {self.anneal_frac*100:.0f}% "
                        f"of {self.total_timesteps:,} steps)"
                    )
            except Exception:
                # If the model doesn't expose target_kl (shouldn't happen for PPO), bail silently
                # rather than crashing the training loop.
                pass

    def _on_step(self) -> bool:
        if self.model is None:
            return True
        try:
            current_total = int(getattr(self.model, "num_timesteps", 0) or 0)
        except Exception:
            return True

        # Progress through the anneal window: 0 at start, 1 at anneal_frac of total
        progress = current_total / max(1, self.total_timesteps)
        if progress >= self.anneal_frac:
            # Past the anneal window — hold the end value
            new_value = float(self.target_kl_end)
        else:
            # Linear interpolation from start → end
            t = progress / max(1e-9, self.anneal_frac)
            new_value = float(self.target_kl_start) + t * (float(self.target_kl_end) - float(self.target_kl_start))

        # Only write if it actually changed (avoid spamming SB3 internals)
        if self._last_value is None or abs(new_value - self._last_value) > 1e-6:
            try:
                self.model.target_kl = new_value
                self._last_value = new_value
            except Exception:
                return True
        return True


__all__ = ["TargetKLAnnealCallback"]
