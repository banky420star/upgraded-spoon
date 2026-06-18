"""LSTM gradient flow diagnostics - lightweight stub.

Provides two optional SB3-friendly callback-style classes:

* :class:`DiagnosticsCallback`  - periodic logger for gradient statistics
* :class:`LSTMGradientDiagnostics` - collects gradient / activation stats
  during :class:`stable_baselines3.PPO.PPO` training.

These classes are imported conditionally by ``training/train_drl.py`` inside a
``try / except``; a previous version of the repo referenced them without the
file existing, which crashed the DRL trainer outright. They live here as stubs
so ``train_drl.py`` can resolve the names and the trainer runs end-to-end.

Both classes accept arbitrary positional/keyword arguments (the trainer passes
``log_interval`` to ``DiagnosticsCallback`` and ``pretrain_loss_reduction`` to
``LSTMGradientDiagnostics``); concrete instrumentation can be added later
without changing the call sites.
"""

from __future__ import annotations

from typing import Any


class DiagnosticsCallback:
    """Lightweight no-op gradient-diagnostics callback.

    Stable-Baselines3-style callback compatible with :class:`CallbackList`.
    Accepts any arguments but only stores the ``log_interval`` so the trainer
    can inspect / override it later.
    """

    def __init__(
        self,
        *args: Any,
        log_interval: int = 1000,
        **kwargs: Any,
    ) -> None:
        self.log_interval = max(1, int(log_interval))
        self._extra = kwargs

    def on_step(self) -> bool:
        """Return True to continue training (never aborts)."""
        return True


class LSTMGradientDiagnostics:
    """Lightweight no-op stand-in for LSTM gradient stats collection.

    Holds the ``pretrain_loss_reduction`` choice (used by the trainer when it
    later logs gradient stats). Concrete implementation lives downstream; the
    trainer must remain runnable without it.
    """

    def __init__(
        self,
        *args: Any,
        pretrain_loss_reduction: str = "mean",
        **kwargs: Any,
    ) -> None:
        if pretrain_loss_reduction not in {"mean", "sum", "none"}:
            pretrain_loss_reduction = "mean"
        self.pretrain_loss_reduction = pretrain_loss_reduction
        self._extra = kwargs

    def collect(self, *_args: Any, **_kwargs: Any) -> dict:
        """Return a minimal stats payload so downstream code never sees None."""
        return {
            "available": False,
            "reason": "gradient_flow_analyzer stub; instrumentation not wired",
        }


__all__ = ["DiagnosticsCallback", "LSTMGradientDiagnostics"]
