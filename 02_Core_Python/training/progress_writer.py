"""Atomic training progress writer for API consumption."""
import json
import os
import time

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")

_MAX_RETRIES = 3
_RETRY_DELAY = 0.1  # seconds


def update_training_progress(trainer_key, data, symbol=None):
    """Write progress for a single trainer to its own JSON file.

    Args:
        trainer_key: "lstm", "ppo", or "dreamer"
        data: dict with progress fields (running, symbol, epoch, loss, etc.)
        symbol: optional per-symbol key to avoid file contention during parallel training
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    if symbol and trainer_key == "ppo":
        path = os.path.join(LOGS_DIR, f"ppo_{symbol}_progress.json")
    else:
        path = os.path.join(LOGS_DIR, f"{trainer_key}_progress.json")
    payload = {**data, "updated_at": time.time()}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    # On Windows, os.replace can fail with PermissionError if another process
    # is reading the file. Retry with a short delay.
    for attempt in range(_MAX_RETRIES):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAY)
            else:
                # Fallback: write directly (non-atomic but better than crashing)
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=2)
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except Exception:
                    pass


# === Training Health Signals (for supervisor, TUI, robust recovery) ===
# Added 2026-05-27 by Training Robustness & Recovery Agent.
# Provides clear, machine-readable "training health" for detecting stalls early
# and enabling bounded auto-recovery with conservative hyperparams.
# Consumers: vps_agi_supervisor.ps1, monitor_tui.py, api_server.py, launchers.
#
# v5+ extension (TUI & Diagnostics Integration): richer live metrics supported
# (approx_kl, explained_variance, loss, ep_rew_mean) via update_live_training_metrics()
# and **extra on heartbeat. Powers real-time KL/loss/reward visibility in TUI
# Deep Dive + "Training" observer card for post-fix runs.


_TRAINING_HEALTH_PATH = os.path.join(LOGS_DIR, "training_health.json")


def update_training_health(data: dict) -> None:
    """Atomically write centralized training health signal.

    Keys (all optional, sensible defaults applied):
      status: "running" | "stalled" | "failed" | "completed" | "recovering"
      current_step: int
      total_timesteps: int
      pct_complete: float
      symbol: str
      last_error: str | None
      conservative_params: bool
      recovery_attempts: int
      early_exit_diagnostics: dict
      last_heartbeat (auto)
      # v5+ richer live signals (emitted by PPOProgressCallback in train_drl + v5 launchers):
      approx_kl: float
      explained_variance: float
      loss: float
      ep_rew_mean: float  # reward health snapshot
      live_metrics: dict  # {'approx_kl': , 'explained_variance':, 'loss':, 'ep_rew_mean': }
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    now = time.time()
    base = {
        "status": "running",
        "last_heartbeat": now,
        "current_step": 0,
        "total_timesteps": 0,
        "pct_complete": 0.0,
        "symbol": None,
        "last_error": None,
        "conservative_params": True,
        "recovery_attempts": 0,
        "early_exit_diagnostics": {},
        "updated_at": now,
    }
    payload = {**base, **(data or {})}
    payload["last_heartbeat"] = now  # always refresh on update
    payload["updated_at"] = now
    tmp = _TRAINING_HEALTH_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    for attempt in range(_MAX_RETRIES):
        try:
            os.replace(tmp, _TRAINING_HEALTH_PATH)
            return
        except PermissionError:
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAY)
            else:
                try:
                    with open(_TRAINING_HEALTH_PATH, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=2)
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except Exception:
                    pass


def read_training_health() -> dict:
    """Read latest training health signal. Returns {} if missing/stale."""
    try:
        if os.path.exists(_TRAINING_HEALTH_PATH):
            with open(_TRAINING_HEALTH_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Consider stale if no heartbeat in last 20 minutes during a run
            if data.get("status") in ("running", "recovering") and (time.time() - data.get("last_heartbeat", 0)) > 1200:
                data["status"] = "stalled"
            return data
    except Exception:
        pass
    return {}


def mark_training_heartbeat(step: int = None, total: int = None, symbol: str = None, **extra):
    """Convenience: heartbeat + optional step update (called from progress callbacks)."""
    data = {"status": "running"}
    if step is not None:
        data["current_step"] = int(step)
    if total is not None:
        data["total_timesteps"] = int(total)
        if total > 0 and step is not None:
            data["pct_complete"] = round(100.0 * step / total, 2)
    if symbol:
        data["symbol"] = symbol
    if extra:
        data.update(extra)
    update_training_health(data)


def mark_training_failed(error: str, diagnostics: dict | None = None):
    """Mark training health as failed with diagnostics for supervisor recovery."""
    update_training_health({
        "status": "failed",
        "last_error": str(error)[:500],
        "early_exit_diagnostics": diagnostics or {},
    })


def mark_training_completed(symbol: str = None, final_metrics: dict | None = None):
    """Mark successful completion."""
    data = {"status": "completed"}
    if symbol:
        data["symbol"] = symbol
    if final_metrics:
        data["final_metrics"] = final_metrics
    update_training_health(data)


def mark_training_recovering(attempt: int):
    """Signal that supervisor/launcher is initiating bounded recovery."""
    update_training_health({
        "status": "recovering",
        "recovery_attempts": int(attempt),
    })


def update_live_training_metrics(
    approx_kl: float | None = None,
    explained_variance: float | None = None,
    loss: float | None = None,
    ep_rew_mean: float | None = None,
    step: int | None = None,
    **extra
) -> None:
    """Dedicated v5+ hook for richer PPO callback signals (approx_kl, explained_variance, loss + reward health).

    These are emitted live by the PPOProgressCallback during post-fix runs (v5 launcher etc.).
    Stores both top-level (easy consumption) and under 'live_metrics' sub-dict.
    Also usable by TUI diagnostics or external writers if desired.
    Safe no-op if all None.
    """
    data: dict = {}
    live = {}
    if approx_kl is not None:
        data["approx_kl"] = float(approx_kl)
        live["approx_kl"] = float(approx_kl)
    if explained_variance is not None:
        data["explained_variance"] = float(explained_variance)
        live["explained_variance"] = float(explained_variance)
    if loss is not None:
        data["loss"] = float(loss)
        live["loss"] = float(loss)
    if ep_rew_mean is not None:
        data["ep_rew_mean"] = float(ep_rew_mean)
        live["ep_rew_mean"] = float(ep_rew_mean)
    if step is not None:
        data["current_step"] = int(step)
    if extra:
        data.update(extra)
    if live:
        data["live_metrics"] = live
    if data:
        update_training_health(data)
