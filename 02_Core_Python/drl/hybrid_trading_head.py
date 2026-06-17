"""
Hybrid Trading Head - Parameterized Action Decoder

Decodes a compact continuous action vector (6 dims, [-1, 1]) into
structured trading parameters: side, entry/exit threshold deltas,
size fraction, TP/SL offsets.

This is the env-side decoder approach (fastest path inside SB3 PPO):
- PPO outputs a standard Box(6,) action
- TradingEnv decodes it into trading semantics
- No custom SB3 policy needed

Reference: plan's bounded-Box decoder design.
"""

from __future__ import annotations

import math
import numpy as np

# Action vector layout (6 dims, each in [-1, 1])
# [side_score, entry_delta, exit_delta, size_raw, tp_delta, sl_delta]

# Action dimension
PARAM_ACTION_DIM = 6

# Decoding constants
SIDE_THRESHOLD = 0.33
ENTRY_DELTA_SCALE = 0.005
EXIT_DELTA_SCALE = 0.005
TP_DELTA_SCALE = 0.01
SL_DELTA_SCALE = 0.01
SIZE_SIGMOID_GAIN = 3.0

DEFAULT_BASE_ENTRY = 0.001
DEFAULT_BASE_EXIT = 0.002
DEFAULT_TP = 0.008
DEFAULT_SL = 0.004


def decode_parameterized_action(
    action: np.ndarray,
    base_entry_threshold: float = DEFAULT_BASE_ENTRY,
    base_exit_threshold: float = DEFAULT_BASE_EXIT,
    base_tp: float = DEFAULT_TP,
    base_sl: float = DEFAULT_SL,
) -> dict:
    action = np.asarray(action, dtype=np.float64).flatten()
    if len(action) < PARAM_ACTION_DIM:
        action = np.pad(action, (0, PARAM_ACTION_DIM - len(action)), mode="constant")

    side_score = float(np.clip(action[0], -1.0, 1.0))
    entry_delta = float(np.clip(action[1], -1.0, 1.0)) * ENTRY_DELTA_SCALE
    exit_delta = float(np.clip(action[2], -1.0, 1.0)) * EXIT_DELTA_SCALE
    size_raw = float(np.clip(action[3], -1.0, 1.0))
    tp_delta = float(np.clip(action[4], -1.0, 1.0)) * TP_DELTA_SCALE
    sl_delta = float(np.clip(action[5], -1.0, 1.0)) * SL_DELTA_SCALE

    if side_score > SIDE_THRESHOLD:
        side = "long"
    elif side_score < -SIDE_THRESHOLD:
        side = "short"
    else:
        side = "flat"

    size_fraction = 1.0 / (1.0 + math.exp(-SIZE_SIGMOID_GAIN * size_raw))
    entry_threshold = max(0.0001, base_entry_threshold + entry_delta)
    exit_threshold = max(0.0001, base_exit_threshold + exit_delta)
    tp = max(0.0001, base_tp + tp_delta)
    sl = max(0.0001, base_sl + sl_delta)

    return {
        "side": side,
        "side_score": side_score,
        "entry_threshold": entry_threshold,
        "exit_threshold": exit_threshold,
        "size_fraction": size_fraction,
        "tp": tp,
        "sl": sl,
        "raw_action": action.copy(),
    }


def compute_action_space():
    import gymnasium as gym
    return gym.spaces.Box(low=-1.0, high=1.0, shape=(PARAM_ACTION_DIM,), dtype=np.float32)
