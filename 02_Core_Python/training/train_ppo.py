#!/usr/bin/env python3
"""
train_ppo.py — Thin wrapper around train_drl.py for PPO training.
Ensures macOS/Wine compatibility by using mt5_compat instead of direct MetaTrader5.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Ensure mt5_compat is loaded before any direct MetaTrader5 import
from Python.mt5_compat import mt5  # noqa: F401

from training.train_drl import train_drl

if __name__ == "__main__":
    train_drl()
