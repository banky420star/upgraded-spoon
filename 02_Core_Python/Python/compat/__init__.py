"""
Compatibility shims for Chain Gambler.

This module provides compatibility fixes for different Python/NumPy/PyTorch versions.
Import at the top of entry points before using numpy/pytorch-dependent code.
"""

from .numpy_fix import ensure_numpy_compatibility

__all__ = ["ensure_numpy_compatibility"]