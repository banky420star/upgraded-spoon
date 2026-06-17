"""
NumPy Compatibility Shim

Fixes incompatibility between NumPy 1.x and 2.x pickle files.
Models saved with NumPy 2.x reference numpy._core, but NumPy 1.x uses numpy.core.

Usage:
    from Python.compat.numpy_fix import ensure_numpy_compatibility
    ensure_numpy_compatibility()

    # Or simply import the compat module:
    import Python.compat  # Applies fixes automatically
"""
import sys


def ensure_numpy_compatibility():
    """
    Create module aliases so pickle can find numpy._core in NumPy 1.x.
    Call this before loading pickled models.
    """
    try:
        import numpy as _np
        if not hasattr(_np, '_core'):
            # NumPy 1.x detected - create aliases for NumPy 2.x compatibility
            import numpy.core as _np_core
            sys.modules['numpy._core'] = _np_core
            sys.modules['numpy._core.numeric'] = _np_core.numeric
            sys.modules['numpy._core._multiarray_umath'] = _np_core._multiarray_umath
            return True
    except ImportError:
        pass
    return False


# Auto-apply on import for convenience
_COMPATIBILITY_APPLIED = ensure_numpy_compatibility()


if _COMPATIBILITY_APPLIED:
    # Log for debugging (but avoid circular import)
    try:
        import logging
        logging.debug("NumPy 1.x compatibility aliases applied for pickle compatibility")
    except:
        pass
