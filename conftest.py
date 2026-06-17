"""Repo-root conftest.py — inserted onto sys.path for all tests.

Many tests in this repo (in both `tests/` and `02_Core_Python/tests/`) import
from `Python.*` and `training.*`, which are subpackages of `02_Core_Python/`.
This conftest puts that directory on sys.path so pytest can collect the tests
regardless of which subdirectory the user invokes pytest from.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_CORE_PYTHON = _REPO_ROOT / "02_Core_Python"

if _CORE_PYTHON.is_dir() and str(_CORE_PYTHON) not in sys.path:
    sys.path.insert(0, str(_CORE_PYTHON))

# Skip tests that import modules that no longer exist in the codebase.
# These are orphan tests from prior refactors; they fail at collection and
# abort the run. Add the path relative to repo root.
collect_ignore_glob = [
    "02_Core_Python/tests/test_sync_rainforest_model.py",  # imports deleted `training.sync_rainforest_model`
]
