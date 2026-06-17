r"""Shared pytest infrastructure for 02_Core_Python/tests.

Responsibilities:
  1. Add 02_Core_Python/Python to sys.path at conftest-import time so tests can do
     ``from training_analyzer import TrainingAnalyzer, COOLDOWN_SECONDS`` without per-file
     sys.path gymnastics. Works for this directory and any subdir pytest descends into.
  2. Pre-import the most-used SUT symbols so collection-time ImportErrors surface early.
  3. Expose the three shared test helpers as pytest **fixtures**. Tests request them by name,
     e.g.::
         def test_x(monkeypatch, clock_mock, build_ollama_response, assert_cooldown_bounds):
     Each fixture returns the underlying class / callable; test bodies use the returned object
     as if it were inline.

Why fixtures rather than conftest module-level callables:
  - pytest auto-injects only *fixtures* into test function arguments. Naked callables at
    conftest module level are NOT auto-injected into adjacent tests; consumers would need a
    ``from conftest import ClockMock`` import - which conftest's namespace quirks discourage.
  - Each fixture returns the helper, so the test instantiates it (preserving the in-test
    ``monkeypatch.setattr(...)`` pattern that the existing test relied on).

Adding helpers later: define them at module level (the leading-underscore section below) +
add a one-line ``@pytest.fixture`` wrapper. Tests request by fixture name; nothing else changes.
"""

from __future__ import annotations

import pathlib
import sys
from unittest.mock import MagicMock

import pytest
import requests as _requests


# ===== 1. sys.path setup at conftest import time =================================================
_HERE = pathlib.Path(__file__).resolve()
_PYTHON_PKG = str(_HERE.parent.parent / "Python")
if _PYTHON_PKG not in sys.path:
    sys.path.insert(0, _PYTHON_PKG)


# ===== 2. NO module-level SUT pre-import -- conftest purpose is sys.path + fixtures ==========
# Pre-importing training_analyzer here would force EVERY test in tests/ to pay its load cost
# (requests, torch, broker SDKs, ...) and block collection-time if any dep is missing. Tests
# declare `from training_analyzer import TrainingAnalyzer` themselves; the sys.path mutation
# above is enough to make that work.


# ==================================================================================================
# Shared helper implementations (the leading-underscore names keep them out of test namespaces
# accidentally; the @pytest.fixture block below exposes the public-API aliases).
# ==================================================================================================


class _ClockMock:
    """Stateful stand-in for ``time.time``. Returns the same value between ``advance`` calls.

    Usage in a test::

        clock = clock_mock(start=10_000.0)
        monkeypatch.setattr("Python.training_analyzer.time.time", clock)
        clock.advance(30)   # next call to time.time() returns 10_030.0
    """

    __slots__ = ("_now",)

    def __init__(self, start: float = 10_000.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        """Move the clock forward by ``seconds`` (negative values are allowed)."""
        self._now += seconds


def _build_ollama_response(
    status_code: int = 200,
    payload=None,
    *,
    json_side_effect=None,
):
    """Build a ``MagicMock`` of ``requests.Response`` shaped like an Ollama /api/generate reply.

    - ``status_code >= 400`` -> ``raise_for_status`` raises ``requests.exceptions.HTTPError``
      (a ``RequestException``) so the SUT's ``except requests.exceptions.RequestException``
      branch fires without a real HTTP exchange.
    - ``status_code < 400``  -> ``raise_for_status`` returns ``None``; ``response.json()``
      returns ``{"response": payload}`` -- the exact shape that ``_call_ollama`` extracts via
      ``response.json().get("response", "")``.
    - ``json_side_effect``    -> set on ``response.json``; typically a ``ValueError`` to drive
      the SUT's ``except (ValueError,)`` (post-success branch) into the 15s short-cooldown path.
    """
    resp = MagicMock()
    resp.status_code = status_code
    if status_code >= 400:
        def _raise():
            raise _requests.exceptions.HTTPError(f"{status_code} Client Error")
        resp.raise_for_status.side_effect = _raise
    else:
        resp.raise_for_status.return_value = None
    if json_side_effect is not None:
        resp.json.side_effect = json_side_effect
    else:
        # Match Ollama /api/generate response shape: {"response": payload}
        resp.json.return_value = (
            {"response": payload} if payload is not None else {"response": "ok-text"}
        )
    return resp


def _assert_cooldown_bounds(
    ta,
    *,
    base_time: float,
    expected_offset: float,
    slack: float = 1.0,
) -> None:
    """Assert ``ta._ollama_cooldown_until`` is approximately ``base_time + expected_offset``.

    Slack (default 1.0s) absorbs integer-vs-float clock noise; raise it for CI under heavy load.
    """
    val = ta._ollama_cooldown_until
    lo = base_time + expected_offset - slack
    hi = base_time + expected_offset + slack
    assert lo <= val <= hi, (
        f"cooldown_until={val} not within [{lo}, {hi}] "
        f"(base={base_time}, expected+={expected_offset} +/- {slack})"
    )


# ==================================================================================================
# Fixture exports -- tests request these by name; the public API aliases mirror the
# implementation names without the underscore prefix.
# ==================================================================================================


@pytest.fixture
def clock_mock():
    """Returns the :class:`_ClockMock` class for tests to instantiate and monkeypatch."""
    return _ClockMock


@pytest.fixture
def build_ollama_response():
    """Returns :func:`_build_ollama_response` for tests to use (or override)."""
    return _build_ollama_response


@pytest.fixture
def assert_cooldown_bounds():
    """Returns :func:`_assert_cooldown_bounds` for tests to use."""
    return _assert_cooldown_bounds

# =====================================================================
# 5. loguru -> stdlib bridge so caplog sees TrainingAnalyzer emissions
# =====================================================================
# loguru bypasses the stdlib logging handler chain. We attach a stdlib
# `logging.Handler` instance to loguru as a SINK; loguru builds a
# stdlib LogRecord per emission and calls handler.emit(record). Our emit
# re-dispatches into logging.getLogger(record.name).handle(record), so
# the record climbs the stdlib hierarchy to root where caplog listens.
# Reasoning: this satisfies the user request to capture via caplog while
# leaving production logging unchanged (no app-side rewrite).
import logging  # stdlib logger used by the bridge
from loguru import logger as _loguru_logger

class _LoguruCaplogHandler(logging.Handler):
    """A stdlib handler used as a loguru sink; re-dispatches the record."""

    def emit(self, record):  # noqa: D401
        logging.getLogger(record.name).handle(record)

@pytest.fixture(autouse=True, scope="function")
def _loguru_caplog_bridge():
    """Bridge loguru emissions into stdlib for the lifetime of one test."""
    handler = _LoguruCaplogHandler(level=logging.DEBUG)
    handler_id = _loguru_logger.add(
        handler,
        level="DEBUG",
        format="{message}",
        backtrace=False,
        diagnose=False,
    )
    yield handler
    try:
        _loguru_logger.remove(handler_id)
    except ValueError:
        pass


# --- ADDED for test_api_server_health.py HERMETIC tests ---
# Fixture-only: does NOT gate the whole test directory. The hard gate
# is at module-scope in test_api_server_health.py; sibling cooldown +
# logging tests are unaffected.


@pytest.fixture
def live_state_factory(monkeypatch, tmp_path):
    """Write a controlled ``live_state.json`` to ``tmp_path/`` and
    monkeypatch ``Python.api_server.ROOT`` to point at it.

    Returns a factory: ``live_state_factory(content) -> Path`` where
    ``content`` is either a dict (auto-JSON-serialized) or a raw
    ``str`` (e.g. malformed JSON for fail-soft tests).
    """
    import json as _json
    import sys as _sys

    def _factory(content):
        if isinstance(content, dict):
            text_out = _json.dumps(content)
        else:
            text_out = content
        fp = tmp_path / "live_state.json"
        fp.write_text(text_out, encoding="utf-8")
        target = _sys.modules.get("Python.api_server")
        if target is not None:
            monkeypatch.setattr(target, "ROOT", str(tmp_path))
        return fp

    return _factory
