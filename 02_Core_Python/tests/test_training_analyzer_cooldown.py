"""Locks the circuit-breaker cooldown contract for TrainingAnalyzer._call_ollama.

Shared helpers come from tests/conftest.py:
  - clock_mock            fixture returns the ClockMock class
  - build_ollama_response fixture returns the MagicMock-response builder
  - assert_cooldown_bounds fixture returns the bounds-assertion helper

Subject under test:
 02_Core_Python/Python/training_analyzer.py
  v4 _call_ollama contract (refresh-now cooldown):
    - cold start: success resets _ollama_cooldown_until to 0.0
    - raise_for_status HTTPError (404/500): caught, arms +60s, returns None
    - requests.ConnectionError: caught, arms +60s, returns None
    - json.decode ValueError: caught, arms +15s, returns None
    - within cooldown: short-circuits (no requests.post call)
    - slow call: time_post is fresh at the cooldown arm site (refresh-now)
"""
from __future__ import annotations

import pathlib, sys

import pytest
import requests  # used by test_04

# sys.path is configured by conftest.py (02_Core_Python/Python/ exposed at collection time)

from training_analyzer import COOLDOWN_SECONDS, TrainingAnalyzer  # noqa: E402


def _make_ta(monkeypatch, clock):
    monkeypatch.setattr("training_analyzer.time.time", clock)
    ta = TrainingAnalyzer()
    ta.ollama_url = "http://localhost:11434/api/generate"
    ta.model_name = "qwen3:4b"
    return ta





def test_01_cold_success_resets_cooldown(
    monkeypatch, clock_mock, build_ollama_response, assert_cooldown_bounds,
):
    """A successful 200 response resets _ollama_cooldown_until to 0.0."""
    clock = clock_mock(start=10_000.0)
    ta = _make_ta(monkeypatch, clock)
    ok = build_ollama_response(status_code=200, payload="hello")
    monkeypatch.setattr("training_analyzer.requests.post", lambda *a, **kw: ok)
    result = ta._call_ollama("anything")
    assert result == "hello"
    assert ta._ollama_cooldown_until == 0.0


def test_02_post_404_arms_cooldown_60s(
    monkeypatch, clock_mock, build_ollama_response, assert_cooldown_bounds,
):
    """A 404 triggers raise_for_status -> HTTPError; v4 catches it, arms +60s, returns None."""
    clock = clock_mock(start=10_000.0)
    ta = _make_ta(monkeypatch, clock)
    bad = build_ollama_response(status_code=404)
    monkeypatch.setattr("training_analyzer.requests.post", lambda *a, **kw: bad)
    result = ta._call_ollama("anything")
    assert result.startswith("Analysis unavailable: ollama unreachable") and "HTTPError" in result
    assert_cooldown_bounds(ta, base_time=10_000.0, expected_offset=float(COOLDOWN_SECONDS))


def test_03_post_500_arms_cooldown_60s(
    monkeypatch, clock_mock, build_ollama_response, assert_cooldown_bounds,
):
    """A 500 triggers raise_for_status -> HTTPError; v4 catches it, arms +60s, returns None."""
    clock = clock_mock(start=10_000.0)
    ta = _make_ta(monkeypatch, clock)
    bad = build_ollama_response(status_code=500)
    monkeypatch.setattr("training_analyzer.requests.post", lambda *a, **kw: bad)
    result = ta._call_ollama("anything")
    assert result.startswith("Analysis unavailable: ollama unreachable") and "HTTPError" in result
    assert_cooldown_bounds(ta, base_time=10_000.0, expected_offset=float(COOLDOWN_SECONDS))





def test_04_connection_error_arms_cooldown_60s(
    monkeypatch, clock_mock, assert_cooldown_bounds,
):
    """requests.ConnectionError is caught, cooldown +60s armed, returns None."""
    clock = clock_mock(start=10_000.0)
    ta = _make_ta(monkeypatch, clock)

    def boom(*a, **kw):
        raise requests.ConnectionError("simulated")
    monkeypatch.setattr("training_analyzer.requests.post", boom)
    result = ta._call_ollama("anything")
    assert result.startswith("Analysis unavailable: ollama unreachable") and "ConnectionError" in result
    assert_cooldown_bounds(ta, base_time=10_000.0, expected_offset=float(COOLDOWN_SECONDS))


def test_05_within_cooldown_short_circuits(
    monkeypatch, clock_mock, build_ollama_response, assert_cooldown_bounds,
):
    """Within cooldown window, _call_ollama returns the cache string without calling requests.post."""
    clock = clock_mock(start=10_000.0)
    ta = _make_ta(monkeypatch, clock)
    # 1. arm via 500
    bad = build_ollama_response(status_code=500)
    monkeypatch.setattr("training_analyzer.requests.post", lambda *a, **kw: bad)
    ta._call_ollama("anything")
    assert_cooldown_bounds(ta, base_time=10_000.0, expected_offset=float(COOLDOWN_SECONDS))
    # 2. advance inside the window
    clock.advance(5.0)
    called = {"n": 0}
    def guard(*a, **kw):
        called["n"] += 1
        return build_ollama_response(status_code=200, payload="should-not-fire")
    monkeypatch.setattr("training_analyzer.requests.post", guard)
    cached = ta._call_ollama("anything")
    assert called["n"] == 0, "requests.post must NOT be called during cooldown"
    assert isinstance(cached, str) and cached, "short-circuit returns a non-empty cache string"


def test_06_recovery_after_window_resets_cooldown(
    monkeypatch, clock_mock, build_ollama_response, assert_cooldown_bounds,
):
    """After window elapses, a successful call resets _ollama_cooldown_until to 0.0."""
    clock = clock_mock(start=10_000.0)
    ta = _make_ta(monkeypatch, clock)
    # arm
    bad = build_ollama_response(status_code=500)
    monkeypatch.setattr("training_analyzer.requests.post", lambda *a, **kw: bad)
    ta._call_ollama("anything")
    assert_cooldown_bounds(ta, base_time=10_000.0, expected_offset=float(COOLDOWN_SECONDS))
    # advance past window
    clock.advance(float(COOLDOWN_SECONDS) + 5.0)
    # recover
    ok = build_ollama_response(status_code=200, payload="recovered")
    monkeypatch.setattr("training_analyzer.requests.post", lambda *a, **kw: ok)
    result = ta._call_ollama("anything")
    assert result == "recovered"
    assert ta._ollama_cooldown_until == 0.0





def test_07_slow_call_uses_refresh_now_at_arm_site(
    monkeypatch, clock_mock, build_ollama_response, assert_cooldown_bounds,
):
    """Refresh-now regression guard: the cooldown offset uses fresh time.time()
    at the arm site, NOT the wall-clock value captured at method entry.
    Modeled by patching time.time() to advance itself by 12s per call.
    """
    clock = clock_mock(start=10_000.0)
    ta = _make_ta(monkeypatch, clock)

    def advancing_time():
        before = clock()
        clock.advance(12.0)
        return before + 12.0
    monkeypatch.setattr("training_analyzer.time.time", advancing_time)

    bad = build_ollama_response(status_code=500)
    monkeypatch.setattr("training_analyzer.requests.post", lambda *a, **kw: bad)
    ta._call_ollama("anything")
    # Method entry saw 10000.0; the +60s offset must be measured from the fresh
    # sample at the arm site, which advancing_time returns as +12s -> 10012.0.
    assert_cooldown_bounds(ta, base_time=10_024.0, expected_offset=float(COOLDOWN_SECONDS))


def test_08_post_success_json_decode_value_error_arms_15s(
    monkeypatch, clock_mock, build_ollama_response, assert_cooldown_bounds,
):
    """response.json() raising ValueError triggers the +15s (min(COOLDOWN_SECONDS, 15)) arm."""
    clock = clock_mock(start=10_000.0)
    ta = _make_ta(monkeypatch, clock)
    bad = build_ollama_response(
        status_code=200, json_side_effect=ValueError("bad json"),
    )
    monkeypatch.setattr("training_analyzer.requests.post", lambda *a, **kw: bad)
    result = ta._call_ollama("anything")
    assert result == "Analysis unavailable: malformed ollama response"
    assert_cooldown_bounds(
        ta, base_time=10_000.0, expected_offset=min(float(COOLDOWN_SECONDS), 15.0),
    )


def test_09_post_500_then_success_arms_then_resets_logs(
    monkeypatch, clock_mock, build_ollama_response, assert_cooldown_bounds,
):
    """End-to-end: arm via 500, advance past window, recover via 200; verify reset + payload."""
    clock = clock_mock(start=10_000.0)
    ta = _make_ta(monkeypatch, clock)
    # arm
    bad = build_ollama_response(status_code=500)
    monkeypatch.setattr("training_analyzer.requests.post", lambda *a, **kw: bad)
    ta._call_ollama("anything")
    assert_cooldown_bounds(ta, base_time=10_000.0, expected_offset=float(COOLDOWN_SECONDS))
    # advance past window (+1s buffer)
    clock.advance(float(COOLDOWN_SECONDS) + 1.0)
    # recover
    ok = build_ollama_response(status_code=200, payload="final")
    monkeypatch.setattr("training_analyzer.requests.post", lambda *a, **kw: ok)
    result = ta._call_ollama("anything")
    assert result == "final"
    assert ta._ollama_cooldown_until == 0.0

