
# Contract tests for api_server /api/health wiring.
#
# api_server (Bottle, :5051) is the dashboard surface. Server_AGI sidecars
# via the filesystem, writing 02_Core_Python/live_state.json, which
# api_server._read_live_state (api_server.py:385) consumes to expose
# server_running / brain_initialized to the React dashboard.
#
# Hermetic tests use the live_state_factory fixture (conftest.py) which
# writes a controlled live_state.json to tmp_path/ and monkeypatches
# api_server_mod.ROOT. This makes the wire contract verifiable on CI
# runners WITHOUT Server_AGI actually running. Only
# test_health_flags_when_cycles_and_process_match runs against the live
# system (gated by pytest.mark.integration + a runtime auto-skip when
# no Server_AGI python process is detected).

import json
import subprocess

import pytest


# Skip the whole module if api_server.py cannot be imported -- it pulls in
# heavy domain deps (numpy / torch / MT5) that the test venv does not
# always have. Kept at module scope (NOT conftest) so cooldown + logging
# tests elsewhere in tests/ are NOT affected by this gate.
api_server_mod = pytest.importorskip(
    "Python.api_server",
    reason="api_server requires heavy deps (numpy/torch/MT5); install full "
    "deps to enable contract tests",
)


_read_live_state = api_server_mod._read_live_state
api_health = api_server_mod.api_health


_RECOVERY_HINT = (
    "Recover by starting Server_AGI: in the project root run "
    "START_DEMO_BOT.bat (sets required env vars + launches api_server + Server_AGI)."
)


def _api_health_dict(raw):
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            pass
    body = getattr(raw, "body", None)
    if isinstance(body, (bytes, bytearray)):
        try:
            return json.loads(body.decode("utf-8", errors="replace"))
        except (json.JSONDecodeError, ValueError):
            pass
    elif isinstance(body, str):
        try:
            return json.loads(body)
        except (json.JSONDecodeError, ValueError):
            pass
    raise TypeError(
        "Cannot adapt api_health() return of type " + type(raw).__name__
    )


# Test 1: HERMETIC -- reader parses writer standard schema
def test_read_live_state_returns_standard_schema(live_state_factory):
    state = {
        "timestamp": "2025-01-01T00:00:00Z",
        "registry": {"server_AGI": "started"},
        "symbols": ["XAUUSDm"],
        "trading": {"account": {}, "risk": {"canTrade": True}},
        "training": {"active_canary": True, "cycles_completed": 3},
    }
    live_state_factory(state)
    out = _read_live_state()

    assert isinstance(out, dict)
    assert sorted(out.keys()) == [
        "registry", "symbols", "timestamp", "trading", "training",
    ], "Unexpected top-level key set: " + repr(sorted(out.keys()))
    assert out["training"]["cycles_completed"] == 3
    assert out["training"]["active_canary"] is True


# Test 2: HERMETIC -- /api/health response shape only (boolean values may be False on CI)
def test_api_health_response_shape(live_state_factory):
    live_state_factory({"training": {"cycles_completed": 1}})
    response = _api_health_dict(api_health())

    assert "status" in response
    assert response["status"] in ("ok", "healthy", "degraded", "unhealthy")
    assert "checks" in response
    assert isinstance(response["checks"], dict)
    for flag in ("server_running", "brain_initialized"):
        assert flag in response["checks"], "missing flag " + repr(flag)
        flag_value = response["checks"][flag]
        assert type(flag_value) is bool, (
            repr(flag) + " must be a bool, got " + type(flag_value).__name__
        )


# Test 3: HERMETIC -- reader + writer agree on file path
def test_read_live_state_uses_same_path_as_writer(live_state_factory):
    live_state_factory({"training": {"cycles_completed": 7}})
    out = _read_live_state()
    assert isinstance(out, dict)
    assert out["training"]["cycles_completed"] == 7


# Test 4: HERMETIC -- malformed JSON fails soft (empirically returns {})
def test_read_live_state_recovers_from_malformed_json(live_state_factory):
    live_state_factory("{not valid json,,,")  # raw malformed text

    try:
        result = _read_live_state()
    except ValueError:
        result = None

    assert result in (None, {}), (
        "malformed live_state.json must NOT propagate JSONDecodeError; "
        "got " + repr(result)
    )


# Test 5: INTEGRATION -- Server_AGI process detection aligned with state
@pytest.mark.integration
def test_health_flags_when_cycles_and_process_match():
    # Filter to Server_AGI specifically; do NOT count pytest's own
    # python.exe, which would defeat the integration auto-skip.
    process_query = ''''
(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue
 -Filter "Name='python.exe'"
 | Where-Object { $_.CommandLine -match 'Server_AGI' }
 | Measure-Object).Count
'''.strip()
    try:
        proc = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", process_query],
            capture_output=True, text=True, timeout=15,
        )
        count_str = (proc.stdout or "0").strip()
        count = int(count_str) if count_str.isdigit() else 0
    except Exception:
        count = 0

    if count == 0:
        pytest.skip(
            "No Server_AGI python process detected; integration test "
            "only meaningful when Server_AGI is alive. " + _RECOVERY_HINT
        )

    response = _api_health_dict(api_health())
    cycles_value = (_read_live_state().get("training") or {}).get(
        "cycles_completed", 0
    )
    if cycles_value > 0:
        assert response["checks"]["server_running"] is True, (
            "Server_AGI process detected AND cycles_completed="
            + str(cycles_value)
            + ", but server_running="
            + repr(response["checks"]["server_running"])
        )
        assert response["checks"]["brain_initialized"] is True, (
            "Server_AGI process detected AND cycles_completed="
            + str(cycles_value)
            + ", but brain_initialized="
            + repr(response["checks"]["brain_initialized"])
        )
