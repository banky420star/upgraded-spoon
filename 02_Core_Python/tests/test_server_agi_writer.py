# Contract tests for Server_AGI._write_live_state -- the writer side of
# the api_server <-> Server_AGI live_state.json wire.
#
# Server_AGI writes 02_Core_Python/live_state.json via os.path.join(BASE_DIR, "live_state.json").
# api_server._read_live_state (api_server.py:385) consumes it to expose
# server_running / brain_initialized to the React dashboard.
#
# The reader-side tests in test_api_server_health.py lock the CONSUMER
# contract. This file locks the PRODUCER contract so writer drift
# surfaces immediately on CI -- the contract is no longer inferred only
# from _read_live_state.
#
# Hermetic tests monkeypatch Server_AGI.BASE_DIR (writer side) AND
# api_server.ROOT (reader side, used by Test 4 round-trip) to the
# same tmp_path so the writer writes produce a file the reader can
# immediately consume.

import json
import types

import pytest

# Skip the whole module if Server_AGI.py cannot be imported -- it pulls
# in MT5/numpy/torch. Same gate pattern as the reader test file.
server_agi_mod = pytest.importorskip(
    "Python.Server_AGI",
    reason="Server_AGI requires heavy deps (MT5/numpy/torch); install "
    "full deps to enable writer contract tests",
)
# Round-trip test (Test 4) also needs the reader side. Both modules
# must succeed for any of these tests to even run.
api_server_mod = pytest.importorskip(
    "Python.api_server",
    reason="Round-trip test needs api_server module; install to enable",
)

# Gate numpy behind importorskip -- the stub below exercises the writer's
# _json_default fallback by donating a numpy scalar to a nested dict.
np = pytest.importorskip("numpy")

_write_live_state = server_agi_mod._write_live_state
_read_live_state = api_server_mod._read_live_state

EXPECTED_TOP_KEYS = ["registry", "symbols", "timestamp", "training", "trading"]


@pytest.fixture
def writer_base_dir(tmp_path, monkeypatch):
    # Monkeypatch BOTH sides: writer (Server_AGI.BASE_DIR) AND reader
    # (api_server.ROOT) to the same tmp_path. Tests assert on tmp_path /
    # live_state.json immediately after the writer runs.
    monkeypatch.setattr(server_agi_mod, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(api_server_mod, "ROOT", str(tmp_path))
    return tmp_path


# -- Stub builders (types.SimpleNamespace = getattr-friendly, no class) --

def _stub_risk():
    return types.SimpleNamespace(
        _mt5_balance=10000.0,
        _current_equity=10500.0,
        canTrade=True,
    )


def _stub_symbols():
    return ["XAUUSDm"]


def _stub_last_symbol_state():
    # Deliberately include a numpy float64 so the writer's _json_default
    # fallback is exercised on this run. A regression in _json_default
    # would cause the writer to silently fail (try/except + logger.warning)
    # and Test A's file-exists assertion catches the latent crash site.
    return {
        "XAUUSDm": {
            "signal": "buy",
            "regime": "trending",
            "score": np.float64(0.8734),
        },
    }


def _stub_models():
    return {
        "XAUUSDm": {
            "champion": {"model_id": "v3", "sharpe": 1.2},
            "canary": {"model_id": "v4", "sharpe": 1.0},
        },
    }


def _stub_training_state(active=True, cycles=42):
    return {
        "active_canary": active,
        "cycles_completed": cycles,
        "observed": 100,
    }


# Test A: HERMETIC -- writer drops the file at BASE_DIR/live_state.json
# and parse-roundtrip succeeds.
def test_write_live_state_writes_to_base_dir(writer_base_dir):
    tmp = writer_base_dir
    _write_live_state(
        _stub_risk(),
        _stub_symbols(),
        _stub_last_symbol_state(),
        _stub_models(),
        _stub_training_state(),
    )
    fp = tmp / "live_state.json"
    # Existence assertion FIRST: writer wraps its body in try/except and
    # silently logs warnings on failure. Without this assert a regression
    # in _json_default (for example) would surface as a parse AssertionError
    # downstream, hiding the real crash site.
    assert fp.exists()
    parsed = json.loads(fp.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    assert sorted(parsed.keys()) == sorted(EXPECTED_TOP_KEYS)


# Test B: HERMETIC -- full args; deep-check that the writer preserves
# theta values end-to-end.
def test_write_live_state_emits_standard_writer_schema(writer_base_dir):
    tmp = writer_base_dir
    cycles = 42
    _write_live_state(
        _stub_risk(),
        _stub_symbols(),
        _stub_last_symbol_state(),
        _stub_models(),
        _stub_training_state(active=True, cycles=cycles),
    )
    fp = tmp / "live_state.json"
    assert fp.exists()
    parsed = json.loads(fp.read_text(encoding="utf-8"))

    # Theta-by-theta equality -- this IS the contract being locked.
    assert parsed["training"]["active_canary"] is True
    assert parsed["training"]["cycles_completed"] == cycles
    # canTrade propagates from risk.canTrade into trading.risk.canTrade
    assert parsed["trading"]["risk"]["canTrade"] is True
    # timestamp is numeric (time.time output)
    assert isinstance(parsed["timestamp"], (int, float))
    # Symbol round-tripped into the symbols payload
    if isinstance(parsed["symbols"], dict):
        assert "XAUUSDm" in parsed["symbols"]
    else:
        assert "XAUUSDm" in [str(s) for s in parsed["symbols"]]


# Test C: HERMETIC -- training_state=None must not crash; training key
# is always present and is a dict (writer internal default -- exact key
# set is implementation detail and not part of the public contract).
def test_write_live_state_default_when_training_state_is_None(writer_base_dir):
    tmp = writer_base_dir
    _write_live_state(
        _stub_risk(),
        _stub_symbols(),
        _stub_last_symbol_state(),
        _stub_models(),
        training_state=None,
    )
    fp = tmp / "live_state.json"
    assert fp.exists()
    parsed = json.loads(fp.read_text(encoding="utf-8"))
    assert "training" in parsed
    val = parsed["training"]
    assert isinstance(val, dict), (
        "writer must always emit training as a dict, even when "
        "training_state=None; got " + type(val).__name__
    )


# Test D: HERMETIC -- writer writes, reader reads back, values preserved
# end-to-end through the same tmp_path.
def test_writer_then_reader_round_trip(writer_base_dir):
    tmp = writer_base_dir
    cycles = 17
    _write_live_state(
        _stub_risk(),
        _stub_symbols(),
        _stub_last_symbol_state(),
        _stub_models(),
        _stub_training_state(active=False, cycles=cycles),
    )
    out = _read_live_state()
    assert isinstance(out, dict)
    assert out["training"]["cycles_completed"] == cycles
    assert out["training"]["active_canary"] is False
