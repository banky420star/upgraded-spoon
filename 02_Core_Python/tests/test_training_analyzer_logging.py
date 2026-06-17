"""End-to-end tests locking TrainingAnalyzer._call_ollama's no-spam logging contract.

Captures loguru emissions via pytest's caplog fixture (the loguru -> stdlib bridge
in tests/conftest.py reroutes them into stdlib logging). Contract under test:
  - First failure emits exactly ONE warning containing "Muting further Ollama calls".
  - During cooldown, subsequent calls emit zero NEW such warnings and one
    DEBUG "suppressed by cooldown" line per suppressed call.
  - First successful call after the cooldown expires emits exactly ONE
    INFO "Ollama recovered".
  - A json-decode ValueError failure emits exactly ONE warning
    "JSON decode failed" and arms a 15-second cooldown (NOT the default 60 s).
"""
import logging
import pytest

# conftest.py owns the sys.path setup; this import works because of that.
import training_analyzer  # noqa: F401  -- path comes from tests/conftest.py


COOLDOWN_SECONDS = getattr(training_analyzer, 'COOLDOWN_SECONDS', 60)


def make_ta(monkeypatch, clock):
    """Instantiate TrainingAnalyzer with time stubbed by `clock` (a ClockMock)."""
    ta = training_analyzer.TrainingAnalyzer.__new__(training_analyzer.TrainingAnalyzer)
    ta._ollama_cooldown_until = 0.0
    ta.model = getattr(training_analyzer, 'DEFAULT_MODEL', 'qwen3:4b')
    monkeypatch.setattr('time.time', clock)
    return ta



def count_msg(caplog, level_name, substring):
    """Count caplog records whose level+message substring match."""
    return sum(
        1 for r in caplog.records
        if r.levelname == level_name and substring in r.getMessage()
    )


def test_01_first_failure_fires_one_muting_warning(
    caplog, monkeypatch, clock_mock, build_ollama_response, assert_cooldown_bounds,
):
    """A single HTTPError (500) triggers exactly one Muting WARNING and arms +60 s."""
    clock = clock_mock(start=10_000.0)
    ta = make_ta(monkeypatch, clock)
    bad = build_ollama_response(status_code=500)
    monkeypatch.setattr('training_analyzer.requests.post', lambda *a, **kw: bad)

    with caplog.at_level(logging.DEBUG, logger='training_analyzer'):
        ta._call_ollama('anything')

    assert_cooldown_bounds(ta, base_time=10_000.0, expected_offset=float(COOLDOWN_SECONDS))
    muting = count_msg(caplog, 'WARNING', 'Muting further Ollama calls')
    assert muting == 1, f'lifetime Muting WARNING count: got {muting}, want 1'


def test_02_no_new_warnings_during_cooldown_window(
    caplog, monkeypatch, clock_mock, build_ollama_response,
):
    """After arming, 5 in-window calls emit zero NEW Muting WARNINGS and
    exactly 5 suppressed-by-cooldown DEBUG lines."""
    clock = clock_mock(start=10_000.0)
    ta = make_ta(monkeypatch, clock)
    bad = build_ollama_response(status_code=500)
    posts = []

    def fake_post(*a, **kw):
        posts.append('post')
        return bad

    monkeypatch.setattr('training_analyzer.requests.post', fake_post)

    with caplog.at_level(logging.DEBUG, logger='training_analyzer'):
        ta._call_ollama('anything')
        assert count_msg(caplog, 'WARNING', 'Muting further Ollama calls') == 1
        for _tick in range(2, 7):
            clock.advance(1.0)
            ta._call_ollama('anything')

    assert len(posts) == 1, (
        'only the arming call may hit requests.post during cooldown'
    )
    assert count_msg(caplog, 'WARNING', 'Muting further Ollama calls') == 1
    assert count_msg(caplog, 'DEBUG', 'suppressed by cooldown') == 5


def test_03_recovery_fires_one_recovered_info(
    caplog, monkeypatch, clock_mock, build_ollama_response,
):
    """First successful call AFTER the cooldown expires fires exactly 1 recovered-INFO."""
    clock = clock_mock(start=10_000.0)
    ta = make_ta(monkeypatch, clock)
    bad = build_ollama_response(status_code=500)
    ok = build_ollama_response(status_code=200, payload='hello')
    responses = iter([bad, ok])
    monkeypatch.setattr('training_analyzer.requests.post', lambda *a, **kw: next(responses))

    with caplog.at_level(logging.DEBUG, logger='training_analyzer'):
        ta._call_ollama('anything')
        clock.advance(float(COOLDOWN_SECONDS) + 5.0)
        result = ta._call_ollama('anything')

    assert result == 'hello'
    assert ta._ollama_cooldown_until == 0.0
    assert count_msg(caplog, 'INFO', 'Ollama recovered') == 1


def test_04_json_decode_value_error_emits_one_warning_with_15s(
    caplog, monkeypatch, clock_mock, build_ollama_response, assert_cooldown_bounds,
):
    """json.ValueError arms a 15-second (NOT default 60 s) cooldown and
    one dedicated JSON-decode WARNING."""
    clock = clock_mock(start=10_000.0)
    ta = make_ta(monkeypatch, clock)
    json_value_error = build_ollama_response(
        status_code=200,
        payload=None,
        json_side_effect=ValueError('bad json'),
    )
    monkeypatch.setattr('training_analyzer.requests.post', lambda *a, **kw: json_value_error)

    with caplog.at_level(logging.DEBUG, logger='training_analyzer'):
        ta._call_ollama('anything')

    assert_cooldown_bounds(ta, base_time=10_000.0, expected_offset=15.0)
    json_warns = count_msg(caplog, 'WARNING', 'JSON decode failed')
    assert json_warns == 1
    assert count_msg(caplog, 'WARNING', 'Muting further Ollama calls') == 0
