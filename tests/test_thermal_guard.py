"""Tests for the thermal guard used by long-running batch jobs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "thermal_guard.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("thermal_guard_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_read_temps_returns_numbers_or_empty():
    module = _load_module()
    temps = module.read_temps()
    assert isinstance(temps, list)
    assert all(isinstance(t, float) for t in temps)
    assert all(t > -50 for t in temps), f"implausible reading: {temps}"


def test_max_temp_matches_read_temps():
    module = _load_module()
    temps = module.read_temps()
    observed = module.max_temp()
    if temps:
        assert observed == max(temps)
    else:
        assert observed is None


def test_guard_does_not_pause_when_cool(monkeypatch):
    """A cool machine must never stall the batch loop."""
    module = _load_module()
    monkeypatch.setattr(module, "max_temp", lambda: 40.0)

    guard = module.ThermalGuard(check_every=1, hot_c=84.0, verbose=False)
    assert guard.tick() is False
    assert guard.pause_count == 0
    assert guard.max_observed_c == 40.0


def test_guard_records_peak_temperature(monkeypatch):
    module = _load_module()
    guard = module.ThermalGuard(check_every=1, hot_c=200.0, verbose=False)
    for value in (50.0, 65.0, 61.0):
        monkeypatch.setattr(module, "max_temp", lambda v=value: v)
        guard.tick()
    assert guard.max_observed_c == 65.0


def test_guard_pauses_when_hot(monkeypatch):
    """Above the threshold the guard must wait, not keep grinding."""
    module = _load_module()
    monkeypatch.setattr(module, "max_temp", lambda: 95.0)

    waited = []
    monkeypatch.setattr(
        module, "wait_until_cool", lambda **kw: waited.append(kw) or True
    )

    guard = module.ThermalGuard(check_every=1, hot_c=84.0, verbose=False)
    assert guard.tick() is True
    assert guard.pause_count == 1
    assert waited, "wait_until_cool should be called when hot"


def test_guard_respects_check_every(monkeypatch):
    """Temperature is only read on the configured cadence."""
    module = _load_module()
    calls = []

    def _fake():
        calls.append(1)
        return 40.0

    monkeypatch.setattr(module, "max_temp", _fake)
    guard = module.ThermalGuard(check_every=5, hot_c=84.0, verbose=False)

    for _ in range(4):
        guard.tick()
    assert len(calls) == 0

    guard.tick()
    assert len(calls) == 1


def test_guard_survives_missing_sensor(monkeypatch):
    """No readable sensor must disable pausing, not crash the run."""
    module = _load_module()
    monkeypatch.setattr(module, "max_temp", lambda: None)

    guard = module.ThermalGuard(check_every=1, hot_c=84.0, verbose=False)
    assert guard.tick() is False
    assert guard.pause_count == 0
    assert guard.max_observed_c is None


def test_wait_until_cool_returns_false_when_cool(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "max_temp", lambda: 50.0)
    assert module.wait_until_cool(cool_c=72.0, verbose=False) is False


def test_wait_until_cool_waits_then_returns(monkeypatch):
    """Hot now, cool later: the call must report that it paused."""
    module = _load_module()
    readings = iter([95.0, 95.0, 60.0, 60.0, 60.0])
    monkeypatch.setattr(module, "max_temp", lambda: next(readings, 60.0))
    monkeypatch.setattr(module.time, "sleep", lambda _s: None)

    assert (
        module.wait_until_cool(
            cool_c=72.0, hot_c=84.0, poll_seconds=0, verbose=False
        )
        is True
    )


def test_wait_until_cool_gives_up(monkeypatch):
    """A stuck sensor must not hang a multi-hour run forever."""
    module = _load_module()
    monkeypatch.setattr(module, "max_temp", lambda: 99.0)
    monkeypatch.setattr(module.time, "sleep", lambda _s: None)

    ticks = iter([0.0] + [10_000.0] * 50)
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks, 10_000.0))

    assert (
        module.wait_until_cool(
            cool_c=72.0,
            poll_seconds=0,
            max_wait_seconds=5.0,
            verbose=False,
        )
        is True
    )