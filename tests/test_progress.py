"""The progress reporter must never interfere with a run."""

import time

from laserttm.progress import ProgressReporter, _format_duration


def test_disabled_reporter_is_inert():
    p = ProgressReporter(1000, enabled=False)
    for i in range(1000):
        p.update(i + 1)
    p.close()
    assert p._root is None


def test_auto_mode_off_under_pytest():
    # pytest captures stdout, so auto mode must resolve to disabled and
    # never open a window during test runs.
    p = ProgressReporter(10, enabled=None)
    p.update(5)
    p.close()
    assert p._root is None


def test_forced_reporter_survives_headless(monkeypatch):
    # Forcing enabled=True on a machine with no display must degrade
    # gracefully rather than raise. Simulate headless by making the
    # tkinter import fail.
    import sys

    monkeypatch.setitem(sys.modules, "tkinter", None)
    p = ProgressReporter(10, enabled=True, create_delay=0.0,
                         min_interval=0.0)
    time.sleep(0.01)
    for i in range(10):
        p.update(i + 1)
    p.close()
    assert p.enabled is False
    assert p._root is None


def test_format_duration():
    assert _format_duration(0) == "0:00"
    assert _format_duration(65) == "1:05"
    assert _format_duration(3725) == "1:02:05"


def test_solver_accepts_show_progress(tmp_path):
    from laserttm import surface_point_solver

    results = surface_point_solver({
        "Pavg": 10, "f_rep": 5e6, "simDuration": 2 / 5e6,
        "makePlots": False, "showProgress": False,
        "outputDir": str(tmp_path),
    })
    assert results["nPulses"] == 2
