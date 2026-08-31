"""Validate single_pulse_visualizer against the MATLAB golden fixture."""

from __future__ import annotations

import pytest

from laserttm import single_pulse_visualizer

from .conftest import load_fixture

SCALAR_CHECKS = [
    ("peakTe_C", 2e-3, 0.5),
    ("peakTl_C", 2e-3, 0.5),
    ("finalTe_C", 2e-3, 0.5),
    ("finalTl_C", 2e-3, 0.5),
    ("maxInv_C", 5e-2, 0.5),
]


def test_against_matlab_fixture(tmp_path):
    fx = load_fixture("single_pulse_baseline")
    ref = fx["results"]

    cfg = dict(fx["cfg"])
    cfg["outputDir"] = str(tmp_path)
    cfg["makePlots"] = False
    cfg["saveFigures"] = False

    results = single_pulse_visualizer(cfg)

    assert results["solverId"] == "single_pulse"
    assert results["contractVersion"] == "v1"
    assert bool(results["invDetected"]) == bool(ref["invDetected"])

    for field, rtol, atol in SCALAR_CHECKS:
        got = float(results[field])
        want = float(ref[field])
        err = abs(got - want)
        rel = err / max(abs(want), 1e-300)
        print(f"    single_pulse:{field}: abs err {err:.3e}, rel err {rel:.3e}")
        assert got == pytest.approx(want, rel=rtol, abs=atol), f"{field}"
