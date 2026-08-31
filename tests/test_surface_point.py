"""Validate surface_point_solver against MATLAB golden fixtures.

The MATLAB Phase 1/Phase 2 algorithms are ported step-identically, so
agreement should be limited by libm/round-off drift over ~1e5 adaptive
steps, far inside the tolerances asserted here. The projected steady state
goes through Nelder-Mead (fminsearch vs scipy), so it gets a looser bound.
"""

from __future__ import annotations

import numpy as np
import pytest

from laserttm import surface_point_solver

from .conftest import load_fixture

CASES = ["surface_point_baseline", "surface_point_cu", "surface_point_square",
         "surface_point_au"]

# (field, rtol, atol) — temperatures in deg C, energies in J/m^2
SCALAR_CHECKS = [
    ("peakTe_C", 1e-6, 1e-4),
    ("peakTl_C", 1e-6, 1e-4),
    ("finalResid_C", 1e-6, 1e-4),
    ("absorbedAreal_J_m2", 1e-6, 0.0),
    ("depthEnergy_J_m2", 1e-6, 0.0),
    ("projectedSteadyState_C", 1e-4, 1e-2),
]

ARRAY_CHECKS = [
    ("TeqVals_C", 1e-6, 1e-4),
    ("TresidVals_C", 1e-6, 1e-4),
]


def _cfg_from_fixture(fx: dict, tmp_path) -> dict:
    cfg = dict(fx["cfg"])
    cfg["outputDir"] = str(tmp_path)  # keep test output out of the fixtures dir
    cfg["makePlots"] = False
    cfg["saveFigures"] = False
    return cfg


@pytest.mark.parametrize("case", CASES)
def test_against_matlab_fixture(case, tmp_path):
    fx = load_fixture(case)
    ref = fx["results"]

    results = surface_point_solver(_cfg_from_fixture(fx, tmp_path))

    assert results["solverId"] == "surface_point"
    assert results["contractVersion"] == "v1"
    assert int(results["nPulses"]) == int(ref["nPulses"])
    assert int(results["peakPulse"]) == int(ref["peakPulse"])

    for field, rtol, atol in SCALAR_CHECKS:
        got = float(results[field])
        want = float(ref[field])
        assert got == pytest.approx(want, rel=rtol, abs=atol), (
            f"{case}: {field}: python={got!r} matlab={want!r}"
        )

    for field, rtol, atol in ARRAY_CHECKS:
        got = np.asarray(results[field], dtype=float).ravel()
        want = np.asarray(ref[field], dtype=float).ravel()
        assert got.shape == want.shape, f"{case}: {field}: shape mismatch"
        np.testing.assert_allclose(
            got, want, rtol=rtol, atol=atol, err_msg=f"{case}: {field}"
        )


def test_output_file_written(tmp_path):
    fx = load_fixture("surface_point_baseline")
    results = surface_point_solver(_cfg_from_fixture(fx, tmp_path))
    import ntpath
    import os

    assert os.path.exists(results["outputFile"])
    # Same filename construction as MATLAB (which wrote this fixture's file).
    # The fixture stores a Windows path; ntpath.basename splits it correctly
    # on every platform (os.path.basename would not on Linux).
    assert os.path.basename(results["outputFile"]) == \
        ntpath.basename(str(fx["results"]["outputFile"]))
