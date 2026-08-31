"""Validate scanning_beam_solver against MATLAB golden fixtures.

The per-pulse deposit uses the step-identical RK4 single-pulse response; the
depth CN and ADI sweeps are the same linear updates MATLAB performs with
sparse backslash, so agreement is limited by round-off accumulated over the
pulse train. The 36000-pulse baseline is marked slow (run with `-m slow`).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from laserttm import scanning_beam_solver

from .conftest import load_fixture

CASES = [
    pytest.param("scanning_small", id="scanning_small"),
    pytest.param("scanning_baseline", id="scanning_baseline",
                 marks=pytest.mark.slow),
]

SCALAR_CHECKS = [
    ("dTeq_single", 1e-6, 1e-4),
    ("pulseSpacing", 1e-12, 0.0),
]

ARRAY_CHECKS = [
    ("xGrid", 1e-12, 0.0),
    ("yGrid", 1e-12, 0.0),
    ("Tpeak_map", 1e-5, 1e-2),
    ("Tsurf", 1e-5, 1e-2),
    ("peakT_history", 1e-5, 1e-2),
]


@pytest.mark.parametrize("case", CASES)
def test_against_matlab_fixture(case, tmp_path):
    fx = load_fixture(case)
    ref = fx["results"]

    results = scanning_beam_solver(dict(fx["cfg"]), str(tmp_path),
                                   save_plots=False)

    assert results["solverId"] == "scanning_beam"
    assert results["contractVersion"] == "v1"
    assert int(results["nPulses"]) == int(ref["nPulses"])

    for field, rtol, atol in SCALAR_CHECKS:
        got = float(results[field])
        want = float(ref[field])
        err = abs(got - want)
        rel = err / max(abs(want), 1e-300)
        print(f"    {case}:{field}: abs err {err:.3e}, rel err {rel:.3e}")
        assert got == pytest.approx(want, rel=rtol, abs=atol), (
            f"{case}: {field}: python={got!r} matlab={want!r}"
        )

    for field, rtol, atol in ARRAY_CHECKS:
        got = np.asarray(results[field], dtype=float)
        want = np.asarray(ref[field], dtype=float)
        assert got.shape == want.shape, f"{case}: {field}: shape mismatch"
        err = np.max(np.abs(got - want))
        print(f"    {case}:{field}: max abs err {err:.3e}")
        np.testing.assert_allclose(got, want, rtol=rtol, atol=atol,
                                   err_msg=f"{case}: {field}")

    import ntpath

    assert os.path.exists(results["outputFile"])
    # ntpath handles the fixture's Windows path on every platform
    assert os.path.basename(results["outputFile"]) == \
        ntpath.basename(str(ref["outputFile"]))
    assert os.path.exists(results["matPath"])
