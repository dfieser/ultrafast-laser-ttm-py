"""Validate radial_profile_solver against MATLAB golden fixtures.

The 'scale' mode (used by both fixtures) is built entirely from the
step-identical hand-rolled kernels (RK4 pulse phase, depth CN with k(T),
cylindrical radial CN), so agreement is limited only by libm/round-off drift
and the tolerances are as tight as the 0D solver's.
"""

from __future__ import annotations

import numpy as np
import pytest

from laserttm import radial_profile_solver

from .conftest import load_fixture

CASES = ["radial_profile_small", "radial_profile_baseline",
         "radial_profile_independent"]

SCALAR_CHECKS = [
    ("peakTeq_C", 1e-6, 1e-4),
    ("finalResid_C", 1e-6, 1e-4),
]

ARRAY_CHECKS = [
    ("rGrid_um", 1e-9, 1e-9),
    ("finalRadialProfile_C", 1e-6, 1e-4),
]


def _cfg_from_fixture(fx: dict, tmp_path) -> dict:
    cfg = dict(fx["cfg"])
    cfg["outputDir"] = str(tmp_path)
    cfg["makePlots"] = False
    cfg["saveFigures"] = False
    return cfg


@pytest.mark.parametrize("case", CASES)
def test_against_matlab_fixture(case, tmp_path):
    fx = load_fixture(case)
    ref = fx["results"]

    results = radial_profile_solver(_cfg_from_fixture(fx, tmp_path))

    assert results["solverId"] == "radial_profile"
    assert results["contractVersion"] == "v1"
    assert results["mode"] == ref["mode"]
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
        got = np.asarray(results[field], dtype=float).ravel()
        want = np.asarray(ref[field], dtype=float).ravel()
        assert got.shape == want.shape, f"{case}: {field}: shape mismatch"
        np.testing.assert_allclose(
            got, want, rtol=rtol, atol=atol, err_msg=f"{case}: {field}"
        )

    import ntpath
    import os

    assert os.path.exists(results["outputFile"])
    # ntpath handles the fixture's Windows path on every platform
    assert os.path.basename(results["outputFile"]) == \
        ntpath.basename(str(ref["outputFile"]))
