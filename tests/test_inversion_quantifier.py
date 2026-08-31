"""Validate inversion_quantifier against the MATLAB golden fixture.

The quantifier is a statistics layer over the depth solver, so tolerances
mirror the depth-profile test: stiff-integrator-level agreement on the
underlying per-pulse data, and correspondingly bounded statistics.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from laserttm import inversion_quantifier

from .conftest import load_fixture

SCALAR_CHECKS = [
    ("meanInv_K", 5e-2, 0.5),
    ("maxInv_K", 5e-2, 0.5),
    ("minInv_K", 5e-2, 0.5),
    ("stdInv_K", 2e-1, 0.3),
    ("invSlope_KperPulse", 2e-1, 0.05),
    ("corrBaseTempInv", 1e-2, 5e-3),
    ("meanInvFraction", 5e-2, 1e-3),
    ("peakTe_C", 2e-3, 0.5),
    ("peakTl_C", 2e-3, 0.5),
    ("finalResid_C", 2e-3, 0.5),
]

ARRAY_CHECKS = [
    ("invMaxPerPulse_K", 5e-2, 0.5),
    ("TePeak_C", 2e-3, 0.5),
    ("TlPeak_C", 2e-3, 0.5),
    ("Tbase_C", 2e-3, 0.5),
    ("Teq_C", 2e-3, 0.5),
    ("Tresid_C", 2e-3, 0.5),
    ("tMaxInv_s", 5e-2, 2e-12),
    ("tOnset_s", 5e-2, 2e-12),
    ("invDuration_s", 5e-2, 5e-12),
    ("Te_atMaxInv_C", 2e-3, 2.0),
    ("Tl_atMaxInv_C", 2e-3, 2.0),
]


def test_against_matlab_fixture(tmp_path):
    fx = load_fixture("inversion_baseline")
    ref = fx["results"]

    cfg = dict(fx["cfg"])
    cfg["outputDir"] = str(tmp_path)
    cfg["makePlots"] = False
    cfg["saveFigures"] = False

    results = inversion_quantifier(cfg)

    assert results["solverId"] == "inversion_quantifier"
    assert results["contractVersion"] == "v1"
    assert int(results["nPulses"]) == int(ref["nPulses"])
    assert int(results["nInvPulses"]) == int(ref["nInvPulses"])

    for field, rtol, atol in SCALAR_CHECKS:
        got = float(results[field])
        want = float(ref[field])
        err = abs(got - want)
        rel = err / max(abs(want), 1e-300)
        print(f"    inversion:{field}: abs err {err:.3e}, rel err {rel:.3e}")
        assert got == pytest.approx(want, rel=rtol, abs=atol), (
            f"{field}: python={got!r} matlab={want!r}"
        )

    for field, rtol, atol in ARRAY_CHECKS:
        got = np.asarray(results[field], dtype=float).ravel()
        want = np.asarray(ref[field], dtype=float).ravel()
        assert got.shape == want.shape, f"{field}: shape mismatch"
        assert np.array_equal(np.isnan(got), np.isnan(want)), \
            f"{field}: NaN pattern mismatch"
        mask = np.isfinite(want)
        np.testing.assert_allclose(got[mask], want[mask], rtol=rtol, atol=atol,
                                   err_msg=field)

    import ntpath

    assert os.path.exists(results["outputFile"])
    # ntpath handles the fixture's Windows path on every platform
    assert os.path.basename(results["outputFile"]) == \
        ntpath.basename(str(ref["outputFile"]))
    # The wrapped depth solver also writes its own output
    assert os.path.exists(results["depthOutputFile"])
    assert "depthResults" not in results["inputConfig"]
