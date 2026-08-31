"""Validate depth_profile_solver against MATLAB golden fixtures.

Unlike the 0D solver (hand-rolled RK4, ported step-identically, bit-level
agreement), Phase 1 here is a stiff integrator: MATLAB ode15s (NDF) vs
SciPy BDF. Both run at RelTol 1e-6 / AbsTol 0.1 K, so per-pulse surface
temperatures agree at integrator-tolerance level, and the tolerances below
reflect that. The Phase 2 CN coast is step-identical.
"""

from __future__ import annotations

import numpy as np
import pytest

from laserttm import depth_profile_solver

from .conftest import load_fixture

CASES = ["depth_profile_small", "depth_profile_baseline"]

SCALAR_CHECKS = [
    ("peakTe_C", 2e-3, 0.5),
    ("peakTl_C", 2e-3, 0.5),
    ("finalResid_C", 2e-3, 0.5),
]

ARRAY_CHECKS = [
    ("TePeakPerPulse_C", 2e-3, 0.5),
    ("TlPeakPerPulse_C", 2e-3, 0.5),
    ("TeqVals_C", 2e-3, 0.5),
    ("TresidVals_C", 2e-3, 0.5),
    ("baseTempPerPulse_C", 2e-3, 0.5),
]

# Inversion metrics ride on sharp interpolated features of the transient;
# they get physical (absolute) tolerances rather than tight relative ones.
INV_CHECKS = [
    ("invMaxPerPulse_K", 5e-2, 0.5),          # [K]
    ("tMaxInvPerPulse_s", 5e-2, 2e-12),       # [s]
    ("tInvOnsetPerPulse_s", 5e-2, 2e-12),     # [s]
    ("invDurationPerPulse_s", 5e-2, 5e-12),   # [s]
    ("Te_atMaxInvPerPulse_C", 2e-3, 2.0),     # [degC]
    ("Tl_atMaxInvPerPulse_C", 2e-3, 2.0),     # [degC]
]


def _cfg_from_fixture(fx: dict, tmp_path) -> dict:
    cfg = dict(fx["cfg"])
    cfg["outputDir"] = str(tmp_path)
    cfg["makePlots"] = False
    cfg["saveFigures"] = False
    return cfg


def _report(case, field, got, want):
    got = np.atleast_1d(np.asarray(got, dtype=float)).ravel()
    want = np.atleast_1d(np.asarray(want, dtype=float)).ravel()
    both = np.isfinite(got) & np.isfinite(want)
    if both.any():
        abs_err = float(np.max(np.abs(got[both] - want[both])))
        denom = np.maximum(np.abs(want[both]), 1e-300)
        rel_err = float(np.max(np.abs(got[both] - want[both]) / denom))
        print(f"    {case}:{field}: max abs err {abs_err:.3e}, max rel err {rel_err:.3e}")
    return got, want


@pytest.mark.parametrize("case", CASES)
def test_against_matlab_fixture(case, tmp_path):
    fx = load_fixture(case)
    ref = fx["results"]

    results = depth_profile_solver(_cfg_from_fixture(fx, tmp_path))

    assert results["solverId"] == "depth_profile"
    assert results["contractVersion"] == "v1"
    assert int(results["nPulses"]) == int(ref["nPulses"])

    for field, rtol, atol in SCALAR_CHECKS:
        got, want = _report(case, field, results[field], ref[field])
        assert got[0] == pytest.approx(want[0], rel=rtol, abs=atol), f"{case}: {field}"

    for field, rtol, atol in ARRAY_CHECKS + INV_CHECKS:
        got, want = _report(case, field, results[field], ref[field])
        assert got.shape == want.shape, f"{case}: {field}: shape mismatch"
        # NaN patterns must match (e.g. pulses with no inversion detected)
        assert np.array_equal(np.isnan(got), np.isnan(want)), \
            f"{case}: {field}: NaN pattern mismatch"
        mask = np.isfinite(want)
        np.testing.assert_allclose(got[mask], want[mask], rtol=rtol, atol=atol,
                                   err_msg=f"{case}: {field}")
