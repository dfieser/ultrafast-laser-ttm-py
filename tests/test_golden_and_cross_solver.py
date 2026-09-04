"""Golden inversion values, cross-solver agreement, and a regression grep.

The inversion numbers below are the library's own at 0.4.0 for the
documented showcase settings, tungsten at 12 W, 1 MHz, 500 fs into an
80 um spot. They differ between the two depth-resolved solvers on
purpose: single_pulse runs the constant electron conductivity of the
MATLAB visualizer, depth_profile runs tungsten's measured k(T) table, and
the hotter, more conductive electrons of the former hand energy to the
lattice faster.
"""

import contextlib
import io
import pathlib
import re

import numpy as np
import pytest

from laserttm.physics import refine_peak
from laserttm.runtools import get_solver

SHOWCASE = {"material": "W", "Pavg": 12.0, "f_rep": 1e6, "tau_FWHM": 500e-15,
            "spotRadius": 80e-6, "makePlots": False, "verbose": False}


def _run(solver_id, cfg):
    with contextlib.redirect_stdout(io.StringIO()):
        return get_solver(solver_id)(cfg)


def test_golden_single_pulse_inversion(tmp_path):
    res = _run("single_pulse", {**SHOWCASE, "outputDir": str(tmp_path)})
    assert res["maxInv_K"] == pytest.approx(208.1, rel=0.02)
    assert res["tMaxInv_s"] == pytest.approx(15.0e-12, rel=0.05)
    assert res["peakTl_C"] == pytest.approx(2498.0, rel=0.02)
    assert res["warnings"] == []


def test_golden_depth_profile_first_pulse_inversion(tmp_path):
    res = _run("depth_profile", {**SHOWCASE, "nPulses": 2,
                                 "outputDir": str(tmp_path)})
    assert res["invMaxPerPulse_K"][0] == pytest.approx(173.1, rel=0.02)
    assert res["tMaxInvPerPulse_s"][0] == pytest.approx(17.6e-12, rel=0.05)
    assert res["TlPeakPerPulse_C"][0] == pytest.approx(2925.0, rel=0.02)


def test_the_zero_d_solvers_agree_at_beam_centre(tmp_path):
    """surface_point and radial_profile in scale mode run the same 0D
    model at the centre, so their first-pulse equilibrium is identical;
    the radial coast then loses heat sideways, so its residual is lower."""
    sp = _run("surface_point", {**SHOWCASE, "nPulses": 3,
                                "outputDir": str(tmp_path / "sp")})
    rp = _run("radial_profile", {**SHOWCASE, "nPulses": 3, "Nr": 24,
                                 "outputDir": str(tmp_path / "rp")})
    assert rp["TeqVals_C"][0] == pytest.approx(sp["TeqVals_C"][0], rel=1e-9)
    assert rp["peakTl_C"] == pytest.approx(rp["peakTeq_C"])
    assert rp["TresidVals_C"][0] < sp["TresidVals_C"][0]
    assert rp["TresidVals_C"][0] == pytest.approx(sp["TresidVals_C"][0],
                                                   rel=0.05)


def test_the_depth_solvers_agree_within_their_conductivity_models(tmp_path):
    """single_pulse and depth_profile solve the same first pulse with
    different electron conductivity models; their peaks and inversion
    timing agree to the tens of percent that difference costs, not more."""
    single = _run("single_pulse", {**SHOWCASE, "outputDir": str(tmp_path)})
    depth = _run("depth_profile", {**SHOWCASE, "nPulses": 1,
                                   "outputDir": str(tmp_path)})
    assert depth["TlPeakPerPulse_C"][0] == pytest.approx(single["peakTl_C"],
                                                         rel=0.2)
    assert depth["invMaxPerPulse_K"][0] == pytest.approx(single["maxInv_K"],
                                                         rel=0.2)
    assert depth["tMaxInvPerPulse_s"][0] == pytest.approx(single["tMaxInv_s"],
                                                          rel=0.2)


def test_peak_refinement_is_well_conditioned_late_in_a_train():
    """The parabola through three samples fitted in absolute time lost
    everything to cancellation a millisecond into a train: the vertex
    value came back as the rounding remainder of 1e19-sized terms,
    exactly 256 K or 512 K on the agent's plot. The local fit returns the
    peak the three samples describe."""
    t1 = 5.0e-4 + 1.5e-11
    t3 = np.array([t1 - 1.1e-12, t1, t1 + 1.2e-12])
    d3 = np.array([154.0, 155.0, 154.6])
    t_peak, d_peak = refine_peak(t3, d3)
    assert 155.0 <= d_peak < 155.2
    assert t3[0] < t_peak < t3[2]
    # The exact parabola through the three points is recovered.
    a, b, c = np.polyfit(t3 - t1, d3, 2)
    assert d_peak == pytest.approx(c - b * b / (4 * a), rel=1e-9)
    assert t_peak - t1 == pytest.approx(-b / (2 * a), rel=1e-6)


def test_peak_refinement_declines_when_there_is_no_interior_peak():
    assert refine_peak([0.0, 1.0, 2.0], [1.0, 2.0, 3.0]) is None   # rising
    assert refine_peak([0.0, 1.0, 2.0], [3.0, 2.0, 1.0]) is None   # falling
    assert refine_peak([0.0, 0.0, 1.0], [1.0, 2.0, 1.0]) is None   # bad grid


def test_nothing_speaks_of_a_steady_state():
    """Release 0.3.0 removed the steady-state projection for good."""
    root = pathlib.Path(__file__).resolve().parents[1]
    word = "stead" + "y state"
    pattern = re.compile(word.replace(" ", "[ _-]?"), re.IGNORECASE)
    offenders = []
    for folder in ("src", "docs", "examples"):
        for path in (root / folder).rglob("*"):
            if (path.suffix in (".py", ".md", ".txt", ".json") and path.is_file()
                    and pattern.search(path.read_text(encoding="utf-8",
                                                      errors="replace"))):
                offenders.append(str(path.relative_to(root)))
    for name in ("README.md", "AGENTS.md", "CONTRIBUTING.md"):
        text = (root / name).read_text(encoding="utf-8", errors="replace")
        # AGENTS.md states the rule itself; every other file stays silent.
        if name != "AGENTS.md" and pattern.search(text):
            offenders.append(name)
    assert offenders == []
