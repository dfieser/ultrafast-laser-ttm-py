"""A run that leaves the model's range of validity says so in the envelope.

The solvers have no phase change or ablation. A lattice above the melting
point is therefore a statement about deposited energy, not a temperature
the material could have, and every consumer, including one that never sees
the console, must be told.
"""

import contextlib
import io
import warnings

import pytest

from laserttm.physics import validity_warnings
from laserttm.runtools import get_solver

OUT_OF_RANGE = {"material": "W", "Pavg": 50.0, "f_rep": 1e6,
                "tau_FWHM": 500e-15, "spotRadius": 80e-6}  # 0.50 J/cm^2 peak


def _run(solver_id, cfg):
    with contextlib.redirect_stdout(io.StringIO()):
        return get_solver(solver_id)(cfg)


def test_the_helper_is_quiet_below_and_explicit_above_the_melting_point():
    assert validity_warnings(3000.0, 3422.0, "W", emit=False) == []
    assert validity_warnings(float("nan"), 3422.0, "W", emit=False) == []
    msgs = validity_warnings(8561.0, 3422.0, "w", emit=False)
    assert len(msgs) == 1
    assert "8561 C" in msgs[0] and "W (3422 C)" in msgs[0]
    assert "no phase change" in msgs[0]
    with pytest.warns(UserWarning, match="melting point"):
        validity_warnings(8561.0, 3422.0, "W")


def test_surface_point_warns_above_the_melting_point(tmp_path):
    cfg = {**OUT_OF_RANGE, "simDuration": 2e-6, "makePlots": False,
           "outputDir": str(tmp_path)}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = _run("surface_point", cfg)
    assert res["peakTl_C"] > res["materialProps"]["Tmelt_C"]
    # 0.50 J/cm^2 is past both the melting point and the 0.44 J/cm^2
    # ablation threshold, and each concern is its own diagnostic.
    codes = [d["code"] for d in res["diagnostics"] if d["level"] == "warning"]
    assert codes == ["above_melting", "above_ablation"]
    assert res["warnings"] == [d["message"] for d in res["diagnostics"]
                               if d["level"] == "warning"]
    assert "melting point of W" in res["warnings"][0]
    assert "ablation threshold" in res["warnings"][1]
    assert res["meltDetected"] is True and res["meltPulse"] == 1
    assert any("melting point" in str(w.message) for w in caught)


def test_single_pulse_warns_above_the_melting_point(tmp_path):
    cfg = {**OUT_OF_RANGE, "makePlots": False, "outputDir": str(tmp_path)}
    res = _run("single_pulse", cfg)
    assert res["peakTl_C"] > res["materialProps"]["Tmelt_C"]
    assert [d["code"] for d in res["diagnostics"]] == [
        "above_melting", "above_ablation"]
    assert res["meltPulse"] == 1


def test_a_run_inside_the_model_stays_silent(tmp_path):
    cfg = {"f_rep": 5e6, "simDuration": 2 / 5e6, "makePlots": False,
           "outputDir": str(tmp_path)}  # the 1 W default, far below melting
    res = _run("surface_point", cfg)
    assert res["peakTl_C"] < res["materialProps"]["Tmelt_C"]
    assert res["warnings"] == []
    assert res["meltDetected"] is False and res["meltPulse"] == 0
    # Informational diagnostics never masquerade as warnings.
    assert all(d["level"] == "info" for d in res["diagnostics"])
