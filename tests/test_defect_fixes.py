"""Regression tests for the defects confirmed by the adversarial review."""

import json
import os

import pytest

from laserttm import schema
from laserttm.config import safe_tag

TINY = {"Pavg": 10, "f_rep": 5e6, "tau_FWHM": 500e-15,
        "simDuration": 2 / 5e6, "makePlots": False}


# --- null and empty config values mean "use the default" -------------------

def test_null_and_empty_values_resolve_to_defaults():
    report = schema.validate_config(
        "surface_point", {"simDuration": None, "Pavg": "", "material": None})
    assert report["ok"]
    assert report["resolved"]["simDuration"] == 100e-6
    assert report["resolved"]["Pavg"] == 1.0
    assert report["estimate"]["nPulses"] == 100


def test_estimate_run_survives_null_values():
    assert schema.estimate_run(
        "radial_profile", {"simDuration": None})["nPulses"] == 18000
    assert schema.estimate_run(
        "scanning_beam", {"v_scan": "", "scanLength": None})["nPulses"] == 36000


# --- enum validation matches the read sites, which lowercase ---------------

@pytest.mark.parametrize("solver_id,key,value", [
    ("surface_point", "pulseProfile", "GAUSSIAN"),
    ("surface_point", "pulseProfile", "Exp"),
    ("radial_profile", "radialSolveMode", "Scale"),
    ("radial_profile", "depthProfile", "BOX"),
])
def test_enum_validation_is_case_insensitive(solver_id, key, value):
    assert schema.validate_config(solver_id, {key: value})["ok"]


def test_wrong_enum_is_still_rejected():
    assert not schema.validate_config(
        "surface_point", {"pulseProfile": "sech2"})["ok"]


# --- case tags cannot break filenames --------------------------------------

def test_safe_tag_neutralizes_path_separators():
    assert safe_tag("A=0.55/run 1") == "A_0.55_run_1"
    assert safe_tag(r"..\evil") == ".._evil"  # no separator survives
    assert safe_tag("sweep_A0.40-x") == "sweep_A0.40-x"
    assert safe_tag("") == ""


def test_every_solver_declares_case_tag():
    for solver_id in schema.SOLVER_IDS:
        assert "caseTag" in schema.solver_schema(solver_id).params, solver_id


def test_surface_point_applies_the_tag(tmp_path):
    from laserttm import surface_point_solver

    results = surface_point_solver(
        {**TINY, "caseTag": "tag/with sep", "outputDir": str(tmp_path)})
    name = os.path.basename(results["outputFile"])
    assert name.startswith("tag_with_sep__TTM_")
    assert os.path.exists(results["outputFile"])


def test_untagged_basename_is_unchanged(tmp_path):
    from laserttm import surface_point_solver

    results = surface_point_solver({**TINY, "outputDir": str(tmp_path)})
    assert os.path.basename(results["outputFile"]).startswith("TTM_5_MHz")


# --- inert keys say so ------------------------------------------------------

def test_inert_keys_carry_a_note():
    for solver_id in ("surface_point", "scanning_beam"):
        params = schema.solver_schema(solver_id).params
        assert "No effect" in params["kTable"].notes, solver_id
    for solver_id in ("surface_point", "depth_profile", "single_pulse",
                      "inversion_quantifier", "scanning_beam"):
        note = schema.solver_schema(solver_id).params["T_melt_C"].notes
        assert "radial" in note, solver_id


# --- named figures start fresh each run ------------------------------------

def test_figures_do_not_accumulate_across_runs(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from laserttm import radial_profile_solver

    cfg = {**TINY, "Nr": 8, "makePlots": True}
    radial_profile_solver({**cfg, "outputDir": str(tmp_path / "a")})
    first = len(plt.figure("Radial_TTM_Center_Temperature_Timeline").axes[0].lines)
    radial_profile_solver({**cfg, "outputDir": str(tmp_path / "b")})
    second = len(plt.figure("Radial_TTM_Center_Temperature_Timeline").axes[0].lines)
    plt.close("all")
    assert first == second


# --- MCP runs survive a server restart -------------------------------------

def test_mcp_run_is_reachable_after_restart(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    monkeypatch.setenv("LASERTTM_RUNS_DIR", str(tmp_path / "runs"))
    import time

    from laserttm import mcp_server

    started = mcp_server.start_run("surface_point",
                                   {**TINY, "simDuration": 2 / 5e6})
    run_id = started["run_id"]
    deadline = time.time() + 300
    while time.time() < deadline:
        if mcp_server.check_run(run_id)["status"] != "running":
            break
        time.sleep(0.5)
    assert mcp_server.check_run(run_id)["status"] == "done"

    # Simulate a restart: the in-memory registry is gone.
    monkeypatch.setattr(mcp_server, "_RUNS", {})
    status = mcp_server.check_run(run_id)
    assert status["status"] == "done"
    assert status["solver"] == "surface_point"
    out = mcp_server.get_results(run_id)
    assert out["results"]["solverId"] == "surface_point"

    with open(os.path.join(status["run_dir"], "run.json"),
              encoding="utf-8") as f:
        run_json = json.load(f)
    assert run_json["solver"] == "surface_point"

    with pytest.raises(KeyError):
        mcp_server.check_run("never-existed")


# --- absent inversion still honors the output contract ---------------------

def test_zero_inversion_run_still_writes_its_report(tmp_path):
    from laserttm import inversion_quantifier

    results = inversion_quantifier(
        {"Pavg": 1e-3, "f_rep": 5e6, "tau_FWHM": 500e-15,
         "simDuration": 2 / 5e6, "Nz": 60, "Lz": 400e-9,
         "makePlots": False, "outputDir": str(tmp_path)})
    assert results["nInvPulses"] == 0
    assert os.path.exists(results["outputFile"])
    assert results["outputDir"] == str(tmp_path)
    with open(results["outputFile"], encoding="utf-8") as f:
        text = f.read()
    assert "No significant inversion" in text
