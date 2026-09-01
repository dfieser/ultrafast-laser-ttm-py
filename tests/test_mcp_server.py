"""Smoke tests for the MCP server's job pattern (requires the mcp extra)."""

import time

import pytest

pytest.importorskip("mcp")


def _tiny_cfg(tmp_path):
    return {
        "material": "W",
        "Pavg": 10,
        "spotRadius": 100e-6,
        "f_rep": 5e6,
        "tau_FWHM": 500e-15,
        "simDuration": 2 / 5e6,
        "outputDir": str(tmp_path / "outputs"),
    }


def test_list_solvers():
    from laserttm import mcp_server

    solvers = mcp_server.list_solvers()
    assert "radial_profile" in solvers
    assert "surface_point" in solvers
    # An agent picks a solver from this alone, so it must carry routing
    # guidance, not just a one-line description.
    assert solvers["radial_profile"]["whenToUse"]
    assert solvers["radial_profile"]["whenNotToUse"]
    assert solvers["radial_profile"]["minimalConfig"]


def test_describe_solver_gives_units_and_defaults():
    from laserttm import mcp_server

    described = mcp_server.describe_solver("depth_profile")
    assert described["params"]["spotRadius"]["unit"] == "m"
    assert described["params"]["Nz"]["default"] == 200


def test_validate_config_catches_a_typo_without_running():
    from laserttm import mcp_server

    report = mcp_server.validate_config("surface_point", {"pavg": 10})
    assert not report["ok"]
    assert report["errors"][0]["code"] == "unknown_key"
    assert "'Pavg'" in report["errors"][0]["suggestion"]


def test_list_materials_says_which_solvers_accept_each():
    from laserttm import mcp_server

    rows = {m["key"]: m for m in mcp_server.list_materials()}
    assert "depth_profile" not in rows["AU"]["solvers"]
    assert "depth_profile" in rows["W"]["solvers"]


def test_start_run_rejects_a_bad_config_before_spawning():
    """A typo must raise here, not produce a plausible run on defaults."""
    from laserttm import mcp_server

    with pytest.raises(ValueError, match="Pavg"):
        mcp_server.start_run("surface_point", {"pavg": 10})


def test_run_quick_refuses_a_run_that_cannot_fit_its_timeout():
    from laserttm import mcp_server

    out = mcp_server.run_quick("radial_profile",
                               {"simDuration": 1.0, "f_rep": 40e6},
                               timeout_s=5)
    assert out["ok"] is False
    assert out["use"] == "start_run"


def test_job_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setenv("LASERTTM_RUNS_DIR", str(tmp_path / "runs"))
    from laserttm import mcp_server

    started = mcp_server.start_run("surface_point", _tiny_cfg(tmp_path))
    run_id = started["run_id"]
    assert started["status"] in ("running", "done")

    deadline = time.time() + 300  # generous: child may JIT-compile kernels
    while time.time() < deadline:
        status = mcp_server.check_run(run_id)
        if status["status"] != "running":
            break
        time.sleep(0.5)
    assert status["status"] == "done", status.get("error", status)

    out = mcp_server.get_results(run_id)
    assert out["status"] == "done"
    assert out["results"]["solverId"] == "surface_point"
    assert out["results"]["nPulses"] == 2
    assert "finalResid_C" in out["results"]

    log = mcp_server.get_log(run_id)
    assert "Surface TTM" in log


def test_unknown_solver_rejected_before_spawn():
    from laserttm import mcp_server

    with pytest.raises(KeyError):
        mcp_server.start_run("not_a_solver", {})


def test_unknown_run_id():
    from laserttm import mcp_server

    with pytest.raises(KeyError):
        mcp_server.check_run("does-not-exist")
