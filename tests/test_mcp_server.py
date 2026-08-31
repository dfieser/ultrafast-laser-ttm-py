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
