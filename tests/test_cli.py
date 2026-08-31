"""Smoke tests for the `laserttm` command-line interface."""

import json

from laserttm.cli import main


def test_list_runs():
    assert main(["list"]) == 0


def test_version_runs():
    assert main(["version"]) == 0


def test_run_surface_point(tmp_path):
    cfg = {
        "solver": "surface_point",
        "material": "W",
        "Pavg": 10,
        "spotRadius": 100e-6,
        "f_rep": 5e6,
        "tau_FWHM": 500e-15,
        "simDuration": 3 / 5e6,
        "outputDir": str(tmp_path / "outputs"),
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    out_path = tmp_path / "results.json"

    assert main(["run", str(cfg_path), "--out", str(out_path)]) == 0

    results = json.loads(out_path.read_text())
    assert results["solverId"] == "surface_point"
    assert results["nPulses"] == 3
    assert isinstance(results["finalResid_C"], float)


def test_run_npz_output(tmp_path):
    cfg = {
        "solver": "surface_point",
        "Pavg": 10,
        "f_rep": 5e6,
        "simDuration": 2 / 5e6,
        "outputDir": str(tmp_path / "outputs"),
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    out_path = tmp_path / "results.npz"

    assert main(["run", str(cfg_path), "--out", str(out_path)]) == 0

    import numpy as np
    data = np.load(out_path)
    assert "TresidVals_C" in data


def test_unknown_solver_errors(tmp_path):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"solver": "nope"}))
    try:
        main(["run", str(cfg_path)])
    except KeyError as exc:
        assert "Unknown solver" in str(exc)
    else:
        raise AssertionError("expected KeyError for unknown solver")
