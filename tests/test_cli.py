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


def test_unknown_solver_exits_cleanly(tmp_path, capsys):
    """A bad solver id gets a message naming the valid ids, not a traceback."""
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"solver": "nope"}))

    assert main(["run", str(cfg_path)]) == 4
    assert "Unknown solver" in capsys.readouterr().err


def test_describe_lists_keys_with_units(capsys):
    assert main(["describe", "radial_profile"]) == 0
    out = capsys.readouterr().out
    assert "spotRadius [m]" in out
    assert "radialSolveMode" in out


def test_describe_json_is_machine_readable(capsys):
    assert main(["describe", "depth_profile", "--json"]) == 0
    described = json.loads(capsys.readouterr().out)
    assert described["params"]["Nz"]["default"] == 200


def test_schema_command_emits_json_schema(capsys):
    assert main(["schema", "surface_point"]) == 0
    js = json.loads(capsys.readouterr().out)
    assert js["additionalProperties"] is False


def test_materials_command(capsys):
    assert main(["materials"]) == 0
    assert "melting point" in capsys.readouterr().out


def test_validate_rejects_a_typo(tmp_path, capsys):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"solver": "surface_point", "pavg": 10}))

    assert main(["validate", str(cfg_path)]) == 2
    assert "Did you mean 'Pavg'" in capsys.readouterr().err


def test_validate_accepts_a_good_config(tmp_path, capsys):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(
        {"solver": "surface_point", "Pavg": 10, "f_rep": 5e6,
         "simDuration": 3 / 5e6}))

    assert main(["validate", str(cfg_path)]) == 0
    assert "valid" in capsys.readouterr().out


def test_run_refuses_a_bad_config_before_solving(tmp_path, capsys):
    """The point of validating first: a typo must not cost a full run."""
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(
        {"solver": "surface_point", "spotRadius": 100}))

    assert main(["run", str(cfg_path)]) == 2
    assert "micrometres" in capsys.readouterr().err


def test_dry_run_reports_cost_without_running(tmp_path, capsys):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(
        {"solver": "surface_point", "f_rep": 5e6, "simDuration": 50 / 5e6}))

    assert main(["run", str(cfg_path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "50 pulses" in out
    assert "Nothing was run" in out


def test_config_with_a_byte_order_mark_is_accepted(tmp_path):
    """PowerShell writes a BOM by default, so configs routinely have one."""
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(
        {"solver": "surface_point", "f_rep": 5e6, "simDuration": 1 / 5e6}),
        encoding="utf-8-sig")

    assert main(["run", str(cfg_path), "--dry-run"]) == 0
