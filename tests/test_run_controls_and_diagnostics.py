"""Run controls, melt flags, coded diagnostics, and the validate report.

writeReport, reportHistory, verbose and showProgress=False put the caller
in charge of what a run writes and prints. Every solver flags melting the
same way, every warning carries a code, and validate_config shows the
pulse-train numbers a config implies before any run is spent on it.
"""

import contextlib
import io
import os
import subprocess
import sys

import pytest

from laserttm import schema
from laserttm.runtools import get_solver

MELT = {"material": "W", "Pavg": 50.0, "f_rep": 1e6, "tau_FWHM": 500e-15,
        "spotRadius": 80e-6, "makePlots": False, "verbose": False}
MILD = {"material": "W", "Pavg": 1.0, "f_rep": 5e6, "tau_FWHM": 500e-15,
        "spotRadius": 80e-6, "makePlots": False, "verbose": False}


def _run(solver_id, cfg, capture=False):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = get_solver(solver_id)(cfg)
    return (res, buf.getvalue()) if capture else res


# ------------------------------------------------------------ run controls


def test_write_report_off_writes_nothing(tmp_path):
    res = _run("surface_point", {**MILD, "nPulses": 2, "writeReport": False,
                                 "outputDir": str(tmp_path)})
    assert res["outputFile"] is None
    assert os.listdir(tmp_path) == []
    assert res["TresidVals_C"].size == 2


@pytest.mark.parametrize("solver_id, extra", [
    ("surface_point", {"nPulses": 2}),
    ("depth_profile", {"nPulses": 2, "Nz": 60, "Lz": 400e-9}),
    ("single_pulse", {}),
])
def test_report_history_off_keeps_the_summary_only(tmp_path, solver_id, extra):
    res = _run(solver_id, {**MILD, **extra, "reportHistory": False,
                           "outputDir": str(tmp_path)})
    with open(res["outputFile"], encoding="utf-8") as f:
        report = f.read()
    assert "XY Data" not in report
    assert "reportHistory=False" in report
    assert res["time_s"].size > 0            # the history itself is kept


@pytest.mark.parametrize("solver_id, extra", [
    ("surface_point", {"nPulses": 2}),
    ("depth_profile", {"nPulses": 2, "Nz": 60, "Lz": 400e-9}),
    ("radial_profile", {"nPulses": 2, "Nr": 8}),
    ("single_pulse", {}),
    ("inversion_quantifier", {"nPulses": 2, "Nz": 60, "Lz": 400e-9}),
])
def test_verbose_off_is_silent(tmp_path, solver_id, extra):
    _, out = _run(solver_id, {**MILD, **extra, "outputDir": str(tmp_path)},
                  capture=True)
    assert out == ""


def test_verbose_off_is_silent_for_the_scanning_solver(tmp_path):
    _, out = _run("scanning_beam", {"f_rep": 5e6, "v_scan": 1.0,
                                    "scanLength": 4e-7, "makePlots": False,
                                    "verbose": False,
                                    "outputDir": str(tmp_path)}, capture=True)
    assert out == ""


def test_show_progress_false_silences_the_per_pulse_lines(tmp_path):
    cfg = {**MILD, "nPulses": 3, "verbose": True, "outputDir": str(tmp_path)}
    _, chatty = _run("surface_point", {**cfg, "showProgress": None},
                     capture=True)
    _, quiet = _run("surface_point", {**cfg, "showProgress": False},
                    capture=True)
    assert "Pulse 3/3" in chatty
    assert "Pulse 3/3" not in quiet
    assert "Results" in quiet            # the summary still prints


# ------------------------------------------------------------- melt flags


@pytest.mark.parametrize("solver_id, extra", [
    ("surface_point", {"nPulses": 2}),
    ("depth_profile", {"nPulses": 2, "Nz": 60, "Lz": 400e-9}),
    ("radial_profile", {"nPulses": 2, "Nr": 8}),
    ("single_pulse", {}),
])
def test_fifty_watts_sets_the_melt_flag_on_pulse_one(tmp_path, solver_id,
                                                     extra):
    res = _run(solver_id, {**MELT, **extra, "outputDir": str(tmp_path)})
    assert res["Tmelt_C"] == pytest.approx(3422.0)
    assert res["peakTl_C"] > res["Tmelt_C"]
    assert res["meltDetected"] is True
    assert res["meltPulse"] == 1
    codes = [d["code"] for d in res["diagnostics"]]
    assert "above_melting" in codes and "above_ablation" in codes
    assert res["peakFluence_J_m2"] == pytest.approx(4973.59, rel=1e-4)
    assert res["pulseEnergy_J"] == pytest.approx(50e-6)
    assert res["absorbedFluence_J_m2"] == pytest.approx(0.55 * 4973.59, rel=1e-4)


def test_scanning_melt_flag(tmp_path):
    res = _run("scanning_beam", {"material": "W", "Pavg": 50.0, "f_rep": 1e6,
                                 "tau_FWHM": 500e-15, "spotRadius": 80e-6,
                                 "v_scan": 1.0, "scanLength": 3e-6,
                                 "makePlots": False, "verbose": False,
                                 "outputDir": str(tmp_path)})
    assert res["meltDetected"] is True and res["meltPulse"] == 1
    assert "above_ablation" in [d["code"] for d in res["diagnostics"]]


def test_the_quantifier_passes_the_flags_through(tmp_path):
    res = _run("inversion_quantifier", {**MELT, "nPulses": 2, "Nz": 60,
                                        "Lz": 400e-9,
                                        "outputDir": str(tmp_path)})
    assert res["meltDetected"] is True and res["meltPulse"] == 1
    assert res["diagnostics"] == res["depthResults"]["diagnostics"]
    assert res["Tmelt_C"] == pytest.approx(3422.0)


def test_every_diagnostic_has_the_validate_shape(tmp_path):
    res = _run("depth_profile", {**MELT, "nPulses": 2, "Nz": 60, "Lz": 400e-9,
                                 "enableRadialProfile": True,
                                 "simDuration": 2e-6,
                                 "outputDir": str(tmp_path)})
    for d in res["diagnostics"]:
        assert set(d) >= {"code", "level", "message", "suggestion"}
        assert d["level"] in ("warning", "info")
    assert res["warnings"] == [d["message"] for d in res["diagnostics"]
                               if d["level"] == "warning"]


def test_ablation_threshold_can_be_overridden_or_absent(tmp_path):
    quiet = _run("surface_point", {**MELT, "nPulses": 1,
                                   "ablationThreshold": 1e6,
                                   "outputDir": str(tmp_path / "a")})
    assert "above_ablation" not in [d["code"] for d in quiet["diagnostics"]]
    copper = _run("surface_point", {**MELT, "material": "Cu", "nPulses": 1,
                                    "outputDir": str(tmp_path / "b")})
    assert "above_ablation" not in [d["code"] for d in copper["diagnostics"]]
    assert copper["materialProps"]["ablationThreshold_J_m2"] is None


# --------------------------------------------------------------- validate


def test_validate_shows_derived_quantities_and_the_ablation_warning():
    report = schema.validate_config("depth_profile", MELT)
    assert report["ok"]
    d = report["derived"]
    assert d["pulseEnergy_J"] == pytest.approx(50e-6)
    assert d["peakFluence_J_cm2"] == pytest.approx(0.4974, rel=1e-3)
    assert d["absorbedFluence_J_m2"] == pytest.approx(0.55 * 4973.59, rel=1e-4)
    assert d["period_s"] == pytest.approx(1e-6)
    (abl,) = [w for w in report["warnings"] if w["code"] == "above_ablation"]
    assert "0.44 J/cm^2" in abl["message"]
    assert schema.derived_quantities("depth_profile", MELT) == d


def test_validate_keeps_the_estimate_when_another_key_is_wrong():
    report = schema.validate_config("surface_point",
                                    {**MILD, "nPulses": 4, "pavg": 3})
    assert not report["ok"]
    assert report["estimate"]["nPulses"] == 4
    assert report["derived"]["nPulses"] == 4


def test_validate_names_the_coast_step_count_it_will_use():
    report = schema.validate_config("surface_point", {**MILD, "f_rep": 1e6})
    (coast,) = [w for w in report["warnings"] if w["code"] == "coarse_coast"]
    assert "549" in coast["message"]
    fine = schema.validate_config("surface_point", {**MILD, "f_rep": 18e6})
    assert "coarse_coast" not in [w["code"] for w in fine["warnings"]]


def test_estimate_reports_warmup_and_history():
    est = schema.estimate_run("depth_profile", {"nPulses": 10})
    assert est["warmup_s"] > 0
    assert est["historyIncluded"] is True
    lean = schema.estimate_run("depth_profile",
                               {"nPulses": 10, "storeHistory": False})
    assert lean["historyIncluded"] is False
    assert lean["estRuntime_s"] <= est["estRuntime_s"]
    # The depth solver's cost follows its fine-stage step count, tau.
    short = schema.estimate_run("depth_profile",
                                {"nPulses": 10, "tau_FWHM": 100e-15})
    long = schema.estimate_run("depth_profile",
                               {"nPulses": 10, "tau_FWHM": 500e-15})
    assert short["estRuntime_s"] > 3 * long["estRuntime_s"]


def test_scanning_derived_includes_the_pulse_spacing():
    d = schema.derived_quantities("scanning_beam",
                                  {"f_rep": 5e6, "v_scan": 2.0})
    assert d["pulseSpacing_m"] == pytest.approx(4e-7)


# -------------------------------------------------------------------- CLI


def test_describe_survives_a_cp1252_console():
    env = {**os.environ, "PYTHONIOENCODING": "cp1252:strict",
           "PYTHONUTF8": "0"}
    proc = subprocess.run(
        [sys.executable, "-m", "laserttm.cli", "describe", "depth_profile"],
        capture_output=True, text=True, env=env, encoding="cp1252",
        errors="replace", check=False)
    assert proc.returncode == 0, proc.stderr
    assert "depth_profile: Depth profile" in proc.stdout
    assert "—" not in proc.stdout


def test_validate_prints_the_pulse_train_numbers(tmp_path, capsys):
    import json

    from laserttm.cli import main

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"solver": "depth_profile", **MELT}))
    assert main(["validate", str(cfg_path)]) == 0
    captured = capsys.readouterr()
    assert "peak fluence 0.4974 J/cm^2" in captured.out
    assert "ablation threshold" in captured.err
