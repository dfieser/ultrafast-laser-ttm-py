"""nPulses, exact snapshot instants, chosen profile pulses, map history.

Before 0.4.0 the pulse count was round(simDuration * f_rep) with the last
fraction of a period silently coasted, snapshots landed on the nearest
integrator output, residual profiles were kept at twelve pulses nobody
chose, and the scanning solver kept no map history at all.
"""

import contextlib
import io

import numpy as np
import pytest

from laserttm import schema
from laserttm.runtools import get_solver

BASE = {"material": "W", "Pavg": 12.0, "f_rep": 1e6, "tau_FWHM": 500e-15,
        "spotRadius": 80e-6, "makePlots": False, "verbose": False}
DEPTH = {**BASE, "Nz": 60, "Lz": 400e-9}


def _run(solver_id, cfg):
    with contextlib.redirect_stdout(io.StringIO()):
        return get_solver(solver_id)(cfg)


# --------------------------------------------------------------- nPulses


@pytest.mark.parametrize("solver_id, extra", [
    ("surface_point", {}),
    ("depth_profile", {"Nz": 60, "Lz": 400e-9}),
])
def test_npulses_fires_the_count_and_simduration_ends_the_run(tmp_path,
                                                              solver_id, extra):
    """nPulses = 2 with simDuration = 1.2 periods fires two pulses and the
    trace ends at 1.2 periods. Without nPulses the same duration would
    have rounded to a single pulse."""
    res = _run(solver_id, {**BASE, **extra, "nPulses": 2,
                           "simDuration": 1.2e-6, "outputDir": str(tmp_path)})
    assert res["nPulses"] == 2
    assert res["time_s"][-1] == pytest.approx(1.2e-6)
    assert res["resolvedConfig"]["nPulses"] == 2
    assert res["resolvedConfig"]["simDuration"] == pytest.approx(1.2e-6)


def test_npulses_alone_sets_the_duration(tmp_path):
    res = _run("surface_point", {**BASE, "nPulses": 3,
                                 "outputDir": str(tmp_path)})
    assert res["nPulses"] == 3
    assert res["resolvedConfig"]["simDuration"] == pytest.approx(3e-6)
    assert res["time_s"][-1] == pytest.approx(3e-6)
    assert schema.pulse_count("surface_point", {**BASE, "nPulses": 3}) == 3


def test_one_pulse_shorter_than_a_period_is_allowed(tmp_path):
    res = _run("surface_point", {**BASE, "nPulses": 1, "simDuration": 0.5e-6,
                                 "outputDir": str(tmp_path)})
    assert res["nPulses"] == 1
    assert res["time_s"][-1] == pytest.approx(0.5e-6)


def test_radial_and_quantifier_accept_npulses(tmp_path):
    rad = _run("radial_profile", {**BASE, "nPulses": 2, "Nr": 8,
                                  "outputDir": str(tmp_path)})
    assert rad["nPulses"] == 2 and rad["TeqVals_C"].size == 2
    inv = _run("inversion_quantifier", {**DEPTH, "nPulses": 2,
                                        "outputDir": str(tmp_path)})
    assert inv["nPulses"] == 2


def test_validate_flags_a_fractional_period_and_not_an_explicit_count():
    report = schema.validate_config("surface_point",
                                    {**BASE, "simDuration": 1.2e-6})
    codes = [w["code"] for w in report["warnings"]]
    assert "truncated_period" in codes
    assert report["estimate"]["nPulses"] == 1

    report = schema.validate_config(
        "surface_point", {**BASE, "nPulses": 2, "simDuration": 1.2e-6})
    assert "truncated_period" not in [w["code"] for w in report["warnings"]]
    assert report["estimate"]["nPulses"] == 2
    assert report["resolved"]["simDuration"] == pytest.approx(1.2e-6)

    whole = schema.validate_config("surface_point",
                                   {**BASE, "simDuration": 3 / 1e6})
    assert "truncated_period" not in [w["code"] for w in whole["warnings"]]


# ------------------------------------------------------------- snapshots


def test_dense_single_pulse_snapshots_are_exact_and_distinct(tmp_path):
    delays = np.arange(0.0, 20e-12, 0.05e-12)
    res = _run("single_pulse", {**BASE, "snapshotDelays": delays.tolist(),
                                "outputDir": str(tmp_path)})
    got = res["snapshotDelays_s"]
    assert got.size == delays.size
    assert np.all(np.diff(got) > 0)
    np.testing.assert_allclose(got, delays, atol=1e-18)
    assert res["TeSnapshots_C"].shape == (delays.size, res["zGrid_m"].size)


def test_depth_profile_first_pulse_snapshots_are_exact(tmp_path):
    delays = [0.0, 0.3e-12, 0.35e-12, 7e-12, 33e-12]
    res = _run("depth_profile", {**DEPTH, "nPulses": 2,
                                 "snapshotDelays": delays,
                                 "outputDir": str(tmp_path)})
    np.testing.assert_allclose(res["snapshotDelays_s"], delays, atol=1e-18)


def test_snapshots_beyond_the_window_are_skipped_not_clamped(tmp_path):
    res = _run("single_pulse", {**BASE, "snapshotDelays": [1e-12, 5.0],
                                "outputDir": str(tmp_path)})
    np.testing.assert_allclose(res["snapshotDelays_s"], [1e-12], atol=1e-18)


def test_profile_snapshot_pulses_can_be_chosen(tmp_path):
    res = _run("depth_profile", {**DEPTH, "nPulses": 5,
                                 "profileSnapshotPulses": [2, 4, 9],
                                 "enableRadialProfile": True,
                                 "outputDir": str(tmp_path)})
    assert res["profileSnapshotPulses"].tolist() == [2, 4]
    assert res["profileSnapshots_C"].shape == (2, res["zGridDiff_m"].size)
    assert res["crossSections_C"].shape[0] == 2
    assert res["radialSurfaceProfiles_C"].shape[0] == 2
    # Each profile describes the end of its pulse's period.
    trep = 1.0 / BASE["f_rep"]
    offset = 5.0 * BASE["tau_FWHM"]
    np.testing.assert_allclose(
        res["profileSnapshotTimes_s"],
        [offset + 2 * trep - offset, offset + 4 * trep - offset], rtol=1e-9)


def test_profile_snapshot_default_is_the_log_spaced_set(tmp_path):
    res = _run("depth_profile", {**DEPTH, "nPulses": 3,
                                 "outputDir": str(tmp_path)})
    assert res["profileSnapshotPulses"].tolist() == [1, 2, 3]
    assert res["profileSnapshotTimes_s"][-1] == pytest.approx(3e-6)


def test_profile_snapshot_pulses_outside_the_run_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="profileSnapshotPulses"):
        _run("depth_profile", {**DEPTH, "nPulses": 2,
                               "profileSnapshotPulses": [7],
                               "outputDir": str(tmp_path)})


# ----------------------------------------------------- scanning history


SCAN = {"f_rep": 5e6, "v_scan": 1.0, "scanLength": 4e-6, "makePlots": False,
        "verbose": False}


def test_scanning_map_snapshots(tmp_path):
    res = _run("scanning_beam", {**SCAN, "mapSnapshotPulses": [1, 5, 20, 99],
                                 "outputDir": str(tmp_path)})
    assert res["nPulses"] == 20
    assert res["mapSnapshotPulses"].tolist() == [1, 5, 20]
    assert res["TsurfSnapshots_C"].shape == (
        3, res["yGrid"].size, res["xGrid"].size)
    np.testing.assert_allclose(res["mapSnapshotTimes_s"],
                               np.array([1, 5, 20]) / SCAN["f_rep"])
    np.testing.assert_array_equal(res["TsurfSnapshots_C"][-1], res["Tsurf_C"])
    assert np.all(res["TsurfSnapshots_C"][0] <= res["Tpeak_map_C"] + 1e-9)


def test_scanning_snapshots_do_not_change_the_solution(tmp_path):
    """A snapshot only adds a chunk boundary to the loop."""
    plain = _run("scanning_beam", {**SCAN, "outputDir": str(tmp_path / "a")})
    split = _run("scanning_beam", {**SCAN, "mapSnapshotPulses": [3, 7, 11],
                                   "outputDir": str(tmp_path / "b")})
    np.testing.assert_array_equal(plain["Tsurf"], split["Tsurf"])
    np.testing.assert_array_equal(plain["Tpeak_map"], split["Tpeak_map"])
    np.testing.assert_array_equal(plain["peakT_history"],
                                  split["peakT_history"])
    assert plain["TsurfSnapshots_C"].shape == (
        0, plain["yGrid"].size, plain["xGrid"].size)
