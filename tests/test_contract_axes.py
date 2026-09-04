"""Every array a solver returns declares its axes, and the axes fit.

A consumer animating a depth snapshot or a scanning map needs to know
which axis is which and where its coordinates live. The registry says so
through ``dims``, one results key per axis, and this file holds the live
arrays to that declaration for every solver and every gating flag.
"""

import contextlib
import io

import numpy as np
import pytest

from laserttm import schema
from laserttm.runtools import get_solver

FREP = 5e6
DUR = 2 / FREP


def _run(solver_id, cfg):
    with contextlib.redirect_stdout(io.StringIO()):
        return get_solver(solver_id)(cfg)


def _rows(solver_id):
    return {f.name: f
            for f in schema.RESULT_ENVELOPE + schema.RESULTS[solver_id]}


def _axis_length(res, dim):
    value = res[dim]
    if isinstance(value, np.ndarray):
        return value.shape[0]
    return int(value)


def _check_axes(solver_id, res):
    rows = _rows(solver_id)
    for key, value in res.items():
        row = rows[key]
        if isinstance(value, np.ndarray):
            assert row.kind == "array", f"{solver_id}.{key} is not declared an array"
            assert row.dims, f"{solver_id}.{key} declares no dims"
            assert value.ndim == len(row.dims), (
                f"{solver_id}.{key}: ndim {value.ndim} vs dims {row.dims}")
            for axis, dim in zip(value.shape, row.dims):
                assert dim in res, f"{solver_id}.{key}: axis key {dim} missing"
                assert axis == _axis_length(res, dim), (
                    f"{solver_id}.{key}: axis {dim} has {axis} entries, "
                    f"{dim} has {_axis_length(res, dim)}")
        else:
            assert row.kind != "array", (
                f"{solver_id}.{key} is declared an array but is "
                f"{type(value).__name__}")


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("axes"))
    base = {"f_rep": FREP, "simDuration": DUR, "makePlots": False,
            "outputDir": out, "verbose": False}
    results = {
        "surface_point": _run("surface_point", dict(base)),
        "radial_profile": _run("radial_profile", {**base, "Nr": 8}),
        "depth_profile": _run("depth_profile",
                              {**base, "Nz": 60, "Lz": 400e-9,
                               "enableRadialProfile": True}),
        "single_pulse": _run("single_pulse",
                             {"f_rep": FREP, "makePlots": False,
                              "outputDir": out, "verbose": False}),
        "scanning_beam": _run("scanning_beam",
                              {"f_rep": FREP, "v_scan": 1.0,
                               "scanLength": 6e-7, "makePlots": False,
                               "outputDir": out, "verbose": False,
                               "mapSnapshotPulses": [1, 3]}),
    }
    results["inversion_quantifier"] = _run(
        "inversion_quantifier",
        {**base, "depthResults": results["depth_profile"]})
    return results


def test_every_registry_array_declares_dims_of_known_keys():
    for solver_id in schema.SOLVER_IDS:
        rows = _rows(solver_id)
        for row in rows.values():
            if row.kind == "array":
                assert row.dims, f"{solver_id}.{row.name} has no dims"
                for dim in row.dims:
                    assert dim in rows, f"{solver_id}.{row.name}: {dim}"
                    assert rows[dim].kind in ("array", "scalar")
            else:
                assert row.dims is None, f"{solver_id}.{row.name}"


def test_live_arrays_match_their_declared_axes(runs):
    for solver_id, res in runs.items():
        _check_axes(solver_id, res)


def test_describe_results_carries_dims():
    by_name = {f["name"]: f for f in schema.describe_results("depth_profile")}
    assert by_name["TeSnapshots_C"]["dims"] == ["snapshotDelays_s", "zGrid_m"]
    assert by_name["crossSections_C"]["dims"] == [
        "profileSnapshotPulses", "radialGrid_um", "zGridDiff_m"]
    assert "dims" not in by_name["peakTl_C"]


@pytest.mark.parametrize("radial", [False, True])
@pytest.mark.parametrize("history", [False, True])
def test_depth_profile_emits_exactly_the_documented_keys(tmp_path, radial,
                                                          history):
    """For every combination of gating flags the emitted key set equals
    the documented one: non-gated keys always, gated keys iff their gate
    is on. storeHistory only empties arrays, it never removes keys."""
    res = _run("depth_profile", {
        "f_rep": FREP, "simDuration": DUR, "Nz": 60, "Lz": 400e-9,
        "makePlots": False, "outputDir": str(tmp_path), "verbose": False,
        "enableRadialProfile": radial, "storeHistory": history})
    rows = _rows("depth_profile")
    expected = {name for name, row in rows.items()
                if row.gated_by is None
                or (row.gated_by == "enableRadialProfile" and radial)}
    assert set(res) == expected
    _check_axes("depth_profile", res)
    assert (res["time_s"].size > 0) is history


def test_scanning_maps_have_celsius_twins_and_positions(runs):
    scan = runs["scanning_beam"]
    np.testing.assert_allclose(scan["Tpeak_map_C"], scan["Tpeak_map"] - 273.15)
    np.testing.assert_allclose(scan["Tsurf_C"], scan["Tsurf"] - 273.15)
    np.testing.assert_allclose(scan["peakT_history_C"],
                               scan["peakT_history"] - 273.15)
    assert scan["peakTl_C"] == scan["peakT_C"]
    # One beam position per pulse, spaced by v_scan / f_rep from x = 0.
    assert scan["beamX_m"].shape == (scan["nPulses"],)
    np.testing.assert_allclose(np.diff(scan["beamX_m"]), scan["pulseSpacing"])
    assert scan["beamX_m"][0] == 0.0
