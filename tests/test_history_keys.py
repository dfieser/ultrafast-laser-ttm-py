"""The depth solvers return the arrays their figures are drawn from.

A consumer that animates the electron-lattice inversion or replots a depth
profile needs the surface traces and the depth snapshots as arrays, not a
report file. These keys are additive on the v1 contract.
"""

import contextlib
import io

import numpy as np

from laserttm.runtools import get_solver

INVERTING = {"material": "W", "Pavg": 12.0, "f_rep": 1e6,
             "tau_FWHM": 500e-15, "spotRadius": 80e-6, "makePlots": False}


def _run(solver_id, cfg):
    with contextlib.redirect_stdout(io.StringIO()):
        return get_solver(solver_id)(cfg)


def test_single_pulse_returns_traces_and_snapshots(tmp_path):
    res = _run("single_pulse", {**INVERTING, "outputDir": str(tmp_path)})
    nz = res["resolvedConfig"]["Nz"]
    n_snap = res["snapshotDelays_s"].size
    assert 0 < n_snap <= len(res["resolvedConfig"]["snapshotDelays"])
    assert res["TeSnapshots_C"].shape == (n_snap, nz)
    assert res["TlSnapshots_C"].shape == (n_snap, nz)
    assert res["zGrid_m"].shape == (nz,)
    assert res["time_s"].size == res["Te_C"].size == res["Tl_C"].size > 0
    assert np.all(np.diff(res["time_s"]) >= 0)
    # Each snapshot's surface node lies within the surface trace's range.
    for k in range(n_snap):
        assert res["Te_C"].min() <= res["TeSnapshots_C"][k, 0] <= res["Te_C"].max()
        assert res["Tl_C"].min() <= res["TlSnapshots_C"][k, 0] <= res["Tl_C"].max()
    # The inversion the scalars report is visible in the trace itself.
    assert res["invDetected"]
    diff = res["Tl_C"] - res["Te_C"]
    assert diff.max() > res["invThreshold_K"]
    assert abs(diff.max() - res["maxInv_K"]) < 1e-6


def test_depth_profile_returns_snapshots_traces_and_profiles(tmp_path):
    cfg = {**INVERTING, "simDuration": 3e-6, "outputDir": str(tmp_path)}
    res = _run("depth_profile", cfg)
    nz = res["resolvedConfig"]["Nz"]
    n_snap = res["snapshotDelays_s"].size
    assert n_snap > 0
    assert res["TeSnapshots_C"].shape == (n_snap, nz)
    assert res["TlSnapshots_C"].shape == (n_snap, nz)
    assert res["time_s"].size == res["Te_C"].size == res["Tl_C"].size > 0
    pulses = res["profileSnapshotPulses"]
    assert pulses[0] == 1 and pulses[-1] == res["nPulses"]
    assert res["profileSnapshots_C"].shape == (pulses.size, res["zGridDiff_m"].size)
    # Residual profiles rise at the surface with pulse count and fall with depth.
    assert res["profileSnapshots_C"][-1, 0] >= res["profileSnapshots_C"][0, 0]
    assert res["profileSnapshots_C"][-1, 0] >= res["profileSnapshots_C"][-1, -1]


def test_depth_profile_without_history_keeps_the_keys_but_empty(tmp_path):
    cfg = {**INVERTING, "simDuration": 2e-6, "storeHistory": False,
           "outputDir": str(tmp_path)}
    res = _run("depth_profile", cfg)
    assert res["time_s"].size == 0 and res["Te_C"].size == 0
    assert res["TeSnapshots_C"].shape[0] == res["snapshotDelays_s"].size > 0
