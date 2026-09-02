"""Every solver's results carry the shared envelope.

caseTag, resolvedConfig, materialProps and warnings answer the questions
a machine consumer otherwise has to guess at: what config was actually
in force, which physical numbers the material resolved to, and whether
the run raised a validity concern. All additive on the v1 contract.
"""

import contextlib
import io
import json

import pytest

from laserttm.runtools import get_solver, to_jsonable

FREP = 5e6
DUR = 2 / FREP  # two pulses

ENVELOPE = ("caseTag", "resolvedConfig", "materialProps", "warnings")


def _run(solver_id, cfg):
    with contextlib.redirect_stdout(io.StringIO()):
        return get_solver(solver_id)(cfg)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("envelope"))
    base = {"f_rep": FREP, "simDuration": DUR, "makePlots": False,
            "outputDir": out, "caseTag": "env"}
    results = {
        "surface_point": _run("surface_point", dict(base)),
        "radial_profile": _run("radial_profile", {**base, "Nr": 8}),
        "depth_profile": _run("depth_profile", dict(base)),
        "single_pulse": _run("single_pulse",
                             {"f_rep": FREP, "makePlots": False,
                              "outputDir": out, "caseTag": "env"}),
        "scanning_beam": _run("scanning_beam",
                              {"f_rep": FREP, "v_scan": 1.0,
                               "scanLength": 4e-7, "makePlots": False,
                               "outputDir": out, "caseTag": "env"}),
    }
    results["inversion_quantifier"] = _run(
        "inversion_quantifier",
        {**base, "depthResults": results["depth_profile"]})
    return results


def test_every_solver_returns_the_envelope(runs):
    for solver_id, res in runs.items():
        for key in ENVELOPE:
            assert key in res, f"{solver_id} is missing {key}"
        assert res["caseTag"] == "env", solver_id
        assert isinstance(res["resolvedConfig"], dict), solver_id
        assert isinstance(res["warnings"], list), solver_id


def test_resolved_config_merges_defaults_with_the_caller(runs):
    for solver_id, res in runs.items():
        rc = res["resolvedConfig"]
        # A default the tiny configs never set is present and in force.
        assert rc["material"] == "W", solver_id
        # The caller's own value wins over the default.
        assert rc["f_rep"] == FREP, solver_id


def test_material_props_carry_the_resolved_record(runs):
    for solver_id, res in runs.items():
        mp = res["materialProps"]
        assert mp is not None, solver_id
        assert mp["material"] == "w", solver_id
        assert mp["gamma"] == pytest.approx(137.3), solver_id
        assert mp["Tmelt_C"] == pytest.approx(3422.0), solver_id
        assert "kModel" in mp, solver_id
    # The wrapper passes the depth run's record through unchanged.
    assert (runs["inversion_quantifier"]["materialProps"]
            == runs["depth_profile"]["materialProps"])


def test_the_new_uniform_scalars(runs):
    sp = runs["surface_point"]
    assert sp["wallTime_s"] > 0
    assert sp["finalTl_C"] == pytest.approx(sp["Tl_C"][-1])

    single = runs["single_pulse"]
    assert single["nPulses"] == 1
    assert single["maxInv_K"] == single["maxInv_C"]
    assert single["finalResid_C"] == pytest.approx(single["finalTl_C"])
    assert single["invThreshold_K"] == 0.5

    radial = runs["radial_profile"]
    assert len(radial["TeqVals_C"]) == radial["nPulses"]
    assert radial["nPulsesRequested"] == radial["nPulses"]
    assert radial["earlyStopped"] is False

    depth = runs["depth_profile"]
    assert depth["invThreshold_K"] == 0.5
    assert depth["peakPulse"] >= 1
    assert depth["energyMismatch_pct"] >= 0.0

    scan = runs["scanning_beam"]
    assert scan["peakT_C"] > 0.0
    assert scan["simDuration_s"] == pytest.approx(DUR)

    inv = runs["inversion_quantifier"]
    assert inv["invThreshold_K"] == 0.5
    # The pre-computed depth results dict must not leak into the resolved
    # config: the key stays at its schema default of None.
    assert inv["resolvedConfig"]["depthResults"] is None


def test_the_envelope_serializes(runs):
    for res in runs.values():
        payload = to_jsonable(
            {k: res[k] for k in ENVELOPE}, max_array=10)
        json.dumps(payload)  # must not raise
