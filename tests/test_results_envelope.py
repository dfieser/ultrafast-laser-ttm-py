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


def test_the_conductivity_model_claim_matches_what_each_solver_ran(runs):
    """kModel must state the model actually used, not the material's best.

    Tungsten carries a measured k(T) table, but only the depth and radial
    solvers run it. The 0D solvers and single_pulse run a constant k, and
    their record has to say so.
    """
    measured = {"depth_profile", "radial_profile", "inversion_quantifier"}
    for solver_id, res in runs.items():
        k_model = res["materialProps"]["kModel"]
        if solver_id in measured:
            assert k_model.startswith("measured"), (solver_id, k_model)
        else:
            assert k_model.startswith("constant"), (solver_id, k_model)


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


def test_every_emitted_key_is_documented_and_vice_versa(runs):
    """The registry in schema.py is the results contract, enforced.

    A solver emitting a key with no registry row fails here, and so does
    a registry row no solver honors. Gated rows may be absent when their
    config key is off, which it is in these tiny runs.
    """
    from laserttm import schema

    for solver_id, res in runs.items():
        emitted = set(res)
        rows = {f.name: f
                for f in schema.RESULT_ENVELOPE + schema.RESULTS[solver_id]}

        undocumented = emitted - set(rows)
        assert not undocumented, (
            f"{solver_id} returns undocumented keys: {sorted(undocumented)}. "
            "Add a ResultField row in schema.py in the same commit.")

        missing = {name for name, f in rows.items()
                   if name not in emitted and f.gated_by is None}
        assert not missing, (
            f"{solver_id} does not return documented keys: "
            f"{sorted(missing)}")


def test_gated_keys_appear_when_their_gate_is_on(runs, tmp_path):
    cfg = {"f_rep": FREP, "simDuration": DUR, "makePlots": False,
           "outputDir": str(tmp_path), "enableRadialProfile": True}
    res = _run("depth_profile", cfg)
    from laserttm import schema
    gated = {f.name for f in schema.RESULTS["depth_profile"]
             if f.gated_by == "enableRadialProfile"}
    assert gated <= set(res)
    assert res["warnings"] == [] or all(
        isinstance(w, str) for w in res["warnings"])


def test_describe_results_and_the_solver_description():
    from laserttm import schema

    fields = schema.describe_results("depth_profile")
    names = [f["name"] for f in fields]
    assert names[0] == "solver"          # envelope first
    assert "invMaxPerPulse_K" in names
    by_name = {f["name"]: f for f in fields}
    assert by_name["radialGrid_um"]["gatedBy"] == "enableRadialProfile"
    assert by_name["gamma"]["prefer"] == "materialProps"

    described = schema.describe_solver("single_pulse", section="results")
    assert {f["name"] for f in described["results"]} == {
        f.name for f in (schema.RESULT_ENVELOPE
                         + schema.RESULTS["single_pulse"])}

    with pytest.raises(KeyError):
        schema.describe_results("no_such_solver")


def test_the_contract_doc_is_current():
    """docs/results-contract.md must be regenerated with registry edits."""
    import importlib.util
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "generate_contract", root / "docs" / "generate_contract.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)

    on_disk = (root / "docs" / "results-contract.md").read_text(
        encoding="utf-8")
    assert on_disk == gen.render(), (
        "docs/results-contract.md is stale. Run "
        "'python docs/generate_contract.py' and commit the result.")
