"""The input schema: defaults, discovery, and validation.

The default snapshot below is transcribed from the solver source as it stood
before the schema existed. It is what makes pointing the solvers at the
schema a provable no-op rather than a hopeful one, so treat a change here as
a deliberate behavior change, not a test to update.
"""

import pytest

from laserttm import schema

# solver id -> {key: default}, copied from the get_cfg_field call sites.
SNAPSHOT = {
    "surface_point": {
        "material": "W", "Pavg": 1.0, "spotRadius": 80e-6, "f_rep": 1e6,
        "tau_FWHM": 500e-15, "pulseProfile": "gaussian", "absorbance": 0.55,
        "T0_C": 25.0, "simDuration": 100e-6, "Leff": 100e-9,
        "depthProfile": "exponential", "dzTarget": 500e-9, "Ndiff": 100,
        # legacyDeposit False is the 0.2.0 energy-conservation default; True
        # reproduces the MATLAB deposit the fixtures pin.
        "legacyDeposit": False,
        "makePlots": True, "saveFigures": False, "outputDir": None,
        "caseTag": "", "storeHistory": True,
        "showProgress": None,
        "gamma_manual": 137.3, "Cl_manual": 2.54e6, "G_manual": 1.65e17,
        "kl_manual": 174.0, "kTable": "auto",
        "gamma": None, "Cl": None, "G": None, "kl": None, "T_melt_C": None,
    },
    "depth_profile": {
        "material": "W", "Pavg": 40.0, "spotRadius": 100e-6, "f_rep": 18e6,
        "tau_FWHM": 100e-15, "pulseProfile": "gaussian", "absorbance": 0.55,
        "T0_C": 25.0, "simDuration": 100e-6, "Lz": 1000e-9, "Nz": 200,
        "enableRadialProfile": True, "Nr_radial": 20, "rMax_factor": 3.0,
        "dzTarget_diff": 500e-9, "Ndiff": 100, "relTol": 1e-6, "absTol": 1e-1,
        "storeHistory": True,
        "makePlots": True, "saveFigures": False, "outputDir": None,
        "caseTag": "", "showProgress": None,
        "gamma_manual": 137.3, "Cl_manual": 2.54e6, "G_manual": 1.65e17,
        "ke0_manual": 150.0, "kl_manual": 24.0, "alpha_opt_manual": 5.88e7,
        "kTable": "auto", "gamma": None, "Cl": None, "G": None, "kl": None,
        "ke0": None, "alpha_opt": None, "delta_opt": None, "T_melt_C": None,
    },
    "radial_profile": {
        "material": "W", "Pavg": 40.0, "spotRadius": 100e-6, "f_rep": 18e6,
        "tau_FWHM": 100e-15, "pulseProfile": "gaussian", "absorbance": 0.55,
        "T0_C": 25.0, "simDuration": 1e-3, "Leff": 100e-9,
        "depthProfile": "exponential", "dzTarget": 500e-9, "Ndiff": 100,
        "Nr": 80, "rMax_factor": 5.0, "radialSolveMode": "scale",
        "earlyStopMeltRadius_um": 0.0, "earlyStopT_melt_C": None,
        "earlyStopCheckInterval": 100, "legacyDeposit": False,
        "storeHistory": True,
        "makePlots": True, "saveFigures": False, "outputDir": None,
        "caseTag": "", "showProgress": None,
        "gamma_manual": 137.3, "Cl_manual": 2.54e6, "G_manual": 1.65e17,
        "kl_manual": 174.0, "kTable": "auto",
        "gamma": None, "Cl": None, "G": None, "kl": None, "T_melt_C": None,
    },
    "single_pulse": {
        "material": "W", "Pavg": 1.0, "spotRadius": 80e-6, "f_rep": 1e6,
        "tau_FWHM": 100e-15, "pulseProfile": "gaussian", "absorbance": 0.55,
        "T0_C": 25.0, "Lz": 1000e-9, "Nz": 200, "relTol": 1e-6,
        "absTol": 1e-1, "makePlots": True, "saveFigures": False,
        "outputDir": None, "caseTag": "",
        "gamma_manual": 137.3, "Cl_manual": 2.54e6, "G_manual": 1.65e17,
        "ke0_manual": 150.0, "kl_manual": 24.0, "alpha_opt_manual": 5.88e7,
        "kTable": "auto", "gamma": None, "Cl": None, "G": None, "kl": None,
        "ke0": None, "alpha_opt": None, "delta_opt": None, "T_melt_C": None,
    },
    "scanning_beam": {
        "material": "W", "Pavg": 40.0, "spotRadius": 100e-6, "f_rep": 18e6,
        "tau_FWHM": 100e-15, "pulseProfile": "gaussian", "absorbance": 0.55,
        "T0_C": 25.0, "v_scan": 1.0, "scanLength": 2e-3, "Leff": 100e-9,
        "depthProfile": "exponential", "dzTarget": 500e-9, "Ndiff": 100,
        "NadiPerGap": 10, "legacyDeposit": False,
        "Nx": 120, "Ny": 60, "xPad": 3.0, "yExtent": 5.0,
        "makePlots": True, "saveFigures": False, "outputDir": None,
        "caseTag": "", "showProgress": None,
        "gamma_manual": 137.3, "Cl_manual": 2.54e6, "G_manual": 1.65e17,
        "kl_manual": 174.0, "kTable": "auto",
        "gamma": None, "Cl": None, "G": None, "kl": None, "T_melt_C": None,
    },
}


@pytest.mark.parametrize("solver_id", sorted(SNAPSHOT))
def test_defaults_match_the_source_snapshot(solver_id):
    got = schema.defaults(solver_id)
    expected = SNAPSHOT[solver_id]
    assert set(got) - {"snapshotDelays"} == set(expected), (
        f"{solver_id} parameter set drifted from the snapshot")
    for key, value in expected.items():
        assert got[key] == value, f"{solver_id}.{key}"


def test_snapshot_delays_default_is_the_matlab_set():
    for solver_id in ("depth_profile", "single_pulse"):
        assert schema.defaults(solver_id)["snapshotDelays"] == (
            0.0, 0.5e-12, 1e-12, 2e-12, 5e-12, 10e-12, 50e-12, 200e-12)


def test_every_solver_is_registered():
    from laserttm.runtools import SOLVER_DESCRIPTIONS

    assert set(schema.SOLVER_IDS) == set(SOLVER_DESCRIPTIONS)


def test_kl_means_different_things_in_the_two_families():
    """The collision is real, so the schema states it rather than hiding it."""
    lattice = schema.solver_schema("radial_profile").params["kl"]
    optical = schema.solver_schema("depth_profile").params["kl"]
    assert "total" in lattice.summary.lower()
    assert "lattice" in optical.summary.lower()
    assert optical.notes and lattice.notes


def test_ndiff_is_documented_as_a_floor_only_in_the_radial_solver():
    assert "minimum" in schema.solver_schema(
        "radial_profile").params["Ndiff"].notes.lower()
    assert not schema.solver_schema("depth_profile").params["Ndiff"].notes


def test_rmax_factor_is_marked_plot_only_in_the_depth_solver():
    depth = schema.solver_schema("depth_profile").params["rMax_factor"]
    radial = schema.solver_schema("radial_profile").params["rMax_factor"]
    assert not depth.affects_numerics
    assert radial.affects_numerics


# ----------------------------  discovery  ---------------------------------


def test_describe_solver_carries_units_and_ranges():
    described = schema.describe_solver("depth_profile")
    spot = described["params"]["spotRadius"]
    assert spot["unit"] == "m"
    assert spot["range"] == [1e-9, 1e-2]
    assert described["whenToUse"] and described["whenNotToUse"]


def test_describe_solver_sections():
    assert "params" not in schema.describe_solver("surface_point", "examples")
    assert "examples" in schema.describe_solver("surface_point", "examples")


def test_list_solvers_gives_routing_information():
    rows = {r["id"]: r for r in schema.list_solvers()}
    assert set(rows) == set(schema.SOLVER_IDS)
    for row in rows.values():
        assert row["whenToUse"] and row["whenNotToUse"]


def test_json_schema_rejects_unknown_properties():
    js = schema.json_schema("radial_profile")
    assert js["additionalProperties"] is False
    assert js["properties"]["radialSolveMode"]["enum"] == ["scale",
                                                           "independent"]
    assert js["properties"]["spotRadius"]["x-unit"] == "m"


def test_unknown_solver_names_the_known_ones():
    with pytest.raises(KeyError, match="radial_profile"):
        schema.solver_schema("radial")


# ----------------------------  validation  --------------------------------


def test_valid_config_passes_with_an_estimate():
    result = schema.validate_config("surface_point",
                                    {"Pavg": 10, "f_rep": 5e6,
                                     "simDuration": 50 / 5e6})
    assert result["ok"]
    assert result["errors"] == []
    assert result["estimate"]["nPulses"] == 50


def test_misspelled_key_is_rejected_not_ignored():
    result = schema.validate_config("surface_point", {"pavg": 10})
    assert not result["ok"]
    (problem,) = result["errors"]
    assert problem["code"] == "unknown_key"
    assert "'Pavg'" in problem["suggestion"]
    assert "case-sensitive" in problem["suggestion"]


def test_key_belonging_to_another_solver_says_which():
    result = schema.validate_config("depth_profile", {"Leff": 100e-9})
    (problem,) = result["errors"]
    assert problem["code"] == "unknown_key"
    assert "radial_profile" in problem["suggestion"]


def test_close_misspelling_is_suggested():
    result = schema.validate_config("radial_profile", {"spotRadus": 1e-4})
    assert "spotRadius" in result["errors"][0]["suggestion"]


def test_unit_slip_is_named():
    result = schema.validate_config("depth_profile", {"spotRadius": 100})
    (problem,) = result["errors"]
    assert problem["code"] == "unit_slip"
    assert "micrometres" in problem["suggestion"]
    assert "100e-06" in problem["suggestion"] or "0.0001" in problem["suggestion"]


def test_percentage_absorbance_is_caught():
    result = schema.validate_config("surface_point", {"absorbance": 55})
    (problem,) = result["errors"]
    assert "percentage" in problem["suggestion"]


def test_all_problems_are_reported_at_once():
    result = schema.validate_config(
        "depth_profile", {"pavg": 10, "spotRadius": 100, "Nz": 1})
    assert len(result["errors"]) == 3
    assert {p["code"] for p in result["errors"]} == {
        "unknown_key", "unit_slip", "out_of_range"}


def test_bad_enum_lists_the_choices():
    result = schema.validate_config("radial_profile",
                                    {"radialSolveMode": "fast"})
    (problem,) = result["errors"]
    assert problem["code"] == "bad_enum"
    assert "independent" in problem["suggestion"]


def test_empty_values_fall_back_like_matlab():
    result = schema.validate_config("surface_point",
                                    {"material": "", "showProgress": None})
    assert result["ok"]


def test_unusual_but_legal_values_warn_without_failing():
    result = schema.validate_config("surface_point", {"tau_FWHM": 5e-11})
    assert result["ok"]
    assert any(w["code"] == "unusual_value" for w in result["warnings"])


def test_expensive_run_is_flagged():
    result = schema.validate_config("radial_profile",
                                    {"simDuration": 1.0, "f_rep": 40e6})
    assert result["ok"]
    assert any(w["code"] == "expensive_run" for w in result["warnings"])


# -----------------------------  estimates  --------------------------------


def test_estimate_counts_pulses_the_way_the_solvers_do():
    assert schema.estimate_run("radial_profile", {})["nPulses"] == 18000
    assert schema.estimate_run("surface_point", {})["nPulses"] == 100
    assert schema.estimate_run("single_pulse", {})["nPulses"] == 1
    assert schema.estimate_run("scanning_beam", {})["nPulses"] == 36000


def test_independent_mode_is_estimated_as_more_expensive():
    scaled = schema.estimate_run("radial_profile", {"Nr": 80})
    independent = schema.estimate_run(
        "radial_profile", {"Nr": 80, "radialSolveMode": "independent"})
    assert independent["estRuntime_s"] > scaled["estRuntime_s"] * 10


def test_enum_choices_match_what_the_code_accepts():
    """An enum the schema advertises but no consumer implements is worse than
    no schema: validation passes and the run then fails."""
    from laserttm.kernels import _PROFILE_CODES
    from laserttm.materials import MATERIALS
    from laserttm.radial_profile import _SOLVE_MODES

    # Material names are matched case-insensitively, so compare them that way.
    expected = {
        "pulseProfile": set(_PROFILE_CODES),
        "radialSolveMode": set(_SOLVE_MODES),
        "material": set(MATERIALS) | {"custom"},
    }
    for sid in schema.SOLVER_IDS:
        params = schema.solver_schema(sid).params
        for key, accepted in expected.items():
            if key in params:
                choices = {c.lower() for c in params[key].choices}
                assert choices == accepted, f"{sid}.{key}"


def test_every_advertised_pulse_profile_actually_runs():
    from laserttm.kernels import profile_code

    for name in schema.PARAMS["pulseProfile"].choices:
        profile_code(name)   # raises if the kernel does not implement it


def test_zero_pulse_config_is_rejected_with_the_fix_named():
    """simDuration below half a pulse period rounds to no pulses. It used to
    validate cleanly and then fail deep inside the solver."""
    result = schema.validate_config(
        "surface_point", {"f_rep": 1e6, "simDuration": 1e-9})
    assert not result["ok"]
    (problem,) = [p for p in result["errors"] if p["code"] == "no_pulses"]
    assert "simDuration = N / f_rep" in problem["suggestion"]


def test_zero_pulse_scan_names_the_scanning_keys():
    result = schema.validate_config(
        "scanning_beam", {"scanLength": 1e-9, "v_scan": 1.0, "f_rep": 1e3})
    assert not result["ok"]
    (problem,) = [p for p in result["errors"] if p["code"] == "no_pulses"]
    assert "scanLength" in problem["suggestion"]


def test_one_pulse_is_still_allowed():
    result = schema.validate_config(
        "surface_point", {"f_rep": 1e6, "simDuration": 1 / 1e6})
    assert result["ok"]
    assert result["estimate"]["nPulses"] == 1


def test_radial_node_minimum_matches_the_stencil():
    """The cylindrical Crank-Nicolson stencil divides by (Nr - 2), so two
    nodes is not a usable grid."""
    assert schema.PARAMS["Nr"].minimum == 3
    assert not schema.validate_config("radial_profile", {"Nr": 2})["ok"]
    assert schema.validate_config("radial_profile", {"Nr": 3})["ok"]


def test_require_pulses_guards_direct_python_calls():
    schema.require_pulses("surface_point", 1)          # fine
    with pytest.raises(ValueError, match="0 pulses"):
        schema.require_pulses("surface_point", 0)
    with pytest.raises(ValueError, match="scanLength"):
        schema.require_pulses("scanning_beam", 0)


def test_schema_import_stays_light():
    """Discovery must not drag in numba or scipy: it is called to decide
    whether a run is worth starting."""
    import subprocess
    import sys

    code = ("import sys, laserttm.schema as s; "
            "s.describe_solver('depth_profile'); "
            "print(int(any(m in sys.modules for m in "
            "('numba', 'scipy', 'matplotlib'))))")
    out = subprocess.run([sys.executable, "-c", code], check=True,
                         capture_output=True, text=True)
    assert out.stdout.strip() == "0"
