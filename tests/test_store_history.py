"""storeHistory=False must change memory behavior only, never the physics."""

import numpy as np

from laserttm import radial_profile_solver


def _base_cfg(tmp_path, **overrides):
    cfg = {
        "material": "W",
        "Pavg": 40,
        "spotRadius": 100e-6,
        "f_rep": 18e6,
        "tau_FWHM": 500e-15,
        "absorbance": 0.55,
        "Nr": 24,
        "rMax_factor": 4,
        "radialSolveMode": "scale",
        "makePlots": False,
        "simDuration": 15 / 18e6,
        "outputDir": str(tmp_path),
    }
    cfg.update(overrides)
    return cfg


def test_scale_mode_results_identical(tmp_path):
    with_hist = radial_profile_solver(_base_cfg(tmp_path / "a"))
    without = radial_profile_solver(
        _base_cfg(tmp_path / "b", storeHistory=False))

    np.testing.assert_array_equal(with_hist["finalRadialProfile_C"],
                                  without["finalRadialProfile_C"])
    assert with_hist["finalResid_C"] == without["finalResid_C"]
    assert with_hist["peakTeq_C"] == without["peakTeq_C"]


def test_independent_mode_results_identical(tmp_path):
    with_hist = radial_profile_solver(
        _base_cfg(tmp_path / "a", radialSolveMode="independent",
                  Nr=12, simDuration=5 / 18e6))
    without = radial_profile_solver(
        _base_cfg(tmp_path / "b", radialSolveMode="independent",
                  Nr=12, simDuration=5 / 18e6, storeHistory=False))

    np.testing.assert_array_equal(with_hist["finalRadialProfile_C"],
                                  without["finalRadialProfile_C"])
    assert with_hist["finalResid_C"] == without["finalResid_C"]


def test_surface_point_results_identical(tmp_path):
    from laserttm import surface_point_solver

    cfg = {"Pavg": 10, "f_rep": 5e6, "tau_FWHM": 500e-15,
           "simDuration": 5 / 5e6, "makePlots": False}
    with_hist = surface_point_solver(
        {**cfg, "outputDir": str(tmp_path / "a")})
    without = surface_point_solver(
        {**cfg, "outputDir": str(tmp_path / "b"), "storeHistory": False})

    for key in ("finalResid_C", "peakTe_C", "peakTl_C", "peakPulse",
                "absorbedAreal_J_m2",
                "depthEnergy_J_m2"):
        assert with_hist[key] == without[key], key
    np.testing.assert_array_equal(with_hist["TeqVals_C"],
                                  without["TeqVals_C"])
    np.testing.assert_array_equal(with_hist["TresidVals_C"],
                                  without["TresidVals_C"])
    assert without["time_s"].size == 0
    assert with_hist["time_s"].size > 0

    with open(without["outputFile"], encoding="utf-8") as f:
        report = f.read()
    assert "XY Data" not in report
    assert "not retained" in report


def test_depth_profile_results_identical(tmp_path):
    from laserttm import depth_profile_solver

    cfg = {"Pavg": 40, "f_rep": 18e6, "tau_FWHM": 500e-15,
           "simDuration": 3 / 18e6, "Nz": 60, "Lz": 400e-9,
           "makePlots": False, "enableRadialProfile": True}
    with_hist = depth_profile_solver(
        {**cfg, "outputDir": str(tmp_path / "a")})
    without = depth_profile_solver(
        {**cfg, "outputDir": str(tmp_path / "b"), "storeHistory": False})

    for key in ("finalResid_C", "peakTe_C", "peakTl_C", "nPulses"):
        assert with_hist[key] == without[key], key
    for key in ("TeqVals_C", "TresidVals_C", "invMaxPerPulse_K",
                "TePeakPerPulse_C", "TlPeakPerPulse_C"):
        np.testing.assert_array_equal(with_hist[key], without[key])
    # The radial view is part of the solution, not the history.
    np.testing.assert_array_equal(with_hist["radialSurfaceProfiles_C"],
                                  without["radialSurfaceProfiles_C"])

    with open(without["outputFile"], encoding="utf-8") as f:
        report = f.read()
    assert "XY Data" not in report
