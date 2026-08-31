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
