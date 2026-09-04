"""The depth solver's radial view is part of the solution, not a figure.

It used to be computed inside plot_depth_profile, so it existed only when
figures were drawn. Both the CLI and the MCP server default to no figures,
which meant enableRadialProfile silently did nothing under the two most
common entry points, and the accuracy warning never reached those callers.
"""

import warnings

import numpy as np
import pytest

from laserttm import depth_profile_solver

BASE = {
    "material": "W", "Pavg": 40, "spotRadius": 100e-6, "f_rep": 18e6,
    "tau_FWHM": 500e-15, "simDuration": 3 / 18e6,
    "Nz": 60, "Lz": 400e-9, "makePlots": False,
    "enableRadialProfile": True,   # off by default since 0.4.0
}

RADIAL_KEYS = ("radialGrid_um", "radialFluenceRatio",
               "radialSurfaceProfiles_C", "crossSection_C",
               "crossSections_C", "lateralDiffusionLength_m",
               "lateralDiffusionRatio")


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    out = tmp_path_factory.mktemp("radial_view")
    return depth_profile_solver({**BASE, "outputDir": str(out)})


def test_radial_view_is_returned_without_figures(results):
    for key in RADIAL_KEYS:
        assert key in results, key


def test_radial_grid_spans_the_requested_extent(results):
    r_um = results["radialGrid_um"]
    assert r_um.size == 20                      # Nr_radial default
    assert r_um[0] == 0.0
    # rMax_factor default 3, spot radius 100 um
    assert r_um[-1] == pytest.approx(300.0)


def test_fluence_ratio_is_gaussian_in_radius(results):
    r_m = results["radialGrid_um"] * 1e-6
    expected = np.exp(-2.0 * r_m**2 / (100e-6) ** 2)
    np.testing.assert_allclose(results["radialFluenceRatio"], expected)
    assert results["radialFluenceRatio"][0] == pytest.approx(1.0)


def test_surface_profiles_are_hottest_at_the_centre(results):
    profiles = results["radialSurfaceProfiles_C"]
    assert profiles.shape[1] == results["radialGrid_um"].size
    for profile in profiles:
        assert profile[0] == profile.max()
        assert profile[0] > profile[-1]


def test_cross_section_is_radius_by_depth(results):
    section = results["crossSection_C"]
    assert section.shape[0] == results["radialGrid_um"].size
    # Beam centre, surface node, is the hottest point of the map.
    assert section[0, 0] == pytest.approx(section.max())


def test_centre_of_the_radial_view_matches_the_1d_solution(results):
    """The view is a scaling of the depth solve, so r=0 must reproduce it."""
    centre = results["radialSurfaceProfiles_C"][-1][0]
    assert centre == pytest.approx(results["crossSection_C"][0, 0])


def test_disabling_it_really_disables_it(tmp_path):
    results = depth_profile_solver(
        {**BASE, "enableRadialProfile": False, "outputDir": str(tmp_path)})
    for key in RADIAL_KEYS:
        assert key not in results


def test_it_is_off_by_default(tmp_path):
    cfg = {k: v for k, v in BASE.items() if k != "enableRadialProfile"}
    results = depth_profile_solver({**cfg, "outputDir": str(tmp_path)})
    assert "crossSection_C" not in results
    assert results["resolvedConfig"]["enableRadialProfile"] is False


def test_cross_sections_are_the_time_resolved_cross_section(results):
    """A snapshot at the last pulse equals crossSection_C, and every
    snapshot is a radius-by-depth map on the same grids."""
    stack = results["crossSections_C"]
    assert stack.shape == (results["profileSnapshotPulses"].size,
                           results["radialGrid_um"].size,
                           results["zGridDiff_m"].size)
    np.testing.assert_array_equal(stack[-1], results["crossSection_C"])
    assert results["lateralDiffusionRatio"] == pytest.approx(
        results["lateralDiffusionLength_m"] / BASE["spotRadius"])


def test_accuracy_warning_reaches_callers_who_never_asked_for_figures(tmp_path):
    """Lateral diffusion above 10% of the spot invalidates the scaling."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = depth_profile_solver({**BASE, "spotRadius": 5e-6,
                                    "outputDir": str(tmp_path)})
    assert any("Lateral diffusion" in str(w.message) for w in caught)
    assert "lateral_diffusion" in [d["code"] for d in res["diagnostics"]]
    # The threshold is a config key, so the same run can be told it is fine.
    quiet = depth_profile_solver({**BASE, "spotRadius": 5e-6,
                                  "lateralDiffusionWarnRatio": 100.0,
                                  "outputDir": str(tmp_path / "quiet")})
    assert "lateral_diffusion" not in [d["code"] for d in quiet["diagnostics"]]
