"""The default deposit conserves the pulse energy on any grid.

The MATLAB reference raises the surface node to Teq and lets the rise
decay over Leff. On a grid coarser than Leff that injects roughly
(dz/2)/Leff times the intended layer energy per pulse, and it always
drops the electron bath's share. legacyDeposit=True reproduces the
reference exactly, and the golden-fixture tests run with it; the default
deposit is exact by construction, and these tests are its guarantee.
"""

import contextlib
import io

import numpy as np
import pytest

from laserttm import physics
from laserttm.kernels import _deposit_amp
from laserttm.runtools import get_solver

GAMMA, CL = 137.3, 2.54e6  # tungsten
_trapz = physics.trapezoid


def _grid(dz, n):
    return np.arange(n) * dz


def _layer_energy(teq, t_base, leff):
    return (physics.bath_energy_density(teq, teq, GAMMA, CL)
            - physics.bath_energy_density(t_base, t_base, GAMMA, CL)) * leff


def test_the_corrected_exponential_deposit_is_exact_on_a_coarse_grid():
    dz, leff, teq = 500e-9, 23e-9, 3000.0
    z = _grid(dz, 101)
    exp_z, box = physics.depth_deposit_shape(z, leff)
    w = physics.deposit_shape_weight(z, leff, True)
    tz0 = np.full(z.size, 300.0)
    amp = physics.deposit_amplitude(teq, tz0[0], GAMMA, CL, leff, w)
    tz = physics.deposit_pulse(tz0.copy(), teq, exp_z, box, True,
                               amplitude=amp)
    deposited = CL * _trapz(tz - tz0, z)
    assert deposited == pytest.approx(_layer_energy(teq, 300.0, leff),
                                      rel=1e-12)


def test_the_corrected_box_deposit_is_exact_on_a_coarse_grid():
    dz, leff, teq = 500e-9, 23e-9, 3000.0
    z = _grid(dz, 101)
    exp_z, box = physics.depth_deposit_shape(z, leff)
    w = physics.deposit_shape_weight(z, leff, False)
    tz0 = np.full(z.size, 300.0)
    amp = physics.deposit_amplitude(teq, tz0[0], GAMMA, CL, leff, w)
    tz = physics.deposit_pulse(tz0.copy(), teq, exp_z, box, False,
                               amplitude=amp)
    deposited = CL * _trapz(tz - tz0, z)
    assert deposited == pytest.approx(_layer_energy(teq, 300.0, leff),
                                      rel=1e-12)


def test_the_legacy_deposit_overshoots_by_the_predicted_factor():
    dz, leff, teq = 500e-9, 23e-9, 3000.0
    z = _grid(dz, 101)
    exp_z, box = physics.depth_deposit_shape(z, leff)
    tz0 = np.full(z.size, 300.0)
    tz = physics.deposit_pulse(tz0.copy(), teq, exp_z, box, True)
    deposited = CL * _trapz(tz - tz0, z)
    lattice_layer = CL * (teq - 300.0) * leff
    # The exponential's trapezoid weight collapses to ~dz/2 on this grid.
    assert deposited / lattice_layer == pytest.approx((dz / 2) / leff,
                                                      rel=0.01)


def test_the_corrected_deposit_reduces_to_the_legacy_one_when_resolved():
    # With the electron share removed (gamma -> 0), the only difference
    # left is the shape's discretization, which vanishes as dz/Leff does.
    dz, leff, teq = 1e-9, 100e-9, 3000.0
    z = _grid(dz, 5001)  # 5 um deep, 50 Leff
    w = physics.deposit_shape_weight(z, leff, True)
    assert w == pytest.approx(leff, rel=1e-4)
    amp = physics.deposit_amplitude(teq, 300.0, 1e-12, CL, leff, w)
    assert amp == pytest.approx(teq - 300.0, rel=1e-4)


def test_the_jitted_amplitude_matches_the_physics_one():
    args = (3123.4, 456.7, GAMMA, CL, 23e-9, 260e-9)
    assert _deposit_amp(*args) == pytest.approx(
        physics.deposit_amplitude(*args), rel=1e-14)


def _run(solver_id, cfg):
    with contextlib.redirect_stdout(io.StringIO()):
        return get_solver(solver_id)(cfg)


def test_surface_point_energy_bookkeeping_closes_by_default(tmp_path):
    base = {"f_rep": 5e6, "simDuration": 2 / 5e6, "makePlots": False,
            "outputDir": str(tmp_path)}
    corrected = _run("surface_point", dict(base))
    legacy = _run("surface_point", {**base, "legacyDeposit": True})

    # At the defaults (dzTarget 500 nm, Leff 100 nm) the legacy surplus is
    # about 2.5x per pulse. Conservation collapses the mismatch and the
    # accumulated residual comes down with it.
    assert corrected["energyMismatch_pct"] < 5.0
    assert legacy["energyMismatch_pct"] > 100.0
    assert corrected["finalResid_C"] < legacy["finalResid_C"]

    # The 0D pulse physics is untouched: the first pulse's equilibrium
    # temperature is identical in both modes, since the deposit only
    # happens after it is recorded.
    assert corrected["TeqVals_C"][0] == legacy["TeqVals_C"][0]


def test_radial_profile_residual_comes_down_by_default(tmp_path):
    base = {"f_rep": 5e6, "simDuration": 2 / 5e6, "makePlots": False,
            "Nr": 8, "outputDir": str(tmp_path)}
    corrected = _run("radial_profile", dict(base))
    legacy = _run("radial_profile", {**base, "legacyDeposit": True})
    assert corrected["finalResid_C"] < legacy["finalResid_C"]
    assert corrected["TeqVals_C"][0] == legacy["TeqVals_C"][0]
