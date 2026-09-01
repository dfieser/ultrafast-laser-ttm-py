"""The shared physics identities, and their equality with the inline forms
they replaced."""

import numpy as np
import pytest

from laserttm import physics

GAMMA, CL = 137.3, 2.54e6


def test_equilibration_conserves_energy_exactly():
    """Putting Teq back through the energy density returns the energy it
    was derived from. This is the defining property of the identity."""
    rng = np.random.default_rng(7)
    te = rng.uniform(300.0, 30000.0, 200)
    tl = rng.uniform(300.0, 4000.0, 200)
    utot = physics.bath_energy_density(te, tl, GAMMA, CL)
    teq = physics.equilibrium_temperature(utot, GAMMA, CL)
    back = physics.bath_energy_density(teq, teq, GAMMA, CL)
    np.testing.assert_allclose(back, utot, rtol=1e-12)


def test_equilibrium_lies_between_the_baths():
    teq = physics.equilibrate(10000.0, 400.0, GAMMA, CL)
    assert 400.0 < teq < 10000.0
    # Equal baths equilibrate to themselves.
    assert physics.equilibrate(500.0, 500.0, GAMMA, CL) == pytest.approx(500.0)


def test_equilibrate_matches_the_inline_form_bitwise():
    """The helper must be character-identical to what the solvers inlined."""
    te, tl = 8234.567, 512.345
    utot = 0.5 * GAMMA * te**2 + CL * tl
    inline = (-CL + np.sqrt(CL**2 + 2.0 * GAMMA * utot)) / GAMMA
    assert physics.equilibrate(te, tl, GAMMA, CL) == inline


def test_deposit_shapes():
    z = np.arange(6) * 100e-9
    decay, box = physics.depth_deposit_shape(z, 250e-9)
    np.testing.assert_array_equal(decay, np.exp(-z / 250e-9))
    np.testing.assert_array_equal(box, [True, True, True, False, False, False])


def test_exponential_deposit_matches_the_inline_form_bitwise():
    z = np.arange(8) * 500e-9
    decay, box = physics.depth_deposit_shape(z, 100e-9)
    tz = np.linspace(400.0, 300.0, 8)
    teq = 900.0
    inline = tz + (teq - tz[0]) * decay
    out = physics.deposit_pulse(tz.copy(), teq, decay, box, exponential=True)
    np.testing.assert_array_equal(out, inline)
    # The surface node lands exactly on teq.
    assert out[0] == pytest.approx(teq)


def test_box_deposit_mutates_in_place_like_the_inline_form():
    z = np.arange(8) * 50e-9
    decay, box = physics.depth_deposit_shape(z, 120e-9)
    tz = np.full(8, 350.0)
    out = physics.deposit_pulse(tz, 800.0, decay, box, exponential=False)
    assert out is tz
    np.testing.assert_array_equal(tz[:3], 800.0)
    np.testing.assert_array_equal(tz[3:], 350.0)


def test_deposit_works_on_a_2d_column_view():
    """The radial independent mode deposits into columns of a 2D field."""
    z = np.arange(5) * 500e-9
    decay, box = physics.depth_deposit_shape(z, 100e-9)
    tz_all = np.full((5, 3), 300.0)
    tz_all[:, 1] = physics.deposit_pulse(tz_all[:, 1], 700.0, decay, box,
                                         exponential=True)
    assert tz_all[0, 1] == pytest.approx(700.0)
    np.testing.assert_array_equal(tz_all[:, 0], 300.0)
    np.testing.assert_array_equal(tz_all[:, 2], 300.0)


def test_derive_laser_matches_the_inline_expressions():
    dl = physics.derive_laser(pavg=70.0, f_rep=40e6, spot_radius=150e-6,
                              absorbance=0.4, t0_c=25.0,
                              gamma=GAMMA, g_ep=1.65e17,
                              sim_duration=5000 / 40e6)
    assert dl.t0_k == 25.0 + 273.15
    assert dl.pulse_energy == 70.0 / 40e6
    assert dl.peak_fluence == 2.0 * dl.pulse_energy / (np.pi * 150e-6**2)
    assert dl.absorbed_fluence == 0.4 * dl.peak_fluence
    assert dl.period == 1.0 / 40e6
    assert dl.tau_eph == GAMMA * dl.t0_k / 1.65e17
    assert dl.n_pulses == 5000


def test_derive_laser_without_a_duration_is_untimed():
    dl = physics.derive_laser(pavg=1.0, f_rep=1e6, spot_radius=80e-6,
                              absorbance=0.55, t0_c=25.0,
                              gamma=GAMMA, g_ep=1.65e17)
    assert dl.n_pulses is None


def test_matlab_round_is_half_away_from_zero():
    assert physics.matlab_round(2.5) == 3
    assert physics.matlab_round(2.4999) == 2
    assert physics.matlab_round(0.5) == 1
