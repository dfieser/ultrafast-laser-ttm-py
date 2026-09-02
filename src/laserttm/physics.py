"""The physics vocabulary shared by every solver.

The six solvers are different discretizations of one model, and the
closed-form identities they share used to be written out separately in
each, up to five copies of a single expression. Each identity now has one
home here, so a correction or a refinement lands everywhere at once.

Everything in this module is a plain closed-form expression, kept
character-identical to the inline forms it replaced. The integrators
themselves stay where they are: the adaptive RK4 and the stiff BDF phases
are genuinely different numerics, and kernels.py remains line-faithful to
the MATLAB reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def matlab_round(x: float) -> int:
    """MATLAB round() for the positive arguments used here."""
    return int(np.floor(x + 0.5))


# ==========================  Two-bath energetics  ==========================


def bath_energy_density(te, tl, gamma: float, cl: float):
    """Combined energy density of the electron and lattice baths [J/m^3].

    The electron bath carries 0.5*gamma*Te^2 under the Sommerfeld capacity
    Ce(Te) = gamma*Te, and the lattice carries Cl*Tl.
    """
    return 0.5 * gamma * te**2 + cl * tl


def equilibrium_temperature(utot, gamma: float, cl: float):
    """The common temperature the two baths reach for a given energy [K].

    Inverts bath_energy_density with Te = Tl = Teq, taking the physical
    root of the quadratic. Energy-conserving by construction: putting the
    result back through bath_energy_density returns utot exactly.
    """
    return (-cl + np.sqrt(cl**2 + 2.0 * gamma * utot)) / gamma


def equilibrate(te, tl, gamma: float, cl: float):
    """Equilibrium temperature of two baths at Te and Tl [K]."""
    return equilibrium_temperature(bath_energy_density(te, tl, gamma, cl),
                                   gamma, cl)


def energy_mismatch_pct(absorbed_areal: float, stored_areal: float) -> float:
    """Relative energy-bookkeeping mismatch, in percent of the absorbed.

    With the energy-conserving deposit this reduces to boundary losses
    and integrator error. Under legacyDeposit it also carries the
    coarse-grid deposit surplus described in deposit_pulse.
    """
    return abs(absorbed_areal - stored_areal) / max(abs(absorbed_areal),
                                                    np.finfo(float).eps) * 100.0


# ==========================  Energy deposition  ============================


def depth_deposit_shape(z_grid: np.ndarray, leff: float):
    """Loop-invariant deposit shapes for a grid: (exponential, box mask).

    The exponential profile is exp(-z/Leff); the box profile heats every
    node within Leff of the surface.
    """
    return np.exp(-z_grid / leff), z_grid <= leff


def deposit_shape_weight(z_grid: np.ndarray, leff: float,
                         exponential: bool) -> float:
    """Trapezoid weight of the deposit shape on this grid [m].

    The thickness the grid actually assigns to the deposit: the same
    trapezoid the solvers use for their energy bookkeeping, applied to
    the shape deposit_pulse paints. On a grid that resolves Leff this
    approaches Leff; on a coarser one it approaches dz/2, which is what
    made the legacy deposit over-inject energy.
    """
    if exponential:
        return float(trapezoid(np.exp(-z_grid / leff), z_grid))
    return float(trapezoid((z_grid <= leff).astype(np.float64), z_grid))


def deposit_amplitude(teq, t_surf, gamma: float, cl: float, leff: float,
                      shape_weight: float):
    """Lattice-temperature amplitude carrying the layer's full energy [K].

    The pulse leaves a layer of thickness Leff equilibrated at teq where
    the surface previously sat at t_surf. Its energy above the old state,
    electron and lattice bath both, is deposited as lattice heat, since
    the electrons hand their share to the lattice within picoseconds
    while the inter-pulse coast lasts microseconds:

        E_areal = [u(teq) - u(t_surf)] * Leff
        amp     = E_areal / (Cl * shape_weight)

    so that Cl * trapezoid(amp * shape) equals E_areal exactly, on any
    grid. kernels._deposit_amp is the jitted twin of this expression for
    the scanning loop; a test pins the two together.
    """
    e_areal = (bath_energy_density(teq, teq, gamma, cl)
               - bath_energy_density(t_surf, t_surf, gamma, cl)) * leff
    return e_areal / (cl * shape_weight)


def deposit_pulse(tz: np.ndarray, teq, exp_decay_z: np.ndarray,
                  box_mask_z: np.ndarray, exponential: bool, *,
                  amplitude=None) -> np.ndarray:
    """Deposit one pulse's temperature rise into a depth profile.

    With ``amplitude`` (from deposit_amplitude) the deposit conserves
    energy on any grid: the shape is scaled so the grid's own trapezoid
    of the added heat equals the layer energy exactly. On a coarse grid
    the surface node then reads as a control-volume average rather than
    the true surface temperature; shrink dzTarget to resolve early-time
    surface values.

    With ``amplitude=None`` the legacy MATLAB deposit runs instead: the
    surface node is raised to teq and the rise decays by the shape. That
    form is exact only when the grid resolves Leff. When it does not,
    the surface control volume absorbs roughly (dz/2)/Leff times the
    requested energy, and it always drops the electron bath's share of
    the layer energy. It is kept so the golden fixtures keep proving the
    port line-faithful, selected by the ``legacyDeposit`` config key.
    """
    if amplitude is None:
        if exponential:
            return tz + (teq - tz[0]) * exp_decay_z
        tz[box_mask_z] = teq
        return tz
    if exponential:
        return tz + amplitude * exp_decay_z
    tz[box_mask_z] = tz[box_mask_z] + amplitude
    return tz


# ========================  Derived pulse-train values  =====================


@dataclass(frozen=True)
class DerivedLaser:
    """Quantities every solver derives from the laser inputs, in SI."""

    t0_k: float              # initial temperature [K]
    pulse_energy: float      # Pavg / f_rep [J]
    peak_fluence: float      # 2 Ep / (pi w0^2) for a Gaussian [J/m^2]
    absorbed_fluence: float  # A * peak fluence [J/m^2]
    period: float            # 1 / f_rep [s]
    tau_eph: float           # electron-phonon time gamma*T0/G at T0 [s]
    n_pulses: int | None     # round(simDuration * f_rep), None if untimed


def derive_laser(*, pavg: float, f_rep: float, spot_radius: float,
                 absorbance: float, t0_c: float, gamma: float, g_ep: float,
                 sim_duration: float | None = None) -> DerivedLaser:
    """Derive the shared pulse-train quantities from the config inputs.

    Each expression is the one the solvers carried individually, written
    once. sim_duration of None, used by the single-pulse solver, leaves
    n_pulses as None.
    """
    t0 = t0_c + 273.15
    ep = pavg / f_rep
    f_peak = 2.0 * ep / (np.pi * spot_radius**2)
    return DerivedLaser(
        t0_k=t0,
        pulse_energy=ep,
        peak_fluence=f_peak,
        absorbed_fluence=absorbance * f_peak,
        period=1.0 / f_rep,
        tau_eph=gamma * t0 / g_ep,
        n_pulses=(matlab_round(sim_duration * f_rep)
                  if sim_duration is not None else None),
    )
