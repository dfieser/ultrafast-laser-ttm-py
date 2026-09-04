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

import warnings
from dataclasses import dataclass

import numpy as np

trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def matlab_round(x: float) -> int:
    """MATLAB round() for the positive arguments used here."""
    return int(np.floor(x + 0.5))


# Surface inversion threshold [K]: a pulse counts as inverted when the
# surface Tl - Te exceeds this. One home, so the depth solver's per-pulse
# metrics, the single-pulse detector, and the inversion quantifier's
# counts can never disagree.
INV_THRESHOLD_K = 0.5


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


# ========================  Peak refinement  ================================


def refine_peak(t3, d3):
    """Sub-sample peak of a sampled curve from the three samples around it.

    Fits the parabola through ``(t3[i], d3[i])`` for the sample before, at
    and after a discrete maximum and returns ``(t_peak, d_peak)`` of its
    vertex, or None when the vertex falls outside the three samples or the
    three points are not concave.

    The fit is done in time measured from the middle sample. Fitting in
    absolute time is catastrophically ill-conditioned for this library:
    a pulse a millisecond into a train sampled at picosecond spacing puts
    t^2 terms near 1e19 in the vertex value, whose rounding remainder is
    hundreds of kelvin. The MATLAB reference fits in absolute time, and
    the port inherited spikes of exactly 256 K and 512 K from it until
    0.4.0.
    """
    h1 = float(t3[1] - t3[0])
    h2 = float(t3[2] - t3[1])
    if h1 <= 0.0 or h2 <= 0.0:
        return None
    d0, d1, d2 = float(d3[0]), float(d3[1]), float(d3[2])
    # d(tau) = d1 + b*tau + a*tau^2 through (-h1, d0), (0, d1), (h2, d2)
    a = ((d2 - d1) / h2 + (d0 - d1) / h1) / (h1 + h2)
    if a >= 0.0:
        return None
    b = (d2 - d1) / h2 - a * h2
    tau_peak = -b / (2.0 * a)
    if not -h1 <= tau_peak <= h2:
        return None
    return float(t3[1]) + tau_peak, d1 - b * b / (4.0 * a)


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
                 sim_duration: float | None = None,
                 n_pulses: int | None = None) -> DerivedLaser:
    """Derive the shared pulse-train quantities from the config inputs.

    Each expression is the one the solvers carried individually, written
    once. sim_duration of None, used by the single-pulse solver, leaves
    n_pulses as None. An explicit n_pulses (the nPulses config key) wins
    over the count derived from sim_duration.
    """
    t0 = t0_c + 273.15
    ep = pavg / f_rep
    f_peak = 2.0 * ep / (np.pi * spot_radius**2)
    if n_pulses is not None:
        count: int | None = int(n_pulses)
    elif sim_duration is not None:
        count = matlab_round(sim_duration * f_rep)
    else:
        count = None
    return DerivedLaser(
        t0_k=t0,
        pulse_energy=ep,
        peak_fluence=f_peak,
        absorbed_fluence=absorbance * f_peak,
        period=1.0 / f_rep,
        tau_eph=gamma * t0 / g_ep,
        n_pulses=count,
    )


def coast_substeps(n_diff_min: int, alpha: float, gap: float, dz: float,
                   fourier_max: float = 0.5) -> int:
    """Crank-Nicolson substeps for a coast of length gap [s].

    At least n_diff_min, and enough to keep the Fourier number
    alpha*dt/dz^2 at or below fourier_max. Crank-Nicolson is stable at any
    step, but on the sharp profile a pulse leaves it rings for the first
    few steps, and at a Fourier number of several the samples inside the
    first 100 ns of a coast are off by half. The end-of-period state is
    insensitive to the count, which is why the fixtures at 18 MHz, where
    the number is 0.15, are unaffected.
    """
    return max(int(n_diff_min),
               int(np.ceil(alpha * gap / (fourier_max * dz**2))))


def melt_pulse(peaks_k, t_melt_c: float) -> int:
    """1-based index of the first pulse whose peak [K] passed the melting
    point, or 0 when none did."""
    hot = np.flatnonzero(np.asarray(peaks_k, dtype=float) - 273.15 > t_melt_c)
    return int(hot[0]) + 1 if hot.size else 0


def diagnostic(code: str, message: str, suggestion: str = "",
               key: str | None = None, level: str = "warning") -> dict:
    """One run diagnostic in the shape validate_config reports problems.

    level is "warning" for a concern about the result and "info" for a
    notice about what the solver did; only warnings are raised through
    warnings.warn and listed under the results key ``warnings``.
    """
    out = {"code": code, "level": level, "message": message,
           "suggestion": suggestion}
    if key is not None:
        out["key"] = key
    return out


def diagnostic_messages(diags) -> list[str]:
    """The warning-level messages of a diagnostics list, as plain strings."""
    return [d["message"] for d in diags if d.get("level", "warning") == "warning"]


def validity_diagnostics(peak_t_c: float, t_melt_c: float, material: str,
                         *, melt_pulse: int = 0,
                         peak_fluence: float | None = None,
                         ablation_threshold: float | None = None,
                         extra=(), emit: bool = True) -> list[dict]:
    """Diagnostics about a run that left the model's range of validity.

    The solvers have no phase change and no ablation: a lattice above the
    melting point keeps heating as a solid, so every temperature past that
    point is a statement about deposited energy, not about the material.
    Each entry carries a code (above_melting, above_ablation, plus any
    passed in extra), a message and a suggestion. With emit, every message
    is also raised through warnings.warn so console users see it.
    """
    out: list[dict] = []
    if np.isfinite(peak_t_c) and peak_t_c > t_melt_c:
        where = f" First on pulse {melt_pulse}." if melt_pulse else ""
        out.append(diagnostic(
            "above_melting",
            f"Peak lattice temperature {peak_t_c:.0f} C exceeds the melting "
            f"point of {str(material).upper()} ({t_melt_c:.0f} C). The model "
            f"has no phase change or ablation, so temperatures above the "
            f"melting point are not physical: the fluence is above the "
            f"melting threshold for these settings.{where}",
            "Lower Pavg, raise f_rep or enlarge spotRadius until peakTl_C "
            "stays below Tmelt_C.", key="Pavg"))
    if (peak_fluence is not None and ablation_threshold is not None
            and peak_fluence > ablation_threshold):
        out.append(diagnostic(
            "above_ablation",
            f"Peak fluence {peak_fluence / 1e4:.3g} J/cm^2 exceeds the "
            f"single-shot ablation threshold of {str(material).upper()} "
            f"({ablation_threshold / 1e4:.2g} J/cm^2). The model has no "
            "ablation, so the run describes deposited energy, not the "
            "material's response.",
            "Bring the peak fluence under the threshold, or set "
            "ablationThreshold to your own value.", key="Pavg"))
    out.extend(extra)
    if emit:
        for d in out:
            if d.get("level", "warning") == "warning":
                warnings.warn(d["message"], stacklevel=2)
    return out


def validity_warnings(peak_t_c: float, t_melt_c: float, material: str,
                      *, emit: bool = True, **kwargs) -> list[str]:
    """The warning messages of validity_diagnostics, as plain strings."""
    return diagnostic_messages(validity_diagnostics(
        peak_t_c, t_melt_c, material, emit=emit, **kwargs))

