"""Human-readable unit formatting helpers.

Direct ports of the smartTime/smartFreq/smartEnergy/smartLength helpers that
are duplicated across the MATLAB solvers; this module is their single home.
Each returns ``(value, unit)``.
"""

from __future__ import annotations


def smart_time(t_s: float) -> tuple[float, str]:
    """Convert a time in seconds to a value and display unit."""
    abs_t = abs(t_s)
    if abs_t < 1e-12:
        return t_s * 1e15, "fs"
    if abs_t < 1e-9:
        return t_s * 1e12, "ps"
    if abs_t < 1e-6:
        return t_s * 1e9, "ns"
    if abs_t < 1e-3:
        return t_s * 1e6, "us"
    if abs_t < 1:
        return t_s * 1e3, "ms"
    return t_s, "s"


def smart_freq(f_hz: float) -> tuple[float, str]:
    """Convert a frequency in Hz to a value and display unit."""
    abs_f = abs(f_hz)
    if abs_f < 1e3:
        return f_hz, "Hz"
    if abs_f < 1e6:
        return f_hz / 1e3, "kHz"
    if abs_f < 1e9:
        return f_hz / 1e6, "MHz"
    return f_hz / 1e9, "GHz"


def smart_energy(e_j: float) -> tuple[float, str]:
    """Convert an energy in joules to a value and display unit."""
    abs_e = abs(e_j)
    if abs_e < 1e-9:
        return e_j * 1e12, "pJ"
    if abs_e < 1e-6:
        return e_j * 1e9, "nJ"
    if abs_e < 1e-3:
        return e_j * 1e6, "uJ"
    if abs_e < 1:
        return e_j * 1e3, "mJ"
    return e_j, "J"


def smart_length(l_m: float) -> tuple[float, str]:
    """Convert a length in metres to a value and display unit."""
    abs_l = abs(l_m)
    if abs_l < 1e-9:
        return l_m * 1e12, "pm"
    if abs_l < 1e-6:
        return l_m * 1e9, "nm"
    if abs_l < 1e-3:
        return l_m * 1e6, "um"
    if abs_l < 1:
        return l_m * 1e3, "mm"
    return l_m, "m"
