"""Material property records shared by every solver.

One table replaces the per-solver preset dicts. Solvers come in two families
and need different subsets of the same record:

* lattice family (surface point, radial, scanning) uses gamma, Cl, G and the
  total conductivity ``k_total``
* optical family (depth profile, single pulse) additionally needs the
  electron/lattice conductivity split (``ke0``, ``kl``) and the optical
  absorption coefficient ``alpha_opt``

``k_total`` is ``ke0 + kl`` wherever both are known, which is why the two
families historically carried the same numbers in two shapes.

Any preset field can be overridden per run without switching to
``material="custom"``: pass ``gamma``, ``Cl``, ``G``, ``kl``, ``ke0``,
``alpha_opt`` (or ``delta_opt``, its reciprocal, in metres), or ``T_melt_C``
alongside the material name. ``material="custom"`` still starts from the
``*_manual`` fields, as in the MATLAB reference.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .config import get_cfg_field
from .kernels import KHYBRID_W_K, KHYBRID_W_T


@dataclass(frozen=True)
class Material:
    """Physical properties of one material.

    ``ke0``, ``kl`` and ``alpha_opt`` are None for materials that only carry
    the lattice-family parameters; the depth-resolved solvers reject those.
    """

    key: str
    gamma: float             # electron heat capacity coefficient [J m^-3 K^-2]
    cl: float                # lattice heat capacity      [J m^-3 K^-1]
    g_ep: float              # electron-phonon coupling   [W m^-3 K^-1]
    k_total: float           # total conductivity         [W m^-1 K^-1]
    t_melt_c: float          # melting point              [deg C]
    ke0: float | None = None        # electron conductivity term [W m^-1 K^-1]
    kl: float | None = None         # lattice conductivity term  [W m^-1 K^-1]
    alpha_opt: float | None = None  # optical absorption coeff   [m^-1]
    measured_k_table: bool = False  # a tabulated k(T) curve exists

    @property
    def has_optical(self) -> bool:
        return self.alpha_opt is not None and self.ke0 is not None

    @property
    def delta_opt(self) -> float | None:
        """Optical penetration depth [m]."""
        return None if self.alpha_opt is None else 1.0 / self.alpha_opt


# Melting points are the accepted handbook values; every other number is the
# value the per-solver preset tables carried before they were merged here.
MATERIALS: dict[str, Material] = {
    "w": Material("w", 137.3, 2.54e6, 1.65e17, 174.0, 3422.0,
                  ke0=150.0, kl=24.0, alpha_opt=5.88e7,
                  measured_k_table=True),
    "cu": Material("cu", 98.0, 3.45e6, 0.90e17, 401.0, 1085.0,
                   ke0=390.0, kl=11.0, alpha_opt=7.09e7),
    "al": Material("al", 136.0, 2.42e6, 2.40e17, 237.0, 660.0,
                   ke0=220.0, kl=17.0, alpha_opt=1.22e8),
    # Gold carries only the lattice-family parameters: the electron/lattice
    # conductivity split and alpha_opt for Au are not part of the reference
    # toolbox, so the depth-resolved solvers reject it rather than guess.
    "au": Material("au", 67.0, 2.49e6, 1.40e16, 317.0, 1064.0),
}

_W = MATERIALS["w"]

_CONSTANT_K_T = np.array([1.0, 1e12])


def _custom(cfg, needs_optical: bool) -> Material:
    """Build the ``material='custom'`` record from the ``*_manual`` fields."""
    gamma = float(get_cfg_field(cfg, "gamma_manual", _W.gamma))
    cl = float(get_cfg_field(cfg, "Cl_manual", _W.cl))
    g_ep = float(get_cfg_field(cfg, "G_manual", _W.g_ep))
    t_melt = float(get_cfg_field(cfg, "T_melt_C", _W.t_melt_c))
    if needs_optical:
        ke0 = float(get_cfg_field(cfg, "ke0_manual", _W.ke0))
        kl = float(get_cfg_field(cfg, "kl_manual", _W.kl))
        alpha = float(get_cfg_field(cfg, "alpha_opt_manual", _W.alpha_opt))
        return Material("custom", gamma, cl, g_ep, ke0 + kl, t_melt,
                        ke0=ke0, kl=kl, alpha_opt=alpha)
    # Lattice family: kl_manual is the total conductivity, as in MATLAB.
    k_total = float(get_cfg_field(cfg, "kl_manual", _W.k_total))
    return Material("custom", gamma, cl, g_ep, k_total, t_melt)


def _apply_overrides(mat: Material, overrides, needs_optical: bool) -> Material:
    """Apply per-field cfg overrides on top of a preset or custom record."""
    changes: dict[str, float] = {}

    for cfg_key, field in (("gamma", "gamma"), ("Cl", "cl"), ("G", "g_ep"),
                           ("T_melt_C", "t_melt_c")):
        value = get_cfg_field(overrides, cfg_key, None)
        if value is not None:
            changes[field] = float(value)

    # Optical penetration: alpha_opt [1/m] or its reciprocal delta_opt [m].
    delta = get_cfg_field(overrides, "delta_opt", None)
    alpha = get_cfg_field(overrides, "alpha_opt", None)
    if delta is not None and alpha is None:
        alpha = 1.0 / float(delta)
    if alpha is not None:
        if not needs_optical:
            raise ValueError(
                "alpha_opt/delta_opt only apply to the depth-resolved solvers "
                "(depth_profile, single_pulse). The 0D and radial solvers use "
                "the effective deposition depth 'Leff' instead.")
        changes["alpha_opt"] = float(alpha)

    # Conductivity. 'kl' keeps each family's existing meaning: the lattice
    # term where the split is used, the total conductivity otherwise.
    ke0 = get_cfg_field(overrides, "ke0", None)
    kl = get_cfg_field(overrides, "kl", None)
    if ke0 is not None and not needs_optical:
        raise ValueError(
            "ke0 only applies to the depth-resolved solvers; use 'kl' for the "
            "total conductivity of the 0D, radial and scanning solvers.")
    if ke0 is not None or kl is not None:
        if needs_optical:
            new_ke0 = float(ke0) if ke0 is not None else mat.ke0
            new_kl = float(kl) if kl is not None else mat.kl
            changes["ke0"] = new_ke0
            changes["kl"] = new_kl
            changes["k_total"] = new_ke0 + new_kl
        else:
            changes["k_total"] = float(kl)
        # A conductivity that actually differs from the material's own value
        # replaces the tabulated k(T) curve, since the table would otherwise
        # silently ignore what the caller asked for. Restating a preset value
        # changes nothing. An explicit kTable overrides either way.
        if (changes["k_total"] != mat.k_total
                and str(get_cfg_field(overrides, "kTable", "auto")).lower() == "auto"):
            changes["measured_k_table"] = False

    if not changes:
        return mat
    return replace(mat, **changes)


def resolve_material(cfg, *, needs_optical: bool, overrides=None) -> Material:
    """Resolve a run's material record from its config dict.

    ``needs_optical`` selects the solver family. ``overrides`` defaults to
    ``cfg``; pass the caller's raw user dict when the solver merges a defaults
    table into its config, so untouched defaults are not read as overrides.
    """
    name = get_cfg_field(cfg, "material", "W")
    key = str(name).lower()

    if key == "custom":
        mat = _custom(cfg, needs_optical)
    elif key in MATERIALS:
        mat = MATERIALS[key]
        if needs_optical and not mat.has_optical:
            known = ", ".join(sorted(
                k.upper() for k, m in MATERIALS.items() if m.has_optical))
            raise ValueError(
                f'Material "{name}" has no electron conductivity split or '
                f"optical absorption coefficient, which this solver needs. "
                f"Use {known}, or custom with ke0_manual/kl_manual/"
                f"alpha_opt_manual.")
    else:
        known = ", ".join(sorted(
            k.upper() for k, m in MATERIALS.items()
            if m.has_optical or not needs_optical))
        raise ValueError(
            f'Unknown material "{name}". Use {known}, or custom.')

    mat = _apply_overrides(mat, cfg if overrides is None else overrides,
                           needs_optical)

    forced = str(get_cfg_field(cfg, "kTable", "auto")).lower()
    if forced == "measured":
        # Only tungsten has a measured k(T) curve here. Forcing 'measured' on
        # another metal would hand it tungsten's conductivity while every
        # report still stated its own, so refuse instead.
        if not MATERIALS.get(mat.key, mat).measured_k_table:
            with_curve = ", ".join(sorted(
                k.upper() for k, m in MATERIALS.items() if m.measured_k_table))
            raise ValueError(
                f'kTable="measured" is not available for material '
                f'"{name}": no measured k(T) curve ships for it. A measured '
                f"curve exists only for {with_curve}. Use kTable=\"auto\" to "
                f"take this material's own conductivity, or supply kl.")
        mat = replace(mat, measured_k_table=True)
    elif forced == "constant":
        mat = replace(mat, measured_k_table=False)
    elif forced != "auto":
        raise ValueError(
            f'Unknown kTable "{forced}". Use "auto", "measured", or "constant".')
    return mat


def k_table(mat: Material, *, constant_only: bool = False
            ) -> tuple[np.ndarray, np.ndarray]:
    """Return the (T, k) hybrid-conductivity table for a material.

    Tungsten has a measured k(T) curve; everything else uses a flat table at
    the material's total conductivity. ``constant_only`` forces the flat
    table for solvers whose kernel expects a temperature-independent k.
    """
    if mat.measured_k_table and not constant_only:
        return KHYBRID_W_T.copy(), KHYBRID_W_K.copy()
    return _CONSTANT_K_T.copy(), np.array([mat.k_total, mat.k_total])


def k_model_name(mat: Material, *, constant_only: bool = False) -> str:
    """Human-readable description of the conductivity model in use."""
    if mat.measured_k_table and not constant_only:
        return "measured k(T) table (tungsten)"
    return f"constant k = {mat.k_total:.1f} W/mK"
