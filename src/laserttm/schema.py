"""Declarative registry of the library's contract, inputs and results both.

One table describes each config key once, with its unit, default, valid range
and meaning, and records which solvers accept it; a second set of tables
describes every key each solver returns. Everything else is derived: solver
defaults, config validation, the generated documentation, the CLI's
``describe``/``schema`` commands, and the MCP server's discovery tools.

The point is that a caller, human or machine, can learn the complete input
and output surface without reading solver source:

    >>> from laserttm import describe_solver, validate_config
    >>> describe_solver("depth_profile")["params"]["spotRadius"]["unit"]
    'm'
    >>> validate_config("depth_profile", {"spotRadius": 100})["ok"]
    False

Defaults are recorded per solver rather than unified. The six solvers grew
from separate MATLAB scripts and their bare-call defaults genuinely differ,
so the table states that divergence instead of hiding or silently changing
it. Key spellings follow the MATLAB reference exactly, since one-to-one
field correspondence with the published toolbox is a feature; units live
here rather than in renamed keys.

This module imports nothing heavier than the standard library, so discovery
stays fast. Anything needing NumPy, Numba or material data is imported
lazily inside the function that needs it.
"""

from __future__ import annotations

import difflib
import math
from dataclasses import dataclass, field, replace
from typing import Any

__all__ = [
    "PARAMS",
    "RESULTS",
    "RESULT_ENVELOPE",
    "SOLVER_IDS",
    "ParamSpec",
    "ResultField",
    "SolverSchema",
    "defaults",
    "describe_results",
    "describe_solver",
    "effective_config",
    "estimate_run",
    "json_schema",
    "list_solvers",
    "require_pulses",
    "solver_schema",
    "validate_config",
]

_DEFAULT_SNAPSHOT_DELAYS = (0.0, 0.5e-12, 1e-12, 2e-12, 5e-12,
                            10e-12, 50e-12, 200e-12)


@dataclass(frozen=True)
class ParamSpec:
    """One config key: what it means, what it accepts, what it defaults to."""

    name: str
    kind: str                      # float | int | bool | enum | str | array | path | any
    summary: str
    default: Any = None
    unit: str | None = None
    minimum: float | None = None   # hard bound; outside it is an error
    maximum: float | None = None
    typical: tuple[float, float] | None = None   # soft window; outside it warns
    choices: tuple[str, ...] | None = None
    group: str = "run"             # laser | material | geometry | grid | run | output
    affects_numerics: bool = True
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        out = {
            "name": self.name,
            "kind": self.kind,
            "summary": self.summary,
            "default": self.default,
            "group": self.group,
            "affectsNumerics": self.affects_numerics,
        }
        if self.unit:
            out["unit"] = self.unit
        if self.minimum is not None or self.maximum is not None:
            out["range"] = [self.minimum, self.maximum]
        if self.typical is not None:
            out["typical"] = list(self.typical)
        if self.choices is not None:
            out["choices"] = list(self.choices)
        if self.notes:
            out["notes"] = self.notes
        return out


def _p(name: str, kind: str, summary: str, **kw: Any) -> ParamSpec:
    return ParamSpec(name=name, kind=kind, summary=summary, **kw)


# ===========================  Parameter catalog  ===========================
# The base spec for every key in the library. Per-solver differences in
# default or meaning are applied in the solver tables further down.

_CATALOG: tuple[ParamSpec, ...] = (
    # ---- laser -----------------------------------------------------------
    _p("Pavg", "float", "Average laser power.", unit="W", default=40.0,
       minimum=1e-9, maximum=1e7, typical=(0.1, 1000.0), group="laser"),
    _p("spotRadius", "float",
       "Beam radius where intensity falls to 1/e^2 of its peak.",
       unit="m", default=100e-6, minimum=1e-9, maximum=1e-2,
       typical=(1e-6, 1e-3), group="laser"),
    _p("f_rep", "float", "Pulse repetition rate.", unit="Hz", default=18e6,
       minimum=1e-3, maximum=1e12, typical=(1e3, 1e9), group="laser"),
    _p("tau_FWHM", "float", "Pulse duration, full width at half maximum.",
       unit="s", default=100e-15, minimum=1e-18, maximum=1e-6,
       typical=(1e-15, 1e-11), group="laser"),
    _p("pulseProfile", "enum", "Temporal shape of a single pulse.",
       default="gaussian", choices=("gaussian", "square", "exp"),
       group="laser",
       notes="'exp' is a one-sided exponential, zero before the pulse and "
             "exp(-t/tau)/tau after it. There tau is tau_FWHM used as a decay "
             "constant rather than a full width."),
    _p("absorbance", "float",
       "Absorbed fraction of incident fluence, A in 0 to 1.",
       default=0.55, minimum=0.0, maximum=1.0, typical=(0.05, 0.95),
       group="laser"),
    _p("simDuration", "float",
       "Simulated time. Pulse count is round(simDuration * f_rep).",
       unit="s", default=100e-6, minimum=1e-15, maximum=1e3, group="laser"),
    _p("T0_C", "float", "Initial and ambient temperature.", unit="degC",
       default=25.0, minimum=-273.15, maximum=5000.0, group="laser"),

    # ---- material --------------------------------------------------------
    _p("material", "enum",
       "Material preset, or 'custom' to supply properties by hand.",
       default="W", choices=("W", "Cu", "Al", "Au", "custom"),
       group="material"),
    _p("gamma", "float",
       "Electron heat capacity coefficient. Overrides the preset value.",
       unit="J m^-3 K^-2", default=None, minimum=1.0, maximum=1e5,
       group="material"),
    _p("Cl", "float",
       "Lattice heat capacity. Overrides the preset value.",
       unit="J m^-3 K^-1", default=None, minimum=1e3, maximum=1e8,
       group="material"),
    _p("G", "float",
       "Electron-phonon coupling. Overrides the preset value.",
       unit="W m^-3 K^-1", default=None, minimum=1e12, maximum=1e20,
       group="material"),
    _p("kl", "float",
       "Thermal conductivity. Overrides the preset value.",
       unit="W m^-1 K^-1", default=None, minimum=1e-2, maximum=1e4,
       group="material"),
    _p("ke0", "float",
       "Electron conductivity term. Overrides the preset value.",
       unit="W m^-1 K^-1", default=None, minimum=1e-2, maximum=1e4,
       group="material"),
    _p("alpha_opt", "float",
       "Optical absorption coefficient. Overrides the preset value.",
       unit="1/m", default=None, minimum=1e3, maximum=1e12,
       group="material"),
    _p("delta_opt", "float",
       "Optical penetration depth, the reciprocal of alpha_opt.",
       unit="m", default=None, minimum=1e-12, maximum=1e-3,
       typical=(1e-9, 1e-6), group="material"),
    _p("T_melt_C", "float",
       "Melting point. Overrides the preset value.",
       unit="degC", default=None, minimum=-273.15, maximum=6000.0,
       group="material"),
    _p("kTable", "enum",
       "Conductivity model: 'auto' follows the material, 'measured' forces "
       "the tabulated k(T) curve, 'constant' forces a single value.",
       default="auto", choices=("auto", "measured", "constant"),
       group="material"),
    _p("gamma_manual", "float", "Electron heat capacity for material='custom'.",
       unit="J m^-3 K^-2", default=137.3, minimum=1.0, maximum=1e5,
       group="material"),
    _p("Cl_manual", "float", "Lattice heat capacity for material='custom'.",
       unit="J m^-3 K^-1", default=2.54e6, minimum=1e3, maximum=1e8,
       group="material"),
    _p("G_manual", "float", "Electron-phonon coupling for material='custom'.",
       unit="W m^-3 K^-1", default=1.65e17, minimum=1e12, maximum=1e20,
       group="material"),
    _p("kl_manual", "float", "Conductivity for material='custom'.",
       unit="W m^-1 K^-1", default=174.0, minimum=1e-2, maximum=1e4,
       group="material"),
    _p("ke0_manual", "float",
       "Electron conductivity term for material='custom'.",
       unit="W m^-1 K^-1", default=150.0, minimum=1e-2, maximum=1e4,
       group="material"),
    _p("alpha_opt_manual", "float",
       "Optical absorption coefficient for material='custom'.",
       unit="1/m", default=5.88e7, minimum=1e3, maximum=1e12,
       group="material"),

    # ---- geometry and grids ---------------------------------------------
    _p("Leff", "float",
       "Effective deposition depth: pulse energy is deposited over this "
       "thickness. Distinct from the optical penetration depth used by the "
       "depth-resolved solvers.",
       unit="m", default=100e-9, minimum=1e-12, maximum=1e-3,
       typical=(1e-9, 1e-6), group="geometry"),
    _p("depthProfile", "enum",
       "How pulse energy is distributed with depth.",
       default="exponential", choices=("exponential", "box"),
       group="geometry"),
    _p("Lz", "float", "Depth of the fine two-temperature grid.", unit="m",
       default=1000e-9, minimum=1e-9, maximum=1e-2, group="grid"),
    _p("Nz", "int", "Node count on the fine depth grid.", default=200,
       minimum=3, maximum=100000, group="grid"),
    _p("dzTarget", "float", "Spacing of the coarse depth diffusion grid.",
       unit="m", default=500e-9, minimum=1e-12, maximum=1e-3, group="grid"),
    _p("dzTarget_diff", "float",
       "Spacing of the coarse depth diffusion grid.",
       unit="m", default=500e-9, minimum=1e-12, maximum=1e-3, group="grid"),
    _p("Ndiff", "int",
       "Crank-Nicolson substeps per inter-pulse period.",
       default=100, minimum=1, maximum=int(1e7), group="grid"),
    _p("Nr", "int", "Radial node count.", default=80, minimum=3,
       maximum=100000, group="grid",
       notes="The cylindrical Crank-Nicolson stencil needs at least three "
             "nodes: a centre, one interior node, and the fixed outer "
             "boundary."),
    _p("rMax_factor", "float",
       "Radial extent as a multiple of the spot radius.",
       default=5.0, minimum=0.5, maximum=100.0, group="grid"),
    _p("Nr_radial", "int", "Radial node count for the derived radial view.",
       default=20, minimum=2, maximum=100000, group="grid"),
    _p("snapshotDelays", "array",
       "Delays after the pulse centre at which depth snapshots are captured.",
       unit="s", default=_DEFAULT_SNAPSHOT_DELAYS, group="grid"),
    _p("relTol", "float", "Relative tolerance of the stiff BDF integrator.",
       default=1e-6, minimum=1e-14, maximum=1.0, group="grid"),
    _p("absTol", "float", "Absolute tolerance of the stiff BDF integrator.",
       unit="K", default=1e-1, minimum=1e-14, maximum=1e3, group="grid"),

    # ---- solver behavior -------------------------------------------------
    _p("radialSolveMode", "enum",
       "'scale' solves one 0D model at beam centre and scales it radially. "
       "'independent' solves a 0D model at every radial node, capturing the "
       "nonlinear response at each local fluence, at much higher cost.",
       default="scale", choices=("scale", "independent"), group="run"),
    _p("enableRadialProfile", "bool",
       "Derive a radial surface view from the depth solution.",
       default=True, group="run", affects_numerics=False),
    _p("earlyStopMeltRadius_um", "float",
       "Stop once the melt radius reaches this value. 0 disables the check.",
       unit="um", default=0.0, minimum=0.0, maximum=1e6, group="run"),
    _p("earlyStopT_melt_C", "float",
       "Melt temperature for the early-stop check. Defaults to the "
       "material's own melting point.",
       unit="degC", default=None, minimum=-273.15, maximum=6000.0,
       group="run"),
    _p("earlyStopCheckInterval", "int",
       "Pulses between early-stop checks.", default=100, minimum=1,
       maximum=int(1e9), group="run"),
    _p("depthResults", "any",
       "A depth_profile results dict to analyze. When omitted, the depth "
       "solver is run first using the rest of this config.",
       default=None, group="run"),

    # ---- scanning beam ---------------------------------------------------
    _p("v_scan", "float", "Scan speed of the beam across the surface.",
       unit="m/s", default=1.0, minimum=1e-9, maximum=1e4, group="laser"),
    _p("scanLength", "float", "Length of the scan line.", unit="m",
       default=2e-3, minimum=1e-9, maximum=10.0, group="geometry"),
    _p("Nx", "int", "Surface grid nodes along the scan direction.",
       default=120, minimum=3, maximum=100000, group="grid"),
    _p("Ny", "int", "Surface grid nodes across the scan direction.",
       default=60, minimum=3, maximum=100000, group="grid"),
    _p("xPad", "float",
       "Padding beyond each end of the scan, in spot radii.",
       default=3.0, minimum=0.1, maximum=100.0, group="grid"),
    _p("yExtent", "float",
       "Half-width of the surface grid across the scan, in spot radii.",
       default=5.0, minimum=0.1, maximum=100.0, group="grid"),
    _p("NadiPerGap", "int",
       "Alternating-direction implicit steps per inter-pulse gap.",
       default=10, minimum=1, maximum=10000, group="grid"),

    # ---- run controls and output ----------------------------------------
    _p("makePlots", "bool", "Create matplotlib figures.", default=True,
       group="output", affects_numerics=False),
    _p("saveFigures", "bool",
       "Write figures to the output directory. Implies makePlots.",
       default=False, group="output", affects_numerics=False),
    _p("outputDir", "path",
       "Directory for the text report and any figures. Defaults to "
       "'outputs' under the working directory.",
       default=None, group="output", affects_numerics=False),
    _p("caseTag", "str",
       "Prefix for output filenames, for keeping a sweep's runs apart.",
       default="", group="output", affects_numerics=False),
    _p("storeHistory", "bool",
       "Keep the per-pulse time histories. False bounds memory on long "
       "runs and disables the timeline figures. Physics is unaffected.",
       default=True, group="run", affects_numerics=False),
    _p("legacyDeposit", "bool",
       "Reproduce the MATLAB reference deposit exactly. The default "
       "deposit conserves the pulse energy on any grid, which the "
       "reference does not once the grid is coarser than Leff.",
       default=False, group="run",
       notes="The legacy deposit raises the surface node to Teq and lets "
             "the rise decay over Leff. On a grid with dz > 2*Leff that "
             "injects roughly (dz/2)/Leff times the intended energy per "
             "pulse and inflates heat accumulation. Set True only to "
             "reproduce MATLAB toolbox output, as the golden-fixture "
             "tests do."),
    _p("showProgress", "bool",
       "Show the progress window. None auto-detects a usable display.",
       default=None, group="run", affects_numerics=False),
)

PARAMS: dict[str, ParamSpec] = {spec.name: spec for spec in _CATALOG}


# ============================  Solver schemas  =============================


@dataclass(frozen=True)
class SolverSchema:
    """The complete input contract for one solver."""

    id: str
    title: str
    summary: str
    when_to_use: str
    when_not_to_use: str
    params: dict[str, ParamSpec]
    files: tuple[str, ...] = ()
    seconds_per_pulse: float = 1e-3
    examples: dict[str, dict] = field(default_factory=dict)

    def defaults(self) -> dict[str, Any]:
        return {name: spec.default for name, spec in self.params.items()}


def _params(*names: str, **overrides: dict) -> dict[str, ParamSpec]:
    """Build a solver's parameter table from catalog names.

    ``overrides`` specializes a spec for this solver, e.g. a different
    default or a note about a different meaning.
    """
    table: dict[str, ParamSpec] = {}
    for name in names:
        spec = PARAMS[name]
        if name in overrides:
            spec = replace(spec, **overrides[name])
        table[name] = spec
    return table


_MATERIAL_LATTICE = ("material", "gamma", "Cl", "G", "kl", "T_melt_C",
                     "kTable", "gamma_manual", "Cl_manual", "G_manual",
                     "kl_manual")
_MATERIAL_OPTICAL = ("material", "gamma", "Cl", "G", "kl", "ke0",
                     "alpha_opt", "delta_opt", "T_melt_C", "kTable",
                     "gamma_manual", "Cl_manual", "G_manual", "kl_manual",
                     "ke0_manual", "alpha_opt_manual")

_KL_TOTAL = {"summary": "Total thermal conductivity. Overrides the preset.",
             "notes": "This solver family uses a single conductivity, so kl "
                      "is the total value."}
_KL_LATTICE = {"summary": "Lattice conductivity term, the kl of ke0 + kl. "
                          "Overrides the preset.",
               "notes": "This solver splits conductivity into ke0 + kl, so "
                        "kl is the lattice component only, not the total."}
# kl_manual follows the same split, so its default differs by family: the
# lattice term for the depth-resolved solvers, the total for the others.
_KTABLE_INERT = {
    "notes": "No effect in this solver: it uses a single constant "
             "conductivity rather than a k(T) table."}
_TMELT_INERT = {
    "notes": "Stored on the material record, but only the radial solver's "
             "melt early stop reads it."}
_KL_MANUAL_LATTICE = {
    "default": 24.0,
    "summary": "Lattice conductivity term for material='custom'.",
    "notes": "The lattice component of ke0 + kl, not the total.",
}

_SURFACE_POINT = SolverSchema(
    id="surface_point",
    title="Surface point",
    summary="0D electron and lattice temperature at the surface, with "
            "inter-pulse diffusion into the depth.",
    when_to_use="Pulse-accumulation studies and parameter sweeps, where the "
                "surface temperature history matters and depth resolution "
                "does not. The fastest solver.",
    when_not_to_use="When you need temperature versus depth, the electron "
                    "lattice inversion resolved in space, or any radial or "
                    "scanned geometry.",
    params=_params(
        *_MATERIAL_LATTICE, "Pavg", "spotRadius", "f_rep", "tau_FWHM",
        "pulseProfile", "absorbance", "T0_C", "simDuration",
        "Leff", "depthProfile", "dzTarget", "Ndiff", "legacyDeposit",
        "storeHistory",
        "makePlots", "saveFigures", "outputDir", "caseTag", "showProgress",
        kl=_KL_TOTAL, kTable=_KTABLE_INERT, T_melt_C=_TMELT_INERT,
        Pavg={"default": 1.0}, spotRadius={"default": 80e-6},
        f_rep={"default": 1e6}, tau_FWHM={"default": 500e-15},
    ),
    files=("TTM_<params>_<n>p_<profile>.txt",),
    seconds_per_pulse=2e-4,
    examples={
        "minimal": {"material": "W", "Pavg": 10, "f_rep": 5e6,
                    "simDuration": 50 / 5e6},
        "fast_smoke": {"simDuration": 10 / 1e6, "makePlots": False},
    },
)

_DEPTH_PROFILE = SolverSchema(
    id="depth_profile",
    title="Depth profile",
    summary="1D depth-resolved electron and lattice temperatures, solved "
            "per pulse with a stiff integrator.",
    when_to_use="The main multi-pulse workflow. Use it when you need "
                "temperature versus depth, per-pulse peaks, or the "
                "electron lattice inversion.",
    when_not_to_use="Large pulse counts, where the stiff solve per pulse is "
                    "expensive. Use surface_point for accumulation only, or "
                    "radial_profile for footprint questions.",
    params=_params(
        *_MATERIAL_OPTICAL, "Pavg", "spotRadius", "f_rep", "tau_FWHM",
        "pulseProfile", "absorbance", "T0_C", "simDuration",
        "Lz", "Nz", "snapshotDelays", "enableRadialProfile", "Nr_radial",
        "rMax_factor", "dzTarget_diff", "Ndiff", "relTol", "absTol",
        "storeHistory",
        "makePlots", "saveFigures", "outputDir", "caseTag", "showProgress",
        kl=_KL_LATTICE, kl_manual=_KL_MANUAL_LATTICE,
        T_melt_C=_TMELT_INERT,
        rMax_factor={"default": 3.0, "affects_numerics": False,
                     "notes": "Sets the extent of the derived radial view "
                              "only. It does not affect the depth solution."},
        Nr_radial={"affects_numerics": False,
                   "notes": "Node count of the derived radial view only."},
    ),
    files=("TTM_1D_Result_<params>_<n>p_<profile>.txt",),
    seconds_per_pulse=0.135,
    examples={
        "minimal": {"material": "W", "Pavg": 40, "simDuration": 10 / 18e6},
        "penetration_depth": {"material": "W", "delta_opt": 23e-9,
                              "simDuration": 100 / 18e6},
    },
)

_RADIAL_PROFILE = SolverSchema(
    id="radial_profile",
    title="Radial profile",
    summary="Radial surface temperature under a Gaussian spot, with depth "
            "and radial diffusion between pulses.",
    when_to_use="Melt radius and heat-affected footprint studies, and any "
                "question about how far heat spreads sideways.",
    when_not_to_use="When temperature versus depth is the quantity of "
                    "interest, or when a single point suffices.",
    params=_params(
        *_MATERIAL_LATTICE, "Pavg", "spotRadius", "f_rep", "tau_FWHM",
        "pulseProfile", "absorbance", "T0_C", "simDuration",
        "Leff", "depthProfile", "dzTarget", "Ndiff", "Nr", "rMax_factor",
        "radialSolveMode", "earlyStopMeltRadius_um", "earlyStopT_melt_C",
        "earlyStopCheckInterval", "legacyDeposit", "storeHistory",
        "makePlots", "saveFigures", "outputDir", "caseTag", "showProgress",
        kl=_KL_TOTAL,
        simDuration={"default": 1e-3},
        Ndiff={"notes": "A minimum here. The solver raises it as needed to "
                        "keep the Fourier number at or below 0.5, unlike the "
                        "other solvers where it is the exact substep count."},
    ),
    files=("TTM_Radial_Result_<params>_<n>p_<profile>.txt",),
    seconds_per_pulse=1.2e-2,
    examples={
        "minimal": {"material": "W", "Pavg": 40, "simDuration": 100 / 18e6},
        "melt_pool": {"material": "W", "Pavg": 70, "f_rep": 40e6,
                      "spotRadius": 150e-6, "tau_FWHM": 500e-15,
                      "absorbance": 0.4, "simDuration": 5000 / 40e6,
                      "storeHistory": False},
    },
)

_SINGLE_PULSE = SolverSchema(
    id="single_pulse",
    title="Single pulse",
    summary="One pulse, depth-resolved, with snapshots at chosen delays.",
    when_to_use="Inspecting the early-time electron lattice dynamics of a "
                "single pulse, and teaching figures.",
    when_not_to_use="Anything involving heat accumulation over many pulses.",
    params=_params(
        *_MATERIAL_OPTICAL, "Pavg", "spotRadius", "f_rep", "tau_FWHM",
        "pulseProfile", "absorbance", "T0_C", "Lz", "Nz", "snapshotDelays",
        "relTol", "absTol",
        "makePlots", "saveFigures", "outputDir", "caseTag",
        kl=_KL_LATTICE, kl_manual=_KL_MANUAL_LATTICE,
        T_melt_C=_TMELT_INERT,
        Pavg={"default": 1.0}, spotRadius={"default": 80e-6},
        f_rep={"default": 1e6,
               "notes": "Only sets the pulse energy, Pavg / f_rep. This "
                        "solver always runs exactly one pulse."},
        kTable={"notes": "Inert here: this solver always uses a constant "
                         "electron conductivity, matching the MATLAB "
                         "single-pulse visualizer."},
    ),
    files=("TTM1D_<params>_1p_<profile>.txt",),
    seconds_per_pulse=2.0,
    examples={"minimal": {"material": "W", "Pavg": 1.0}},
)

_INVERSION = SolverSchema(
    id="inversion_quantifier",
    title="Inversion analysis",
    summary="Per-pulse statistics of the electron lattice temperature "
            "inversion, computed from a depth-profile solution.",
    when_to_use="Quantifying how the Tl > Te inversion evolves across a "
                "pulse train.",
    when_not_to_use="When you only need one run's temperatures. Call "
                    "depth_profile directly.",
    params=_params(
        *_MATERIAL_OPTICAL, "Pavg", "spotRadius", "f_rep", "tau_FWHM",
        "pulseProfile", "absorbance", "T0_C", "simDuration",
        "Lz", "Nz", "snapshotDelays", "dzTarget_diff", "Ndiff",
        "relTol", "absTol", "depthResults",
        "makePlots", "saveFigures", "outputDir", "caseTag", "showProgress",
        kl=_KL_LATTICE, kl_manual=_KL_MANUAL_LATTICE,
        T_melt_C=_TMELT_INERT,
    ),
    files=("Inversion_Analysis_<params>_<n>p.txt",),
    seconds_per_pulse=0.135,
    examples={"minimal": {"material": "W", "simDuration": 20 / 18e6}},
)

_SCANNING_BEAM = SolverSchema(
    id="scanning_beam",
    title="Scanning beam",
    summary="2D surface temperature under a beam moving along a scan line.",
    when_to_use="Translating stationary results to a scanned process, and "
                "peak-temperature maps along a track.",
    when_not_to_use="When depth resolution or the single-pulse transient "
                    "matters.",
    params=_params(
        *_MATERIAL_LATTICE, "Pavg", "spotRadius", "f_rep", "tau_FWHM",
        "pulseProfile", "absorbance", "T0_C", "v_scan", "scanLength",
        "Leff", "depthProfile", "dzTarget", "Ndiff", "NadiPerGap",
        "legacyDeposit",
        "Nx", "Ny", "xPad", "yExtent",
        "makePlots", "saveFigures", "outputDir", "caseTag", "showProgress",
        kl=_KL_TOTAL, kTable=_KTABLE_INERT, T_melt_C=_TMELT_INERT,
    ),
    files=("TTMmov_<params>_<n>p.txt", "TTMmov_<params>_<n>p_surface.mat"),
    seconds_per_pulse=2e-5,
    examples={
        "minimal": {"material": "W", "Pavg": 40, "v_scan": 1.0,
                    "scanLength": 2e-3},
    },
)

_SCHEMAS: dict[str, SolverSchema] = {
    s.id: s for s in (_SURFACE_POINT, _DEPTH_PROFILE, _RADIAL_PROFILE,
                      _SINGLE_PULSE, _INVERSION, _SCANNING_BEAM)
}

SOLVER_IDS: tuple[str, ...] = tuple(_SCHEMAS)


# ==============================  Accessors  ================================


def solver_schema(solver_id: str) -> SolverSchema:
    """Return the schema for a solver id, case-insensitively."""
    key = str(solver_id).strip().lower()
    if key not in _SCHEMAS:
        known = ", ".join(SOLVER_IDS)
        raise KeyError(f"Unknown solver '{solver_id}'. Known solvers: {known}")
    return _SCHEMAS[key]


def defaults(solver_id: str) -> dict[str, Any]:
    """Every default for one solver, as a plain dict."""
    return solver_schema(solver_id).defaults()


def describe_solver(solver_id: str, section: str = "all") -> dict[str, Any]:
    """Full description of one solver: inputs, results, files, examples.

    ``section`` selects a subset: 'inputs', 'results', 'files',
    'examples' or 'all'.
    """
    sch = solver_schema(solver_id)
    out: dict[str, Any] = {
        "id": sch.id,
        "title": sch.title,
        "summary": sch.summary,
        "whenToUse": sch.when_to_use,
        "whenNotToUse": sch.when_not_to_use,
    }
    if section in ("all", "inputs"):
        out["params"] = {n: s.as_dict() for n, s in sch.params.items()}
    if section in ("all", "results"):
        out["results"] = describe_results(sch.id)
    if section in ("all", "files"):
        out["files"] = list(sch.files)
    if section in ("all", "examples"):
        out["examples"] = dict(sch.examples)
    return out


def list_solvers() -> list[dict[str, Any]]:
    """One routing record per solver: what it does and when to reach for it."""
    return [
        {
            "id": s.id,
            "title": s.title,
            "summary": s.summary,
            "whenToUse": s.when_to_use,
            "whenNotToUse": s.when_not_to_use,
            "nParams": len(s.params),
            "minimalConfig": s.examples.get("minimal", {}),
        }
        for s in _SCHEMAS.values()
    ]


# ===========================================================================
# The results contract: every key a solver returns, described once.
# ===========================================================================


@dataclass(frozen=True)
class ResultField:
    """One results-dict key: what it holds, in what unit, when it appears."""

    name: str
    kind: str                # scalar | array | str | bool | dict | path | list
    unit: str | None
    summary: str
    gated_by: str | None = None   # config key that must be enabled for it
    prefer: str | None = None     # the newer spelling to reach for instead

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "kind": self.kind,
                               "unit": self.unit, "summary": self.summary}
        if self.gated_by:
            out["gatedBy"] = self.gated_by
        if self.prefer:
            out["prefer"] = self.prefer
        return out


_r = ResultField

# Keys present in every solver's results.
RESULT_ENVELOPE: tuple[ResultField, ...] = (
    _r("solver", "str", None, "Human-readable solver name."),
    _r("solverId", "str", None, "The id used by run(), the CLI and MCP."),
    _r("contractVersion", "str", None,
       "Results-contract generation, 'v1' since the first release."),
    _r("material", "str", None, "Material key as given in the config."),
    _r("caseTag", "str", None,
       "Sanitized caseTag echoed back, empty when unset."),
    _r("resolvedConfig", "dict", None,
       "The config actually in force: defaults overlaid with the caller's "
       "values under the same empty-means-default semantics the solver "
       "itself applies."),
    _r("materialProps", "dict", None,
       "The resolved material record: gamma, Cl, G, the two conductivity "
       "terms spelled apart as kTotal_W_mK and kLattice_W_mK, optical "
       "properties, melting point, and the conductivity model in use. "
       "None only when a pre-0.1.22 depthResults dict was supplied."),
    _r("warnings", "list", None,
       "Validity warnings raised during the run, for consumers that never "
       "see the console. Includes a peak lattice temperature above the "
       "melting point, which the model cannot represent."),
    _r("nPulses", "scalar", None, "Number of pulses simulated."),
    _r("wallTime_s", "scalar", "s", "Wall-clock solve time."),
    _r("outputFile", "path", None, "The text report written by this run."),
    _r("outputDir", "path", None, "Directory holding every file written."),
    _r("inputConfig", "dict", None,
       "The caller's config exactly as passed, unmerged.",
       prefer="resolvedConfig"),
)

# Keys specific to one solver, beyond the envelope.
RESULTS: dict[str, tuple[ResultField, ...]] = {
    "surface_point": (
        _r("time_s", "array", "s",
           "Sample times over the whole train; empty when storeHistory "
           "is off."),
        _r("Te_K", "array", "K", "Electron temperature at each sample."),
        _r("Tl_K", "array", "K", "Lattice temperature at each sample."),
        _r("Te_C", "array", "degC", "Electron temperature, Celsius."),
        _r("Tl_C", "array", "degC", "Lattice temperature, Celsius."),
        _r("peakPulse", "scalar", None,
           "1-based pulse on which the electron peak occurred."),
        _r("peakTe_C", "scalar", "degC", "Peak electron temperature."),
        _r("peakTl_C", "scalar", "degC", "Peak lattice temperature."),
        _r("finalTe_C", "scalar", "degC",
           "Electron temperature at the last sample."),
        _r("finalTl_C", "scalar", "degC",
           "Lattice temperature at the last sample."),
        _r("finalResid_C", "scalar", "degC",
           "Residual surface temperature after the final inter-pulse "
           "diffusion."),
        _r("TeqVals_C", "array", "degC",
           "Post-pulse equilibrium temperature, one per pulse."),
        _r("TresidVals_C", "array", "degC",
           "Residual temperature after each inter-pulse diffusion."),
        _r("absorbedAreal_J_m2", "scalar", "J/m^2",
           "Energy absorbed per unit area over the run."),
        _r("depthEnergy_J_m2", "scalar", "J/m^2",
           "Energy stored in the depth grid at the end."),
        _r("energyMismatch_pct", "scalar", "%",
           "Bookkeeping mismatch between the two; expected to be nonzero "
           "in this hybrid 0D+1D model."),
        _r("makePlots", "bool", None, "Echo of the plotting switch.",
           prefer="resolvedConfig"),
        _r("saveFigures", "bool", None, "Echo of the figure switch.",
           prefer="resolvedConfig"),
        _r("figureFile", "path", None, "The saved timeline figure.",
           gated_by="makePlots"),
    ),
    "depth_profile": (
        _r("peakPulse", "scalar", None,
           "1-based pulse on which the electron peak occurred."),
        _r("peakTe_C", "scalar", "degC", "Peak surface electron temperature."),
        _r("peakTl_C", "scalar", "degC", "Peak surface lattice temperature."),
        _r("finalResid_C", "scalar", "degC",
           "Residual surface temperature after the final diffusion."),
        _r("invDetected", "bool", None,
           "Whether any surface inversion (Tl > Te) exceeded the "
           "threshold."),
        _r("maxInv_K", "scalar", "K",
           "Largest surface Tl - Te over the run. Zero when none."),
        _r("invThreshold_K", "scalar", "K",
           "Threshold above which Tl - Te counts as an inversion."),
        _r("absorbedAreal_J_m2", "scalar", "J/m^2",
           "Energy absorbed per unit area over the run."),
        _r("depthEnergy_J_m2", "scalar", "J/m^2",
           "Energy stored in the diffusion grid at the end."),
        _r("energyMismatch_pct", "scalar", "%",
           "Bookkeeping mismatch between the two; expected in the hybrid "
           "fine+coarse model."),
        _r("TePeakPerPulse_C", "array", "degC",
           "Peak surface electron temperature, one per pulse."),
        _r("TlPeakPerPulse_C", "array", "degC",
           "Peak surface lattice temperature, one per pulse."),
        _r("TeqVals_C", "array", "degC",
           "Post-pulse equilibrium temperature, one per pulse."),
        _r("TresidVals_C", "array", "degC",
           "Residual temperature after each inter-pulse diffusion."),
        _r("baseTempPerPulse_C", "array", "degC",
           "Baseline temperature each pulse started from."),
        _r("invMaxPerPulse_K", "array", "K",
           "Largest surface Tl - Te within each pulse."),
        _r("tMaxInvPerPulse_s", "array", "s",
           "Time of the largest inversion, relative to each pulse centre; "
           "NaN where none."),
        _r("tInvOnsetPerPulse_s", "array", "s",
           "Inversion onset time relative to each pulse centre; NaN where "
           "none."),
        _r("invDurationPerPulse_s", "array", "s",
           "How long the inversion lasted in each pulse. Zero where none."),
        _r("Te_atMaxInvPerPulse_C", "array", "degC",
           "Electron temperature at the moment of largest inversion; NaN "
           "where none."),
        _r("Tl_atMaxInvPerPulse_C", "array", "degC",
           "Lattice temperature at the moment of largest inversion; NaN "
           "where none."),
        _r("time_s", "array", "s",
           "Surface sample times over the whole train; empty when "
           "storeHistory is off."),
        _r("Te_C", "array", "degC",
           "Surface electron temperature at each sample; empty when "
           "storeHistory is off."),
        _r("Tl_C", "array", "degC",
           "Surface lattice temperature at each sample; empty when "
           "storeHistory is off."),
        _r("zGrid_m", "array", "m", "Fine depth grid of the snapshots."),
        _r("snapshotDelays_s", "array", "s",
           "Delay of each first-pulse snapshot from the pulse centre, as "
           "sampled."),
        _r("TeSnapshots_C", "array", "degC",
           "Electron temperature versus depth at each first-pulse "
           "snapshot, shape (nSnapshots, Nz)."),
        _r("TlSnapshots_C", "array", "degC",
           "Lattice temperature versus depth at each first-pulse "
           "snapshot, shape (nSnapshots, Nz)."),
        _r("zGridDiff_m", "array", "m",
           "Coarse diffusion grid of the residual profiles."),
        _r("profileSnapshotPulses", "array", None,
           "1-based pulses at which a residual depth profile was kept, "
           "logarithmically spaced, at most 12."),
        _r("profileSnapshots_C", "array", "degC",
           "Residual temperature versus depth after each snapshot pulse, "
           "shape (nProfiles, NzDiff)."),
        _r("f_rep", "scalar", "Hz", "Echo of the repetition rate.",
           prefer="resolvedConfig"),
        _r("Pavg", "scalar", "W", "Echo of the average power.",
           prefer="resolvedConfig"),
        _r("tau_FWHM", "scalar", "s", "Echo of the pulse width.",
           prefer="resolvedConfig"),
        _r("spotRadius", "scalar", "m", "Echo of the spot radius.",
           prefer="resolvedConfig"),
        _r("absorbance", "scalar", None, "Echo of the absorbance.",
           prefer="resolvedConfig"),
        _r("T0_C", "scalar", "degC", "Echo of the initial temperature.",
           prefer="resolvedConfig"),
        _r("F_peak", "scalar", "J/m^2", "Peak fluence at beam centre."),
        _r("gamma", "scalar", "J/(m^3 K^2)",
           "Electron heat-capacity coefficient used.",
           prefer="materialProps"),
        _r("Cl", "scalar", "J/(m^3 K)", "Lattice heat capacity used.",
           prefer="materialProps"),
        _r("G", "scalar", "W/(m^3 K)", "Electron-phonon coupling used.",
           prefer="materialProps"),
        _r("ke0", "scalar", "W/(m K)",
           "Electron conductivity coefficient used.",
           prefer="materialProps"),
        _r("kl", "scalar", "W/(m K)",
           "Lattice conductivity term used; the lattice-only share here, "
           "unlike the total the 0D solvers call kl.",
           prefer="materialProps"),
        _r("alpha_opt", "scalar", "1/m",
           "Optical absorption coefficient used.", prefer="materialProps"),
        _r("Trep", "scalar", "s", "Pulse period, 1/f_rep."),
        _r("simDuration", "scalar", "s", "Simulated duration.",
           prefer="resolvedConfig"),
        _r("radialGrid_um", "array", "um",
           "Radial positions of the scaled view.",
           gated_by="enableRadialProfile"),
        _r("radialFluenceRatio", "array", None,
           "Gaussian fluence ratio at each radius.",
           gated_by="enableRadialProfile"),
        _r("radialSurfaceProfiles_C", "array", "degC",
           "Surface temperature versus radius at each snapshot pulse.",
           gated_by="enableRadialProfile"),
        _r("crossSection_C", "array", "degC",
           "Depth-by-radius temperature map at the last snapshot.",
           gated_by="enableRadialProfile"),
        _r("lateralDiffusionLength_m", "scalar", "m",
           "Lateral diffusion length the radial scaling assumes small.",
           gated_by="enableRadialProfile"),
    ),
    "radial_profile": (
        _r("mode", "str", None,
           "Which radial algorithm ran: 'scale' or 'independent'."),
        _r("nPulsesRequested", "scalar", None,
           "Pulses the config asked for. nPulses is what actually ran."),
        _r("earlyStopped", "bool", None,
           "True when the melt-radius early stop ended the run before "
           "nPulsesRequested."),
        _r("peakTeq_C", "scalar", "degC",
           "Largest post-pulse equilibrium temperature at beam centre."),
        _r("finalResid_C", "scalar", "degC",
           "Residual centre temperature after the final diffusion."),
        _r("TeqVals_C", "array", "degC",
           "Post-pulse equilibrium temperature at centre, one per pulse."),
        _r("TresidVals_C", "array", "degC",
           "Residual centre temperature after each inter-pulse diffusion."),
        _r("rGrid_um", "array", "um", "Radial grid positions."),
        _r("finalRadialProfile_C", "array", "degC",
           "Residual surface temperature versus radius after the last "
           "pulse."),
        _r("spotRadius_um", "scalar", "um", "Echo of the spot radius.",
           prefer="resolvedConfig"),
    ),
    "single_pulse": (
        _r("peakTe_C", "scalar", "degC", "Peak surface electron temperature."),
        _r("peakTl_C", "scalar", "degC", "Peak surface lattice temperature."),
        _r("finalTe_C", "scalar", "degC",
           "Surface electron temperature at the end."),
        _r("finalTl_C", "scalar", "degC",
           "Surface lattice temperature at the end."),
        _r("finalResid_C", "scalar", "degC",
           "Same value as finalTl_C, under the cross-solver name."),
        _r("invDetected", "bool", None,
           "Whether the surface inversion exceeded the threshold."),
        _r("maxInv_C", "scalar", "K",
           "Largest surface Tl - Te; a temperature difference despite the "
           "historical _C spelling.", prefer="maxInv_K"),
        _r("maxInv_K", "scalar", "K",
           "Largest surface Tl - Te. Zero when none."),
        _r("invThreshold_K", "scalar", "K",
           "Threshold above which Tl - Te counts as an inversion."),
        _r("tInvOnset_s", "scalar", "s",
           "Inversion onset relative to the pulse centre. NaN when none."),
        _r("tMaxInv_s", "scalar", "s",
           "Time of the largest inversion relative to the pulse centre; "
           "NaN when none."),
        _r("time_s", "array", "s", "Surface sample times."),
        _r("Te_C", "array", "degC",
           "Surface electron temperature at each sample."),
        _r("Tl_C", "array", "degC",
           "Surface lattice temperature at each sample."),
        _r("zGrid_m", "array", "m", "Depth grid of the snapshots."),
        _r("snapshotDelays_s", "array", "s",
           "Delay of each snapshot from the pulse centre, as sampled."),
        _r("TeSnapshots_C", "array", "degC",
           "Electron temperature versus depth at each snapshot, shape "
           "(nSnapshots, Nz)."),
        _r("TlSnapshots_C", "array", "degC",
           "Lattice temperature versus depth at each snapshot, shape "
           "(nSnapshots, Nz)."),
        _r("absorbedAreal_J_m2", "scalar", "J/m^2",
           "Energy absorbed per unit area."),
        _r("depthEnergy_J_m2", "scalar", "J/m^2",
           "Energy stored in the depth grid at the end."),
        _r("energyMismatch_pct", "scalar", "%",
           "Bookkeeping mismatch between the two."),
    ),
    "scanning_beam": (
        _r("Tpeak_map", "array", "K",
           "Peak temperature ever reached at each surface point."),
        _r("Tsurf", "array", "K", "Final surface temperature map."),
        _r("peakT_history", "array", "K",
           "Peak surface temperature after each pulse."),
        _r("xGrid", "array", "m", "Surface grid x positions."),
        _r("yGrid", "array", "m", "Surface grid y positions."),
        _r("peakT_C", "scalar", "degC",
           "Largest temperature anywhere on the map."),
        _r("pulseSpacing", "scalar", "m",
           "Distance the beam moves between pulses, v_scan/f_rep."),
        _r("simDuration_s", "scalar", "s",
           "Scan duration, scanLength/v_scan."),
        _r("wallTime", "scalar", "s", "Same value as wallTime_s.",
           prefer="wallTime_s"),
        _r("dTeq_single", "scalar", "K",
           "Single-pulse equilibrium temperature rise used by the "
           "superposition."),
        _r("params", "dict", None,
           "The defaults-merged parameter dict this solver ran from.",
           prefer="resolvedConfig"),
        _r("outPath", "path", None, "Same value as outputFile.",
           prefer="outputFile"),
        _r("matPath", "path", None,
           "MATLAB-compatible .mat with the surface maps."),
    ),
    "inversion_quantifier": (
        _r("nInvPulses", "scalar", None,
           "Pulses whose inversion exceeded the threshold."),
        _r("invThreshold_K", "scalar", "K",
           "Threshold above which Tl - Te counts as an inversion."),
        _r("meanInv_K", "scalar", "K",
           "Mean of the per-pulse maximum inversions. NaN when none."),
        _r("maxInv_K", "scalar", "K", "Largest inversion in any pulse."),
        _r("minInv_K", "scalar", "K",
           "Smallest inversion among inverted pulses."),
        _r("stdInv_K", "scalar", "K",
           "Standard deviation of the per-pulse maxima."),
        _r("invSlope_KperPulse", "scalar", "K/pulse",
           "Linear trend of inversion magnitude across the train."),
        _r("corrBaseTempInv", "scalar", None,
           "Correlation between baseline temperature and inversion "
           "magnitude. NaN below 3 inverted pulses."),
        _r("meanInvFraction", "scalar", None,
           "Mean inversion as a fraction of the electron excursion."),
        _r("invMaxPerPulse_K", "array", "K",
           "Largest surface Tl - Te within each pulse."),
        _r("TePeak_C", "array", "degC",
           "Peak surface electron temperature, one per pulse."),
        _r("TlPeak_C", "array", "degC",
           "Peak surface lattice temperature, one per pulse."),
        _r("Tbase_C", "array", "degC",
           "Baseline temperature each pulse started from."),
        _r("Teq_C", "array", "degC",
           "Post-pulse equilibrium temperature, one per pulse."),
        _r("Tresid_C", "array", "degC",
           "Residual temperature after each inter-pulse diffusion."),
        _r("tMaxInv_s", "array", "s",
           "Time of the largest inversion per pulse. NaN where none."),
        _r("tOnset_s", "array", "s",
           "Inversion onset per pulse. NaN where none."),
        _r("invDuration_s", "array", "s",
           "Inversion duration per pulse. Zero where none."),
        _r("Te_atMaxInv_C", "array", "degC",
           "Electron temperature at the largest inversion. NaN where "
           "none."),
        _r("Tl_atMaxInv_C", "array", "degC",
           "Lattice temperature at the largest inversion. NaN where none."),
        _r("peakTe_C", "scalar", "degC",
           "Peak surface electron temperature, from the depth run."),
        _r("peakTl_C", "scalar", "degC",
           "Peak surface lattice temperature, from the depth run."),
        _r("finalResid_C", "scalar", "degC",
           "Final residual temperature, from the depth run."),
        _r("depthResults", "dict", None,
           "The full depth_profile results this analysis ran on."),
        _r("depthOutputFile", "path", None,
           "The depth run's own text report."),
    ),
}


def describe_results(solver_id: str) -> list[dict[str, Any]]:
    """Every key in one solver's results dict, with unit and meaning.

    The shared envelope comes first, then the solver's own fields. A
    ``gatedBy`` entry names the config key that must be enabled for the
    field to appear; ``prefer`` points at the newer spelling of a value
    kept for compatibility.
    """
    sch = solver_schema(solver_id)
    return [f.as_dict() for f in RESULT_ENVELOPE + RESULTS[sch.id]]


def _matlab_round(x: float) -> int:
    """MATLAB round(): half away from zero, matching the solvers."""
    return math.floor(x + 0.5)


def effective_config(solver_id: str, cfg: dict | None = None) -> dict[str, Any]:
    """The config actually in force: defaults overlaid with the caller's cfg.

    This is what every solver returns as ``resolvedConfig``, and the same
    merge validate_config reports as ``resolved``. None and empty values
    fall back to the default, exactly as get_cfg_field treats them.
    """
    return _effective(solver_schema(solver_id), cfg)


def _effective(sch: SolverSchema, cfg: dict | None) -> dict[str, Any]:
    """Defaults overlaid with the config, honoring MATLAB empty semantics.

    A key set to None or an empty string/collection means "use the default",
    exactly as config.get_cfg_field treats it at the solver's own read sites.
    """
    merged = sch.defaults()
    for key, value in (cfg or {}).items():
        if value is None or (isinstance(value, (str, list, tuple, dict))
                             and len(value) == 0):
            continue
        merged[key] = value
    return merged


def require_pulses(solver_id: str, n_pulses: int) -> None:
    """Reject a config that rounds to no pulses, before the solver runs.

    The pulse loop would simply never execute, leaving every accumulator
    empty, and the failure would surface much later as an unhelpful error
    from concatenating nothing.
    """
    if n_pulses >= 1:
        return
    if solver_id == "scanning_beam":
        fix = ("Lengthen scanLength, lower v_scan, or raise f_rep so that "
               "scanLength / v_scan * f_rep is at least 1.")
    else:
        fix = ("Raise simDuration to at least one pulse period, 1 / f_rep. "
               "For N pulses use simDuration = N / f_rep.")
    raise ValueError(
        f"{solver_id} was asked to simulate 0 pulses, so there is nothing to "
        f"solve. {fix}")


def estimate_run(solver_id: str, cfg: dict | None = None) -> dict[str, Any]:
    """Pulse count and rough runtime for a config, without running anything.

    The runtime is a coarse per-solver rate, meant for choosing between a
    blocking call and a background job, not for benchmarking.
    """
    sch = solver_schema(solver_id)
    merged = _effective(sch, cfg)

    if sch.id == "single_pulse":
        n_pulses = 1
    elif sch.id == "scanning_beam":
        v_scan = float(merged.get("v_scan") or 1.0)
        n_pulses = _matlab_round(
            float(merged["scanLength"]) / v_scan * float(merged["f_rep"]))
    else:
        n_pulses = _matlab_round(
            float(merged["simDuration"]) * float(merged["f_rep"]))

    if sch.id == "radial_profile" and str(
            merged.get("radialSolveMode", "scale")).lower() == "independent":
        rate = sch.seconds_per_pulse * float(merged.get("Nr") or 80)
    else:
        rate = sch.seconds_per_pulse

    seconds = n_pulses * rate
    return {
        "solver": sch.id,
        "nPulses": n_pulses,
        "estRuntime_s": round(seconds, 1),
        "recommend": "run_quick" if seconds < 45 else "start_run",
        "basis": "coarse per-solver rate; order of magnitude only",
    }


# ==============================  Validation  ===============================

_UNIT_SLIPS: dict[str, tuple[tuple[float, str], ...]] = {
    "m": ((1e-6, "micrometres"), (1e-9, "nanometres"), (1e-3, "millimetres")),
    "s": ((1e-15, "femtoseconds"), (1e-12, "picoseconds"),
          (1e-9, "nanoseconds"), (1e-6, "microseconds")),
    "Hz": ((1e6, "megahertz"), (1e3, "kilohertz"), (1e9, "gigahertz")),
    "1/m": ((1e6, "per micrometre"),),
}


def _unit_slip_hint(spec: ParamSpec, value: float) -> str | None:
    """If a decade slip would land the value in range, name the likeliest.

    Several slips can be in range at once, so a value that also lands in the
    typical window wins. Failing that the candidates are ordered by how often
    the mistake is made for that dimension.
    """
    candidates = list(_UNIT_SLIPS.get(spec.unit or "", ()))
    if spec.name == "absorbance":
        candidates = [(0.01, "a percentage")]

    low = -math.inf if spec.minimum is None else spec.minimum
    high = math.inf if spec.maximum is None else spec.maximum
    hits = [(value * factor, label) for factor, label in candidates
            if low <= value * factor <= high]
    if not hits:
        return None

    if spec.typical:
        preferred = [h for h in hits
                     if spec.typical[0] <= h[0] <= spec.typical[1]]
        if len(preferred) == 1:
            hits = preferred

    scaled, label = hits[0]
    return (f"That value looks like {label}. Pass SI units: "
            f"{spec.name} = {scaled:g}.")


def _suggest_key(key: str, sch: SolverSchema) -> str:
    """Explain an unrecognized key as helpfully as the schema allows."""
    lowered = {name.lower(): name for name in sch.params}
    if key.lower() in lowered:
        return (f"Did you mean '{lowered[key.lower()]}'? "
                "Config keys are case-sensitive.")

    elsewhere = [s.id for s in _SCHEMAS.values()
                 if key in s.params and s.id != sch.id]
    if elsewhere:
        return (f"'{key}' is read by {', '.join(elsewhere)}, not by "
                f"{sch.id}. Run describe_solver('{sch.id}') for the keys "
                "this solver accepts.")

    close = difflib.get_close_matches(key, list(sch.params), n=3, cutoff=0.6)
    if close:
        return "Did you mean " + " or ".join(f"'{c}'" for c in close) + "?"
    return (f"{sch.id} accepts: " + ", ".join(sorted(sch.params)) + ".")


def validate_config(solver_id: str, cfg: dict | None = None) -> dict[str, Any]:
    """Check a config against a solver's schema without running anything.

    Returns ``ok``, ``errors``, ``warnings``, the ``resolved`` config with
    defaults merged, and a runtime ``estimate``. Every problem is reported
    at once, each with a machine-readable ``code`` and a suggested fix.
    """
    sch = solver_schema(solver_id)
    cfg = dict(cfg or {})
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for key, value in cfg.items():
        spec = sch.params.get(key)
        if spec is None:
            errors.append({
                "code": "unknown_key", "key": key, "value": value,
                "message": f"Unknown key '{key}' for solver '{sch.id}'.",
                "suggestion": _suggest_key(key, sch),
            })
            continue

        if value is None or (isinstance(value, (str, list, tuple, dict))
                             and len(value) == 0):
            continue  # MATLAB empty semantics: falls back to the default

        if spec.kind == "enum" and spec.choices:
            # Read sites lowercase enum values, so validation must too.
            if str(value).lower() not in {c.lower() for c in spec.choices}:
                errors.append({
                    "code": "bad_enum", "key": key, "value": value,
                    "message": f"'{value}' is not a valid {key}.",
                    "suggestion": "Use one of: "
                                  + ", ".join(spec.choices) + ".",
                })
            continue

        if spec.kind in ("float", "int"):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append({
                    "code": "bad_type", "key": key, "value": value,
                    "message": f"{key} must be a number, got "
                               f"{type(value).__name__}.",
                    "suggestion": f"{key} is {spec.summary.lower()}",
                })
                continue
            number = float(value)
            below = spec.minimum is not None and number < spec.minimum
            above = spec.maximum is not None and number > spec.maximum
            if below or above:
                unit = f" {spec.unit}" if spec.unit else ""
                hint = _unit_slip_hint(spec, number)
                errors.append({
                    "code": "unit_slip" if hint else "out_of_range",
                    "key": key, "value": value,
                    "message": f"{key} = {value:g} is outside the plausible "
                               f"range {spec.minimum:g} to {spec.maximum:g}"
                               f"{unit}.",
                    "suggestion": hint or f"{key} is {spec.summary.lower()}",
                })
            elif spec.typical and not (spec.typical[0] <= number
                                       <= spec.typical[1]):
                warnings.append({
                    "code": "unusual_value", "key": key, "value": value,
                    "message": f"{key} = {value:g} is outside the typical "
                               f"range {spec.typical[0]:g} to "
                               f"{spec.typical[1]:g}"
                               + (f" {spec.unit}." if spec.unit else "."),
                    "suggestion": "This is allowed. Check it is intended.",
                })

    resolved = _effective(sch, cfg)
    estimate = estimate_run(sch.id, cfg) if not errors else None

    # A config that rounds to no pulses at all is not runnable: the pulse loop
    # never executes and the solver fails later on an empty result. Catch it
    # here, where the fix can be named.
    if estimate and estimate["nPulses"] < 1:
        if sch.id == "scanning_beam":
            fix = ("Lengthen scanLength, lower v_scan, or raise f_rep so that "
                   "scanLength / v_scan * f_rep is at least 1.")
        else:
            fix = ("Raise simDuration to at least one pulse period, "
                   "1 / f_rep. For N pulses use simDuration = N / f_rep.")
        errors.append({
            "code": "no_pulses",
            "key": "scanLength" if sch.id == "scanning_beam" else "simDuration",
            "value": resolved.get(
                "scanLength" if sch.id == "scanning_beam" else "simDuration"),
            "message": "This config simulates 0 pulses, so there is nothing "
                       "to solve.",
            "suggestion": fix,
        })
        estimate = None

    if estimate and estimate["estRuntime_s"] > 600:
        warnings.append({
            "code": "expensive_run", "key": "simDuration",
            "value": estimate["nPulses"],
            "message": f"This config is {estimate['nPulses']} pulses, very "
                       f"roughly {estimate['estRuntime_s'] / 60:.0f} minutes.",
            "suggestion": "Run it as a background job rather than a "
                          "blocking call.",
        })

    return {
        "ok": not errors,
        "solver": sch.id,
        "errors": errors,
        "warnings": warnings,
        "resolved": resolved,
        "estimate": estimate,
    }


_JSON_KINDS = {"float": "number", "int": "integer", "bool": "boolean",
               "enum": "string", "str": "string", "path": "string",
               "array": "array"}


def json_schema(solver_id: str) -> dict[str, Any]:
    """A JSON Schema for one solver's config, for typed machine consumers."""
    sch = solver_schema(solver_id)
    props: dict[str, Any] = {}
    for name, spec in sch.params.items():
        prop: dict[str, Any] = {"description": spec.summary}
        if spec.kind in _JSON_KINDS and spec.kind != "any":
            prop["type"] = _JSON_KINDS[spec.kind]
        if spec.choices:
            prop["enum"] = list(spec.choices)
        if spec.minimum is not None:
            prop["minimum"] = spec.minimum
        if spec.maximum is not None:
            prop["maximum"] = spec.maximum
        if spec.default is not None:
            prop["default"] = (list(spec.default)
                               if isinstance(spec.default, tuple)
                               else spec.default)
        if spec.unit:
            prop["x-unit"] = spec.unit
        props[name] = prop
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": f"laserttm {sch.id} config",
        "description": sch.summary,
        "type": "object",
        "additionalProperties": False,
        "properties": props,
    }


def materials_table() -> list[dict[str, Any]]:
    """Every material preset with its properties, units and solver support."""
    from .materials import MATERIALS  # heavy import, kept out of module load

    rows = []
    for key, mat in MATERIALS.items():
        rows.append({
            "key": key.upper(),
            "gamma": mat.gamma,
            "Cl": mat.cl,
            "G": mat.g_ep,
            "kTotal": mat.k_total,
            "T_melt_C": mat.t_melt_c,
            "ke0": mat.ke0,
            "kl": mat.kl,
            "alpha_opt": mat.alpha_opt,
            "delta_opt": mat.delta_opt,
            "measuredKTable": mat.measured_k_table,
            "solvers": (list(SOLVER_IDS) if mat.has_optical else
                        ["surface_point", "radial_profile", "scanning_beam"]),
        })
    return rows


UNITS = {
    "gamma": "J m^-3 K^-2", "Cl": "J m^-3 K^-1", "G": "W m^-3 K^-1",
    "kTotal": "W m^-1 K^-1", "ke0": "W m^-1 K^-1", "kl": "W m^-1 K^-1",
    "alpha_opt": "1/m", "delta_opt": "m", "T_melt_C": "degC",
}
