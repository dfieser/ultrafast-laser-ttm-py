"""laserttm — two-temperature model solvers for ultrafast pulsed-laser heating of metals.

Python port of the Ultrafast Laser TTM Toolbox (MATLAB reference
implementation for doi:10.1007/s11665-026-14738-6).
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

# Solver entry points, resolved lazily (PEP 562): the solver modules import
# numba and scipy, which costs about a second per process. `laserttm list`,
# `laserttm version`, and MCP server startup never touch them.
_SOLVER_EXPORTS = {
    "depth_profile_solver": "depth_profile",
    "inversion_quantifier": "inversion_quantifier",
    "radial_profile_solver": "radial_profile",
    "scanning_beam_solver": "scanning_beam",
    "single_pulse_visualizer": "single_pulse",
    "surface_point_solver": "surface_point",
}

# Single source of truth is the VERSION file at the repository root: the
# build backend stamps it into the package metadata (see [tool.hatch.version]
# in pyproject.toml), and the release workflow bumps it on every push.
try:
    __version__ = _pkg_version("laserttm")
except PackageNotFoundError:  # running from a source tree without install
    import os as _os

    try:
        with open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                "..", "..", "VERSION")) as _f:
            __version__ = _f.read().strip()
    except OSError:
        __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    "depth_profile_solver",
    "inversion_quantifier",
    "radial_profile_solver",
    "scanning_beam_solver",
    "single_pulse_visualizer",
    "surface_point_solver",
]


def __getattr__(name: str):
    module_name = _SOLVER_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value  # cache so the import runs once
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_SOLVER_EXPORTS))
