"""laserttm — two-temperature model solvers for ultrafast pulsed-laser heating of metals.

Python port of the Ultrafast Laser TTM Toolbox (MATLAB reference
implementation for doi:10.1007/s11665-026-14738-6).
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from .depth_profile import depth_profile_solver
from .inversion_quantifier import inversion_quantifier
from .radial_profile import radial_profile_solver
from .scanning_beam import scanning_beam_solver
from .single_pulse import single_pulse_visualizer
from .surface_point import surface_point_solver

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
