"""laserttm — two-temperature model solvers for ultrafast pulsed-laser heating of metals.

Python port of the Ultrafast Laser TTM Toolbox (MATLAB reference
implementation for doi:10.1007/s11665-026-14738-6).
"""

from .depth_profile import depth_profile_solver
from .single_pulse import single_pulse_visualizer
from .surface_point import surface_point_solver

__version__ = "0.1.0.dev0"

__all__ = [
    "__version__",
    "depth_profile_solver",
    "single_pulse_visualizer",
    "surface_point_solver",
]
