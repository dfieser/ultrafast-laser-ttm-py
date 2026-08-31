"""Shared plumbing for the CLI and MCP server: solver registry + serialization.

The registry maps stable solver ids (the ``solverId`` field of the v1 result
contract) to their entry-point callables. Serialization turns a solver's
results dict — which mixes scalars, strings, and NumPy arrays — into
JSON-safe structures for machine consumers.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

import numpy as np

# One registry row per solver: (module, attribute, description). Solver
# modules pull in numba and scipy, so the callables are resolved lazily:
# listing solvers must stay import-free to keep the CLI and MCP server
# startup fast.
_REGISTRY: dict[str, tuple[str, str, str]] = {
    "surface_point": (
        "surface_point", "surface_point_solver",
        ("0D surface Te/Tl with inter-pulse depth diffusion "
         "(fastest accumulation studies)")),
    "depth_profile": (
        "depth_profile", "depth_profile_solver",
        ("1D depth-resolved Te(z,t), Tl(z,t) with stiff BDF "
         "integration (main multi-pulse workflow)")),
    "radial_profile": (
        "radial_profile", "radial_profile_solver",
        ("radial surface temperature under a Gaussian spot "
         "(melt-radius and footprint studies)")),
    "single_pulse": (
        "single_pulse", "single_pulse_visualizer",
        "one pulse with depth snapshots at chosen delays"),
    "inversion_quantifier": (
        "inversion_quantifier", "inversion_quantifier",
        "per-pulse Tl>Te inversion statistics"),
    "scanning_beam": (
        "scanning_beam", "scanning_beam_solver",
        "2D surface temperature under a moving beam"),
}

SOLVER_DESCRIPTIONS: dict[str, str] = {
    sid: row[2] for sid, row in _REGISTRY.items()
}


def get_solver(solver_id: str) -> Callable[[dict | None], dict]:
    """Return the solver callable for a registry id (case-insensitive)."""
    key = solver_id.strip().lower()
    if key not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown solver '{solver_id}'. Known solvers: {known}")
    module_name, attr, _ = _REGISTRY[key]
    return getattr(import_module(f".{module_name}", __package__), attr)


def __getattr__(name: str) -> Any:
    # SOLVERS stays available for callers that want the full id->callable
    # map; building it imports every solver, so it is resolved on demand.
    if name == "SOLVERS":
        return {sid: get_solver(sid) for sid in _REGISTRY}
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def to_jsonable(value: Any, max_array: int = 10000) -> Any:
    """Recursively convert a results value into JSON-serializable types.

    NumPy scalars become Python scalars; arrays up to ``max_array`` elements
    become lists, larger ones a shape/dtype summary with head and tail
    samples so payloads stay bounded.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size <= max_array:
            return value.tolist()
        return {
            "_type": "ndarray-summary",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "head": value.ravel()[:20].tolist(),
            "tail": value.ravel()[-20:].tolist(),
            "min": float(np.nanmin(value)),
            "max": float(np.nanmax(value)),
        }
    if isinstance(value, dict):
        return {str(k): to_jsonable(v, max_array) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v, max_array) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def summarize_results(results: dict, max_array: int = 200) -> dict:
    """Compact JSON-safe view of a results dict (large arrays summarized)."""
    return to_jsonable(results, max_array=max_array)


def save_results_npz(results: dict, path: str) -> str:
    """Save the array-valued fields of a results dict to a ``.npz`` file."""
    arrays = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            arrays[k] = v
        elif isinstance(v, (int, float, np.generic)):
            arrays[k] = np.asarray(v)
    np.savez(path, **arrays)
    return path
