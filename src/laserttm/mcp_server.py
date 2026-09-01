"""MCP server exposing the laserttm solvers as tools.

Requires the optional ``mcp`` dependency (``pip install laserttm[mcp]``).
Start it over stdio with the ``laserttm-mcp`` console script, e.g. for
Claude Code::

    claude mcp add laserttm -- laserttm-mcp

Because multi-pulse TTM runs can take minutes to hours, the server uses a
job pattern instead of blocking tool calls: ``start_run`` launches a solver
in a background worker process and returns a run id; ``check_run`` polls
status and progress; ``get_results`` returns the compact results summary
once the run is done (full arrays are saved next to it as ``results.npz``).
``run_quick`` is a convenience wrapper for short runs that waits up to a
timeout before falling back to the job pattern.

Each run gets its own directory under ``~/.laserttm/runs/<run_id>/``
(override the root with the ``LASERTTM_RUNS_DIR`` environment variable)
containing ``log.txt``, ``summary.json``, ``results.npz``, and the solver's
usual text output.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
import uuid
from collections import deque
from typing import Any

# Files each run directory holds, shared by the worker and the tools.
_LOG_NAME = "log.txt"
_SUMMARY_NAME = "summary.json"
_ERROR_NAME = "error.json"
_RESULTS_NAME = "results.npz"

try:
    from mcp.server.mcpserver import MCPServer  # mcp >= 2
except ImportError:  # pragma: no cover - depends on installed mcp version
    try:
        from mcp.server.fastmcp import FastMCP as MCPServer  # mcp 1.x
    except ImportError as exc:
        raise ImportError(
            "The MCP server requires the optional 'mcp' dependency. "
            "Install it with: pip install laserttm[mcp]"
        ) from exc


def _runs_root() -> str:
    root = os.environ.get("LASERTTM_RUNS_DIR")
    if not root:
        root = os.path.join(os.path.expanduser("~"), ".laserttm", "runs")
    os.makedirs(root, exist_ok=True)
    return root


def _run_job(solver_id: str, cfg: dict, run_dir: str) -> None:
    """Worker-process entry point: run one solver, persist its outputs."""
    os.environ.setdefault("MPLBACKEND", "Agg")  # headless: never open windows
    import sys
    import traceback

    log_path = os.path.join(run_dir, _LOG_NAME)
    with open(log_path, "w", buffering=1, encoding="utf-8") as log:
        sys.stdout = log
        sys.stderr = log
        try:
            from .runtools import get_solver, save_results_npz, summarize_results

            cfg = dict(cfg)
            # Figure windows are meaningless in a server; keep saveFigures
            # working but never block on interactive plots.
            if not cfg.get("saveFigures"):
                cfg.setdefault("makePlots", False)
            cfg.setdefault("outputDir", run_dir)

            results = get_solver(solver_id)(cfg)
            save_results_npz(results, os.path.join(run_dir, _RESULTS_NAME))
            summary = summarize_results(results, max_array=200)
            with open(os.path.join(run_dir, _SUMMARY_NAME), "w") as f:
                json.dump(summary, f, indent=2)
        except Exception:
            with open(os.path.join(run_dir, _ERROR_NAME), "w") as f:
                json.dump({"traceback": traceback.format_exc()}, f, indent=2)
            raise


# In-memory registry of runs started by this server process.
_RUNS: dict[str, dict[str, Any]] = {}

mcp = MCPServer(
    "laserttm",
    instructions=(
        "Two-temperature model (TTM) solvers for ultrafast pulsed-laser "
        "heating of metals.\n\n"
        "All inputs are SI: metres, seconds, hertz, watts. A 100 micron spot "
        "is spotRadius=100e-6, not 100. Keys are case-sensitive and unknown "
        "keys are rejected rather than ignored.\n\n"
        "Protocol: describe_solver to learn a solver's keys, then "
        "validate_config to check a config in milliseconds, then start_run "
        "plus check_run plus get_results. Use run_quick only when "
        "validate_config estimates a short run.\n\n"
        "Choosing a solver: pulse accumulation at one point -> "
        "surface_point; temperature versus depth and the electron lattice "
        "inversion -> depth_profile; melt radius or heat-affected footprint "
        "-> radial_profile; one pulse with depth snapshots -> single_pulse; "
        "inversion statistics across a pulse train -> "
        "inversion_quantifier; a moving beam -> scanning_beam."
    ),
)


def _run_dir(run_id: str) -> str:
    if run_id not in _RUNS:
        raise KeyError(f"Unknown run id '{run_id}'. Known: {sorted(_RUNS)}")
    return _RUNS[run_id]["dir"]


def _status(run_id: str) -> dict[str, Any]:
    run_dir = _run_dir(run_id)
    run = _RUNS[run_id]
    if os.path.exists(os.path.join(run_dir, _SUMMARY_NAME)):
        status = "done"
    elif os.path.exists(os.path.join(run_dir, _ERROR_NAME)):
        status = "failed"
    elif run["process"].is_alive():
        status = "running"
    else:
        status = "failed"  # died without writing results
    return {
        "run_id": run_id,
        "solver": run["solver"],
        "status": status,
        "elapsed_s": round(time.time() - run["started"], 1),
        "run_dir": run_dir,
    }


def _render_problems(solver: str, problems: list[dict[str, Any]]) -> str:
    """One message carrying every problem and its fix.

    An error revealing one of three mistakes would cost three round trips.
    """
    head = (f"{len(problems)} problem(s) with the config for '{solver}'. "
            f"Call describe_solver('{solver}') for every accepted key with "
            "its unit, default and valid range.")
    body = "\n".join(
        f"\n  {i}. {p['message']}\n     {p['suggestion']}"
        for i, p in enumerate(problems, 1))
    return head + "\n" + body


def _log_tail(run_dir: str, lines: int) -> str:
    log_path = os.path.join(run_dir, _LOG_NAME)
    if not os.path.exists(log_path):
        return ""
    with open(log_path, encoding="utf-8", errors="replace") as f:
        return "".join(deque(f, maxlen=lines))


@mcp.tool()
def list_solvers() -> dict[str, dict[str, Any]]:
    """List the solvers with guidance on which one answers which question.

    Keyed by solver id. Each entry says what the solver computes, when to
    use it, when not to, how many config keys it accepts, and a minimal
    working config. Call describe_solver for the full key list with units,
    defaults and valid ranges.
    """
    from . import schema

    return {row["id"]: {k: v for k, v in row.items() if k != "id"}
            for row in schema.list_solvers()}


@mcp.tool()
def describe_solver(solver: str, section: str = "all") -> dict[str, Any]:
    """Everything a caller needs to drive one solver correctly.

    Returns every accepted config key with its type, unit, default, valid
    range and meaning, plus the files the solver writes and runnable example
    configs. Call this before composing a config for a solver you have not
    used yet, rather than guessing key names or reading source.

    section limits the reply to 'inputs', 'files' or 'examples'.
    """
    from . import schema

    return schema.describe_solver(solver, section)


@mcp.tool()
def validate_config(solver: str,
                    config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Check a config without running anything. Takes milliseconds.

    Returns ok, a list of errors and warnings each with a machine-readable
    code and a suggested fix, the resolved config with defaults merged, and
    an estimate of pulse count and runtime.

    Catches the failures that otherwise cost a full run: a misspelled or
    wrong-case key, a key that belongs to a different solver, a unit slip
    such as spotRadius=100 meaning microns, and a config far more expensive
    than intended. Always call this before start_run when composing a config
    by hand.
    """
    from . import schema

    return schema.validate_config(solver, config)


@mcp.tool()
def list_materials() -> list[dict[str, Any]]:
    """Material presets with their property values, units and solver support.

    Gold carries only the lattice-family properties, so the depth-resolved
    solvers reject it. Any single property can be overridden per run without
    switching to material='custom'.
    """
    from . import schema

    return schema.materials_table()


@mcp.tool()
def start_run(solver: str, config: dict[str, Any] | None = None,
              case_tag: str = "") -> dict[str, Any]:
    """Start a solver in a background worker and return its run id.

    solver is an id from list_solvers. config is that solver's cfg dict, whose
    keys are described by describe_solver. case_tag prefixes the output
    filenames, which is how a sweep keeps its runs apart.

    The config is validated first, so a typo raises here instead of producing
    a plausible-looking run on default parameters. The reply carries the pulse
    count and runtime estimate alongside the run id. Poll with check_run and
    fetch the outcome with get_results.
    """
    from . import schema
    from .runtools import get_solver

    get_solver(solver)  # validate the id before spawning
    report = schema.validate_config(solver, config)
    if not report["ok"]:
        raise ValueError(_render_problems(solver, report["errors"]))

    config = dict(config or {})
    if case_tag:
        config["caseTag"] = case_tag

    run_id = uuid.uuid4().hex[:12]
    run_dir = os.path.join(_runs_root(), run_id)
    os.makedirs(run_dir, exist_ok=True)

    proc = mp.get_context("spawn").Process(
        target=_run_job, args=(solver.strip().lower(), config or {}, run_dir),
        daemon=True)
    proc.start()
    _RUNS[run_id] = {"process": proc, "solver": solver,
                     "started": time.time(), "dir": run_dir}
    out = _status(run_id)
    out["estimate"] = report["estimate"]
    if report["warnings"]:
        out["warnings"] = report["warnings"]
    return out


@mcp.tool()
def check_run(run_id: str, log_lines: int = 15) -> dict[str, Any]:
    """Check a run's status (running / done / failed) and recent log output."""
    out = _status(run_id)
    out["log_tail"] = _log_tail(out["run_dir"], log_lines)
    if out["status"] == "failed":
        err_path = os.path.join(out["run_dir"], _ERROR_NAME)
        if os.path.exists(err_path):
            with open(err_path) as f:
                out["error"] = json.load(f).get("traceback", "")
    return out


@mcp.tool()
def get_results(run_id: str) -> dict[str, Any]:
    """Return a finished run's results summary.

    Scalars and small arrays are inlined; large arrays are summarized with
    shape, range, and samples. The full arrays are on disk at
    ``<run_dir>/results.npz`` and the solver's own text output is in the
    same directory.
    """
    out = _status(run_id)
    if out["status"] != "done":
        return out
    with open(os.path.join(out["run_dir"], _SUMMARY_NAME)) as f:
        out["results"] = json.load(f)
    out["results_npz"] = os.path.join(out["run_dir"], _RESULTS_NAME)
    return out


@mcp.tool()
def get_log(run_id: str, tail_lines: int = 60) -> str:
    """Return the last lines of a run's captured solver output."""
    return _log_tail(_run_dir(run_id), tail_lines)


@mcp.tool()
def cancel_run(run_id: str) -> dict[str, Any]:
    """Terminate a running job. Its partial outputs stay in the run dir."""
    run = _RUNS.get(run_id)
    if run is None:
        raise KeyError(f"Unknown run id '{run_id}'.")
    if run["process"].is_alive():
        run["process"].terminate()
        run["process"].join(timeout=10)
    return _status(run_id)


@mcp.tool()
def run_quick(solver: str, config: dict[str, Any] | None = None,
              timeout_s: float = 60) -> dict[str, Any]:
    """Run a solver and wait up to timeout_s for it to finish.

    Convenience for short runs. When the estimate already exceeds the
    timeout this refuses immediately and points at start_run, rather than
    burning the timeout to return a half-answer. If a run outlives the
    timeout anyway it keeps going in the background and this returns its run
    id for check_run and get_results.
    """
    from . import schema

    estimate = schema.estimate_run(solver, config)
    if estimate["estRuntime_s"] > timeout_s:
        return {
            "ok": False,
            "reason": f"This config is {estimate['nPulses']} pulses, very "
                      f"roughly {estimate['estRuntime_s']:g} s, which "
                      f"exceeds the {timeout_s:g} s timeout.",
            "estimate": estimate,
            "use": "start_run",
        }

    started = start_run(solver, config)
    run_id = started["run_id"]
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _status(run_id)["status"] != "running":
            break
        time.sleep(0.25)
    return get_results(run_id)


def main() -> None:
    """Console-script entry point: serve over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
