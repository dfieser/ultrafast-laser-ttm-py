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
from typing import Any

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

    log_path = os.path.join(run_dir, "log.txt")
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
            save_results_npz(results, os.path.join(run_dir, "results.npz"))
            summary = summarize_results(results, max_array=200)
            with open(os.path.join(run_dir, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
        except Exception:
            with open(os.path.join(run_dir, "error.json"), "w") as f:
                json.dump({"traceback": traceback.format_exc()}, f, indent=2)
            raise


# In-memory registry of runs started by this server process.
_RUNS: dict[str, dict[str, Any]] = {}

mcp = MCPServer(
    "laserttm",
    instructions="Two-temperature model (TTM) solvers for ultrafast "
                 "pulsed-laser heating of metals. Multi-pulse runs can take "
                 "minutes: prefer start_run + check_run + get_results; use "
                 "run_quick only for small pulse counts.",
)


def _status(run_id: str) -> dict[str, Any]:
    if run_id not in _RUNS:
        raise KeyError(f"Unknown run id '{run_id}'. Known: {sorted(_RUNS)}")
    run = _RUNS[run_id]
    run_dir = run["dir"]
    if os.path.exists(os.path.join(run_dir, "summary.json")):
        status = "done"
    elif os.path.exists(os.path.join(run_dir, "error.json")):
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


def _log_tail(run_dir: str, lines: int) -> str:
    log_path = os.path.join(run_dir, "log.txt")
    if not os.path.exists(log_path):
        return ""
    with open(log_path, encoding="utf-8", errors="replace") as f:
        return "".join(f.readlines()[-lines:])


@mcp.tool()
def list_solvers() -> dict[str, str]:
    """List the available solver ids and what each one computes."""
    from .runtools import SOLVER_DESCRIPTIONS

    return dict(SOLVER_DESCRIPTIONS)


@mcp.tool()
def start_run(solver: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Start a solver in a background worker and return its run id.

    ``solver`` is a registry id from list_solvers. ``config`` is the solver's
    cfg dict with the same field names and defaults as the Python library and
    the MATLAB reference (e.g. material, Pavg, f_rep, spotRadius, tau_FWHM,
    absorbance, Leff, simDuration). Poll with check_run; fetch the outcome
    with get_results.
    """
    from .runtools import get_solver

    get_solver(solver)  # validate the id before spawning
    run_id = uuid.uuid4().hex[:12]
    run_dir = os.path.join(_runs_root(), run_id)
    os.makedirs(run_dir, exist_ok=True)

    proc = mp.get_context("spawn").Process(
        target=_run_job, args=(solver.strip().lower(), config or {}, run_dir),
        daemon=True)
    proc.start()
    _RUNS[run_id] = {"process": proc, "solver": solver,
                     "started": time.time(), "dir": run_dir}
    return _status(run_id)


@mcp.tool()
def check_run(run_id: str, log_lines: int = 15) -> dict[str, Any]:
    """Check a run's status (running / done / failed) and recent log output."""
    out = _status(run_id)
    out["log_tail"] = _log_tail(out["run_dir"], log_lines)
    if out["status"] == "failed":
        err_path = os.path.join(out["run_dir"], "error.json")
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
    with open(os.path.join(out["run_dir"], "summary.json")) as f:
        out["results"] = json.load(f)
    out["results_npz"] = os.path.join(out["run_dir"], "results.npz")
    return out


@mcp.tool()
def get_log(run_id: str, tail_lines: int = 60) -> str:
    """Return the last lines of a run's captured solver output."""
    return _log_tail(_status(run_id)["run_dir"], tail_lines)


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

    Convenience for short runs (small pulse counts). If the run outlives the
    timeout it keeps going in the background and this returns its run id for
    check_run / get_results.
    """
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
